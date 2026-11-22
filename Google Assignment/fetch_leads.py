"""Fetch the opening paragraphs of the articles kept by clean_news.py.

The sub-header of an article is often a few words. This script opens every
article, reads its paragraphs and keeps the first ones, word for word: the
"lead" of the article. When the publisher blocks the script, it tries the copy
kept by the Internet Archive.

The results go to their own CSV file, one row per link. The script can be
stopped and started again: it only fetches the links that are not done yet.
Paywalls and bot protection mean that not every article gets a lead.

Examples:
    python fetch_leads.py
    python fetch_leads.py --in data/google_ad_news_ai_clean.csv
    python fetch_leads.py --retry
"""

import argparse
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}
# "2id_" asks the Internet Archive for its latest copy of a page, as it was saved
ARCHIVE_URL = "https://web.archive.org/web/2id_/{url}"
# Parts of a page that are never the article
SKIPPED_TAGS = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "figure", "figcaption", "button"}
# Words too common to tell if a paragraph is about the headline
FILLER = set(
    "the a an and or of to in on for with from by at as is are was were be been its it this that than how why what "
    "when will can could may not new says say said more over after about into your you our their has have had but "
    "his her they them who which google alphabet".split()
)
# Paragraphs that belong to the page, not to the article
PAGE_FURNITURE = re.compile(
    r"subscribe|sign up|sign in|log in|newsletter|cookie|^advertisement|follow us|all rights reserved|click here"
    r"|read more|getty images|listen to this|min read|share this|©|free account|javascript|ad-?blocker|keep reading"
    r"|unlock|already a (member|subscriber)|terms of (use|service)|privacy policy|disclosure|editor.s note"
    r"|opinions expressed|purchase through links|earn a (small )?commission|affiliate|screenshot|AP Photo"
    r"|via Getty|round-up of|^hello from|confirmation link|spam folder|^by ",
    re.I,
)
AUTHOR_BIOGRAPHY = re.compile(
    r"^.{0,80}\b(is|was) (an?|the) .{0,80}\b(writer|reporter|editor|journalist|correspondent|columnist|contributor|analyst|author|founder|professor)\b",
    re.I,
)
MIN_WORDS, ENOUGH_WORDS, MAX_WORDS = 15, 60, 130
# The Internet Archive refuses visitors that ask too fast: one request at a time, with a pause
ARCHIVE_TURN = threading.Lock()
ARCHIVE_PAUSE = 4
# Set when the Internet Archive keeps refusing: the script stops asking it until the next run
ARCHIVE_REFUSING = threading.Event()


class Paragraphs(HTMLParser):
    """Collect the text of the <p> tags of a page, and the name of its author."""

    def __init__(self):
        super().__init__()
        self.found = []
        self.author = ""
        self.skipping = 0  # how many skipped tags we are inside
        self.current = None  # the pieces of the paragraph being read

    def handle_starttag(self, tag, attrs):
        if tag in SKIPPED_TAGS:
            self.skipping += 1
        elif tag == "p" and not self.skipping:
            self.current = []
        elif tag == "meta":
            attrs = dict(attrs)
            if (attrs.get("name") or attrs.get("property") or "").lower() in ("author", "article:author") and not self.author:
                self.author = " ".join((attrs.get("content") or "").split())

    def handle_endtag(self, tag):
        if tag in SKIPPED_TAGS:
            self.skipping = max(0, self.skipping - 1)
        elif tag == "p" and self.current is not None:
            text = " ".join("".join(self.current).split())
            if text:
                self.found.append(text)
            self.current = None

    def handle_data(self, data):
        if self.current is not None and not self.skipping:
            self.current.append(data)


def story_words(text):
    """The words of a text that say what it is about."""
    return {word for word in re.findall(r"[a-z0-9$%']+", text.lower()) if len(word) > 2 and word not in FILLER}


def lead_of(paragraphs, author, title):
    """Return the opening paragraphs of an article ("" if they are not about the headline)."""
    first_name = author.split()[0] if author else ""
    article = []
    for text in paragraphs:
        words = len(text.split())
        if words < MIN_WORDS or PAGE_FURNITURE.search(text) or AUTHOR_BIOGRAPHY.search(text):
            continue
        if author and (text.startswith(author) or (len(first_name) > 3 and re.match(re.escape(first_name) + "['’]s ", text))):
            continue
        if words < 25 and not re.search(r"[.!?…”\"]$", text):  # a caption or a menu, not a sentence
            continue
        article.append(text)

    # Start with the opening when it is about the headline; if not, with the
    # first of the next paragraphs that is
    headline = story_words(title)
    if len(story_words(" ".join(article[:2])) & headline) >= 2:
        start = 0
    else:
        start = next((i for i, text in enumerate(article[:6]) if len(story_words(text) & headline) >= 2), None)
        if start is None:
            return ""

    lead = []
    for text in article[start : start + 5]:
        if lead and (text[:50] == lead[-1][:50] or text in " ".join(lead)):  # the page repeats its first paragraph
            if len(text) > len(lead[-1]) and text[:50] == lead[-1][:50]:
                lead[-1] = text
            continue
        lead.append(text)
        if sum(len(part.split()) for part in lead) >= ENOUGH_WORDS:
            break
    lead = " ".join(lead)
    if len(lead.split()) > MAX_WORDS:  # cut at the end of a sentence
        cut = " ".join(lead.split()[:MAX_WORDS])
        end = max(cut.rfind(". "), cut.rfind(".” "), cut.rfind("? "))
        lead = cut[: end + 1] if end > 200 else cut + "…"
    return lead


def open_page(address):
    """Open a page. Archive pages wait for their turn, and try once more when told to slow down."""
    if "web.archive.org" not in address:
        return requests.get(address, headers=HEADERS, timeout=30)
    with ARCHIVE_TURN:
        page = requests.get(address, headers=HEADERS, timeout=60)
        if page.status_code == 429:
            time.sleep(30)
            page = requests.get(address, headers=HEADERS, timeout=60)
            if page.status_code == 429:
                ARCHIVE_REFUSING.set()
        time.sleep(ARCHIVE_PAUSE)
        return page


def page_lead(address, title, url):
    """Open a page and return the lead of the article in it ("" if none)."""
    page = open_page(address)
    page.raise_for_status()
    # A link that lands on the home page did not give the article
    landed = page.url.split("id_/", 1)[-1] if "web.archive.org" in page.url else page.url
    if urlparse(landed).path.strip("/") == "" and urlparse(url).path.strip("/") != "":
        return ""
    if "charset" not in page.headers.get("Content-Type", "").lower():
        page.encoding = "utf-8"
    parser = Paragraphs()
    parser.feed(page.text)
    return lead_of(parser.found, parser.author.split(",")[0].strip(), title)


def fetch(article):
    """Return one result row for an article. Never raises: the problem goes in "status"."""
    row = {"link": article["link"], "url": article["url"], "lead": "", "lead_source": "", "status": ""}
    places = [("page", article["url"]), ("archive", ARCHIVE_URL.format(url=article["url"]))]
    for source, address in places:
        if source == "archive" and ARCHIVE_REFUSING.is_set():
            row["status"] += "; archive not asked (it was refusing requests)"
            break
        try:
            row["lead"] = page_lead(address, article["title"], article["url"])
            problem = "" if row["lead"] else "no paragraph about the headline"
        except requests.HTTPError as error:
            problem = f"blocked (HTTP {error.response.status_code})"
        except requests.RequestException as error:
            problem = f"not loaded ({type(error).__name__})"
        if row["lead"]:
            row["lead_source"], row["status"] = source, "ok"
            break
        row["status"] = f"{row['status']}; {source} {problem}".strip("; ")
        time.sleep(1)  # be polite
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="source", default="data/google_ad_news_clean.csv", help="CSV file with link, url and title columns")
    parser.add_argument("--out", default="data/article_leads.csv", help="where the results are saved")
    parser.add_argument("--retry", action="store_true", help="try again the links that failed last time")
    parser.add_argument("--workers", type=int, default=4, help="how many articles to fetch at the same time (default: 4)")
    args = parser.parse_args()

    articles = pd.read_csv(args.source, keep_default_na=False).drop_duplicates(subset="link")
    out = Path(args.out)
    columns = ["link", "url", "lead", "lead_source", "status"]
    done = pd.read_csv(out, keep_default_na=False) if out.exists() else pd.DataFrame(columns=columns)
    if args.retry:
        done = done[done["lead"] != ""]
    todo = articles[~articles["link"].isin(set(done["link"]))][["link", "url", "title"]].to_dict("records")
    print(f"{len(articles)} articles in {args.source}: {len(articles) - len(todo)} already done, {len(todo)} to fetch\n")

    rows = done.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, row in enumerate(pool.map(fetch, todo), start=1):
            rows.append(row)
            if count % 25 == 0 or count == len(todo):
                pd.DataFrame(rows, columns=columns).to_csv(out, index=False)
                print(f"{count}/{len(todo)} fetched")

    results = pd.DataFrame(rows, columns=columns)
    results = results[results["link"].isin(articles["link"])]
    print(f"\nSaved {len(rows)} rows to {out}")
    print(f"\nOf the {len(articles)} articles in {args.source}:")
    print(results["lead_source"].replace("", "no lead").value_counts().to_string())
    if ARCHIVE_REFUSING.is_set():
        print("\nThe Internet Archive refused requests: run again later with --retry to try the missing ones there")


if __name__ == "__main__":
    main()
