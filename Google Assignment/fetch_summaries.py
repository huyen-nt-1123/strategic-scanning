"""Fetch the real address and the sub-header of the articles collected by google_ad_news.py.

Google News gives a redirect link and no summary. For every article this
script finds the publisher's own address, opens the page and reads the short
description the publisher wrote for it (the text shown under the headline in
search results and link previews).

The results go to their own CSV file, one row per link. The script can be
stopped and started again: it only fetches the links that are not done yet.
Some publishers block scripts or sit behind a paywall, so not every article
gets a description.

Examples:
    python fetch_summaries.py
    python fetch_summaries.py --in data/google_ad_news_ai.csv
    python fetch_summaries.py --retry
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlparse

import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
}
# From Europe, Google News answers with a cookie-consent page instead of the
# article. This cookie says the choice was already made.
COOKIES = {"SOCS": "CAISHAgBEhJnd3NfMjAyMzA4MTAtMF9SQzIaAmVuIAEaBgiAo_CmBg"}
DECODE_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
# Where publishers put the description, best first
DESCRIPTION_TAGS = ["og:description", "description", "twitter:description"]


class MetaTags(HTMLParser):
    """Collect the <meta> tags of a page as {name: content}."""

    def __init__(self):
        super().__init__()
        self.found = {}

    def handle_starttag(self, tag, attrs):
        if tag != "meta":
            return
        attrs = dict(attrs)
        name = (attrs.get("property") or attrs.get("name") or "").lower()
        content = " ".join((attrs.get("content") or "").split())
        if name and content and name not in self.found:
            self.found[name] = content


def real_url(link):
    """Turn a Google News redirect link into the publisher's address."""
    article_id = urlparse(link).path.split("/")[-1]
    page = requests.get(
        f"https://news.google.com/rss/articles/{article_id}", headers=HEADERS, cookies=COOKIES, timeout=20
    )
    page.raise_for_status()
    # The page carries a signature and a timestamp that Google asks for
    # before it gives the address away
    signature = re.search(r'data-n-a-sg="([^"]+)"', page.text)
    timestamp = re.search(r'data-n-a-ts="([^"]+)"', page.text)
    if not (signature and timestamp):
        raise ValueError("no signature on the Google News page")

    request = (
        '["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],'
        f'"X","X",1,[1,1,1],1,1,null,0,0,null,0],"{article_id}",{timestamp.group(1)},"{signature.group(1)}"]'
    )
    answer = requests.post(
        DECODE_URL,
        headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
        cookies=COOKIES,
        data=f"f.req={quote(json.dumps([[['Fbv4je', request]]]))}",
        timeout=20,
    )
    answer.raise_for_status()
    return json.loads(json.loads(answer.text.split("\n\n")[1])[0][2])[1]


def page_description(url):
    """Open an article and return the description from its <meta> tags ("" if none)."""
    page = requests.get(url, headers=HEADERS, timeout=20)
    page.raise_for_status()
    if "charset" not in page.headers.get("Content-Type", "").lower():
        page.encoding = "utf-8"
    tags = MetaTags()
    tags.feed(page.text)
    for name in DESCRIPTION_TAGS:
        if name in tags.found:
            return tags.found[name]
    return ""


def fetch(link):
    """Return one result row for a link. Never raises: the problem goes in "status"."""
    row = {"link": link, "url": "", "description": "", "status": "ok"}
    try:
        row["url"] = real_url(link)
    except (requests.RequestException, ValueError, IndexError, KeyError, TypeError) as error:
        row["status"] = f"link not resolved ({type(error).__name__})"
        return row
    try:
        row["description"] = page_description(row["url"])
        if not row["description"]:
            row["status"] = "no description on the page"
    except requests.HTTPError as error:
        row["status"] = f"page blocked (HTTP {error.response.status_code})"
    except requests.RequestException as error:
        row["status"] = f"page not loaded ({type(error).__name__})"
    time.sleep(1)  # be polite
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="source", default="data/google_ad_news.csv", help="CSV file with a link column")
    parser.add_argument("--out", default="data/article_summaries.csv", help="where the results are saved")
    parser.add_argument("--retry", action="store_true", help="try again the links that failed last time")
    parser.add_argument("--workers", type=int, default=4, help="how many articles to fetch at the same time (default: 4)")
    args = parser.parse_args()

    links = pd.read_csv(args.source)["link"].drop_duplicates().tolist()
    out = Path(args.out)
    done = pd.read_csv(out, keep_default_na=False) if out.exists() else pd.DataFrame(columns=["link", "url", "description", "status"])
    if args.retry:
        done = done[done["status"] == "ok"]
    todo = [link for link in links if link not in set(done["link"])]
    print(f"{len(links)} links in {args.source}: {len(links) - len(todo)} already done, {len(todo)} to fetch\n")

    rows = done.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count, row in enumerate(pool.map(fetch, todo), start=1):
            rows.append(row)
            if count % 25 == 0 or count == len(todo):
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{count}/{len(todo)} fetched")

    results = pd.DataFrame(rows)
    results = results[results["link"].isin(links)]
    print(f"\nSaved {len(rows)} rows to {out}")
    print(f"\nOf the {len(links)} links in {args.source}:")
    print(results["status"].str.replace(r" \(.*", "", regex=True).value_counts().to_string())


if __name__ == "__main__":
    main()
