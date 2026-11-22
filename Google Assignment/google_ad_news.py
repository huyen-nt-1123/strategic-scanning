"""Collect news headlines about Google's ad revenue trend from Google News.

Google News returns at most ~100 results per search, so the script splits the
period into small time windows and runs one search per window. The results
are merged, de-duplicated and saved as a CSV file.

Examples:
    python google_ad_news.py
    python google_ad_news.py --ai
    python google_ad_news.py --years 2 --query 'intitle:Google "search market share"'
"""

import argparse
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

RSS_URL = "https://news.google.com/rss/search"
# Google News drops whatever comes after roughly 32 words, including the date
# limits, so the search is split into several short queries instead of one long
# one. intitle:Google keeps only headlines that name Google.
QUERIES = [
    'intitle:Google "ad revenue" search traffic',
    'intitle:Google "search revenue" (traffic OR queries OR usage OR "market share")',
    'intitle:Google search (traffic OR queries OR usage) (ads OR advertising) revenue',
    'intitle:Google "search market share" revenue',
]
AI_QUERY = '(AI OR ChatGPT OR "AI Overviews")'
HEADERS = {"User-Agent": "Mozilla/5.0 (strategic-scanning coursework script)"}


def time_windows(years, window_days):
    """Yield (start, end) dates covering the last `years` years, oldest first."""
    today = date.today()
    start = today - timedelta(days=365 * years)
    while start < today:
        end = min(start + timedelta(days=window_days), today)
        yield start, end
        start = end


def fetch_window(query, start, end):
    """Run one Google News search limited to [start, end) and return its articles."""
    params = {
        "q": f"after:{start} before:{end} {query}",
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    response = requests.get(RSS_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()

    articles = []
    for item in ET.fromstring(response.content).iter("item"):
        source = item.findtext("source", default="")
        title = item.findtext("title", default="")
        # Google News titles look like "Headline - Source name"
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        published = parsedate_to_datetime(item.findtext("pubDate"))
        # Skip anything outside the window, in case Google ignored the date limits
        if not start - timedelta(days=2) <= published.date() <= end + timedelta(days=2):
            continue
        articles.append(
            {
                "published": published,
                "title": title,
                "source": source,
                "link": item.findtext("link", default=""),
                "query": query,
            }
        )
    return articles


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ai", action="store_true", help="only keep articles that also mention AI")
    parser.add_argument("--query", help="use this single search query instead of the default ones")
    parser.add_argument("--years", type=int, default=3, help="how many years back to search (default: 3)")
    parser.add_argument("--window-days", type=int, default=30, help="size of each search window (default: 30)")
    parser.add_argument("--out", help="output CSV path (default: data/google_ad_news[_ai].csv)")
    args = parser.parse_args()

    queries = [args.query] if args.query else QUERIES
    if args.ai:
        queries = [f"{query} {AI_QUERY}" for query in queries]
    out = Path(args.out or f"data/google_ad_news{'_ai' if args.ai else ''}.csv")
    print("Queries:")
    for query in queries:
        print(f"  {query}")
    print()

    articles = []
    for start, end in time_windows(args.years, args.window_days):
        found = []
        for query in queries:
            try:
                found.extend(fetch_window(query, start, end))
            except (requests.RequestException, ET.ParseError) as error:
                print(f"{start} -> {end}: FAILED ({error})")
            time.sleep(1)  # be polite to Google News
        print(f"{start} -> {end}: {len(found)} articles")
        articles.extend(found)

    if not articles:
        print("\nNo articles found. Nothing was saved.")
        return

    df = pd.DataFrame(articles)
    df = df.drop_duplicates(subset="link").drop_duplicates(subset=["title", "source"])
    df = df.sort_values("published").reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\nSaved {len(df)} unique articles to {out}")
    print("\nArticles per year:")
    print(df["published"].dt.year.value_counts().sort_index().to_string())
    print("\nTop sources:")
    print(df["source"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
