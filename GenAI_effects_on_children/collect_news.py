"""Collect news articles about generative AI and young children from Google News.

Google News returns at most ~100 results per search, so the scan period is
split into short time windows (30 days by default) and each query is run once
per window, with "after:" and "before:" limits. This is the same method as
../Google Assignment/google_ad_news.py.

Rules:

- Queries: the "news" rows of config/queries.csv (or the file given with
  --queries), or one query given with --query.
- Search language and country: English, United States by default (--hl en-US,
  --gl US, --ceid US:en).
- Titles: Google adds " - Outlet name" at the end; it is removed.
- Rejected items: each RSS item is checked in this order and rejected with the
  first reason that applies. Rejected items are not dropped silently: they go
  to the rejected file (data/raw/news_rejected.csv by default, next to --out),
  with the common columns plus a `reason` column, and are counted per window
  and in the final summary.
    - "no link": the item has no link, so it has no key for its id (every
      such item would get the same id).
    - "unreadable pubDate": the date is missing or not a date. Only that item
      is rejected; the rest of the window is kept.
    - "date outside window: <date>": the item is dated more than
      --date-tolerance-days days (default 2) outside its window, in case
      Google ignored the date limits. The tolerance is a judgement call (time
      zones, late indexing).
  A rejected row's id is a hash of the query, the window, the link (or the
  title and raw date when there is no link), so a rerun of the same window
  adds no duplicate rejected rows.
- text column, fixed format: "Outlet: <name>." followed, when the RSS
  description adds something to the title, by " <that text>". The RSS
  description is mostly the title and outlet name in HTML; it is cleaned with
  common.clean_text. Later steps (clean.py) strip the "Outlet: <name>." prefix.
- Outlet counts in the summary: an outlet name can itself contain ". "
  ("U.S. News & World Report"), so the outlet is read back from the text by
  the longest outlet name seen in this run's RSS answers that the text starts
  with; for other rows, the whole text after "Outlet: " minus its final dot.
- url: the Google News link, kept as it is (Google's redirect is not
  resolved). The link is also the key of the item's id.
- Restartable: every finished (query, window) pair is written to a progress
  file (data/raw/news_progress.csv by default) and skipped on the next run,
  except windows ending within the last --refresh-days days, which are always
  fetched again so the weekly run picks up new articles. Rows are saved after
  every window, so stopping midway loses nothing. A failed window is not
  marked as done, so the next run tries it again.
- Every failed request is printed and logged in data/run_log.csv (or the file
  given with --log). Printed lines are flushed at once, so progress shows
  during a long run.

Examples:
    python collect_news.py
    python collect_news.py --query '"AI toys" kids' --window-days 14
    python collect_news.py --refresh-days 0 --sleep 2 --date-tolerance-days 0
    python collect_news.py --hl en-GB --gl GB --ceid GB:en
    # test run that writes nothing in data/:
    python collect_news.py --since 2026-10-01 --out /tmp/news_test/news.csv \\
        --progress /tmp/news_test/news_progress.csv --log /tmp/news_test/run_log.csv
"""

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

import common
from common import clean_text, get, load_queries, load_weights, log, make_rows, save_new

COLLECTOR = "news"  # name used in config/queries.csv and in the run log
SOURCE = "Google News"
SOURCE_TYPE = "news"
RSS_URL = "https://news.google.com/rss/search"
# English results as seen from the United States, as in the Google assignment.
DEFAULT_HL, DEFAULT_GL, DEFAULT_CEID = "en-US", "US", "US:en"
LANGUAGE_PARAMS = {"hl": DEFAULT_HL, "gl": DEFAULT_GL, "ceid": DEFAULT_CEID}
# Public release of ChatGPT (30 November 2022): the start of the scan period.
DEFAULT_SINCE = "2022-11-01"
# Google News caps each search at ~100 results; 30-day windows keep most
# windows under that cap.
DEFAULT_WINDOW_DAYS = 30
# Items dated this many days outside their window are still accepted
# (time zones, late indexing); anything further out is rejected. Judgement call.
DATE_TOLERANCE_DAYS = 2
# Windows ending this recently are fetched again on every run: Google keeps
# adding articles to recent windows. 35 covers the latest 30-day window.
DEFAULT_REFRESH_DAYS = 35
# Seconds to wait between two requests, to be polite to Google News.
DEFAULT_SLEEP = 1.0
PROGRESS_COLUMNS = ["query", "window_start", "window_end", "n_items", "time"]
# Reasons an item is rejected, in the order they are checked.
NO_LINK, NO_DATE, OUTSIDE = "no link", "unreadable pubDate", "date outside window"
REASONS = [NO_LINK, NO_DATE, OUTSIDE]


def time_windows(since, until, window_days):
    """Yield (start, end) dates from `since` to `until`, oldest first."""
    start = since
    while start < until:
        end = min(start + timedelta(days=window_days), until)
        yield start, end
        start = end


def load_progress(path):
    """Return the set of (query, window_start, window_end) already collected."""
    if not path.exists():
        return set()
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return set(zip(table["query"], table["window_start"], table["window_end"]))


def save_progress(path, query, start, end, n_items):
    """Append one finished (query, window) pair to the progress file."""
    row = {
        "query": query,
        "window_start": str(start),
        "window_end": str(end),
        "n_items": n_items,
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=PROGRESS_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)


def make_text(outlet, title, description):
    """Build the text column: the outlet name, then whatever the description adds to the title."""
    rest = clean_text(description)
    if outlet and rest.endswith(outlet):
        rest = rest[: -len(outlet)].strip()
    if rest == title:
        rest = ""
    return f"Outlet: {outlet}." + (f" {rest}" if rest else "")


def read_date(raw):
    """Return the date of an RSS pubDate string, or None if it is missing or unreadable."""
    try:
        return parsedate_to_datetime(raw).date()
    except (TypeError, ValueError, IndexError):
        return None


def parse_items(content, query, start, end, tolerance_days=DATE_TOLERANCE_DAYS):
    """Split one RSS answer into (kept, rejected) item dicts; rejected ones have a `reason`."""
    kept, rejected = [], []
    tolerance = timedelta(days=tolerance_days)
    for item in ET.fromstring(content).iter("item"):
        outlet = item.findtext("source", default="").strip()
        title = clean_text(item.findtext("title", default=""))
        # Google News titles look like "Headline - Outlet name"
        if outlet and title.endswith(f" - {outlet}"):
            title = title[: -len(outlet) - 3]
        link = item.findtext("link", default="").strip()
        raw_date = item.findtext("pubDate", default="").strip()
        published = read_date(raw_date)
        row = {
            "key": link,
            "date": published.isoformat() if published else "",
            "title": title,
            "text": make_text(outlet, title, item.findtext("description", default="")),
            "url": link,
            "query": query,
            "outlet": outlet,
        }
        if not link:
            reason = NO_LINK
        elif published is None:
            reason = NO_DATE
        elif not start - tolerance <= published <= end + tolerance:
            reason = f"{OUTSIDE}: {published}"
        else:
            kept.append(row)
            continue
        # one id per (query, window, item), so a rerun adds no duplicate rejected rows
        row["key"] = f"rejected|{query}|{start}|{end}|{link or title + '|' + raw_date}"
        row["reason"] = reason
        rejected.append(row)
    return kept, rejected


def fetch_rss(query, start, end, language=LANGUAGE_PARAMS):
    """Run one search limited to [start, end); return the raw RSS answer, or None if it failed."""
    params = {"q": f"after:{start} before:{end} {query}", **language}
    response = get(RSS_URL, COLLECTOR, params=params)
    return None if response is None else response.content


def fetch_window_all(query, start, end, language=LANGUAGE_PARAMS, tolerance_days=DATE_TOLERANCE_DAYS):
    """Fetch and parse one window; return (kept, rejected), or None if the request or the XML failed."""
    content = fetch_rss(query, start, end, language)
    if content is None:
        return None
    try:
        return parse_items(content, query, start, end, tolerance_days)
    except ET.ParseError as error:
        log(COLLECTOR, "failed", f"{query} {start}->{end}", f"could not read the RSS answer: {error}")
        print(f"  FAILED to read the RSS answer: {error}")
        return None


def fetch_window(query, start, end, language=LANGUAGE_PARAMS, tolerance_days=DATE_TOLERANCE_DAYS):
    """Return only the kept items of one window, or None if it failed (used by check_queries.py)."""
    result = fetch_window_all(query, start, end, language, tolerance_days)
    return None if result is None else result[0]


def rejected_rows(rejected, weights):
    """Turn rejected item dicts into rows with the common columns plus `reason`."""
    rows = make_rows(rejected, SOURCE, SOURCE_TYPE, weights)
    rows["reason"] = [item["reason"] for item in rejected]
    return rows


def reason_counts(rejected):
    """Count rejected items per reason (all dates outside the window count as one reason)."""
    counts = dict.fromkeys(REASONS, 0)
    for item in rejected:
        counts[item["reason"].split(":")[0]] += 1
    return counts


def outlet_of(text, known=()):
    """Return the outlet name of a text column written by make_text ("Outlet: <name>." ...)."""
    text = str(text)
    if not text.startswith("Outlet: "):
        return ""
    # an outlet name may contain ". ", so prefer the longest name seen in this run
    for name in sorted(known, key=len, reverse=True):
        prefix = f"Outlet: {name}."
        if name and (text == prefix or text.startswith(prefix + " ")):
            return name
    return text[len("Outlet: "):].removesuffix(".")


def print_summary(out, rejected_path, n_new, n_failed, run_reasons, known_outlets):
    """Print rows per year and the top 10 outlets of the whole raw file, and the rejected counts."""
    table = pd.read_csv(out, dtype=str, keep_default_na=False)
    print(f"\nRows new: {n_new}   rows total: {len(table)}   failed windows: {n_failed}")
    print(f"Saved to {out}")
    print("\nItems rejected in this run, per reason:")
    for reason, count in run_reasons.items():
        print(f"  {reason}: {count}")
    if rejected_path.exists():
        print(f"Rows in the rejected file: {len(pd.read_csv(rejected_path, usecols=['id']))} ({rejected_path})")
    if table.empty:
        return
    print("\nItems per year:")
    print(table["date"].str[:4].value_counts().sort_index().to_string())
    print("\nTop 10 outlets:")
    print(table["text"].map(lambda text: outlet_of(text, known_outlets)).value_counts().head(10).to_string())


def main():
    """Read the options, collect every (query, window) pair not yet done, save rows and rejects, print a summary."""
    # flush every printed line at once, so progress shows during a long run
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", help="use this single search query instead of the 'news' rows of the queries file")
    parser.add_argument("--since", default=DEFAULT_SINCE,
                        help=f"start of the scan period, YYYY-MM-DD (default: {DEFAULT_SINCE}, the month ChatGPT was released)")
    parser.add_argument("--until", default=str(date.today()), help="end of the scan period, YYYY-MM-DD (default: today)")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
                        help=f"size of each search window in days (default: {DEFAULT_WINDOW_DAYS})")
    parser.add_argument("--refresh-days", type=int, default=DEFAULT_REFRESH_DAYS,
                        help=f"always re-fetch windows ending within this many days (default: {DEFAULT_REFRESH_DAYS})")
    parser.add_argument("--date-tolerance-days", type=int, default=DATE_TOLERANCE_DAYS,
                        help=f"accept items dated up to this many days outside their window; a judgement call "
                             f"(default: {DATE_TOLERANCE_DAYS})")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                        help=f"seconds to wait between requests (default: {DEFAULT_SLEEP})")
    parser.add_argument("--hl", default=DEFAULT_HL, help=f"Google News interface language (default: {DEFAULT_HL})")
    parser.add_argument("--gl", default=DEFAULT_GL, help=f"Google News country (default: {DEFAULT_GL})")
    parser.add_argument("--ceid", default=DEFAULT_CEID, help=f"Google News edition, country:language (default: {DEFAULT_CEID})")
    parser.add_argument("--out", default="data/raw/news.csv", help="raw output CSV (default: data/raw/news.csv)")
    parser.add_argument("--progress", help="progress file (default: news_progress.csv next to --out)")
    parser.add_argument("--rejected", help="rejected items file (default: news_rejected.csv next to --out)")
    common.add_config_options(parser)
    args = parser.parse_args()
    common.use_config_options(args)

    out = Path(args.out)
    progress = Path(args.progress) if args.progress else out.parent / "news_progress.csv"
    rejected_path = Path(args.rejected) if args.rejected else out.parent / "news_rejected.csv"
    language = {"hl": args.hl, "gl": args.gl, "ceid": args.ceid}
    since, until = date.fromisoformat(args.since), date.fromisoformat(args.until)
    refresh_after = date.today() - timedelta(days=args.refresh_days)
    queries = [args.query] if args.query else load_queries(COLLECTOR)
    weights = load_weights()
    done = load_progress(progress)

    print("Queries:")
    for query in queries:
        print(f"  {query}")
    print()

    n_new, n_failed, n_skipped = 0, 0, 0
    run_reasons = dict.fromkeys(REASONS, 0)
    known_outlets = set()
    for query in queries:
        print(f"Query: {query}")
        for start, end in time_windows(since, until, args.window_days):
            if (query, str(start), str(end)) in done and end < refresh_after:
                n_skipped += 1
                continue
            result = fetch_window_all(query, start, end, language, args.date_tolerance_days)
            time.sleep(args.sleep)
            if result is None:
                n_failed += 1
                print(f"  {start} -> {end}: FAILED")
                continue
            items, rejected = result
            known_outlets.update(item["outlet"] for item in items + rejected)
            added = save_new(make_rows(items, SOURCE, SOURCE_TYPE, weights), out)[0] if items else 0
            if rejected:
                save_new(rejected_rows(rejected, weights), rejected_path)
            save_progress(progress, query, start, end, len(items))
            counts = reason_counts(rejected)
            for reason in REASONS:
                run_reasons[reason] += counts[reason]
            detail = ", ".join(f"{reason}: {n}" for reason, n in counts.items() if n)
            print(f"  {start} -> {end}: {len(items)} items, {added} new, {len(rejected)} rejected"
                  + (f" ({detail})" if detail else ""))
            n_new += added

    print(f"\nWindows skipped (already collected): {n_skipped}")
    n_rejected = sum(run_reasons.values())
    if not out.exists():
        print(f"No items kept. Nothing was saved to {out}. Items rejected: {n_rejected}")
        log(COLLECTOR, "done", message=f"0 new, 0 total, {n_rejected} rejected, {n_failed} failed")
        return
    total = len(pd.read_csv(out, usecols=["id"]))
    log(COLLECTOR, "done", message=f"{n_new} new, {total} total, {n_rejected} rejected, {n_failed} failed")
    print_summary(out, rejected_path, n_new, n_failed, run_reasons, known_outlets)


if __name__ == "__main__":
    main()
