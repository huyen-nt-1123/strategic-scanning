"""Collect policy items (institution pages) about GenAI and children from Google News.

Google News RSS is searched with the policy query from config/queries.csv,
restricted to the institution sites listed in config/policy_sites.csv with
the site: operator, one time window at a time.

Rules:

- Query: "<policy query> (site:a OR site:b OR site:c)". Google News drops
  whatever comes after roughly 32 words, so the sites are split into small
  groups (--sites-per-query, a judgement call) and each group is a separate
  search. Groups are cut in the order of config/policy_sites.csv.
- Time windows: from --since (default 2022-11-01, public release of ChatGPT,
  the start of the scan) in fixed windows of --window-days (default 45, since
  policy items are sparse; a judgement call). Windows always have the same
  length, so the same windows are searched on every run.
- Every item that is not kept goes to data/raw/policy_dropped.csv with one
  reason, checked in this order:
  1. "no readable date": the pubDate is missing or malformed.
  2. "dated outside window": dated more than --window-margin-days (default 2,
     a judgement call) outside its window, in case Google ignores the date
     limits.
  3. "no link": the item has no link, so it has no id of its own.
  4. "source not a listed institution site": the <source url="..."> domain is
     not one of the listed sites or a subdomain of one.
  The dropped file has the common columns plus source_url and reason; its
  source is the source name given by Google News, and its id comes from the
  reason and the link (or the title and raw date when there is no link).
- Kept: items whose source domain is one of the listed sites or a subdomain
  of one (www.unicef.org and knowledge.unicef.org both count as unicef.org).
  When several sites match, the longest one wins.
- Columns: the common ones (id, source, source_type, weight, date, title,
  text, url, query). source is the institution name from policy_sites.csv,
  source_type is "policy", the id comes from the Google News link.
- Title: the " - Source name" ending added by Google News is removed.
- Text (fixed format, already used by the rows collected so far, so do not
  change it; clean.py strips the prefix): "Institution: <name>. <description>".
  Google News descriptions usually only repeat the title and the source name;
  in that case the text is just "Institution: <name>.".
- Restartable: each finished (query, sites group, window) is recorded in
  data/raw/policy_progress.csv and skipped on the next run, except windows
  ending within --refresh-days (default 35, a judgement call) of today, which
  are searched again because new items may still appear. Rows are saved
  after every window. The sites group is recorded as its sites joined by
  spaces, so new sites must be added at the END of config/policy_sites.csv:
  a site added in the middle shifts every later group, and those groups are
  then searched again from --since.
- A window that returns 100 items (the Google News maximum) is reported, since
  items may be missing there: use a smaller --window-days if it happens.
- Failed requests are written to the run log (data/run_log.csv, or --log) by
  common.get and printed. The end-of-run log line gives the new rows, new
  dropped rows, failed searches, items dropped per reason and windows at the
  100-item cap.

Examples:
    python collect_policy.py
    python collect_policy.py --window-days 30 --sites-per-query 2
    python collect_policy.py --refresh-days 0 --sleep 2 --window-margin-days 0
    # test run: keep the output, dropped, progress and log files out of data/
    python collect_policy.py --since 2026-09-01 --out /tmp/policy_test/policy.csv --log /tmp/policy_test/run_log.csv
"""

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

import common
from common import (add_config_options, clean_text, get, load_queries, load_weights, log, make_rows,
                    save_new, use_config_options)

# Name used in the run log and in config/queries.csv.
COLLECTOR = "policy"
# Value of the source_type column for every row (and key in source_weights.csv).
SOURCE_TYPE = "policy"
# Google News search as an RSS feed (same as the Google ad news assignment).
RSS_URL = "https://news.google.com/rss/search"
# English results, US edition, so every run asks Google the same thing.
RSS_LANGUAGE = {"hl": "en-US", "gl": "US", "ceid": "US:en"}
# Google News returns at most this many items per search.
RSS_MAX_ITEMS = 100
# Reasons written in the dropped file, in the order they are checked.
REASON_BAD_DATE = "no readable date"
REASON_OUTSIDE = "dated outside window"
REASON_NO_LINK = "no link"
REASON_NOT_LISTED = "source not a listed institution site"
DROP_REASONS = [REASON_BAD_DATE, REASON_OUTSIDE, REASON_NO_LINK, REASON_NOT_LISTED]
# Columns of the progress file: one row per finished search.
PROGRESS_COLUMNS = ["query", "sites", "window_start", "window_end", "n_items", "time"]


def load_sites(path):
    """Return {site: institution} from the policy sites file."""
    table = pd.read_csv(path)
    return dict(zip(table["site"].str.lower().str.strip(), table["institution"].str.strip()))


def site_groups(sites, size):
    """Split the list of sites into groups of `size`."""
    return [sites[i : i + size] for i in range(0, len(sites), size)]


def site_filter(group):
    """Return the site: part of the query for one group of sites."""
    return "(" + " OR ".join(f"site:{site}" for site in group) + ")"


def time_windows(since, window_days, today):
    """Return the list of (start, end) windows of fixed length from `since` until today."""
    windows = []
    start = since
    while start < today:
        windows.append((start, start + timedelta(days=window_days)))
        start += timedelta(days=window_days)
    return windows


def match_site(source_url, sites):
    """Return the listed site that a source url belongs to (domain or subdomain), or None."""
    domain = urlparse(source_url).netloc.lower().split(":")[0]
    matches = [site for site in sites if domain == site or domain.endswith("." + site)]
    return max(matches, key=len) if matches else None


def strip_source(title, source_name):
    """Remove the ' - Source name' ending that Google News adds to titles."""
    if source_name and title.endswith(f" - {source_name}"):
        return title[: -len(source_name) - 3]
    return title


def make_text(institution, description, title, source_name):
    """Return 'Institution: <name>. <description>', without a description that only repeats the title."""
    text = clean_text(description)
    if source_name and text.endswith(source_name):
        text = text[: -len(source_name)].strip()
    if not text or text.lower() == title.lower():
        return f"Institution: {institution}."
    return f"Institution: {institution}. {text}"


def fetch_window(query, start, end):
    """Run one Google News search limited to [start, end); return the list of <item> elements, or None if it failed."""
    params = {"q": f"after:{start} before:{end} {query}", **RSS_LANGUAGE}
    response = get(RSS_URL, COLLECTOR, params=params)
    if response is None:
        return None
    try:
        return list(ET.fromstring(response.content).iter("item"))
    except ET.ParseError as error:
        log(COLLECTOR, "failed", f"{RSS_URL} {params}", f"XML parse error: {error}")
        print(f"  FAILED to read the answer: {error}")
        return None


def read_date(text):
    """Return the date of an RSS pubDate, or None if it is missing or malformed."""
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError):
        return None


def drop_reason(published, link, site, start, end, margin_days):
    """Return why an item is dropped (the first rule it fails, see the docstring), or None if it is kept."""
    margin = timedelta(days=margin_days)
    if published is None:
        return REASON_BAD_DATE
    if not start - margin <= published <= end + margin:
        return REASON_OUTSIDE
    if not link:
        return REASON_NO_LINK
    if site is None:
        return REASON_NOT_LISTED
    return None


def parse_items(items, start, end, query, sites, margin_days):
    """Split RSS items into (kept, dropped): dicts ready for make_rows; each dropped dict has a reason."""
    kept, dropped = [], []
    for item in items:
        source = item.find("source")
        source_name = source.text.strip() if source is not None and source.text else ""
        source_url = source.get("url", "") if source is not None else ""
        raw_date = item.findtext("pubDate", default="")
        published = read_date(raw_date)
        link = (item.findtext("link") or "").strip()
        title = strip_source((item.findtext("title") or "").strip(), source_name)
        site = match_site(source_url, sites)
        reason = drop_reason(published, link, site, start, end, margin_days)
        row = {
            "date": published.strftime("%Y-%m-%d") if published else "",
            "title": title,
            "url": link,
            "query": query,
        }
        if reason is None:
            row["key"] = link
            row["institution"] = sites[site]
            row["text"] = make_text(sites[site], item.findtext("description"), title, source_name)
            kept.append(row)
        else:
            # the reason is part of the key, so one item dropped for two reasons is traced twice
            row["key"] = f"{reason}|{link}" if link else f"{reason}|{title}|{raw_date}"
            row.update(source_name=source_name, source_url=source_url, reason=reason)
            row["text"] = clean_text(item.findtext("description"))
            dropped.append(row)
    return kept, dropped


def kept_rows(kept, weights):
    """Turn kept items into rows with the common columns, one institution at a time."""
    tables = [
        make_rows([item for item in kept if item["institution"] == name], name, SOURCE_TYPE, weights)
        for name in sorted({item["institution"] for item in kept})
    ]
    return pd.concat(tables, ignore_index=True) if tables else make_rows([], "", SOURCE_TYPE, weights)


def dropped_rows(dropped, weights):
    """Turn dropped items into rows with the common columns plus source_url and reason."""
    tables = []
    for name in sorted({item["source_name"] for item in dropped}):
        group = [item for item in dropped if item["source_name"] == name]
        table = make_rows(group, name, SOURCE_TYPE, weights)
        table["source_url"] = [item["source_url"] for item in group]
        table["reason"] = [item["reason"] for item in group]
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else None


def load_progress(path):
    """Return the progress table (empty if the file does not exist)."""
    if Path(path).exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=PROGRESS_COLUMNS)


def done_keys(progress, today, refresh_days):
    """Return the set of (query, sites, start, end) already done, leaving out windows to refresh."""
    limit = str(today - timedelta(days=refresh_days))
    done = progress[progress["window_end"] < limit]
    return set(zip(done["query"], done["sites"], done["window_start"], done["window_end"]))


def record_progress(path, query, sites, start, end, n_items):
    """Add (or replace) the progress row of one finished search."""
    progress = load_progress(path)
    same = (
        (progress["query"] == query)
        & (progress["sites"] == sites)
        & (progress["window_start"] == str(start))
        & (progress["window_end"] == str(end))
    )
    row = {
        "query": query,
        "sites": sites,
        "window_start": str(start),
        "window_end": str(end),
        "n_items": n_items,
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    progress = pd.concat([progress[~same], pd.DataFrame([row])], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    progress.to_csv(path, index=False)


def institution_counts(table, institutions):
    """Return rows per institution, every listed one included, sorted by count (high first) then by name."""
    names = sorted(set(institutions) | set(table["source"]))
    counts = table["source"].value_counts().reindex(names, fill_value=0)
    # a stable sort keeps the alphabetical order among institutions with the same count
    return counts.sort_values(ascending=False, kind="stable")


def main():
    """Search every (query, sites group, window) not done yet, save new rows and dropped items, print a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sites", default="config/policy_sites.csv",
                        help="institution sites file; add new sites at the END, or later site groups are searched "
                             "again from --since (default: config/policy_sites.csv)")
    parser.add_argument("--since", default="2022-11-01", help="first day searched, YYYY-MM-DD (default: 2022-11-01, ChatGPT release)")
    parser.add_argument("--window-days", type=int, default=45,
                        help="length of each search window in days (a judgement call) (default: 45; 90 hit the 100-item cap in 9 windows)")
    parser.add_argument("--sites-per-query", type=int, default=3,
                        help="sites in one search, kept small because Google News cuts long queries (a judgement call) (default: 3)")
    parser.add_argument("--refresh-days", type=int, default=35,
                        help="search again windows ending within this many days of today (a judgement call) (default: 35)")
    parser.add_argument("--window-margin-days", type=int, default=2,
                        help="keep items dated up to this many days outside their window (time zones, late indexing), "
                             "(a judgement call) (default: 2)")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds to wait between searches (default: 1)")
    parser.add_argument("--out", default="data/raw/policy.csv", help="output CSV (default: data/raw/policy.csv)")
    parser.add_argument("--dropped", help="dropped items CSV (default: <out>_dropped.csv next to --out)")
    parser.add_argument("--progress", help="progress CSV (default: <out>_progress.csv next to --out)")
    add_config_options(parser)
    args = parser.parse_args()
    use_config_options(args)
    # print each line at once, so progress shows during a long run (also when the output goes to a file)
    sys.stdout.reconfigure(line_buffering=True)

    out = Path(args.out)
    dropped_path = Path(args.dropped or out.with_name(f"{out.stem}_dropped.csv"))
    progress_path = Path(args.progress or out.with_name(f"{out.stem}_progress.csv"))
    today = date.today()
    since = date.fromisoformat(args.since)

    sites = load_sites(args.sites)
    weights = load_weights()
    queries = load_queries(COLLECTOR)
    groups = site_groups(list(sites), args.sites_per_query)
    windows = time_windows(since, args.window_days, today)
    done = done_keys(load_progress(progress_path), today, args.refresh_days)

    print(f"{len(queries)} queries x {len(groups)} site groups x {len(windows)} windows of {args.window_days} days from {since}")
    new_total, n_dropped, n_failed, n_skipped, full_windows = 0, 0, 0, 0, []
    # a Counter is a dict that counts: reasons[reason] += 1 works even for a reason not seen yet
    reasons = Counter()
    for query in queries:
        for group in groups:
            sites_text = " ".join(group)
            full_query = f"{query} {site_filter(group)}"
            print(f"\nSites: {sites_text}")
            for start, end in windows:
                if (query, sites_text, str(start), str(end)) in done:
                    n_skipped += 1
                    continue
                items = fetch_window(full_query, start, end)
                time.sleep(args.sleep)
                if items is None:
                    n_failed += 1
                    print(f"  {start} -> {end}: FAILED (logged in {common.RUN_LOG})")
                    continue
                if len(items) >= RSS_MAX_ITEMS:
                    full_windows.append(f"{sites_text} {start} -> {end}")
                kept, dropped = parse_items(items, start, end, query, sites, args.window_margin_days)
                window_reasons = Counter(item["reason"] for item in dropped)
                reasons.update(window_reasons)
                added, _ = save_new(kept_rows(kept, weights), out)
                if dropped:
                    added_dropped, _ = save_new(dropped_rows(dropped, weights), dropped_path)
                    n_dropped += added_dropped
                new_total += added
                record_progress(progress_path, query, sites_text, start, end, len(items))
                print(f"  {start} -> {end}: {len(items)} items, {len(kept)} kept, {len(dropped)} dropped "
                      f"({window_reasons[REASON_NOT_LISTED]} not listed, {window_reasons[REASON_OUTSIDE]} outside window, "
                      f"{window_reasons[REASON_NO_LINK]} no link, {window_reasons[REASON_BAD_DATE]} no date), {added} new")

    reason_text = ", ".join(f"{reasons[reason]} {reason}" for reason in DROP_REASONS)
    log(COLLECTOR, "done", message=f"{new_total} new rows, {n_dropped} new dropped, {n_failed} failed searches; "
                                   f"dropped items: {reason_text}; {len(full_windows)} windows at the {RSS_MAX_ITEMS}-item cap")

    print(f"\nSearches skipped (already done): {n_skipped}")
    print(f"Searches failed: {n_failed}")
    print("Items dropped this run, per reason (counted again on every run that searches the window):")
    for reason in DROP_REASONS:
        print(f"  {reason}: {reasons[reason]}")
    if full_windows:
        print(f"Windows that hit the {RSS_MAX_ITEMS}-item limit (items may be missing, try a smaller --window-days):")
        for window in full_windows:
            print(f"  {window}")
    if n_dropped:
        print(f"New dropped items this run: {n_dropped} (saved to {dropped_path})")
    else:
        print("New dropped items this run: 0")
    # save_new writes nothing while no item has ever been kept, so the file may not exist yet
    if not out.exists():
        print(f"No items kept so far. Nothing was saved to {out}.")
        return
    table = pd.read_csv(out, dtype=str, keep_default_na=False)
    print(f"New rows this run: {new_total}")
    print(f"Rows in file: {len(table)} (saved to {out})")
    print("\nItems per institution:")
    print(institution_counts(table, sites.values()).to_string())
    print("\nItems per year:")
    print(table["date"].str[:4].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
