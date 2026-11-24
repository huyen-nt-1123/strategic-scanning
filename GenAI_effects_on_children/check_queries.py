"""Compare search queries before using them: how much each finds, and how much of it is relevant.

For every candidate query in config/query_candidates.csv (columns source,
topic, version, query), the script runs the search the same way the real
collector does (it calls the collectors' own functions) and measures:

- found_total: how many results the source has for the query.
  OpenAlex and PubMed report a total; arXiv and Google News don't, so there it
  is the number of items returned (for Google News, inside the sample windows
  only, see below).
- sampled: how many of those items were looked at. OpenAlex, PubMed and arXiv:
  the first --sample-size results, in each API's default order. Google News
  (news and policy): every item of a few sample time windows (--sample-starts),
  because one search returns at most 100 items.
- share_rule: the share of sampled items that pass the relevance rule clean.py
  applies: title + text has a term of config/ai_terms.csv AND a term of
  config/child_terms.csv (whole words, any case, * = any ending).
- share_young: the share with an explicit 0-8 age term (child_terms groups
  age_0_2, age_3_5, age_6_8, gen_alpha).
- share_older: the share with a teen or older term (config/older_terms.csv).
- est_relevant: found_total x share_rule, an estimate from the sample, for the
  APIs; for Google News it is the exact count of passing items in the windows.
- OpenAlex candidates whose version starts with "title_abstract" are searched
  in titles and abstracts only (collect_academic.py --openalex-mode title_abstract).
- coverage: of all the passing items found by any version of the same source
  and topic (pooled sample), the share this version found. A tighter query with
  a high share_rule but a low coverage misses items the others find. For the
  APIs the samples are only the top results, so coverage is a rough guide.

A second table gives the same measures per (source, version), pooling all the
topics of a version: for comparing the full query set of one version with
another.

Extra columns: complete (False if a request failed), n_duplicates (same title
returned twice), and per source the items the collectors' own rules dropped:
n_offsite and n_outside_window (policy), n_skipped_type (OpenAlex paratext or
retracted).

Google News answers change from one request to the next, so two runs of this
script can differ a little for news and policy. To keep reruns cheap and
restartable, every search answer is saved in --out-dir/cache/ and reused;
--refresh fetches again.

Examples:
    python check_queries.py
    python check_queries.py --sources news,policy
    python check_queries.py --sample-size 100 --refresh
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import collect_academic
import collect_news
import collect_policy
import common

# The age groups of config/child_terms.csv that name children aged 0 to 8.
YOUNG_GROUPS = ["age_0_2", "age_3_5", "age_6_8", "gen_alpha"]
# Sources this script knows how to search.
ALL_SOURCES = ["news", "policy", "openalex", "pubmed", "arxiv"]
# Start of the scan period, as in the collectors (public release of ChatGPT).
SINCE = "2022-11-01"


# ---------- searching ----------

def search_news(query, starts, window_days, sleep):
    """Return (items, found, ok, counts) for one Google News query over the sample windows."""
    items, ok = [], True
    for start in starts:
        found = collect_news.fetch_window(query, start, start + timedelta(days=window_days))
        if found is None:
            ok = False
        items.extend(found or [])
        time.sleep(sleep)
    return items, len(items), ok, {}


def search_policy(query, starts, window_days, sleep, sites, per_query, margin_days):
    """Return (items, found, ok, counts) for one policy query: every site group, every sample window."""
    items, ok, counts = [], True, {"n_offsite": 0, "n_outside_window": 0}
    for group in collect_policy.site_groups(list(sites), per_query):
        full_query = f"{query} {collect_policy.site_filter(group)}"
        for start in starts:
            end = start + timedelta(days=window_days)
            raw = collect_policy.fetch_window(full_query, start, end)
            if raw is None:
                ok = False
            kept, dropped = collect_policy.parse_items(raw or [], start, end, query, sites, margin_days)
            items.extend(kept)
            reasons = [item["reason"] for item in dropped]
            counts["n_offsite"] += sum(reason == "source not a listed institution site" for reason in reasons)
            counts["n_outside_window"] += sum(reason == "dated outside window" for reason in reasons)
            time.sleep(sleep)
    return items, len(items), ok, counts


def openalex_total(query, since, api_key, mode):
    """Return the number of OpenAlex works matching a query since `since` (None if the request failed)."""
    params = {"filter": f"from_publication_date:{since}", "per-page": 1, "select": "id"}
    if mode == "title_abstract":
        params["filter"] += f",title_and_abstract.search:{query}"
    else:
        params["search"] = query
    if api_key:
        params["api_key"] = api_key
    response = common.get(collect_academic.OPENALEX_URL, "check_queries", params=params)
    return response.json()["meta"]["count"] if response is not None else None


def pubmed_total(query, since, until):
    """Return the number of PubMed records matching a query since `since` (None if the request failed)."""
    params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 0, "datetype": "pdat",
              "mindate": since.replace("-", "/"), "maxdate": until.replace("-", "/")}
    response = common.get(collect_academic.PUBMED_SEARCH_URL, "check_queries", params=params)
    return int(response.json()["esearchresult"]["count"]) if response is not None else None


def search(row, args, sites):
    """Run one candidate query and return (items, found_total, ok, counts); ok is False if any request failed."""
    starts = [date.fromisoformat(day) for day in args.sample_starts.split(",")]
    if row.source == "news":
        return search_news(row.query, starts, args.news_window_days, args.sleep)
    if row.source == "policy":
        return search_policy(row.query, starts, args.policy_window_days, args.sleep, sites, args.sites_per_query,
                             args.window_margin_days)
    if row.source == "openalex":
        mode = "title_abstract" if row.version.startswith("title_abstract") else "search"
        items, skipped, ok = collect_academic.fetch_openalex(row.query, args.since, args.sample_size, None,
                                                             args.openalex_key, mode)
        total = openalex_total(row.query, args.since, args.openalex_key, mode)
        return items, total, ok and total is not None, {"n_skipped_type": skipped}
    if row.source == "pubmed":
        items, skipped, ok = collect_academic.fetch_pubmed(row.query, args.since, args.sample_size)
        total = pubmed_total(row.query, args.since, args.until)
        return items, total, ok and total is not None, {"n_skipped_type": skipped}
    if row.source == "arxiv":
        cats = args.arxiv_cats.split(",")
        items, skipped, ok = collect_academic.fetch_arxiv(row.query, args.since, args.sample_size, cats)
        return items, len(items), ok, {"n_skipped_type": skipped}
    raise ValueError(f"unknown source {row.source}")


def cached_search(row, args, sites, cache_dir):
    """Run one candidate, or reuse its saved answer; return (items DataFrame, found_total, ok, counts).

    Only a search where every request worked is saved, so a failed one is
    tried again on the next run. Every option that changes the answer is part
    of the saved file's name.
    """
    settings = [row.source, row.version, row.query, args.since, args.until, args.sample_starts,
                args.sample_size, args.news_window_days, args.policy_window_days, args.sites_per_query,
                args.arxiv_cats, ",".join(sites), args.window_margin_days]
    name = hashlib.sha1("|".join(map(str, settings)).encode()).hexdigest()[:12]
    cache, info_file = cache_dir / f"{name}.csv", cache_dir / f"{name}.json"
    if cache.exists() and info_file.exists() and not args.refresh:
        info = json.loads(info_file.read_text())
        return pd.read_csv(cache, dtype=str, keep_default_na=False), info["found"], True, info["counts"]
    items, found, ok, counts = search(row, args, sites)
    table = pd.DataFrame(items, columns=["key", "date", "title", "text", "url"])
    if ok:
        cache_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache, index=False)
        info_file.write_text(json.dumps({"found": found, "counts": counts}))
    return table, found, ok, counts


# ---------- measuring ----------

def tag(items, patterns):
    """Add the rule columns (ai_terms, child_terms, passes, young, older) to the sampled items."""
    full = items["title"].fillna("") + " " + items["text"].fillna("")
    items = items.copy()
    items["ai_terms"] = full.map(lambda t: "; ".join(common.find_terms(patterns["ai"], t)))
    items["child_terms"] = full.map(lambda t: "; ".join(common.find_terms(patterns["child"], t)))
    items["passes"] = (items["ai_terms"] != "") & (items["child_terms"] != "")
    items["young"] = full.map(lambda t: bool(patterns["young"].search(t)))
    items["older"] = full.map(lambda t: bool(patterns["older"].search(t)))
    # One story can come back under several links: compare on the lower-case title
    # (or on the link when there is no title)
    items["match_key"] = items["title"].fillna("").str.lower().str.strip()
    items.loc[items["match_key"] == "", "match_key"] = items["url"]
    return items


def measure(items, found, pooled_passing):
    """Return the measures of one group of sampled items."""
    unique = items.drop_duplicates("match_key")
    passing = set(unique.loc[unique["passes"], "match_key"])
    n = len(unique)
    return {
        "found_total": found,
        "sampled": n,
        "n_duplicates": len(items) - n,
        "pass_rule": len(passing),
        "share_rule": round(len(passing) / n, 2) if n else None,
        "share_young": round(unique["young"].mean(), 2) if n else None,
        "share_older": round(unique["older"].mean(), 2) if n else None,
        "coverage": round(len(passing) / len(pooled_passing), 2) if pooled_passing else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates", default="config/query_candidates.csv", help="candidate queries (default: config/query_candidates.csv)")
    parser.add_argument("--sources", default=",".join(ALL_SOURCES), help=f"sources to check (default: {','.join(ALL_SOURCES)})")
    parser.add_argument("--since", default=SINCE, help=f"start of the scan period (default: {SINCE})")
    parser.add_argument("--until", default=str(date.today()), help="end of the period for the PubMed count (default: today)")
    parser.add_argument("--sample-starts", default="2023-09-01,2025-03-01,2026-09-15",
                        help="start days of the Google News sample windows, spread over the period (default: 2023-09-01,2025-03-01,2026-09-15)")
    parser.add_argument("--news-window-days", type=int, default=30, help="length of a news sample window (default: 30, as the collector)")
    parser.add_argument("--policy-window-days", type=int, default=45, help="length of a policy sample window (default: 45)")
    parser.add_argument("--window-margin-days", type=int, default=2,
                        help="policy items dated this many days outside their window still count, as the collector (default: 2)")
    parser.add_argument("--sites-per-query", type=int, default=3, help="policy sites per search, as the collector (default: 3)")
    parser.add_argument("--sample-size", type=int, default=200, help="results looked at per API query (default: 200)")
    parser.add_argument("--arxiv-cats", default=collect_academic.ARXIV_CATS, help=f"arXiv categories (default: {collect_academic.ARXIV_CATS})")
    parser.add_argument("--openalex-key", default=os.environ.get("OPENALEX_API_KEY"), help="OpenAlex key (default: OPENALEX_API_KEY, from .env)")
    parser.add_argument("--sleep", type=float, default=1.0, help="seconds between Google News searches (default: 1)")
    parser.add_argument("--sites", default="config/policy_sites.csv", help="policy sites (default: config/policy_sites.csv)")
    parser.add_argument("--ai-terms", default="config/ai_terms.csv", help="AI word list (default: config/ai_terms.csv)")
    parser.add_argument("--child-terms", default="config/child_terms.csv", help="child word list (default: config/child_terms.csv)")
    parser.add_argument("--older-terms", default="config/older_terms.csv", help="teen-or-older word list (default: config/older_terms.csv)")
    parser.add_argument("--young-groups", default=",".join(YOUNG_GROUPS),
                        help=f"groups of the child list that name ages 0-8 (default: {','.join(YOUNG_GROUPS)})")
    parser.add_argument("--refresh", action="store_true", help="search again instead of reusing saved answers")
    parser.add_argument("--out-dir", default="data/query_check", help="output folder (default: data/query_check)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    # This is a check, not a collection: its failed requests go to its own log
    common.RUN_LOG = out_dir / "run_log.csv"
    sources = args.sources.split(",")
    candidates = pd.read_csv(args.candidates)
    candidates = candidates[candidates["source"].isin(sources)]
    if candidates.empty:
        sys.exit(f"No candidate query for the sources {sources} in {args.candidates}.")
    sites = collect_policy.load_sites(args.sites)
    young_groups = args.young_groups.split(",")
    patterns = {
        "ai": common.terms_pattern(common.load_terms(args.ai_terms)),
        "child": common.terms_pattern(common.load_terms(args.child_terms)),
        "young": common.terms_pattern(sum((common.load_terms(args.child_terms, g) for g in young_groups), [])),
        "older": common.terms_pattern(common.load_terms(args.older_terms)),
    }

    # 1. run (or reuse) every search
    samples, found, counts, failed = [], {}, {}, []
    for row in candidates.itertuples():
        print(f"{row.source:8} {row.topic:12} {row.version:9} {row.query[:70]}")
        items, total, ok, counts[row.Index] = cached_search(row, args, sites, out_dir / "cache")
        found[row.Index] = total
        if not ok:
            failed.append(row.Index)
            print("         SOME REQUESTS FAILED: measured on what came back; not saved, so the next run retries")
        items = tag(items, patterns)
        items.insert(0, "candidate", row.Index)
        samples.append(items)
        print(f"         found {total}, sampled {len(items)}, passing the rule {int(items['passes'].sum())}")
    samples = pd.concat(samples, ignore_index=True)
    samples = samples.merge(candidates[["source", "topic", "version", "query"]], left_on="candidate", right_index=True)

    # 2. one row per candidate, coverage against the same source and topic
    rows = []
    for row in candidates.itertuples():
        same_topic = samples[(samples["source"] == row.source) & (samples["topic"] == row.topic)]
        pooled = set(same_topic.loc[same_topic["passes"], "match_key"])
        mine = samples[samples["candidate"] == row.Index]
        rows.append({"source": row.source, "topic": row.topic, "version": row.version, "query": row.query,
                     "complete": row.Index not in failed, **measure(mine, found[row.Index], pooled),
                     **counts[row.Index]})
    per_query = pd.DataFrame(rows)
    # found_total is unknown (empty) when the count request failed
    per_query["found_total"] = pd.to_numeric(per_query["found_total"], errors="coerce")
    has_total = per_query["source"].isin(["openalex", "pubmed"])
    per_query["est_relevant"] = per_query["pass_rule"].astype(float)
    per_query.loc[has_total, "est_relevant"] = (per_query["found_total"] * per_query["share_rule"]).round()

    # 3. one row per (source, version), all topics pooled
    rows = []
    for (source, version), mine in samples.groupby(["source", "version"]):
        same_source = samples[samples["source"] == source]
        pooled = set(same_source.loc[same_source["passes"], "match_key"])
        n_queries = (candidates[(candidates["source"] == source) & (candidates["version"] == version)]).shape[0]
        rows.append({"source": source, "version": version, "queries": n_queries,
                     **measure(mine, None, pooled)})
    per_version = pd.DataFrame(rows).drop(columns="found_total")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(out_dir / "per_query.csv", index=False)
    per_version.to_csv(out_dir / "per_version.csv", index=False)
    columns = ["source", "topic", "version", "query", "date", "title", "url", "passes", "ai_terms", "child_terms", "young", "older"]
    samples[columns].to_csv(out_dir / "samples.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nPer query:")
    print(per_query.drop(columns="query").to_string(index=False))
    print("\nPer source and version (topics pooled):")
    print(per_version.to_string(index=False))
    print(f"\n{len(candidates)} candidate queries checked, {len(samples)} sampled items, "
          f"{len(failed)} with failed requests (see {common.RUN_LOG}).")
    print(f"Saved {out_dir / 'per_query.csv'}, {out_dir / 'per_version.csv'}, {out_dir / 'samples.csv'}")


if __name__ == "__main__":
    main()
