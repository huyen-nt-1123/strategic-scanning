"""Collect academic papers about generative AI and young children.

Three free sources, no API key needed, all written to one raw file with the
common columns (id, source, source_type, weight, date, title, text, url, query):

- OpenAlex (source_type "peer_reviewed"): works API. Two ways to match the
  query (--openalex-mode): "search", OpenAlex's default search, which also
  looks in full texts and applies AND/OR loosely; or "title_abstract", which
  only looks in titles and abstracts. Paging is by page number, because cursor
  paging now needs an API key and page paging does not.
  Abstracts come as a word index and are rebuilt into text. Without a key all
  requests share a small free daily budget per IP address. A free key is read
  from the OPENALEX_API_KEY environment variable, which the local `.env` file
  can fill in (see common.py); --openalex-key also works but leaves the key in
  the shell history. The key is masked (***) in the run log.
- PubMed (source_type "peer_reviewed"): E-utilities esearch (ids) + efetch (XML).
  Each request sends NCBI's `tool` parameter, and `email` only if --email is given.
- arXiv (source_type "preprint"): Atom API, limited to the categories in
  --arxiv-cats (default cs.CY = computers and society, cs.HC = human-computer
  interaction).

Rules:

- Queries come from config/queries.csv: the rows of collector "academic_<source>"
  (e.g. academic_pubmed) if there are any, else the shared "academic" rows,
  because each API has its own search syntax. --query runs one query everywhere.
- Only items published on or after --since (default 2022-11-01, the public
  release of ChatGPT: the start of the scan period).
- At most --max-per-query results are fetched per (source, query). Each query
  prints "N of TOTAL matches", TOTAL being the count the API reports
  (OpenAlex meta.count, PubMed esearch count, arXiv totalResults; arXiv's
  total ignores --since, because its API has no date filter). A query whose
  TOTAL is above the cap is listed as "capped" in the summary, and gets its own
  "capped" line in the run log.
- Fixed order, so a capped query keeps the same results on every run:
  OpenAlex newest publication date first, PubMed newest publication date
  first, arXiv newest submission first. Papers with the same date may still
  come back in a different order between runs (the APIs give no tie rule).
- Item key used for the id: OpenAlex = DOI if present, else the OpenAlex id;
  PubMed = PMID; arXiv = arXiv id without version. The id includes the source
  name, so the same paper found in OpenAlex and PubMed gets two ids; those
  cross-source duplicates are grouped later by clean.py, not here.
- OpenAlex items of type "paratext" (covers, tables of contents...) and items
  flagged as retracted are skipped and counted (each reason apart), not saved.
- PubMed book records (PubmedBookArticle) and records with no PMID are skipped
  and counted, not saved.
- PubMed date: the electronic article date if given, else the journal issue date;
  a missing day becomes 01, a missing month becomes 01.
- arXiv search is literal, so each word of the query is required (AND), unless
  the query is already written in arXiv syntax (contains all:), inside
  the chosen categories. Results come newest first and paging stops at --since.
  An empty page before the reported total is reached counts as a failure.
- An answer that is not valid JSON or XML is logged as a failure; the items
  read before it are kept and the (source, query) is not marked finished.
- Politeness: PubMed allows 3 requests a second without a key (wait 0.4 s);
  arXiv asks for 3 seconds between calls.
- Incremental and restartable: rows are added to the raw file after each
  (source, query), and only ids not already there are added. A progress file
  (next to the output, *_progress.csv) records each finished (source, query);
  a rerun on the same day skips those (--redo to force). A run on another day
  re-queries everything. A (source, query) that failed is not marked finished,
  so a rerun tries it again. Failed requests go to data/run_log.csv (--log).
- One clock: "today" is the UTC date, both for "finished today" and for
  PubMed's latest publication date.

Examples:
    python collect_academic.py
    python collect_academic.py --sources openalex,arxiv --max-per-query 100
    python collect_academic.py --query "AI toys children" --redo
    python collect_academic.py --since 2024-01-01 --email you@example.org
    python collect_academic.py --openalex-mode search
    test run, kept out of data/ (the progress file goes next to --out):
    python collect_academic.py --max-per-query 20 --out /tmp/academic_test.csv --log /tmp/academic_test_log.csv
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import (add_config_options, get, load_queries, load_weights, log, make_rows,
                    save_new, use_config_options)

COLLECTOR = "academic"
OUT_FILE = Path("data/raw/academic.csv")
# Public release of ChatGPT (30 Nov 2022), rounded to the start of that month:
# the start of the scan period.
SINCE = "2022-11-01"
# Above the largest query total seen (OpenAlex, 1,155 on 22 Nov), so nothing is cut;
# the run prints "N of TOTAL matches" and lists any query that hits this cap.
MAX_PER_QUERY = 1500
ALL_SOURCES = ["openalex", "pubmed", "arxiv"]

OPENALEX_URL = "https://api.openalex.org/works"
# OpenAlex allows up to 200 results per page.
OPENALEX_PAGE_SIZE = 200
# Only ask OpenAlex for the fields used here: smaller, faster answers.
OPENALEX_FIELDS = "id,doi,display_name,publication_date,abstract_inverted_index,type,is_retracted"
# OpenAlex types that are not research content.
OPENALEX_SKIP_TYPES = {"paratext"}
# Newest first, so that a query cut by --max-per-query keeps the same (latest)
# papers on every run; OpenAlex's default order (relevance) can change between runs.
OPENALEX_SORT = "publication_date:desc"
# How OpenAlex matches the query. "search": its default search, which also looks
# in the full text and treats AND/OR loosely (finds a lot, much of it loose).
# "title_abstract": only titles and abstracts, with AND/OR and "phrases" applied.
OPENALEX_MODES = ["search", "title_abstract"]

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
# Number of PubMed ids fetched in one efetch call.
PUBMED_BATCH = 100
# NCBI rule without an API key: at most 3 requests per second.
PUBMED_WAIT_SECONDS = 0.4
# Newest publication date first, for the same reason as OPENALEX_SORT
# (PubMed's default order is "best match", which can change between runs).
PUBMED_SORT = "pub_date"
# NCBI asks scripts to name themselves with the `tool` parameter (no personal details).
PUBMED_TOOL = "strategic-scanning-coursework"
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

ARXIV_URL = "https://export.arxiv.org/api/query"
# Computers and society, human-computer interaction: where child/AI studies are filed.
ARXIV_CATS = "cs.CY,cs.HC"
ARXIV_PAGE_SIZE = 100
# arXiv API rule: wait 3 seconds between calls.
ARXIV_WAIT_SECONDS = 3
# XML namespaces of arXiv's answer: Atom for the entries, OpenSearch for the total.
ATOM = {"a": "http://www.w3.org/2005/Atom", "os": "http://a9.com/-/spec/opensearch/1.1/"}

SOURCE_INFO = {
    "openalex": ("OpenAlex", "peer_reviewed"),
    "pubmed": ("PubMed", "peer_reviewed"),
    "arxiv": ("arXiv", "preprint"),
}


def today_utc():
    """Return today's UTC date as YYYY-MM-DD (the one clock used by this script)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def read_json(response, target):
    """Return the JSON of a response, or None after logging a reply that is not JSON."""
    try:
        return response.json()
    except ValueError as error:
        log(COLLECTOR, "failed", target, f"answer is not JSON: {error}")
        print(f"  FAILED to read the answer as JSON: {error}", flush=True)
        return None


# ---------- OpenAlex ----------

def rebuild_abstract(inverted_index):
    """Turn OpenAlex's {word: [positions]} index back into the abstract text."""
    if not inverted_index:
        return ""
    positions = {}
    for word, places in inverted_index.items():
        for place in places:
            positions[place] = word
    return " ".join(positions[place] for place in sorted(positions))


def fetch_openalex(query, since, max_items, email, api_key, mode="search", info=None):
    """Return (items, n_skipped, ok) for one OpenAlex search; `info`, if given, gets total and skip reasons."""
    info = {} if info is None else info
    info.update(total=None, skipped_by_reason={"paratext": 0, "retracted": 0})
    items, skipped, fetched, page = [], 0, 0, 1
    page_size = min(OPENALEX_PAGE_SIZE, max_items)
    while fetched < max_items:
        params = {
            "filter": f"from_publication_date:{since}",
            "sort": OPENALEX_SORT,
            "per-page": page_size,
            "page": page,
            "select": OPENALEX_FIELDS,
        }
        if mode == "title_abstract":
            params["filter"] += f",title_and_abstract.search:{query}"
        else:
            params["search"] = query
        if email:
            params["mailto"] = email
        if api_key:
            params["api_key"] = api_key
        response = get(OPENALEX_URL, COLLECTOR, params=params)
        if response is None:
            return items, skipped, False
        data = read_json(response, f"OpenAlex page {page}: {query}")
        if data is None:
            return items, skipped, False
        info["total"] = data.get("meta", {}).get("count", 0)
        # the last page may hold more than needed: keep only up to max_items
        results = data.get("results", [])[: max_items - fetched]
        if not results:
            break
        fetched += len(results)
        for work in results:
            if work.get("type") in OPENALEX_SKIP_TYPES or work.get("is_retracted"):
                reason = "paratext" if work.get("type") in OPENALEX_SKIP_TYPES else "retracted"
                info["skipped_by_reason"][reason] += 1
                skipped += 1
                continue
            doi = work.get("doi") or ""
            items.append({
                "key": doi.lower() if doi else work["id"],
                "date": work.get("publication_date") or "",
                "title": work.get("display_name") or "",
                "text": rebuild_abstract(work.get("abstract_inverted_index")),
                "url": doi or work["id"],
                "query": query,
            })
        if fetched >= info["total"]:
            break
        page += 1
    return items, skipped, True


# ---------- PubMed ----------

def pubmed_date(article):
    """Return the best YYYY-MM-DD date of a PubmedArticle element ('' if none)."""
    electronic = article.find(".//Article/ArticleDate")
    if electronic is not None and electronic.findtext("Year"):
        return f"{electronic.findtext('Year')}-{int(electronic.findtext('Month', '1')):02d}-" \
               f"{int(electronic.findtext('Day', '1')):02d}"
    pub = article.find(".//Article/Journal/JournalIssue/PubDate")
    if pub is None:
        return ""
    year = pub.findtext("Year")
    if not year:
        # e.g. <MedlineDate>2023 Jan-Feb</MedlineDate>
        parts = (pub.findtext("MedlineDate") or "").split()
        if not parts or not parts[0][:4].isdigit():
            return ""
        year = parts[0][:4]
        month = MONTHS.get(parts[1][:3].lower(), 1) if len(parts) > 1 else 1
        return f"{year}-{month:02d}-01"
    month_text = pub.findtext("Month", "1")
    month = int(month_text) if month_text.isdigit() else MONTHS.get(month_text[:3].lower(), 1)
    day_text = pub.findtext("Day", "1")
    day = int(day_text) if day_text.isdigit() else 1
    return f"{year}-{month:02d}-{day:02d}"


def parse_pubmed(xml_bytes, query):
    """Return (items, n_skipped) from one efetch XML answer; book records and records without PMID are skipped."""
    root = ET.fromstring(xml_bytes)
    items, skipped = [], sum(1 for _ in root.iter("PubmedBookArticle"))
    for article in root.iter("PubmedArticle"):
        pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
        if not pmid:
            skipped += 1
            continue
        title_el = article.find(".//Article/ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""
        parts = []
        for part in article.findall(".//Article/Abstract/AbstractText"):
            text = "".join(part.itertext()).strip()
            label = part.get("Label")
            parts.append(f"{label}: {text}" if label else text)
        items.append({
            "key": pmid,
            "date": pubmed_date(article),
            "title": title,
            "text": " ".join(parts),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "query": query,
        })
    return items, skipped


def pubmed_identity(email):
    """Return the parameters that tell NCBI who calls: always `tool`, `email` only if given."""
    return {"tool": PUBMED_TOOL, **({"email": email} if email else {})}


def fetch_pubmed(query, since, max_items, email=None, info=None):
    """Return (items, n_skipped, ok) for one PubMed search; `info`, if given, gets the total."""
    info = {} if info is None else info
    info.update(total=None)
    params = {
        "db": "pubmed", "term": query, "retmode": "json", "retmax": max_items,
        "sort": PUBMED_SORT, "datetype": "pdat", "mindate": since.replace("-", "/"),
        "maxdate": today_utc().replace("-", "/"), **pubmed_identity(email),
    }
    response = get(PUBMED_SEARCH_URL, COLLECTOR, params=params)
    time.sleep(PUBMED_WAIT_SECONDS)
    if response is None:
        return [], 0, False
    data = read_json(response, f"PubMed esearch: {query}")
    if data is None:
        return [], 0, False
    result = data.get("esearchresult", {})
    info["total"] = int(result.get("count", 0) or 0)
    ids = result.get("idlist", [])
    items, skipped, ok = [], 0, True
    for start in range(0, len(ids), PUBMED_BATCH):
        batch = ids[start:start + PUBMED_BATCH]
        response = get(PUBMED_FETCH_URL, COLLECTOR,
                       params={"db": "pubmed", "id": ",".join(batch), "retmode": "xml",
                               **pubmed_identity(email)})
        time.sleep(PUBMED_WAIT_SECONDS)
        if response is None:
            ok = False
            continue
        try:
            batch_items, batch_skipped = parse_pubmed(response.content, query)
        except ET.ParseError as error:
            log(COLLECTOR, "failed", f"PubMed efetch {batch[0]}..", error)
            print(f"  FAILED to read PubMed XML: {error}", flush=True)
            ok = False
            continue
        items.extend(batch_items)
        skipped += batch_skipped
    return items, skipped, ok


# ---------- arXiv ----------

def arxiv_search_query(query, cats):
    """Build an arXiv search: any of the categories AND every word of the query (or the query as is, if written in arXiv syntax with all:)."""
    cat_part = " OR ".join(f"cat:{cat}" for cat in cats)
    if "all:" in query:
        return f"({cat_part}) AND {query}"
    word_part = " AND ".join(f"all:{word}" for word in query.split())
    return f"({cat_part}) AND {word_part}"


def fetch_arxiv(query, since, max_items, cats, info=None):
    """Return (items, 0, ok) for one arXiv search, newest first, stopping at `since`; `info` gets the total."""
    info = {} if info is None else info
    info.update(total=None, reached_since=False)
    items, start, too_old = [], 0, False
    while start < max_items and not too_old:
        params = {
            "search_query": arxiv_search_query(query, cats),
            "start": start,
            "max_results": min(ARXIV_PAGE_SIZE, max_items - start),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        response = get(ARXIV_URL, COLLECTOR, params=params)
        time.sleep(ARXIV_WAIT_SECONDS)
        if response is None:
            return items, 0, False
        try:
            feed = ET.fromstring(response.content)
        except ET.ParseError as error:
            log(COLLECTOR, "failed", f"arXiv {query}", error)
            print(f"  FAILED to read arXiv XML: {error}", flush=True)
            return items, 0, False
        total_text = (feed.findtext("os:totalResults", "", ATOM) or "").strip()
        info["total"] = int(total_text) if total_text.isdigit() else 0
        entries = feed.findall("a:entry", ATOM)
        if not entries:
            if start < info["total"]:
                # arXiv sometimes answers an empty page in the middle of a result list
                message = f"empty page at start={start} of {info['total']} results"
                log(COLLECTOR, "failed", f"arXiv {query}", message)
                print(f"  FAILED: arXiv gave an {message}", flush=True)
                return items, 0, False
            break
        for entry in entries:
            published = entry.findtext("a:published", "", ATOM)[:10]
            if published < since:
                too_old = True
                break
            # e.g. http://arxiv.org/abs/2301.01234v2 -> 2301.01234
            arxiv_id = entry.findtext("a:id", "", ATOM).split("/abs/")[-1].rsplit("v", 1)[0]
            items.append({
                "key": arxiv_id,
                "date": published,
                "title": entry.findtext("a:title", "", ATOM),
                "text": entry.findtext("a:summary", "", ATOM),
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "query": query,
            })
        start += len(entries)
    info["reached_since"] = too_old
    return items, 0, True


# ---------- queries, progress and summary ----------

def source_queries(key, single_query):
    """Return the queries for one source: --query, else its own rows in config/queries.csv, else the shared academic rows."""
    if single_query:
        return [single_query]
    return load_queries(f"{COLLECTOR}_{key}") or load_queries(COLLECTOR)


def load_progress(path):
    """Return the progress table (source, query, n_items, time), empty if none yet."""
    if Path(path).exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=["source", "query", "n_items", "time"])


def done_today(progress, source, query):
    """True if this (source, query) was finished today (UTC)."""
    today = today_utc()
    match = progress[(progress["source"] == source) & (progress["query"] == query)]
    return any(t.startswith(today) for t in match["time"])


def mark_done(path, source, query, n_items):
    """Append one finished (source, query) to the progress file."""
    row = {"source": source, "query": query, "n_items": n_items,
           "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def is_capped(key, max_items, info):
    """True if the API reports more matches than --max-per-query lets through."""
    if key == "arxiv" and info.get("reached_since"):
        return False  # paging stopped at --since, not at the cap
    return (info.get("total") or 0) > max_items


def skip_reasons_text(key, reasons):
    """Describe the skipped items of one source, by reason."""
    if key == "openalex":
        return f"{reasons['paratext']} paratext, {reasons['retracted']} retracted"
    if key == "pubmed":
        return "book records or no PMID"
    return "none skipped by rule"


def print_summary(out, new_counts, skipped_counts, skip_reasons, failures, capped):
    """Print new and total rows, empty abstracts, skips, capped queries and items per year, per source."""
    if not out.exists():
        print("\nNothing saved.", flush=True)
        return
    table = pd.read_csv(out, dtype=str, keep_default_na=False)
    print("\nSummary", flush=True)
    for key, (source, _) in SOURCE_INFO.items():
        if source not in new_counts:
            continue
        rows = table[table["source"] == source]
        empty = (rows["text"].str.strip() == "").sum()
        print(f"\n{source}: {new_counts[source]} new rows, {len(rows)} rows in file, "
              f"{empty} with empty abstract ({empty / max(len(rows), 1):.0%}), "
              f"{skipped_counts[source]} skipped ({skip_reasons_text(key, skip_reasons[source])}), "
              f"{failures[source]} queries with failures, {len(capped[source])} queries capped", flush=True)
        for query, n_fetched, total in capped[source]:
            print(f"  capped: {n_fetched} of {total} matches: {query}", flush=True)
        years = rows["date"].str[:4].replace("", "no date").value_counts().sort_index()
        print("  items per year: " + ", ".join(f"{y}: {n}" for y, n in years.items()), flush=True)
    print(f"\nSaved to {out} ({len(table)} rows in all)", flush=True)


def main():
    """Read the options, run every (source, query) not finished today, save new rows and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query", help="run this single query instead of those in config/queries.csv")
    parser.add_argument("--sources", default=",".join(ALL_SOURCES),
                        help=f"comma-separated sources among {','.join(ALL_SOURCES)} (default: all)")
    parser.add_argument("--since", default=SINCE,
                        help=f"earliest publication date YYYY-MM-DD (default: {SINCE}, scan start)")
    parser.add_argument("--max-per-query", type=int, default=MAX_PER_QUERY,
                        help=f"most results fetched per source and query (default: {MAX_PER_QUERY})")
    parser.add_argument("--arxiv-cats", default=ARXIV_CATS,
                        help=f"comma-separated arXiv categories (default: {ARXIV_CATS})")
    parser.add_argument("--email", help="contact address sent to OpenAlex (mailto) and PubMed (email) (default: none)")
    parser.add_argument("--openalex-key", default=os.environ.get("OPENALEX_API_KEY"),
                        help="free OpenAlex API key; prefer OPENALEX_API_KEY in .env or the environment, "
                             "because a key typed here stays in the shell history (default: that variable, else none)")
    parser.add_argument("--openalex-mode", choices=OPENALEX_MODES, default="title_abstract",
                        help="search = OpenAlex default search (loose, includes full text); "
                             "title_abstract = only titles and abstracts, AND/OR applied (default: title_abstract, chosen 22 Nov after check_queries.py)")
    parser.add_argument("--redo", action="store_true", help="re-run queries already finished today")
    parser.add_argument("--out", default=str(OUT_FILE), help=f"output CSV (default: {OUT_FILE})")
    add_config_options(parser)
    args = parser.parse_args()
    use_config_options(args)

    out = Path(args.out)
    progress_path = out.with_name(out.stem + "_progress.csv")
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    unknown = set(sources) - set(ALL_SOURCES)
    if unknown:
        parser.error(f"unknown source(s): {', '.join(sorted(unknown))}")
    cats = [c.strip() for c in args.arxiv_cats.split(",") if c.strip()]
    weights = load_weights()
    progress = load_progress(progress_path)

    print(f"Sources: {', '.join(sources)}; since {args.since}; at most {args.max_per_query} per query", flush=True)
    new_counts, skipped_counts, skip_reasons, failures, capped = {}, {}, {}, {}, {}
    for key in sources:
        source, source_type = SOURCE_INFO[key]
        new_counts[source], skipped_counts[source], failures[source], capped[source] = 0, 0, 0, []
        skip_reasons[source] = {"paratext": 0, "retracted": 0}
        print(f"\n{source}", flush=True)
        queries = source_queries(key, args.query)
        for query in queries:
            if not args.redo and done_today(progress, source, query):
                print(f"  already done today, skipped: {query}", flush=True)
                continue
            info = {}
            if key == "openalex":
                items, skipped, ok = fetch_openalex(query, args.since, args.max_per_query, args.email,
                                                    args.openalex_key, args.openalex_mode, info=info)
                for reason, n in info["skipped_by_reason"].items():
                    skip_reasons[source][reason] += n
            elif key == "pubmed":
                items, skipped, ok = fetch_pubmed(query, args.since, args.max_per_query, args.email, info=info)
            else:
                items, skipped, ok = fetch_arxiv(query, args.since, args.max_per_query, cats, info=info)
            n_fetched = len(items) + skipped
            new, _ = save_new(make_rows(items, source, source_type, weights), out)
            new_counts[source] += new
            skipped_counts[source] += skipped
            total = info.get("total")
            if is_capped(key, args.max_per_query, info):
                capped[source].append((query, n_fetched, total))
                log(COLLECTOR, "capped", f"{source}: {query}", f"{n_fetched} of {total} matches kept")
            print(f"  {n_fetched} of {'?' if total is None else total} matches: {len(items)} items, "
                  f"{new} new, {skipped} skipped{'' if ok else ', SOME REQUESTS FAILED'}: {query}", flush=True)
            if ok:
                mark_done(progress_path, source, query, len(items))
            else:
                failures[source] += 1
        # the queries themselves are in the "capped" lines above; here only the counts, which fit the log width
        capped_text = ", ".join(f"{n} of {t}" for _, n, t in capped[source])
        log(COLLECTOR, "done", source,
            f"{new_counts[source]} new rows, {skipped_counts[source]} skipped, "
            f"{failures[source]} queries with failures, {len(capped[source])} queries capped"
            f"{f' ({capped_text})' if capped_text else ''}")

    print_summary(out, new_counts, skipped_counts, skip_reasons, failures, capped)
    print(f"Progress kept in {progress_path}", flush=True)


if __name__ == "__main__":
    main()
