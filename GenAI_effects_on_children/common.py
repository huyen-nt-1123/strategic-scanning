"""Shared helpers for the collectors of the GenAI and children scan.

Every collector (collect_news.py, collect_academic.py, collect_policy.py,
collect_community.py) writes rows with the same columns, so that all later
steps treat every source the same way:

    id, source, source_type, weight, date, title, text, url, query

Rules applied here, for every source:

- id: a short hash of the source name and the item's own key (DOI, PubMed id,
  link...). The same item always gets the same id, which is how a weekly run
  knows what it has already collected.
- Incremental save: new rows are added to the existing raw file; rows whose id
  is already there are skipped, never overwritten.
- weight: read from config/source_weights.csv (a judgement call kept out of
  the code).
- Every request that fails, and a summary of every collector run, is appended
  to data/run_log.csv instead of being dropped silently. Test runs can point
  the log elsewhere with the SCAN_RUN_LOG environment variable.
- Word lists (config/ai_terms.csv, child_terms.csv...) are matched as whole words,
  any case; a term ending in * matches any ending (chatbot* = chatbots); a
  space or hyphen in a term matches either; matches are reported as the term
  written in the list, so counts don't split "chatbot" and "chatbots".
- A source type with no weight in config/source_weights.csv stops the run with
  an error instead of saving an empty weight.
- API keys never go in the code: they are read from environment variables,
  which a local `.env` file (one NAME=value per line, never submitted or
  committed) can fill in. Anything that looks like a key (api_key=..., token=...)
  is replaced by *** before it is written to the log or printed.

This file is imported by the collectors; it is not run on its own.
"""

import hashlib
import html
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# The common columns, in this order, for every raw file.
COLUMNS = ["id", "source", "source_type", "weight", "date", "title", "text", "url", "query"]
# Columns of the run log: one row per failed request and one per finished run.
LOG_COLUMNS = ["time", "collector", "status", "target", "message"]
# Where the judgement-call weights and the editable queries live.
WEIGHTS_FILE = Path("config/source_weights.csv")
QUERIES_FILE = Path("config/queries.csv")
# A test run can set the SCAN_RUN_LOG environment variable to keep its lines
# out of the real log, e.g. SCAN_RUN_LOG=/tmp/test_log.csv python collect_news.py ...
RUN_LOG = Path(os.environ.get("SCAN_RUN_LOG", "data/run_log.csv"))
# Identify the script politely to the APIs (no personal details).
HEADERS = {"User-Agent": "strategic-scanning-coursework/1.0 (python-requests)"}
# Wait this long before retrying a failed request, then double it each time.
RETRY_WAIT_SECONDS = 5
# Private settings (API keys) live in this file, kept out of the submission and of git.
ENV_FILE = Path(".env")
# Parameter names whose values are secret and must never reach the log.
# The name must not be glued to letters before it ("monkey=" is not a key), but
# may follow a "-", "_" or "." ("x-api-key:", "access_token=").
SECRET_PATTERN = re.compile(
    r"((?<![a-z])(?:api[-_]?key|token|secret|password)['\"]?\s*[:=]\s*['\"]?|bearer\s+)[^&'\",\s}<]+",
    re.IGNORECASE)


def load_env_file(path=ENV_FILE):
    """Copy NAME=value lines of the .env file into the environment, unless already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def hide_secrets(text):
    """Replace the value of anything that looks like a key or token with ***."""
    return SECRET_PATTERN.sub(r"\1***", str(text))


# Read the .env file as soon as a collector imports this module, so its
# options can default to the keys it holds.
load_env_file()


def make_id(source, key):
    """Return a stable 16-character id for an item, from its source and own key."""
    return hashlib.sha1(f"{source}|{key}".encode("utf-8")).hexdigest()[:16]


def clean_text(value):
    """Turn HTML or messy text into one plain line: no tags, entities or extra spaces."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def add_config_options(parser):
    """Add the --queries, --weights and --log options shared by every collector."""
    parser.add_argument("--queries", default=str(QUERIES_FILE), help=f"search queries file (default: {QUERIES_FILE})")
    parser.add_argument("--weights", default=str(WEIGHTS_FILE), help=f"source weights file (default: {WEIGHTS_FILE})")
    parser.add_argument("--log", default=str(RUN_LOG),
                        help=f"run log; point test runs elsewhere (default: {RUN_LOG}, or the SCAN_RUN_LOG variable)")


def use_config_options(args):
    """Make the files chosen with --queries, --weights and --log the ones this module reads and writes."""
    global QUERIES_FILE, WEIGHTS_FILE, RUN_LOG
    QUERIES_FILE, WEIGHTS_FILE, RUN_LOG = Path(args.queries), Path(args.weights), Path(args.log)


def load_weights(path=None):
    """Return {source_type: weight} from the weights file."""
    table = pd.read_csv(path or WEIGHTS_FILE)
    return dict(zip(table["source_type"], table["weight"]))


def load_queries(collector, path=None):
    """Return the list of search queries set for one collector in config/queries.csv."""
    table = pd.read_csv(path or QUERIES_FILE)
    return table.loc[table["collector"] == collector, "query"].tolist()


def log(collector, status, target="", message="", path=None):
    """Append one line to the run log (status is e.g. 'failed' or 'done'); the log is RUN_LOG unless `path` is given."""
    path = Path(path or RUN_LOG)
    row = {
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "collector": collector,
        "status": status,
        "target": hide_secrets(target)[:300],
        "message": hide_secrets(message)[:500],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)


def get(url, collector, params=None, retries=3, timeout=30):
    """GET a URL with retries; return the response, or None after logging the failure."""
    wait = RETRY_WAIT_SECONDS
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            # 429 = too many requests, 5xx = server trouble: worth waiting and retrying
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            if attempt == retries:
                target = url if not params else f"{url} {params}"
                log(collector, "failed", target, error)
                print(f"  FAILED after {retries} tries: {hide_secrets(error)}")
                return None
            time.sleep(wait)
            wait *= 2
    return None


def make_rows(items, source, source_type, weights):
    """Turn a list of dicts (with key, date, title, text, url, query) into rows with the common columns."""
    if source_type not in weights:
        raise KeyError(f"source type '{source_type}' has no weight in {WEIGHTS_FILE}")
    rows = []
    for item in items:
        rows.append(
            {
                "id": make_id(source, item["key"]),
                "source": source,
                "source_type": source_type,
                "weight": weights[source_type],
                "date": item.get("date", ""),
                "title": clean_text(item.get("title")),
                "text": clean_text(item.get("text")),
                "url": item.get("url", ""),
                "query": item.get("query", ""),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def save_new(rows, path):
    """Add the rows whose id is not yet in the file; return (new rows added, rows in file)."""
    path = Path(path)
    rows = rows.drop_duplicates(subset="id")
    if rows.empty and not path.exists():
        return 0, 0
    if path.exists():
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        rows = rows[~rows["id"].isin(set(old["id"]))]
        combined = pd.concat([old, rows.astype(str)], ignore_index=True)
    else:
        combined = rows
    combined = combined.sort_values(["date", "id"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return len(rows), len(combined)


def load_terms(path, group=None):
    """Return the terms of a word list in config/ (only one group's terms if `group` is given)."""
    table = pd.read_csv(path)
    if group is not None:
        table = table[table["group"] == group]
    # blank rows and stray spaces would otherwise make a term that never matches
    terms = table["term"].dropna().astype(str).str.strip()
    return terms[terms != ""].tolist()


# For each pattern built by terms_pattern: which config term each named group stands for.
PATTERN_TERMS = {}


def terms_pattern(terms):
    """Build one case-insensitive pattern matching any term as a whole word; a final * means 'any ending'.

    A space or a hyphen inside a term matches a space or a hyphen in the text
    ("pre-k" = "pre k", "large language model" = "large-language model").
    Longer terms are tried first, so "AI tutor" wins over "AI" in "AI tutors".
    An empty list gives a pattern that never matches.
    """
    terms = sorted(set(terms), key=lambda t: (-len(t), t))
    if not terms:
        return re.compile(r"(?!)")
    parts = []
    for number, term in enumerate(terms):
        stem = re.escape(term.rstrip("*"))
        stem = stem.replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]+")
        parts.append(f"(?P<t{number}>" + stem + (r"\w*" if term.endswith("*") else "") + ")")
    # (?<!\w) and (?!\w) are word edges that also work after a final dot, as in "A.I."
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)
    PATTERN_TERMS[pattern.pattern] = {f"t{number}": term for number, term in enumerate(terms)}
    return pattern


def find_terms(pattern, text):
    """Return the sorted list of config terms (as written in the word list) a pattern finds in a text."""
    names = PATTERN_TERMS[pattern.pattern]
    return sorted({names[match.lastgroup] for match in pattern.finditer(str(text))})
