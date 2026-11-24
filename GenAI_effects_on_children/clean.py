"""Clean every collected source with one method: keep relevant items, group same stories.

All raw files have the same columns (id, source, source_type, weight, date,
title, text, url, query), so one set of rules applies to all of them. The rules
run in this order; an item is rejected by the first rule it fails, and the
rejected file says which one:

1. "no title": the title is empty.
2. "no readable date", "before scan period" (before --since), "dated in the
   future" (after --until; e.g. journal issues dated ahead).
3. Relevance: the title + text must have a term of config/ai_terms.csv AND a
   term of config/child_terms.csv (whole words, any case, * = any ending; see
   common.py). Otherwise "no AI term", "no child term" or "no AI or child term".
   Before matching, the label the collectors put at the start of the text
   ("Outlet: <name>." for news, "Institution: <name>." for policy) is removed,
   so an outlet name cannot make an item relevant. The outlet name is kept in
   its own column.
4. Language: the language of the TEXT (abstract) is detected with the
   langdetect library (fixed seed, so the result repeats) when the text has at
   least --min-language-words words; an item whose text is in a language not
   in --languages (default: en) is rejected as "language: <code>". Shorter
   texts (headline-only news and policy items) are not checked: on a few words
   detection is unreliable (on 22-24 Nov every short item it flagged was an
   English headline). An English title over a non-English abstract is
   rejected too: the word lists and the embeddings read the abstract.
5. Clinical use: AI used by clinicians or researchers to diagnose, treat or
   write medical information is out of scope (AI that children, parents or
   teachers use is in scope, health topics included). An item whose TITLE has a
   term of config/clinical_terms.csv is rejected as "clinical use: <term>".
   Only the title is used: abstracts often mention clinicians in passing.
6. Same item twice: two items with the same title (lower case, punctuation and
   spaces ignored) are one item, whatever their source or date (e.g. a paper
   found by both OpenAlex and PubMed). The first is kept; the others are
   rejected as "same title as <id>".
7. Same story: news and policy headlines that report the same story within
   --window-days of each other are grouped, with the word-overlap method of the
   Google ad news assignment (clean_news.py): headline words weighted by how
   rare they are, cosine similarity >= --similarity. Only headlines are
   compared, because Google News gives no summaries. The first item of a story
   is kept; the others are rejected as "same story as <id>".

"The first" means: earliest date, then highest source weight, then id, so the
result does not depend on the order of the raw files.

Every kept item gets story_size (how many items the rules 6 and 7 merged into
it, itself included) and story_source_types (the source types among them), so
later steps can tell a story told once from one told by news, policy and
research.

Outputs: data/clean/items.csv (kept, with the matched terms) and
data/clean/rejected.csv (every other item, with its reason).

Examples:
    python clean.py
    python clean.py --similarity 0.3 --window-days 7
    python clean.py --languages en,fr --min-language-words 30
    python clean.py --since 2023-01-01 --out /tmp/items.csv --rejected /tmp/rejected.csv
"""

import argparse
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from langdetect import DetectorFactory, LangDetectException, detect

import common

RAW_FILES = "data/raw/news.csv,data/raw/policy.csv,data/raw/academic.csv"
# Start of the scan period, as in the collectors (public release of ChatGPT).
SINCE = "2022-11-01"
# Same-story defaults taken from the Google ad news assignment (clean_news.py).
SIMILARITY = 0.25
WINDOW_DAYS = 4
# Source types whose items are news-like headlines, grouped by rule 6.
STORY_SOURCE_TYPES = ["news", "policy"]
# Languages kept by rule 4 (langdetect codes, comma-separated).
LANGUAGES = "en"
# Rule 4 checks only texts with at least this many words; a judgement call
# (below about 20 words langdetect mislabelled English headlines).
MIN_LANGUAGE_WORDS = 20
# langdetect is random unless seeded; a fixed seed makes rule 4 repeatable.
DetectorFactory.seed = 0
# Words that say nothing about which story a headline tells.
STOP_WORDS = {
    "a", "an", "the", "to", "of", "in", "on", "for", "and", "as", "is", "its", "s", "it", "at", "by", "with",
    "says", "say", "said", "new", "from", "that", "this", "how", "why", "what", "are", "be", "will", "could",
    "may", "after", "over", "amid", "into", "than", "more", "has", "have", "was", "were", "their", "they",
    "but", "not", "you", "your", "can", "about", "which", "who", "ai",
}


def split_label(text, source, source_type):
    """Return (outlet, body): the collector's 'Outlet:'/'Institution:' label and the rest of the text."""
    text = str(text)
    if source_type == "policy" and text.startswith(f"Institution: {source}."):
        return source, text[len(f"Institution: {source}."):].strip()
    if source_type == "news" and text.startswith("Outlet: "):
        rest = text[len("Outlet: "):]
        # Nearly always the text is only "Outlet: <name>." (Google gives no summary)
        if rest.endswith(".") and ". " not in rest:
            return rest[:-1], ""
        outlet, _, body = rest.partition(". ")
        return outlet, body.strip()
    return "", text


def title_key(title):
    """Return the title in lower case with only letters and digits, to spot the same title twice."""
    return re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip()


def story_words(text):
    """Return the set of words and numbers of a headline that can identify a story."""
    text = str(text).lower().replace("’", "'").replace("'s", "")
    words = set()
    for word in re.findall(r"[a-z]+|\$?\d+(?:\.\d+)?%?", text):
        if word in STOP_WORDS:
            continue
        # Crude stemming, so that "toys" and "toy" count as one word
        if word.isalpha() and len(word) > 4:
            word = re.sub(r"(ing|ers|er|ed|es|s)$", "", word)
        words.add(word)
    return words


def find_same_stories(titles, dates, similarity, window_days):
    """Return, for each headline, the position of the first headline of its story (dates sorted, oldest first)."""
    counts = [Counter(story_words(title)) for title in titles]
    using = Counter(word for count in counts for word in count)
    vectors = []
    for count in counts:
        vector = {word: n * math.log(len(counts) / using[word]) for word, n in count.items()}
        length = math.sqrt(sum(weight**2 for weight in vector.values())) or 1
        vectors.append({word: weight / length for word, weight in vector.items()})
    window = pd.Timedelta(days=window_days)
    first = list(range(len(titles)))
    for i in range(len(titles)):
        best = similarity
        for j in range(i - 1, -1, -1):
            if dates[i] - dates[j] > window:
                break
            alike = sum(weight * vectors[j].get(word, 0) for word, weight in vectors[i].items())
            if alike >= best:
                best = alike
                first[i] = first[j]
    return first


def load_raw(paths):
    """Read and stack the raw files, sorted so that 'first' means earliest, heaviest source, then id."""
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths]
    items = pd.concat(frames, ignore_index=True)
    items["weight_num"] = pd.to_numeric(items["weight"], errors="coerce")
    items["day"] = pd.to_datetime(items["date"], errors="coerce")
    items = items.sort_values(["day", "weight_num", "id"], ascending=[True, False, True], na_position="last")
    return items.reset_index(drop=True)


def reject_reason(row, since, until):
    """Return the reason from rules 1 and 2 why an item is rejected, or '' if it passes them."""
    if not row["title"].strip():
        return "no title"
    if pd.isna(row["day"]):
        return "no readable date"
    if row["day"] < since:
        return "before scan period"
    if row["day"] > until:
        return "dated in the future"
    return ""


def relevance_reason(ai_terms, child_terms):
    """Return the reason from rule 3, or '' if the item has both kinds of terms."""
    if not ai_terms and not child_terms:
        return "no AI or child term"
    if not ai_terms:
        return "no AI term"
    if not child_terms:
        return "no child term"
    return ""


def language_reason(text, languages, min_words):
    """Return the reason from rule 4 ('language: <code>'), or '' if the text is short or in a kept language."""
    if len(str(text).split()) < min_words:
        return ""
    try:
        code = detect(str(text))
    except LangDetectException:
        return ""
    return "" if code in languages else f"language: {code}"


def merge_groups(kept, first, reason_text):
    """Mark every item whose group starts elsewhere as rejected; add its size and source types to the group's first item."""
    rejected_ids = {}
    for position, start in enumerate(first):
        if start != position:
            rejected_ids[kept.index[position]] = f"{reason_text} {kept.iloc[start]['id']}"
    leader = [kept.index[start] for start in first]
    sizes = pd.Series(leader).groupby(leader).size()
    types = kept.groupby(leader)["story_source_types"].agg(lambda s: ";".join(sorted(set(";".join(s).split(";")))))
    return rejected_ids, sizes, types


def main():
    """Apply the rules to all raw files and save the kept and the rejected items."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default=RAW_FILES, help=f"raw files, comma-separated (default: {RAW_FILES})")
    parser.add_argument("--since", default=SINCE, help=f"first day of the scan period (default: {SINCE})")
    parser.add_argument("--until", default=str(date.today()), help="last day; later dates are rejected (default: today)")
    parser.add_argument("--ai-terms", default="config/ai_terms.csv", help="AI word list (default: config/ai_terms.csv)")
    parser.add_argument("--child-terms", default="config/child_terms.csv", help="child word list (default: config/child_terms.csv)")
    parser.add_argument("--languages", default=LANGUAGES,
                        help=f"languages kept by rule 4, langdetect codes, comma-separated (default: {LANGUAGES})")
    parser.add_argument("--min-language-words", type=int, default=MIN_LANGUAGE_WORDS,
                        help=f"rule 4 checks only texts with at least this many words; a judgement call (default: {MIN_LANGUAGE_WORDS})")
    parser.add_argument("--clinical-terms", default="config/clinical_terms.csv",
                        help="clinical-use word list, matched in titles (default: config/clinical_terms.csv)")
    parser.add_argument("--similarity", type=float, default=SIMILARITY,
                        help=f"how much two headlines must overlap to be the same story, 0-1; a judgement call (default: {SIMILARITY})")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS,
                        help=f"only compare headlines this many days apart; a judgement call (default: {WINDOW_DAYS})")
    parser.add_argument("--out", default="data/clean/items.csv", help="kept items (default: data/clean/items.csv)")
    parser.add_argument("--rejected", default="data/clean/rejected.csv", help="rejected items (default: data/clean/rejected.csv)")
    args = parser.parse_args()

    items = load_raw(args.raw.split(","))
    n_in = len(items)
    patterns = {
        "ai": common.terms_pattern(common.load_terms(args.ai_terms)),
        "child": common.terms_pattern(common.load_terms(args.child_terms)),
        "clinical": common.terms_pattern(common.load_terms(args.clinical_terms)),
    }

    # Remove the collectors' label from the text, keep the outlet in its own column
    labels = [split_label(t, s, st) for t, s, st in zip(items["text"], items["source"], items["source_type"])]
    items["outlet"] = [outlet for outlet, _ in labels]
    items["text"] = [body for _, body in labels]

    # Rules 1 to 5, one row at a time
    full = items["title"] + " " + items["text"]
    items["ai_terms"] = [common.find_terms(patterns["ai"], t) for t in full]
    items["child_terms"] = [common.find_terms(patterns["child"], t) for t in full]
    clinical = [common.find_terms(patterns["clinical"], t) for t in items["title"]]
    since, until = pd.Timestamp(args.since), pd.Timestamp(args.until)
    languages = {code.strip() for code in args.languages.split(",")}
    reasons = []
    for (_, row), ai, child, clin in zip(items.iterrows(), items["ai_terms"], items["child_terms"], clinical):
        reason = reject_reason(row, since, until) or relevance_reason(ai, child)
        if not reason:
            reason = language_reason(row["text"], languages, args.min_language_words)
        if not reason and clin:
            reason = f"clinical use: {', '.join(clin)}"
        reasons.append(reason)
    items["reason"] = reasons
    items["story_source_types"] = items["source_type"]

    # Rule 6: the same title twice, any source and date
    kept = items[items["reason"] == ""].copy()
    keys = kept["title"].map(title_key)
    first_seen = {}
    first = [first_seen.setdefault(key, position) for position, key in enumerate(keys)]
    rejected_ids, sizes, types = merge_groups(kept, first, "same title as")
    items.loc[list(rejected_ids), "reason"] = pd.Series(rejected_ids)
    kept = kept.drop(index=list(rejected_ids))
    kept["story_size"] = sizes.reindex(kept.index).fillna(1).astype(int)
    kept["story_source_types"] = types.reindex(kept.index).fillna(kept["story_source_types"])

    # Rule 7: same story among news and policy headlines
    headlines = kept[kept["source_type"].isin(STORY_SOURCE_TYPES)]
    first = find_same_stories(headlines["title"].tolist(), headlines["day"].tolist(), args.similarity, args.window_days)
    rejected_ids, _, _ = merge_groups(headlines, first, "same story as")
    leaders = [headlines.index[start] for start in first]
    story_size = headlines["story_size"].groupby(leaders).sum()
    story_types = headlines.groupby(leaders)["story_source_types"].agg(
        lambda s: ";".join(sorted(set(";".join(s).split(";")))))
    items.loc[list(rejected_ids), "reason"] = pd.Series(rejected_ids)
    kept = kept.drop(index=list(rejected_ids))
    kept.loc[story_size.index.intersection(kept.index), "story_size"] = story_size
    kept.loc[story_types.index.intersection(kept.index), "story_source_types"] = story_types

    # Save
    for column in ["ai_terms", "child_terms"]:
        kept[column] = kept[column].map("; ".join)
    out_columns = common.COLUMNS + ["outlet", "ai_terms", "child_terms", "story_size", "story_source_types"]
    kept = kept[out_columns]
    rejected = items[items["reason"] != ""]
    rejected = rejected[common.COLUMNS + ["outlet", "reason"]]
    for path, table in [(Path(args.out), kept), (Path(args.rejected), rejected)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    # Summary
    print(f"Items in: {n_in} from {len(args.raw.split(','))} raw files")
    print(f"Kept: {len(kept)}   rejected: {len(rejected)}")
    print("\nKept per source type:")
    print(kept["source_type"].value_counts().to_string())
    print("\nRejected per reason (language, clinical use, same title and same story grouped):")
    short = rejected["reason"].str.replace(r"^(clinical use|same title as|same story as|language).*", r"\1", regex=True)
    print(short.value_counts().to_string())
    print("\nKept per source type and year:")
    print(pd.crosstab(kept["source_type"], kept["date"].str[:4]).to_string())
    # Raw files keep the rows of queries that were later replaced: show them apart
    current = set(pd.read_csv(common.QUERIES_FILE)["query"])
    items["query_status"] = items["query"].map(lambda q: "current query" if q in current else "retired query")
    items["result"] = items["reason"].map(lambda r: "kept" if r == "" else "rejected")
    print("\nKept and rejected, by whether the item's query is still in config/queries.csv:")
    print(pd.crosstab([items["source_type"], items["query_status"]], items["result"], margins=True).to_string())
    multi = kept[kept["story_source_types"].str.contains(";")]
    print(f"\nKept items told by more than one source type: {len(multi)}")
    print(f"\nSaved {args.out} and {args.rejected}")


if __name__ == "__main__":
    main()
