"""Show how a category word is used: its neighbour words and a random sample in context.

A checking tool, not part of the weekly run (like check_queries.py). It helps
decide whether a broad term of config/categories.csv tags what its category
means, or something else (e.g. "literacy" inside "AI literacy").

Input: data/clean/items_tagged.csv (from categorize.py).

Rules:

- A check is written "category=term", with the term exactly as in
  config/categories.csv (e.g. "safety and harmful content=safety"). Only items
  that got that category through that term (column category_terms) are used.
- The term is matched the same way categorize.py matches it (same word
  pattern, same fields, same excluded phrases of config/category_exclusions.csv
  blanked out first), so only the matches that counted are shown.
- Neighbours: for every match, the word just before and the word just after
  (letters, digits, - and '). Counted over all matching items: occurrences
  and the number of different items. Nothing is filtered or judged.
- Sample: --sample-size items drawn at random with a fixed --seed (the same
  input gives the same sample), each with up to 3 matches shown with
  --context characters on each side.

Outputs (folder --out-dir, default data/term_check/):

- neighbours.csv: check, side (before / after), word, occurrences, items
- samples.csv: check, id, source_type, date, title, url, n_matches, contexts

Examples:
    python check_terms.py
    python check_terms.py --check "AI companions and toys=humanoid robot*" --sample-size 40
    python check_terms.py --seed 2 --out-dir /tmp/term_check
"""

import argparse
import re
from pathlib import Path

import pandas as pd

import categorize
import common

# The broad terms flagged on 24 Nov (robot* was then replaced by specific phrases, social robot* is the largest).
DEFAULT_CHECKS = [
    "language and cognitive development=literacy",
    "AI companions and toys=social robot*",
    "safety and harmful content=safety",
]
# Items drawn per check; enough to read in a few minutes.
SAMPLE_SIZE = 25
# Fixed seed so the sample is the same on every run.
SEED = 0
# Characters shown on each side of a match.
CONTEXT = 80
# Most matches shown per sampled item.
MAX_CONTEXTS = 3
# Neighbour words listed per side in the printed summary.
TOP_NEIGHBOURS = 12
# A "word" next to a match.
WORD = r"[\w'’-]+"


def parse_check(check):
    """Split 'category=term' into (category, term)."""
    if "=" not in check:
        raise ValueError(f"check '{check}' must be written category=term")
    category, term = check.split("=", 1)
    return category.strip(), term.strip()


def has_term(category_terms, category, term):
    """Return True if the item got `category` through `term` (column category_terms)."""
    for part in category_terms.split(" | "):
        name, _, terms = part.partition(": ")
        if name == category and term in terms.split(", "):
            return True
    return False


def neighbours(text, match):
    """Return (word before, word after) a match, lower case, '' at the edges."""
    before = re.findall(WORD, text[:match.start()])
    after = re.findall(WORD, text[match.end():])
    return (before[-1].lower() if before else "", after[0].lower() if after else "")


def context(text, match, width):
    """Return the match with `width` characters on each side, the match in [brackets]."""
    start, end = max(0, match.start() - width), min(len(text), match.end() + width)
    return ("..." if start else "") + text[start:match.start()] + "[" + match.group() + "]" + text[match.end():end] + ("..." if end < len(text) else "")


def main():
    """Count the neighbour words of each checked term and save a sample in context."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="inp", default="data/clean/items_tagged.csv", help="tagged items (default: data/clean/items_tagged.csv)")
    parser.add_argument("--check", action="append",
                        help=f"category=term to check; repeat for several (default: {DEFAULT_CHECKS})")
    parser.add_argument("--exclusions", default=categorize.EXCLUSIONS_FILE,
                        help=f"phrases blanked before matching, as in categorize.py (default: {categorize.EXCLUSIONS_FILE})")
    parser.add_argument("--fields", default="title,text", help="fields matched, as in categorize.py (default: title,text)")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help=f"items drawn per check (default: {SAMPLE_SIZE})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"random seed of the sample (default: {SEED})")
    parser.add_argument("--context", type=int, default=CONTEXT, help=f"characters shown on each side of a match (default: {CONTEXT})")
    parser.add_argument("--out-dir", default="data/term_check", help="output folder (default: data/term_check)")
    args = parser.parse_args()

    items = pd.read_csv(args.inp, dtype=str, keep_default_na=False)
    fields = [field.strip() for field in args.fields.split(",")]
    texts = items[fields].agg(" ".join, axis=1)
    exclusions = categorize.load_exclusions(args.exclusions)

    neighbour_rows, sample_rows = [], []
    for check in args.check or DEFAULT_CHECKS:
        category, term = parse_check(check)
        pattern = common.terms_pattern([term])
        tagged = items["category_terms"].map(lambda value: has_term(value, category, term))
        print(f"\n== {check}: {tagged.sum()} items")
        if not tagged.any():
            continue
        matches = {}
        for position in items.index[tagged]:
            text = categorize.blank_phrases(texts[position], exclusions.get(category))
            matches[position] = (text, list(pattern.finditer(text)))
        # neighbour words over all tagged items
        for position, (text, found) in matches.items():
            for match in found:
                before, after = neighbours(text, match)
                neighbour_rows.append({"check": check, "side": "before", "word": before, "id": items.at[position, "id"]})
                neighbour_rows.append({"check": check, "side": "after", "word": after, "id": items.at[position, "id"]})
        # random sample in context
        drawn = items[tagged].sample(n=min(args.sample_size, tagged.sum()), random_state=args.seed)
        for position, row in drawn.iterrows():
            text, found = matches[position]
            sample_rows.append({
                "check": check, "id": row["id"], "source_type": row["source_type"], "date": row["date"],
                "title": row["title"], "url": row["url"], "n_matches": len(found),
                "contexts": " || ".join(context(text, match, args.context) for match in found[:MAX_CONTEXTS]),
            })

    table = pd.DataFrame(neighbour_rows, columns=["check", "side", "word", "id"])
    counts = (table.groupby(["check", "side", "word"], sort=False)
              .agg(occurrences=("id", "size"), items=("id", "nunique")).reset_index()
              .sort_values(["check", "side", "occurrences", "word"], ascending=[True, False, False, True]))
    samples = pd.DataFrame(sample_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / "neighbours.csv", index=False)
    samples.to_csv(out_dir / "samples.csv", index=False)

    for check, group in counts.groupby("check", sort=False):
        print(f"\n{check}")
        for side in ["before", "after"]:
            part = group[group["side"] == side].head(TOP_NEIGHBOURS)
            words = ", ".join(f"{w or '(edge)'} {n}" for w, n in zip(part["word"], part["items"]))
            print(f"  word {side} (items): {words}")
    print(f"\nSaved {out_dir / 'neighbours.csv'} ({len(counts)} rows) and {out_dir / 'samples.csv'} ({len(samples)} rows)")


if __name__ == "__main__":
    main()
