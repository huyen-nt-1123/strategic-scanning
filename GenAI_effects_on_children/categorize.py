"""Tag every clean item with its age band and its known categories.

Input: data/clean/items.csv (from clean.py). Nothing is filtered out: every item
is kept and gets new columns, so no rejected file is needed.

Age band (title + text, word list config/child_terms.csv, its `group` column):

- age_bands: every specific band whose terms the item has, from the groups
  age_0_2 -> "0-2", age_3_5 -> "3-5", age_6_8 -> "6-8", gen_alpha -> "Gen Alpha",
  joined by "; " (an item can mention toddlers and preschoolers).
- age_band: one label per item. The band if there is exactly one; "several"
  if there are more; "children" if there is none but a child_generic term
  (child*, kids...) is there; "school (age not given)" if the item has only
  school_generic terms (K-12, pupil*, classroom*, schools). The last one is
  weaker evidence about children aged 0-8.
- older_terms: terms of config/older_terms.csv (teen*, high school*...) found
  in the item. Not a filter, only a trace: an item can be about both young
  children and teenagers.

Categories (word list config/categories.csv, columns term, category, pestle):

- An item gets every category that has at least --min-terms different terms
  in the fields chosen with --fields (default: title and text). The default of
  1 term is a judgement call: long abstracts match more terms than headlines,
  so a higher value favours news and policy items less.
- categories: the categories found, joined by "; ", or "none" if no category
  matched. "none" items are what the clustering step uses to spot candidate
  new themes.
- pestle: the PESTLE letters of those categories (one letter per category is
  set in the word list; the run stops if a category has two).
- category_terms: which terms matched, as "category: term, term | ...", so
  every tag can be checked.
- below_min_terms: the same, for categories that matched fewer than
  --min-terms terms and so were not given (empty when --min-terms is 1).
- n_categories: how many categories the item has.
- Excluded phrases (config/category_exclusions.csv, columns category, phrase,
  note): before a category's terms are matched, its excluded phrases are
  blanked out of the text, so a term inside them does not count for that
  category (e.g. "literacy" inside "AI literacy" is not language
  development). Only that category is affected; the phrase can still match
  another category's own terms. Phrases are matched like every word list
  (whole words, any case, * = any ending). A missing file means no
  exclusions.
- excluded_phrases: the excluded phrases found, as "category: phrase, ...",
  so every blanked match can be traced.

An item with no child term at all gets an empty age_band (clean.py should
never let one through; the summary counts them).

Second output: data/clean/category_term_counts.csv, one row per category and
term: items_tagged (items the term helped give that category), items_below_min
(items where the term matched but the category stayed under --min-terms) and
share_of_category (items_tagged / items of that category). It shows when one
broad term carries a whole category. A term ending in * is counted as one
term, so its different endings are not shown apart.

The same input always gives the same output (no randomness, rows keep their
order).

Examples:
    python categorize.py
    python categorize.py --fields title
    python categorize.py --min-terms 2 --out /tmp/items_tagged.csv --term-counts /tmp/term_counts.csv
"""

import argparse
from pathlib import Path

import pandas as pd

import common

# Which group of config/child_terms.csv gives which age band label.
AGE_GROUPS = {"age_0_2": "0-2", "age_3_5": "3-5", "age_6_8": "6-8", "gen_alpha": "Gen Alpha"}
# The other groups of the child word list, used when no specific band is found.
GENERIC_GROUP = "child_generic"
SCHOOL_GROUP = "school_generic"
GENERIC_LABEL = "children"
SCHOOL_LABEL = "school (age not given)"
SEVERAL_LABEL = "several"
# Label for an item that matches no known category.
NO_CATEGORY = "none"
# Fields of an item that categories can be matched in.
FIELD_CHOICES = ["title", "text"]
# A category needs at least this many different terms in an item; a judgement call.
MIN_TERMS = 1
# Phrases that must not count for a category (e.g. "AI literacy" for language development).
EXCLUSIONS_FILE = "config/category_exclusions.csv"


def load_exclusions(path):
    """Return {category: pattern of its excluded phrases}; {} if the file does not exist."""
    path = Path(path)
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table["phrase"] = table["phrase"].str.strip()
    table = table[table["phrase"] != ""]
    return {category: common.terms_pattern(group["phrase"].tolist())
            for category, group in table.groupby(table["category"].str.strip(), sort=False)}


def blank_phrases(text, pattern):
    """Replace every match of `pattern` in the text by a space (no change if pattern is None)."""
    return text if pattern is None else pattern.sub(" ", text)


def load_categories(path):
    """Read the category word list; return (terms per category, PESTLE letter per category), in file order."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in ["term", "category", "pestle"]:
        table[column] = table[column].str.strip()
    table = table[table["term"] != ""]
    letters = table.groupby("category", sort=False)["pestle"].unique()
    mixed = letters[letters.map(len) > 1]
    if not mixed.empty:
        raise ValueError(f"{path}: these categories have more than one PESTLE letter: {dict(mixed.map(list))}")
    terms = table.groupby("category", sort=False)["term"].apply(list).to_dict()
    return terms, letters.map(lambda values: values[0]).to_dict()


def load_child_groups(path):
    """Return {term: group} for the child word list, stopping if a group is unknown."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table = table[table["term"].str.strip() != ""]
    known = set(AGE_GROUPS) | {GENERIC_GROUP, SCHOOL_GROUP}
    unknown = set(table["group"]) - known
    if unknown:
        raise ValueError(f"{path}: groups not handled by categorize.py: {sorted(unknown)}")
    return dict(zip(table["term"].str.strip(), table["group"]))


def age_tags(child_terms, term_group):
    """Return (age_bands, age_band) for the child terms found in one item."""
    groups = {term_group[term] for term in child_terms}
    bands = [label for group, label in AGE_GROUPS.items() if group in groups]
    if len(bands) == 1:
        band = bands[0]
    elif bands:
        band = SEVERAL_LABEL
    elif GENERIC_GROUP in groups:
        band = GENERIC_LABEL
    elif SCHOOL_GROUP in groups:
        band = SCHOOL_LABEL
    else:
        band = ""
    return "; ".join(bands), band


def describe(found):
    """Write {category: terms} as 'category: term, term | category: term'."""
    return " | ".join(f"{category}: {', '.join(terms)}" for category, terms in found.items())


def category_tags(found, min_terms, letters):
    """Return (categories, pestle, category_terms, below_min_terms, n_categories) from {category: terms found}."""
    kept = {category: terms for category, terms in found.items() if len(terms) >= min_terms}
    below = {category: terms for category, terms in found.items() if len(terms) < min_terms}
    if not kept:
        return NO_CATEGORY, "", "", describe(below), 0
    letter_list = sorted({letters[category] for category in kept})
    return "; ".join(kept), "".join(letter_list), describe(kept), describe(below), len(kept)


def term_counts(found_per_item, kept_categories):
    """Count, per category and term, the items where the term gave the category and where it stayed below --min-terms."""
    rows = []
    for found, kept in zip(found_per_item, kept_categories):
        for category, terms in found.items():
            status = "items_tagged" if category in kept else "items_below_min"
            rows.extend({"category": category, "term": term, "status": status} for term in terms)
    table = pd.DataFrame(rows, columns=["category", "term", "status"])
    counts = pd.crosstab([table["category"], table["term"]], table["status"])
    counts = counts.reindex(columns=["items_tagged", "items_below_min"], fill_value=0).reset_index()
    counts.columns.name = None
    per_category = pd.Series([c for kept in kept_categories for c in kept]).value_counts()
    counts["share_of_category"] = (counts["items_tagged"] / counts["category"].map(per_category)).fillna(0).round(3)
    return counts.sort_values(["category", "items_tagged", "term"], ascending=[True, False, True]).reset_index(drop=True)


def main():
    """Tag the clean items and save them, with the term counts behind the categories."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="inp", default="data/clean/items.csv", help="clean items (default: data/clean/items.csv)")
    parser.add_argument("--child-terms", default="config/child_terms.csv", help="child word list with age groups (default: config/child_terms.csv)")
    parser.add_argument("--older-terms", default="config/older_terms.csv", help="teenager word list, traced only (default: config/older_terms.csv)")
    parser.add_argument("--categories", default="config/categories.csv", help="category word list (default: config/categories.csv)")
    parser.add_argument("--exclusions", default=EXCLUSIONS_FILE,
                        help=f"phrases that do not count for a category (default: {EXCLUSIONS_FILE})")
    parser.add_argument("--fields", default="title,text",
                        help=f"fields the categories are matched in, comma-separated, from {FIELD_CHOICES} (default: title,text)")
    parser.add_argument("--min-terms", type=int, default=MIN_TERMS,
                        help=f"different terms a category needs in one item; a judgement call (default: {MIN_TERMS})")
    parser.add_argument("--out", default="data/clean/items_tagged.csv", help="tagged items (default: data/clean/items_tagged.csv)")
    parser.add_argument("--term-counts", default="data/clean/category_term_counts.csv",
                        help="items per category and term (default: data/clean/category_term_counts.csv)")
    args = parser.parse_args()

    fields = [field.strip() for field in args.fields.split(",")]
    if not fields or set(fields) - set(FIELD_CHOICES):
        parser.error(f"--fields must be taken from {FIELD_CHOICES}")
    if args.min_terms < 1:
        parser.error("--min-terms must be at least 1")

    items = pd.read_csv(args.inp, dtype=str, keep_default_na=False)
    full = items["title"] + " " + items["text"]
    chosen = items[fields].agg(" ".join, axis=1)

    # Age band, from the same title + text that clean.py matched
    term_group = load_child_groups(args.child_terms)
    child_pattern = common.terms_pattern(list(term_group))
    tags = [age_tags(common.find_terms(child_pattern, text), term_group) for text in full]
    items["age_bands"] = [bands for bands, _ in tags]
    items["age_band"] = [band for _, band in tags]
    older_pattern = common.terms_pattern(common.load_terms(args.older_terms))
    items["older_terms"] = [", ".join(common.find_terms(older_pattern, text)) for text in full]

    # Categories, in the chosen fields
    category_terms, letters = load_categories(args.categories)
    patterns = {category: common.terms_pattern(terms) for category, terms in category_terms.items()}
    exclusions = load_exclusions(args.exclusions)
    unknown = set(exclusions) - set(category_terms)
    if unknown:
        parser.error(f"{args.exclusions}: categories not in {args.categories}: {sorted(unknown)}")
    found_per_item, excluded_per_item = [], []
    for text in chosen:
        found = {category: common.find_terms(pattern, blank_phrases(text, exclusions.get(category)))
                 for category, pattern in patterns.items()}
        found_per_item.append({category: terms for category, terms in found.items() if terms})
        excluded = {category: common.find_terms(pattern, text) for category, pattern in exclusions.items()}
        excluded_per_item.append({category: phrases for category, phrases in excluded.items() if phrases})
    tags = [category_tags(found, args.min_terms, letters) for found in found_per_item]
    for position, column in enumerate(["categories", "pestle", "category_terms", "below_min_terms", "n_categories"]):
        items[column] = [tag[position] for tag in tags]
    items["excluded_phrases"] = [describe(excluded) for excluded in excluded_per_item]
    kept_categories = [[] if tag[0] == NO_CATEGORY else tag[0].split("; ") for tag in tags]
    counts = term_counts(found_per_item, kept_categories)

    # Save
    for path, table in [(Path(args.out), items), (Path(args.term_counts), counts)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    # Summary
    print(f"Items in: {len(items)}   tagged: {len(items)} (nothing is filtered)")
    print(f"Categories matched in: {', '.join(fields)}   min terms per category: {args.min_terms}")
    print("\nItems per age band:")
    print(items["age_band"].replace("", "(no child term found)").value_counts().to_string())
    print("\nItems per age band and source type:")
    print(pd.crosstab(items["age_band"], items["source_type"], margins=True).to_string())
    older = items["older_terms"] != ""
    print(f"\nItems that also have a teenager term (config/older_terms.csv): {older.sum()} of {len(items)}")
    print("\nItems per category (an item can have several):")
    per_category = pd.Series([c for kept in kept_categories for c in kept]).value_counts()
    per_category[NO_CATEGORY] = (items["categories"] == NO_CATEGORY).sum()
    table = pd.DataFrame({"pestle": per_category.index.map(lambda c: letters.get(c, "")),
                          "items": per_category, "share": (per_category / len(items)).round(3)})
    print(table.to_string())
    print("\nShare of items with no category, per source type:")
    print(items.groupby("source_type")["categories"].apply(lambda s: round((s == NO_CATEGORY).mean(), 3)).to_string())
    print(f"\nExcluded phrases ({args.exclusions}), items where each was blanked:")
    blanked = pd.Series([f"{category}: {phrase}" for excluded in excluded_per_item
                         for category, phrases in excluded.items() for phrase in phrases])
    print(blanked.value_counts().to_string() if not blanked.empty else "  none")
    below = (items["below_min_terms"] != "").sum()
    print(f"\nItems with a category left below --min-terms: {below}")
    print("\nItems per number of categories:")
    print(items["n_categories"].value_counts().sort_index().to_string())
    print("\nTop 3 terms per category (share of that category's items):")
    for category, group in counts.groupby("category", sort=False):
        top = ", ".join(f"{t} {s:.0%}" for t, s in zip(group["term"].head(3), group["share_of_category"].head(3)))
        print(f"  {category}: {top}")
    print(f"\nSaved {args.out} and {args.term_counts}")


if __name__ == "__main__":
    main()
