"""Describe one or more groups or topics of the clustering, to read what they hold.

A reading aid run by hand (not part of the weekly run). It reads the outputs of
cluster.py and categorize.py and prints, for the selected groups or topics:

1. their topics: size, label, share of category "none", share of news, top words
2. items per source type and topic
3. items per quarter (by item date)
4. share of items per known category (an item can have several)
5. gap words: words frequent in the selection's items with category "none"
   and rarer in all items (ratio of the two shares, words in at least
   --min-count of those items, English stop words removed). The column
   in_categories says whether config/categories.csv already matches the
   word. They are candidates for the word list, not conclusions: adding one
   is a decision for config/categories.csv.
6. --examples titles per topic, drawn at random with a fixed --seed (the
   same every run)

It also saves the selected items (id, group, topic, label, date, source_type,
age_band, categories, title) to --out, for reading in full.

Group and topic numbers are those of the last cluster.py run and change when
it is rerun: pick them from data/topic_groups.csv or data/topics.csv.

Examples:
    python inspect_topics.py --group 2
    python inspect_topics.py --group 2,4 --examples 10
    python inspect_topics.py --topic 35 --out /tmp/topic35.csv
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import common

# Label written by categorize.py for an item with no known category.
NO_CATEGORY = "none"
# Titles shown per topic, and the seed that draws them.
EXAMPLES = 5
SEED = 0
# Gap words: shown, and how many "none" items a word needs to be shown at all.
GAP_WORDS = 20
MIN_COUNT = 5


def numbers(value):
    """Turn '2,4' into [2, 4]."""
    return [int(part) for part in value.split(",") if part.strip()]


def item_words(text):
    """Return the set of lower-case words (3+ letters, no stop words) of one text."""
    return {w for w in re.findall(r"[a-z][a-z'-]{2,}", str(text).lower()) if w not in ENGLISH_STOP_WORDS}


def gap_words(selected_texts, all_texts, min_count, top, category_pattern):
    """Return the words most over-represented in the selected texts compared with all texts."""
    selected = Counter(w for text in selected_texts for w in item_words(text))
    everywhere = Counter(w for text in all_texts for w in item_words(text))
    rows = []
    for word, count in selected.items():
        if count >= min_count:
            share_selected = count / len(selected_texts)
            share_all = everywhere[word] / len(all_texts)
            rows.append({"word": word, "items": count, "share_here": round(share_selected, 3),
                         "share_all": round(share_all, 3), "ratio": round(share_selected / share_all, 1),
                         "in_categories": bool(common.find_terms(category_pattern, word))})
    table = pd.DataFrame(rows, columns=["word", "items", "share_here", "share_all", "ratio", "in_categories"])
    return table.sort_values(["ratio", "items", "word"], ascending=[False, False, True]).head(top)


def main():
    """Print the description of the selected groups or topics and save their items."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--group", help="group number(s), comma-separated (from data/topic_groups.csv)")
    choice.add_argument("--topic", help="topic number(s), comma-separated (from data/topics.csv)")
    parser.add_argument("--item-topics", default="data/item_topics.csv", help="from cluster.py (default: data/item_topics.csv)")
    parser.add_argument("--topics", default="data/topics.csv", help="from cluster.py (default: data/topics.csv)")
    parser.add_argument("--tagged", default="data/clean/items_tagged.csv",
                        help="from categorize.py, for the item texts (default: data/clean/items_tagged.csv)")
    parser.add_argument("--categories", default="config/categories.csv", help="category word list (default: config/categories.csv)")
    parser.add_argument("--examples", type=int, default=EXAMPLES, help=f"titles shown per topic (default: {EXAMPLES})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"seed for drawing the titles (default: {SEED})")
    parser.add_argument("--gap-words", type=int, default=GAP_WORDS, help=f"gap words shown (default: {GAP_WORDS})")
    parser.add_argument("--min-count", type=int, default=MIN_COUNT,
                        help=f"'none' items a gap word must appear in (default: {MIN_COUNT})")
    parser.add_argument("--out", default=None, help="selected items (default: data/inspect/group_<n>.csv or topic_<n>.csv)")
    args = parser.parse_args()

    items = pd.read_csv(args.item_topics, dtype=str, keep_default_na=False)
    topics = pd.read_csv(args.topics, dtype=str, keep_default_na=False)
    texts = pd.read_csv(args.tagged, dtype=str, keep_default_na=False).set_index("id")
    items["full_text"] = items["id"].map(texts["title"] + " " + texts["text"])
    column, wanted = ("group", args.group) if args.group else ("topic", args.topic)
    picked = items[items[column].astype(int).isin(numbers(wanted))].copy()
    if picked.empty:
        raise SystemExit(f"no item has {column} {wanted} in {args.item_topics}")
    picked["topic"] = picked["topic"].astype(int)

    # 1. Topics of the selection
    info = topics.set_index(topics["topic"].astype(int))
    print(f"Selection: {column} {wanted}   items: {len(picked)} of {len(items)}")
    print("\n1. Topics (size, label, share 'none', share news, top 6 words):")
    for topic, group in picked.groupby("topic"):
        row = info.loc[topic]
        news = (group["source_type"] == "news").mean()
        print(f"  {topic:>3}  {len(group):>4}  {row['label']:<20} {float(row['share_none']):.2f}  {news:.2f}  "
              f"{', '.join(row['top_words'].split(', ')[:6])}")

    print("\n2. Items per source type and topic:")
    print(pd.crosstab(picked["topic"], picked["source_type"], margins=True).to_string())

    print("\n3. Items per quarter:")
    quarter = pd.PeriodIndex(pd.to_datetime(picked["date"], errors="coerce"), freq="Q").astype(str)
    print(quarter.value_counts().sort_index().to_string())

    print("\n4. Share of items per known category:")
    categories = picked["categories"].str.split("; ").explode().value_counts() / len(picked)
    print(categories.round(3).to_string())

    print(f"\n5. Gap words of the {(picked['categories'] == NO_CATEGORY).sum()} items with category 'none' "
          f"(compared with all {len(items)} items):")
    pattern = common.terms_pattern(common.load_terms(args.categories))
    none_texts = picked.loc[picked["categories"] == NO_CATEGORY, "full_text"].tolist()
    if none_texts:
        print(gap_words(none_texts, items["full_text"].tolist(), args.min_count, args.gap_words, pattern).to_string(index=False))
    else:
        print("  no item with category 'none'")

    print(f"\n6. {args.examples} titles per topic (random, seed {args.seed}):")
    for topic, group in picked.groupby("topic"):
        print(f"  topic {topic}:")
        sample = group.sample(min(args.examples, len(group)), random_state=args.seed)
        for row in sample.itertuples():
            print(f"    [{row.source_type}, {row.date}, {row.categories}] {row.title[:110]}")

    out = Path(args.out or f"data/inspect/{column}_{wanted.replace(',', '_')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "group", "topic", "label", "date", "source_type", "age_band", "categories", "title"]
    picked.sort_values(["topic", "date", "id"])[columns].to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
