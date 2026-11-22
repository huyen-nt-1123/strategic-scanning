"""Count the terms used in the articles that do not take a side on the problem statement.

analyze_news.py puts every article in a category. Most of them neither support
nor oppose the two claims ("others" and "no claim"). This script looks at what
those articles talk about, and writes three tables:

1. distinctive   terms used by a bigger share of the scanned articles than of
                 the articles that take a side
2. rising        share of the scanned articles using a term in each period,
                 sorted by the change from the first period to the last one.
                 The terms that go down are at the bottom of the same file.
3. weak signals  terms that are rare, were first seen recently and are used by
                 more than one source

Rules:

- A term is one word, or two words that follow each other in the text. Words
  are lower-cased, a final "'s" is removed, and the common English words in
  STOPWORDS are left out. "ads" and "ad" stay two different terms: nothing is
  done about plurals.
- A term is counted once per article, however many times the article uses it.
  All the figures are numbers of articles, not numbers of occurrences.
- A share is a percentage of the articles of a group or of a period.
- "Recently" is measured from the newest article of the input file, not from
  today's date, so the same input always gives the same output.
- The script only counts. It does not say which terms matter: --min-articles,
  --weak-max, --min-sources, --recent-months and --min-period-articles are
  judgement calls, to be changed from the command line.

Examples:
    python scan_topics.py
    python scan_topics.py --period half
    python scan_topics.py --scan others --text header
    python scan_topics.py --recent-months 6 --weak-max 8
    python scan_topics.py --ignore google,alphabet --show 30
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# Common English words that say nothing about the subject of an article
STOPWORDS = set(
    """
    a about above after again against all also am an and any are aren't as at be because been before being below
    between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
    even few for from further had hadn't has hasn't have haven't having he her here hers herself him himself his
    how however i if in into is isn't it its itself just like may me might more most much must my myself new no
    nor not now of off on once one only or other our ours ourselves out over own per said same say says she
    should shouldn't so some still such than that the their theirs them themselves then there these they this
    those through to too two under until up us very was wasn't we were weren't what when where which while who
    whom whose why will with won't would wouldn't yet you your yours yourself yourselves
    according across ago already among another around back based became become been come comes de el get gets
    going got la last less let made make makes making many mr ms next often read since take takes three told
    use used using via want way well went year years month months week weeks day days today
    """.split()
)

# A word starts with a letter and can hold figures, apostrophes and hyphens ("q3", "zero-click")
WORD = r"[a-z][a-z0-9'-]*"
# Two words only make a term when nothing of this is between them
CHUNK_BREAK = r"[.!?;:,|()\[\]\"“”]|\s[-–—]\s"

# What --text reads, the same choices as in analyze_news.py
TEXTS = {"all": ["title", "description", "lead"], "header": ["title", "description"], "lead": ["lead"]}


def clean_text(text):
    """Lower-case a text and replace the curly apostrophes, to make it easy to match."""
    return str(text).lower().replace("’", "'").replace("‘", "'")


def terms_of(text, ignore):
    """Return the set of terms of a text: its words and its pairs of words."""
    terms = set()
    for chunk in re.split(CHUNK_BREAK, clean_text(text)):
        words = [word.removesuffix("'s").strip("'-") for word in re.findall(WORD, chunk)]
        keep = [len(word) >= 2 and word not in STOPWORDS and word not in ignore for word in words]
        for i, word in enumerate(words):
            if not keep[i]:
                continue
            terms.add(word)
            if i + 1 < len(words) and keep[i + 1]:
                terms.add(f"{word} {words[i + 1]}")
    return terms


def period_of(date, length):
    """Return the period a date is in: '2025', '2025-H1' or '2025-Q3'."""
    if length == "half":
        return f"{date.year}-H{1 if date.month <= 6 else 2}"
    if length == "quarter":
        return f"{date.year}-Q{(date.month - 1) // 3 + 1}"
    return str(date.year)


def term_table(articles, parts, ignore):
    """Return one row per article and term, with the columns of the article next to it."""
    text = articles[parts].agg(" . ".join, axis=1)
    terms = text.apply(lambda value: sorted(terms_of(value, ignore))).explode().dropna().rename("term")
    # The join repeats the row number of an article for each of its terms: number the rows again
    return articles.join(terms, how="inner").reset_index(drop=True)


def distinctive(table, sizes, min_articles):
    """Compare the share of each term in the scanned articles and in the ones that take a side."""
    counts = pd.crosstab(table["term"], table["group"]).reindex(columns=["scan", "compare"], fill_value=0)
    counts = counts[counts["scan"] >= min_articles]
    result = pd.DataFrame({
        "articles_scan": counts["scan"],
        "share_scan": (100 * counts["scan"] / max(sizes["scan"], 1)).round(1),
        "articles_compare": counts["compare"],
        "share_compare": (100 * counts["compare"] / max(sizes["compare"], 1)).round(1),
    })
    result["difference"] = (result["share_scan"] - result["share_compare"]).round(1)
    # No ratio when the other group never uses the term: nothing to divide by
    result["ratio"] = (result["share_scan"] / result["share_compare"].where(result["share_compare"] > 0)).round(1)
    return result.reset_index().sort_values(["difference", "term"], ascending=[False, True], kind="stable")


def rising(table, period_sizes, min_articles):
    """Give the share of each term per period and the change from the first period to the last."""
    periods = list(period_sizes.index)
    counts = pd.crosstab(table["term"], table["period"]).reindex(columns=periods, fill_value=0)
    counts = counts[counts.sum(axis=1) >= min_articles]
    result = (100 * counts / period_sizes).round(1).add_prefix("share_")
    result.insert(0, "articles", counts.sum(axis=1))
    result["change"] = (result[f"share_{periods[-1]}"] - result[f"share_{periods[0]}"]).round(1)
    return result.reset_index().sort_values(["change", "term"], ascending=[False, True], kind="stable")


def weak_signals(table, cutoff, weak_min, weak_max, min_sources):
    """Keep the rare terms first seen after `cutoff` and used by enough sources."""
    table = table.sort_values(["published", "url"], kind="stable")
    result = table.groupby("term").agg(
        articles=("url", "size"),
        sources=("source", "nunique"),
        first_seen=("published", "min"),
        last_seen=("published", "max"),
        example_title=("title", "first"),
        example_url=("url", "first"),
    )
    result = result[
        result["articles"].between(weak_min, weak_max)
        & (result["sources"] >= min_sources)
        & (result["first_seen"] >= cutoff)
    ].reset_index()
    result = result.sort_values(["first_seen", "articles", "term"], ascending=[False, False, True], kind="stable")
    for column in ("first_seen", "last_seen"):
        result[column] = result[column].dt.strftime("%Y-%m-%d")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="source", default="data/matching_analysis_results.csv", help="CSV file written by analyze_news.py")
    parser.add_argument("--out", default="data/topic_scan", help="beginning of the names of the three CSV files to write (default: data/topic_scan)")
    parser.add_argument("--scan", default="others,no claim", help="categories to scan, separated by commas (default: others,no claim)")
    parser.add_argument("--compare", default="support,oppose,both", help="categories to compare them with (default: support,oppose,both)")
    parser.add_argument("--text", choices=TEXTS, default="all", help="what to read: all = everything, header = headline and sub-header, lead = opening paragraphs (default: all)")
    parser.add_argument("--ignore", default="", help="words to leave out on top of the common English words, separated by commas")
    parser.add_argument("--min-articles", type=int, default=5, help="tables 1 and 2: smallest number of scanned articles a term must be in (default: 5)")
    parser.add_argument("--period", choices=["year", "half", "quarter"], default="year", help="table 2: length of a period (default: year)")
    parser.add_argument("--min-period-articles", type=int, default=30, help="table 2: periods with fewer scanned articles are left out (default: 30)")
    parser.add_argument("--weak-min", type=int, default=2, help="table 3: smallest number of articles for a rare term (default: 2)")
    parser.add_argument("--weak-max", type=int, default=5, help="table 3: biggest number of articles for a rare term (default: 5)")
    parser.add_argument("--min-sources", type=int, default=2, help="table 3: smallest number of different sources (default: 2)")
    parser.add_argument("--recent-months", type=int, default=12, help="table 3: a term is recent when first seen in the last N months of the file (default: 12)")
    parser.add_argument("--show", type=int, default=15, help="how many rows of each table to print (default: 15)")
    args = parser.parse_args()

    df = pd.read_csv(args.source, keep_default_na=False)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df["period"] = df["published"].apply(period_of, length=args.period)
    print(f"Loaded {len(df)} articles from {args.source}")

    scan = [name.strip() for name in args.scan.split(",") if name.strip()]
    compare = [name.strip() for name in args.compare.split(",") if name.strip()]
    df["group"] = ""
    df.loc[df["category"].isin(compare), "group"] = "compare"
    df.loc[df["category"].isin(scan), "group"] = "scan"
    sizes = df["group"].value_counts()
    sizes = {group: int(sizes.get(group, 0)) for group in ("scan", "compare")}
    print(f"Scanned: {sizes['scan']} articles ({', '.join(scan)})")
    print(f"Compared with: {sizes['compare']} articles ({', '.join(compare)})")
    if not sizes["scan"]:
        raise SystemExit("No article to scan: check --scan against the `category` column.")

    ignore = {word.strip().lower() for word in args.ignore.split(",") if word.strip()}
    table = term_table(df[df["group"] != ""], TEXTS[args.text], ignore)
    scanned = table[table["group"] == "scan"]
    print(f"{scanned['term'].nunique()} different terms in the scanned articles\n")

    # Table 2 only uses the periods with enough articles for a share to mean something
    all_periods = df.loc[df["group"] == "scan", "period"].value_counts().sort_index()
    period_sizes = all_periods[all_periods >= args.min_period_articles]
    print(f"Scanned articles per period:\n{all_periods.to_string()}")
    left_out = [period for period in all_periods.index if period not in period_sizes.index]
    if left_out:
        print(f"Left out of table 2, fewer than {args.min_period_articles} articles: {', '.join(left_out)}")

    latest = df["published"].max()
    cutoff = latest - pd.DateOffset(months=args.recent_months)

    tables = {"distinctive": distinctive(table, sizes, args.min_articles)}
    if len(period_sizes) >= 2:
        tables["rising"] = rising(scanned[scanned["period"].isin(period_sizes.index)], period_sizes, args.min_articles)
    else:
        print("Table 2 not written: it needs at least two periods with enough articles.")
    tables["weak_signals"] = weak_signals(scanned, cutoff, args.weak_min, args.weak_max, args.min_sources)

    titles = {
        "distinctive": f"1. Distinctive terms (in at least {args.min_articles} scanned articles), biggest difference of share first",
        "rising": f"2. Share per period (in at least {args.min_articles} scanned articles), biggest rise first",
        "weak_signals": (
            f"3. Terms in {args.weak_min} to {args.weak_max} articles, from at least {args.min_sources} sources, "
            f"first seen after {cutoff:%Y-%m-%d} (newest article: {latest:%Y-%m-%d})"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    written = []
    for name, result in tables.items():
        path = f"{args.out}_{name}.csv"
        result.to_csv(path, index=False)
        written.append(f"{len(result)} terms -> {path}")
        print(f"\n{titles[name]}:")
        shown = result.drop(columns=["example_url"], errors="ignore").head(args.show)
        print(shown.to_string(index=False, max_colwidth=60) if len(shown) else "(none)")

    print("\nWrote:")
    print("\n".join(written))


if __name__ == "__main__":
    main()
