"""Check the cleaned headlines against the problem statement.

Problem statement: usage of Google's legacy search is down, maybe because of
AI, and so Google's ad revenue is down. That is two claims, and every article
is checked against each of them:

- search volume is down
- ad revenue is down

The rule is the same for both and uses general vocabulary only, no phrase taken
from a particular headline. The headline and the sub-header are cut into
clauses. A clause says something about a claim when it names the subject
("searches", "ad revenue"...) and has a direction word ("falls", "grows"...)
a few words away:

    subject + down word  -> support   (the claim is true)
    subject + up word    -> oppose    (the claim is false)
    a "not" just before the direction word turns it around

Stance of an article on one claim:

    support    only clauses that support it
    oppose     only clauses that oppose it
    both       clauses on each side
    no claim   the subject is named, but with no direction
    (empty)    the subject is not named

An article can be about both subjects: it then has a stance on each claim, and
is counted in both. The `category` column puts the two claims together, and is
`others` when the article names neither subject.

Everything we have on an article is read: the headline, the sub-header and the
opening paragraphs saved in data/article_leads.csv. --text reads only a part of
it, to compare.

Examples:
    python analyze_news.py
    python analyze_news.py --text header --out data/matching_analysis_results_header.csv
    python analyze_news.py --text lead --out data/matching_analysis_results_leads.csv
    python analyze_news.py --window 4
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# How an article names each subject
SUBJECTS = {
    "search_volume": (
        r"\bsearches\b|\bqueries\b|\bsearching\b"
        r"|\bsearch (volumes?|usage|activity|queries|market share|share)\b"
    ),
    "ad_revenue": (
        r"\b(ads?|advertising|search|network) (ads? |advertising )?(revenues?|sales|income)\b"
        r"|\bsearch (ads?|advertising)\b|\brevenues? from (ads?|advertising|search)\b"
    ),
}

# Direction words, as word beginnings ("declin" finds declines, declined, declining)
DOWN = (
    "fall", "fell", "drop", "declin", "decreas", "plung", "plummet", "slip", "slid", "sink", "sank", "shrink",
    "shrank", "lose", "losing", "loss", "lost", "dip", "erod", "tumbl", "slump", "fading", "fades", "weak",
    "collaps", "crash",
)
UP = (
    "rise", "rising", "rose", "grow", "grew", "increas", "surg", "soar", "jump", "climb", "gain", "record",
    "boost", "lift", "beat", "strong", "accelerat",
)
# "more queries", "less searching": only a direction when right next to the subject
COMPARATIVES = {"more": "up", "higher": "up", "less": "down", "fewer": "down", "lower": "down"}
# "up" and "down" are in many expressions ("teams up", "drying up"). They only count
# as "is up", "was down"... or in front of a figure ("up 11%", "down nearly 20%")
UP_DOWN = r"\b(?:is|are|was|were) (up|down)\b|\b(up|down) (?:by |again |nearly |about |almost |over |roughly )?\$?\d"
NEGATIONS = {"not", "no", "never", "without", "isn't", "aren't", "doesn't", "don't", "didn't", "hasn't", "haven't", "won't"}

# A clause about one of these, which does not name Google, gives their figures
OTHER_COMPANIES = r"\b(meta|facebook|microsoft|bing|amazon|reddit|openai|chatgpt|perplexity|pinterest|snap|tiktok|apple)\b"
GOOGLE = r"\b(google|alphabet)\b"
# A clause with one of these is about what websites get, not about Google: their
# ad revenue, or the searches that send them no visitor
WEBSITE_SIDE = r"\b(publishers?|websites?|sites?|creators?|bloggers?|zero.click)\b"

# Where a sentence is cut into clauses. A dot only counts when a space follows it, to keep "10.4%" whole.
CLAUSE_BREAK = r"[.!?;:|]\s+|,\s+|\s[-–—&]\s|\b(?:while|but|as|despite|although|though|amid|after)\b"


def clean_text(text):
    """Lower-case a text and replace the curly apostrophes, to make it easy to match."""
    return str(text).lower().replace("’", "'").replace("‘", "'")


def direction_of(word, distance, clause):
    """Return 'down', 'up' or '' for a word found `distance` words from the subject."""
    text = word.group()
    if text in COMPARATIVES:
        return COMPARATIVES[text] if distance == 1 else ""
    if text in ("up", "down"):
        counts = any(match.start(1 if match.group(1) else 2) == word.start() for match in re.finditer(UP_DOWN, clause))
        return text if counts else ""
    if text.startswith(DOWN):
        return "down"
    if text.startswith(UP):
        return "up"
    return ""


def read_clause(clause, subject_pattern, window):
    """Return what a clause says about a subject: 'support', 'oppose', 'named' or ''.

    'named' means the subject is there with no direction word within `window`
    words of it.
    """
    subject = re.search(subject_pattern, clause)
    if not subject:
        return ""
    if re.search(OTHER_COMPANIES, clause) and not re.search(GOOGLE, clause):
        return ""
    if re.search(WEBSITE_SIDE, clause):
        return ""

    words = list(re.finditer(r"[a-z']+", clause))
    in_subject = [i for i, word in enumerate(words) if word.start() < subject.end() and word.end() > subject.start()]
    # Closest direction word first
    nearby = sorted(
        (min(abs(i - j) for j in in_subject), i)
        for i in range(len(words))
        if i not in in_subject and min(abs(i - j) for j in in_subject) <= window
    )
    for distance, i in nearby:
        direction = direction_of(words[i], distance, clause)
        if not direction:
            continue
        if any(word.group() in NEGATIONS for word in words[max(0, i - 3):i]):
            direction = "up" if direction == "down" else "down"
        return "support" if direction == "down" else "oppose"
    return "named"


def stance_on(text, subject_pattern, window):
    """Return the stance of a text on one claim, and the clauses it was read from."""
    found = {}
    for clause in re.split(CLAUSE_BREAK, text):
        reading = read_clause(clause, subject_pattern, window)
        if reading:
            found.setdefault(reading, []).append(clause.strip())

    if "support" in found and "oppose" in found:
        stance = "both"
    elif "support" in found:
        stance = "support"
    elif "oppose" in found:
        stance = "oppose"
    elif "named" in found:
        stance = "no claim"
    else:
        stance = ""
    evidence = " | ".join(f"{reading}: {clause}" for reading in ("support", "oppose") for clause in found.get(reading, []))
    return stance, evidence


def overall(stances):
    """Put the stances on the two claims together in one category."""
    stances = set(stances)
    if "both" in stances or {"support", "oppose"} <= stances:
        return "both"
    if "support" in stances:
        return "support"
    if "oppose" in stances:
        return "oppose"
    if "no claim" in stances:
        return "no claim"
    return "others"


# What --text reads. Every sub-header is read, the ones written by Claude too:
# the `description_source` column says which ones they are.
TEXTS = {"all": ["title", "description", "lead"], "header": ["title", "description"], "lead": ["lead"]}


def classify(article, window, parts):
    """Return the stance of an article on each claim and its overall category."""
    text = " . ".join(clean_text(article[part]) for part in parts)

    result = {}
    for name, pattern in SUBJECTS.items():
        result[f"{name}_down"], result[f"{name}_evidence"] = stance_on(text, pattern, window)
        # One yes/no column per subject: an article can be about both
        result[f"about_{name}"] = "yes" if result[f"{name}_down"] else "no"
    result["category"] = overall(result[f"{name}_down"] for name in SUBJECTS)
    return pd.Series(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="source", default="data/google_ad_news_clean.csv", help="cleaned CSV file to analyse")
    parser.add_argument("--out", default="data/matching_analysis_results.csv", help="CSV file to write")
    parser.add_argument("--text", choices=TEXTS, default="all", help="what to read: all = everything, header = headline and sub-header, lead = opening paragraphs (default: all)")
    parser.add_argument("--leads", default="data/article_leads.csv", help="CSV file with the opening paragraphs (default: data/article_leads.csv)")
    parser.add_argument("--window", type=int, default=6, help="how many words apart a subject and a direction word can be (default: 6)")
    args = parser.parse_args()

    df = pd.read_csv(args.source, keep_default_na=False)
    print(f"Loaded {len(df)} articles from {args.source}\n")

    # The opening paragraphs come from their own file, matched on the Google News link
    leads = pd.read_csv(args.leads, keep_default_na=False)[["link", "lead"]].drop_duplicates(subset="link")
    df = df.drop(columns="lead", errors="ignore").merge(leads, on="link", how="left")
    df["lead"] = df["lead"].fillna("")
    if args.text != "header":
        print(f"{(df['lead'] != '').sum()} of them have opening paragraphs\n")

    df = pd.concat([df, df.apply(classify, axis=1, window=args.window, parts=TEXTS[args.text])], axis=1)
    columns = [
        "published", "category", "about_search_volume", "search_volume_down", "about_ad_revenue", "ad_revenue_down",
        "title", "description", "lead", "search_volume_evidence", "ad_revenue_evidence", "source", "url",
        "description_source", "description_weight", "topics", "story_size",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(args.out, index=False)

    order = ["support", "oppose", "both", "no claim", "others"]
    print("Articles per category:")
    print(df["category"].value_counts().reindex(order, fill_value=0).to_string())
    print("\nWhat the articles are about (an article can be about both):")
    about = df[[f"about_{name}" for name in SUBJECTS]].eq("yes")
    print(about.sum().rename(lambda column: column.replace("about_", "").replace("_", " ")).to_string())
    print(f"neither          {(~about.any(axis=1)).sum()}")
    for name in SUBJECTS:
        print(f"\n'{name.replace('_', ' ')} is down':")
        print(df[f"{name}_down"].replace("", "not named").value_counts().reindex(order[:4] + ["not named"], fill_value=0).to_string())
    print("\nCategory per year:")
    print(pd.crosstab(df["published"].str[:4], df["category"]).reindex(columns=order, fill_value=0).to_string())
    print(f"\nWrote {len(df)} articles -> {args.out}")


if __name__ == "__main__":
    main()
