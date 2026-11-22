"""Clean the Google News headlines collected by google_ad_news.py.

The collection script casts a wide net, so most of what it returns only
mentions Google in passing. This script keeps the articles about Google's
search business, Google's ad revenue, or the traffic Google search sends to
publishers, in three steps:

1. Drop sources that publish evergreen explainers, how-to guides or Google's
   own marketing instead of news.
2. Tag every headline with the topics and the noise categories it matches,
   and keep the ones with at least one topic and no noise.
3. Group the articles that report the same story and keep the first one of
   each group. Two outlets rarely word a story the same way, so articles are
   compared on the words they share, in the headline and in the sub-header
   fetched by fetch_summaries.py.

Nothing is thrown away silently: the kept articles and the rejected ones go to
two separate CSV files, each with the reason for the decision.

Examples:
    python clean_news.py
    python clean_news.py --in data/google_ad_news_ai.csv
    python clean_news.py --similarity 0.3
"""

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# Sources that are not news: explainer and statistics pages that get re-dated
# every time they are updated, how-to guides for advertisers, and Google's own
# product marketing.
BLOCKED_SOURCES = {
    "FourWeekMBA",
    "Statista",
    "Semrush",
    "Business Model Analyst",
    "Business Strategy Hub",
    "SmartAsset.com",
    "Shopify",
    "AFFiNCO",
    "SQ Magazine",
    "Amra & Elma",
    "Themeisle",
    "G2 Learning Hub",
    "Enterprise Apps Today",
    "Business.com",
    "blog.google",
    "Google",
    "Google Business Profile",
}

# What we are looking for. A headline needs at least one of these.
# (The search queries all use intitle:Google, so "Google" is already there.)
TOPICS = {
    "search_business": (
        r"\bsearch(es|ing)?\b|searchgpt|perplexity|ai mode|ai overviews?|market share|\bbing\b|safari"
        r"|dominance|google killer|open web|blue links"
    ),
    "ad_revenue": (
        r"\b(ad|ads|advertising) (revenue|sales|business|income|growth|market|spend\w*|dollars|empire|machine|crown|giant|seller|platform)"
        r"|(revenue|sales|billions?|profits?) (from|in|on) (search )?(ads?|advertising)\b"
        r"|search (ads?|advertising)\b|digital advertising|chatgpt ads"
        r"|\bads? (in|into|inside|on|to) (ai overviews?|ai mode|gemini|search|chatbot)"
    ),
    "earnings": (
        r"\b(earnings|revenues?|quarterly|q[1-4]|profits?)\b"
        r"|\$[\d.]+ ?(b|bn|billion|t|trillion)\b.*\b(quarter|sales|year)\b"
    ),
    "publisher_traffic": (
        r"\btraffic\b|\breferrals?\b|zero.click|google zero|click.through|clickthrough"
        r"|\bpublishers?\b|\bnewspapers?\b|\bdailies\b|\bnewsrooms?\b|\bjournalis\w+|\bwebsites?\b"
        r"|news (sites?|industry|media|outlets?|publishers?|scraping)|media (companies|outlets|groups?|briefing)"
        r"|ai (summar\w+|answers?)|independent sites|google discover|\bcrawlers?\b|(french|german|local) media"
    ),
}

# What we are not looking for. Any of these rejects the headline, even when a
# topic matched.
NOISE = {
    "howto": (
        r"^how to\b|: how to\b|\bguide\b|\btips\b|\bstatistics\b|\bstats\b|\btutorial\b|\bchecklist\b|\bcourses?\b"
        r"|\bswot\b|\bsalary\b|marketing strategy|\bbest practices\b|\broundup\b|\beasy steps\b"
        r"|\b\d+ (best )?(ways|tips|tools|facts|examples|software|services)\b|\bvs\.? .*comparison"
    ),
    "ads_product": (
        r"pmax|performance max|demand gen|enhanced conversions|asset groups?|smart bidding|merchant center"
        r"|google ads (adds|rolls|simplifies|launches|tests|updates|introduces|now|is rolling|expands|gets|to|beta|api|editor|bug)"
        r"|keyword planner|ad strength|responsive search ads|broad match|negative keywords|local services ads"
        r"|budget pacing|bulk ad|campaign.level|\bai max\b|search themes|ads agency|target roas"
    ),
    "seo_chatter": (
        r"ranking (volatility|update)|search console|search liaison|helpful content"
        r"|google business profiles?|\bseo (tips|tools|strategy|checklist)\b|structured data"
    ),
    "stock_pick": (
        r"should you buy|stock forecast|price target|\bbuy the (shares|dip|stock)\b|millionaire.maker"
        r"|\bstocks? to (buy|watch|own)\b|\bbuy or sell\b|better buy|\bis \w+ a buy\b"
    ),
    "other_product": (
        r"\bpixel\b|google (maps|play|photos|workspace|drive|wallet|tv|home|fi|flights|classroom|shopping)"
        r"|\bwaymo\b|\byoutube (tv|music|premium|shorts)\b|\bchromebooks?\b|\bnest\b"
        # Google Cloud on its own; earnings stories about the whole company stay in
        r"|^(?!.*(alphabet|search|\bads?\b|advertising|earnings|\bq[1-4]\b)).*google cloud"
    ),
}

# How much two articles must overlap to count as the same story (see find_same_stories)
SIMILARITY = 0.25

# Words that say nothing about which story a headline tells. Every headline
# names Google, so "google" is one of them.
STOP_WORDS = {
    "a", "an", "the", "to", "of", "in", "on", "for", "and", "as", "is", "its", "s", "it", "at", "by", "with",
    "google", "googles", "alphabet", "says", "say", "said", "new", "from", "that", "this", "how", "why", "what",
    "are", "be", "will", "could", "may", "after", "over", "amid", "into", "than", "more", "has", "have", "was",
    "were", "their", "they", "but", "not", "you", "your", "can", "about", "which", "who",
}


def matching_labels(title, patterns):
    """Return the comma-separated names of the patterns found in the headline."""
    title = str(title).lower().replace("’", "'")
    return ", ".join(name for name, pattern in patterns.items() if re.search(pattern, title))


def story_words(text):
    """Return the set of words and numbers of a text that can identify a story."""
    text = str(text).lower().replace("’", "'").replace("'s", "")
    words = set()
    for word in re.findall(r"[a-z]+|\$?\d+(?:\.\d+)?%?", text):
        if word in STOP_WORDS:
            continue
        # Crude stemming, so that "publishers" and "publisher" count as one word
        if word.isalpha() and len(word) > 4:
            word = re.sub(r"(ing|ers|er|ed|es|s)$", "", word)
        words.add(word)
    return words


def find_same_stories(df, similarity, window_days):
    """Return, for each row, the position of the first article of its story.

    A row that starts a story (or is alone) gets its own position. df must be
    sorted by date.

    Every article is turned into a list of weighted words: a word counts more
    when few articles use it ("penske" says more than "search"), and headline
    words count double. Two articles tell the same story when they were
    published within `window_days` of each other and their weighted words
    overlap by at least `similarity` (cosine similarity: 0 = nothing in
    common, 1 = same words).
    """
    counts = []
    for title, description in zip(df["title"], df["description"]):
        in_title = story_words(title)
        counts.append(Counter(in_title) + Counter(in_title | story_words(description)))
    articles_using = Counter(word for count in counts for word in count)

    vectors = []
    for count in counts:
        vector = {word: n * math.log(len(counts) / articles_using[word]) for word, n in count.items()}
        length = math.sqrt(sum(weight**2 for weight in vector.values())) or 1
        vectors.append({word: weight / length for word, weight in vector.items()})

    dates = df["published"].tolist()
    window = pd.Timedelta(days=window_days)
    first_of_story = list(range(len(df)))
    for i in range(len(df)):
        best = similarity
        for j in range(i - 1, -1, -1):
            if dates[i] - dates[j] > window:
                break
            alike = sum(weight * vectors[j].get(word, 0) for word, weight in vectors[i].items())
            if alike >= best:
                best = alike
                first_of_story[i] = first_of_story[j]
    return first_of_story


# A description the publisher wrote counts fully. The ones written afterwards for
# blocked pages count less: they are a second-hand reading of the article, and the
# ones made from a search-result snippet never saw the article at all.
DESCRIPTION_WEIGHTS = {"publisher": 1.0, "written_from_text": 0.5, "written_from_snippet": 0.25}


def description_source(status):
    """Turn the status column of the summaries file into who wrote the description."""
    if status.startswith("summary written from search-result snippet"):
        return "written_from_snippet"
    if status.startswith("summary written from article text"):
        return "written_from_text"
    return "publisher"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="source", default="data/google_ad_news.csv", help="CSV file to clean")
    parser.add_argument("--summaries", default="data/article_summaries.csv", help="CSV file made by fetch_summaries.py (default: data/article_summaries.csv)")
    parser.add_argument("--leads", default="data/article_leads.csv", help="CSV file with the opening paragraphs of the articles (default: data/article_leads.csv)")
    parser.add_argument("--similarity", type=float, default=SIMILARITY, help=f"how much two articles must overlap to count as the same story, 0-1 (default: {SIMILARITY})")
    parser.add_argument("--window-days", type=int, default=4, help="only compare articles published this many days apart (default: 4)")
    args = parser.parse_args()

    source = Path(args.source)
    df = pd.read_csv(source)
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df = df.sort_values("published").reset_index(drop=True)
    print(f"Loaded {len(df)} articles from {source}")

    # Add the real address and the sub-header of each article, when we have them
    if Path(args.summaries).exists():
        summaries = pd.read_csv(args.summaries, keep_default_na=False)[["link", "url", "description", "status"]]
        df = df.merge(summaries.drop_duplicates(subset="link"), on="link", how="left")
    else:
        print(f"{args.summaries} not found: run fetch_summaries.py first to compare sub-headers too")
        df["url"] = df["description"] = df["status"] = ""
    df[["url", "description", "status"]] = df[["url", "description", "status"]].fillna("")
    # A text found under several different headlines describes the site, not the article
    site_text = df.groupby("description")["title"].transform("nunique") > 1
    df.loc[site_text, "description"] = ""
    # Who wrote the description, and how much it should count in the analysis
    df["description_source"] = df["status"].map(description_source).where(df["description"] != "", "")
    df["description_weight"] = df["description_source"].map(DESCRIPTION_WEIGHTS).fillna(0.0)
    df = df.drop(columns="status")
    # The opening paragraphs of each article, when they could be read
    if Path(args.leads).exists():
        leads = pd.read_csv(args.leads, keep_default_na=False)[["link", "lead"]]
        df = df.merge(leads.drop_duplicates(subset="link"), on="link", how="left")
    else:
        df["lead"] = ""
    df["lead"] = df["lead"].fillna("")
    print(f"{(df['description'] != '').sum()} of them have a sub-header\n")

    df["topics"] = df["title"].map(lambda title: matching_labels(title, TOPICS))
    df["noise"] = df["title"].map(lambda title: matching_labels(title, NOISE))
    df["reason"] = ""
    df["story_size"] = 1

    # Step 1: blocked sources
    blocked = df["source"].isin(BLOCKED_SOURCES)
    df.loc[blocked, "reason"] = "blocked source"

    # Step 2: topic and noise patterns
    remaining = df["reason"] == ""
    df.loc[remaining & (df["noise"] != ""), "reason"] = "noise: " + df["noise"]
    df.loc[remaining & (df["noise"] == "") & (df["topics"] == ""), "reason"] = "no topic matched"

    # Step 3: same story, among the articles that are still in
    remaining = df[df["reason"] == ""]
    first_of_story = remaining.index[find_same_stories(remaining, args.similarity, args.window_days)]
    first_of_story = pd.Series(first_of_story, index=remaining.index)
    df.loc[remaining.index, "story_size"] = first_of_story.map(first_of_story.value_counts())
    repeats = first_of_story[first_of_story != first_of_story.index]
    df.loc[repeats.index, "reason"] = "same story as: " + df.loc[repeats, "title"].values

    kept = df[df["reason"] == ""].drop(columns=["noise", "reason"])
    rejected = df[df["reason"] != ""]

    kept_path = source.with_name(f"{source.stem}_clean.csv")
    rejected_path = source.with_name(f"{source.stem}_rejected.csv")
    kept.to_csv(kept_path, index=False)
    rejected.to_csv(rejected_path, index=False)

    print("Rejected:")
    reasons = rejected["reason"].str.replace(r"^same story as: .*", "same story as an earlier article", regex=True)
    print(reasons.value_counts().to_string())
    print(f"\nKept {len(kept)} of {len(df)} articles -> {kept_path}")
    print(f"Rejected {len(rejected)} articles -> {rejected_path}")
    print(f"\nStories covered by several articles: {(kept['story_size'] > 1).sum()}. The biggest:")
    print(kept.nlargest(5, "story_size")[["story_size", "title"]].to_string(index=False, header=False))
    print("\nKept articles per topic (an article can have several):")
    print(kept["topics"].str.split(", ").explode().value_counts().to_string())
    print("\nKept articles per year:")
    print(kept["published"].dt.year.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
