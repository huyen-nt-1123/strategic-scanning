"""Group the tagged items by meaning (embeddings + BERTopic), to find what the known categories miss.

Input: data/clean/items_tagged.csv (from categorize.py). Nothing is filtered
out: every item gets a topic, outliers included.

Rules:

- Tags are kept out of the clustering. Only title + " " + text is embedded
  and only those words make the topic vocabulary. age_band, categories, pestle,
  category_terms and every other tag column are joined to the topics AFTER
  clustering, only to compare them with the topics. (News and policy items have
  an empty text, so for them only the headline is embedded.)
- Method: the sentence model --model (default all-MiniLM-L6-v2, runs on CPU)
  turns each text into a vector of its meaning; UMAP reduces the vectors to
  --umap-components dimensions (default 5) with a fixed --seed; HDBSCAN groups them, a group needing at least
  --min-topic-size items and an item needing --min-samples close neighbours
  to count as a dense point (both judgement calls; a high --min-samples can
  make one topic swallow nearly every item); BERTopic names each topic by its
  top words (c-TF-IDF, English stop words removed).
- Outliers: items HDBSCAN puts in no group get topic -1. They are counted,
  reported and kept in the outputs, never dropped.
- Candidate new theme: a topic (not -1) whose share of items with category
  "none" is above --none-share (default 0.5, "mostly"; a judgement call) gets
  the label "candidate new theme"; other topics get "known categories"; topic
  -1 gets "outliers". It is a label, not a conclusion: the source-type shares
  sit next to it, because headline-only items (news, policy) get "none" more
  often than abstracts.
- Repeatable: the embeddings are saved to --embeddings (.npy) with
  --embedding-ids (.csv: id, text_hash, model) alongside. An item whose id,
  embedded text (sha1 hash) and model are unchanged reuses its saved vector;
  only new or changed texts are embedded. The saved files then hold the
  current items only, in input order. With the same input and saved
  embeddings, two runs write identical files.
- A failure to load the sentence model (e.g. no web access for the first
  download) is written to data/run_log.csv (or SCAN_RUN_LOG) before stopping.
- --use-saved reads the saved topic files instead of recomputing (no model,
  no web access needed; used by the notebook's offline mode). It stops if the
  saved files do not cover the same item ids as the input. The labels are the
  saved ones (--none-share is not applied again).
- Topic numbers are given by size (0 = largest) and can change when the input
  changes: compare topics between runs by their words and items, not their
  numbers.
- Two levels. The topics above are the fine level (for weak signals and
  candidate new themes). The group level joins similar topics into --groups
  broad groups (default 10, a judgement call), for presenting the scan:
  each topic's vector is the mean of its items' (unit-length) embeddings; the
  topics are joined step by step, most similar first, until --groups are
  left (scipy's linkage; --group-linkage "average" = cosine distance, a topic
  far from all others stays a group of its own; "ward" = groups of more even
  size). Every item keeps its
  topic; the group is one more column. Outliers (topic -1) are group -1. If
  --groups is not below the number of topics, each topic is its own group.
  Group numbers are given by size (0 = largest). Groups, like topics, are
  recomputed every run, so they can change when the input changes.
- Group top words: words of the group's items scored as BERTopic scores topic
  words (c-TF-IDF: frequent in the group, rare in the other groups), English
  stop words removed. Outliers are left out of that scoring.

Outputs:

- data/topics.csv: one row per topic (-1 included): topic, topic_name,
  label, group, group_name, size, weighted_size (sum of source weights),
  share_none, top_words (10, best first), example_titles (the 3 items nearest
  to the topic's mean vector), first_date, last_date, share_source_<type>,
  share_category_<name> (an item can have several categories, so these can
  add up to more than 1), share_age_<band>.
- data/topic_groups.csv: one row per group (-1 included): group, group_name,
  n_topics, topics (their numbers), candidate_topics (those labelled
  "candidate new theme"), then the same size, share and date columns as
  topics.csv.
- data/item_topics.csv: one row per item: id, topic, topic_name, label,
  group, group_name, then the tag and descriptive columns joined after
  clustering (source_type, weight, date, title, age_band, categories, pestle,
  n_categories).

Examples:
    python cluster.py
    python cluster.py --min-topic-size 10 --groups 8 --none-share 0.6
    python cluster.py --group-linkage average
    python cluster.py --use-saved
    python cluster.py --topics /tmp/topics.csv --groups-out /tmp/groups.csv --item-topics /tmp/item_topics.csv --embeddings /tmp/emb.npy --embedding-ids /tmp/emb_ids.csv
"""

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

import common

# Small sentence model chosen for the project (about 90 MB, runs on CPU).
MODEL = "all-MiniLM-L6-v2"
# Fixed seed for UMAP, so the same vectors always give the same topics.
SEED = 42
# Smallest topic HDBSCAN may form; a judgement call. Chosen 24 Nov: 10 gave 56
# topics, 20 gives 32 and still keeps the smallest candidate themes (at 30 one was lost).
MIN_TOPIC_SIZE = 20
# HDBSCAN min_samples: how many close neighbours an item needs to count as a dense point.
# A judgement call, measured 24 Nov: left empty (= min topic size, BERTopic's default),
# 3 of 4 runs on 98% samples of the items collapsed into one topic holding 98% of them;
# with 5, every run gave 35-41 topics, the largest holding 10-12% of the items.
MIN_SAMPLES = 5
# Broad groups the topics are joined into, for presenting the scan; a judgement call.
GROUPS = 10
# How topics are joined into groups (scipy's linkage; no randomness); a judgement call.
# "average": mean cosine distance between topics; a topic far from all others stays alone.
# "ward": joins the topics that least increase the spread inside groups; gives groups of
# more even size (computed on the unit-length mean vectors, where Euclidean distance
# orders pairs as cosine distance does).
GROUP_LINKAGES = {"average": "cosine", "ward": "euclidean"}
GROUP_LINKAGE = "ward"
# A topic is a "candidate new theme" when its share of category "none" is above this; a judgement call.
NONE_SHARE = 0.5
# UMAP settings: BERTopic's defaults, written out so they are visible (neighbours and
# dimensions are options, judgement calls; min_dist 0 packs close items tightly, cosine
# compares the direction of the vectors, as the sentence model is trained to).
UMAP_NEIGHBORS = 15
UMAP_COMPONENTS = 5
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
# How many top words (default of --top-words) and example titles each topic shows.
TOP_WORDS = 10
EXAMPLES = 3
# Texts embedded at a time (memory only; does not change the result).
BATCH_SIZE = 64
# Labels written by categorize.py and by this script.
NO_CATEGORY = "none"
OUTLIER_TOPIC = -1
LABEL_OUTLIERS = "outliers"
LABEL_CANDIDATE = "candidate new theme"
LABEL_KNOWN = "known categories"
# Columns of items_tagged.csv joined to the topics after clustering (never embedded).
JOINED_COLUMNS = ["source_type", "weight", "date", "title", "age_band", "categories", "pestle", "n_categories"]


def embedded_text(items):
    """Return the text that is embedded for each item: title + " " + text, nothing else."""
    return (items["title"] + " " + items["text"]).str.strip()


def text_hash(text):
    """Return the sha1 hash of one embedded text, to see when it changes."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_saved_embeddings(vectors_path, ids_path):
    """Return {(id, text_hash, model): vector} from the saved files, or {} if there are none."""
    if not Path(vectors_path).exists() or not Path(ids_path).exists():
        return {}
    vectors = np.load(vectors_path)
    keys = pd.read_csv(ids_path, dtype=str, keep_default_na=False)
    if len(keys) != len(vectors):
        print(f"  {ids_path} and {vectors_path} have different lengths: saved embeddings ignored")
        return {}
    return {(row.id, row.text_hash, row.model): vectors[n] for n, row in enumerate(keys.itertuples())}


def embed(items, texts, model_name, vectors_path, ids_path):
    """Return (vectors, reused, new): saved vectors for unchanged texts, the model for the rest; save them all."""
    hashes = [text_hash(text) for text in texts]
    saved = load_saved_embeddings(vectors_path, ids_path)
    keys = [(item_id, h, model_name) for item_id, h in zip(items["id"], hashes)]
    todo = [n for n, key in enumerate(keys) if key not in saved]
    fresh = {}
    if todo:
        from sentence_transformers import SentenceTransformer  # imported here: --use-saved needs no model
        try:
            model = SentenceTransformer(model_name, device="cpu")
        except Exception as error:
            common.log("cluster", "failed", f"load model {model_name}", error)
            raise
        vectors = model.encode([texts[n] for n in todo], batch_size=BATCH_SIZE, show_progress_bar=False)
        fresh = dict(zip(todo, vectors))
    vectors = np.vstack([fresh[n] if n in fresh else saved[key] for n, key in enumerate(keys)]).astype(np.float32)
    for path in [Path(vectors_path), Path(ids_path)]:
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    pd.DataFrame({"id": items["id"], "text_hash": hashes, "model": model_name}).to_csv(ids_path, index=False)
    return vectors, len(keys) - len(todo), len(todo)


def fit_topics(texts, vectors, min_topic_size, min_samples, seed, neighbors, components, top_words):
    """Run UMAP + HDBSCAN through BERTopic; return (topic per item, {topic: [top words]})."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP
    umap_model = UMAP(n_neighbors=neighbors, n_components=components, min_dist=UMAP_MIN_DIST,
                      metric=UMAP_METRIC, random_state=seed)
    # BERTopic's own HDBSCAN settings, plus min_samples
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size, min_samples=min_samples, metric="euclidean",
                            cluster_selection_method="eom", prediction_data=True)
    model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                     vectorizer_model=CountVectorizer(stop_words="english"),
                     min_topic_size=min_topic_size, top_n_words=top_words)
    topics, _ = model.fit_transform(list(texts), vectors)
    words = {topic: [word for word, _ in model.get_topic(topic) if word] for topic in sorted(set(topics))}
    return list(topics), words


def slug(value):
    """Turn a label into a column-name part: lower case, anything else than letters and digits becomes '_'."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def example_titles(vectors, titles):
    """Return the titles of the EXAMPLES items nearest (cosine) to the mean vector of a topic, joined by ' || '."""
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    centre = unit.mean(axis=0)
    order = np.argsort(-(unit @ centre), kind="stable")[:EXAMPLES]
    return " || ".join(titles[n] for n in order)


def shares(items, group):
    """Return the size, date and share columns of one topic or group of items (shares of the joined tags)."""
    source_types = sorted(items["source_type"].unique())
    categories = sorted({c for value in items["categories"] for c in value.split("; ") if c != NO_CATEGORY})
    bands = sorted(items["age_band"].unique())
    item_categories = group["categories"].str.split("; ")
    row = {
        "size": len(group),
        "weighted_size": round(group["weight"].astype(float).sum(), 2),
        "share_none": round((group["categories"] == NO_CATEGORY).mean(), 3),
        "first_date": group["date"].min(),
        "last_date": group["date"].max(),
    }
    for source_type in source_types:
        row[f"share_source_{slug(source_type)}"] = round((group["source_type"] == source_type).mean(), 3)
    for category in categories:
        row[f"share_category_{slug(category)}"] = round(item_categories.map(lambda cs: category in cs).mean(), 3)
    for band in bands:
        row[f"share_age_{slug(band) or 'none'}"] = round((group["age_band"] == band).mean(), 3)
    return row


def topic_table(items, vectors, words, none_share):
    """Build one row per topic with its label, words, examples and the shares of the joined tags."""
    rows = []
    for topic, group in items.groupby("topic", sort=True):
        row = shares(items, group)
        if topic == OUTLIER_TOPIC:
            label = LABEL_OUTLIERS
        else:
            label = LABEL_CANDIDATE if row["share_none"] > none_share else LABEL_KNOWN
        rows.append({
            "topic": topic,
            "topic_name": group["topic_name"].iloc[0],
            "label": label,
            "top_words": ", ".join(words[topic]),
            "example_titles": example_titles(vectors[group.index], group["title"].tolist()),
        } | row)
    return pd.DataFrame(rows)


def group_topics(items, vectors, n_groups, method):
    """Join similar topics into n_groups groups; return {topic: group}, groups numbered by size (0 = largest)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    topics = sorted(t for t in items["topic"].unique() if t != OUTLIER_TOPIC)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    means = np.vstack([unit[(items["topic"] == t).to_numpy()].mean(axis=0) for t in topics])
    means = means / np.linalg.norm(means, axis=1, keepdims=True).clip(min=1e-12)
    if n_groups >= len(topics):
        cluster_of = list(range(len(topics)))
    else:
        tree = linkage(means, method=method, metric=GROUP_LINKAGES[method])
        cluster_of = list(fcluster(tree, t=n_groups, criterion="maxclust"))
    sizes = items["topic"].value_counts()
    members = {}
    for topic, cluster in zip(topics, cluster_of):
        members.setdefault(cluster, []).append(topic)
    # Biggest group first; ties broken by the smallest topic number, so the order never depends on scipy's labels
    order = sorted(members, key=lambda c: (-sum(sizes[t] for t in members[c]), min(members[c])))
    mapping = {topic: number for number, cluster in enumerate(order) for topic in members[cluster]}
    mapping[OUTLIER_TOPIC] = OUTLIER_TOPIC
    return mapping


def group_words(texts, groups, top_words):
    """Return {group: [top words]} scored by c-TF-IDF over the groups (outliers left out of the scoring)."""
    from sklearn.feature_extraction.text import CountVectorizer
    numbers = sorted(g for g in set(groups) if g != OUTLIER_TOPIC)
    joined = [" ".join(t for t, g in zip(texts, groups) if g == number) for number in numbers]
    vectorizer = CountVectorizer(stop_words="english")
    counts = vectorizer.fit_transform(joined).toarray().astype(float)
    vocabulary = vectorizer.get_feature_names_out()
    # c-TF-IDF as in BERTopic: word frequency in the group x log(1 + average words per group / word frequency overall)
    tf = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    idf = np.log(1 + counts.sum(axis=1).mean() / counts.sum(axis=0).clip(min=1))
    scores = tf * idf
    words = {}
    for row, number in enumerate(numbers):
        best = np.argsort(-scores[row], kind="stable")[:top_words]
        words[number] = [vocabulary[n] for n in best if scores[row, n] > 0]
    words[OUTLIER_TOPIC] = []
    return words


def group_table(items, topics, words):
    """Build one row per group: its name, its topics, the candidate topics in it, and the shares of the joined tags."""
    rows = []
    for number, group in items.groupby("group", sort=True):
        members = topics[topics["group"] == number]
        rows.append({
            "group": number,
            "group_name": group["group_name"].iloc[0],
            "n_topics": len(members),
            "topics": "; ".join(str(t) for t in members["topic"]),
            "candidate_topics": "; ".join(str(t) for t in members.loc[members["label"] == LABEL_CANDIDATE, "topic"]),
            "top_words": ", ".join(words[number]),
        } | shares(items, group))
    return pd.DataFrame(rows)


def group_name(number, words):
    """Name a group like a topic, with a 'g' in front (e.g. 'g2_toys_kids_toy_robot'); -1 is 'g-1_outliers'."""
    return "_".join([f"g{number}"] + (words[:4] if number != OUTLIER_TOPIC else [LABEL_OUTLIERS]))


def topic_name(topic, words):
    """Name a topic by its number and first 4 top words, as BERTopic does (e.g. '3_toys_companion_robot_play')."""
    return "_".join([str(topic)] + words[:4])


def print_summary(items, topics, groups, none_share, files, embedding_note, verb):
    """Print items in, topics, groups, outliers, the candidate new themes and where the files are."""
    real = topics[topics["topic"] != OUTLIER_TOPIC]
    outliers = int((items["topic"] == OUTLIER_TOPIC).sum())
    print(f"Items in: {len(items)}   {embedding_note}")
    print(f"Topics (without -1): {len(real)}   groups (without -1): {(groups['group'] != OUTLIER_TOPIC).sum()}   "
          f"outliers (topic -1): {outliers} ({outliers / len(items):.1%})")
    print("\nGroups (size, topics, share 'none', candidate topics, top 5 words):")
    for row in groups[groups["group"] != OUTLIER_TOPIC].itertuples():
        candidates = row.candidate_topics or "-"
        print(f"  g{row.group:<3} {row.size:>5}  {row.n_topics:>2} topics  {row.share_none:.2f}  "
              f"cand: {candidates:<8}  {', '.join(row.top_words.split(', ')[:5])}")
    print("\nItems per label:")
    print(items["label"].value_counts().to_string())
    source_columns = [c for c in topics.columns if c.startswith("share_source_")]
    print(f"\nOutliers per source type: "
          + ", ".join(f"{t} {n}" for t, n in items.loc[items["topic"] == OUTLIER_TOPIC, "source_type"].value_counts().items()))
    threshold = "labels as saved" if none_share is None else f"share of category 'none' > {none_share}"
    print(f"\nCandidate new themes ({threshold}):")
    candidates = topics[topics["label"] == LABEL_CANDIDATE]
    if candidates.empty:
        print("  none")
    else:
        table = candidates[["topic", "size", "share_none"] + source_columns].copy()
        table.columns = [c.removeprefix("share_source_") for c in table.columns]
        table["top words"] = candidates["top_words"].map(lambda w: ", ".join(w.split(", ")[:5]))
        print(table.to_string(index=False))
    print("\nAll topics (topic, group, size, share 'none', top 5 words):")
    for row in real.sort_values(["group", "topic"]).itertuples():
        print(f"  {row.topic:>3}  g{row.group:<3} {row.size:>4}  {row.share_none:.2f}  "
              f"{', '.join(row.top_words.split(', ')[:5])}")
    print(f"\n{verb} {', '.join(files)}")


def main():
    """Cluster the tagged items (or read the saved topics) and save the topic and item files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="inp", default="data/clean/items_tagged.csv",
                        help="tagged items (default: data/clean/items_tagged.csv)")
    parser.add_argument("--model", default=MODEL, help=f"sentence model (default: {MODEL})")
    parser.add_argument("--min-topic-size", type=int, default=MIN_TOPIC_SIZE,
                        help=f"smallest topic HDBSCAN may form; a judgement call (default: {MIN_TOPIC_SIZE})")
    parser.add_argument("--umap-neighbors", type=int, default=UMAP_NEIGHBORS,
                        help=f"UMAP n_neighbors, BERTopic's default; a judgement call (default: {UMAP_NEIGHBORS})")
    parser.add_argument("--umap-components", type=int, default=UMAP_COMPONENTS,
                        help=f"dimensions UMAP reduces to, BERTopic's default; a judgement call (default: {UMAP_COMPONENTS})")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                        help=f"HDBSCAN min_samples; lower resists one topic swallowing everything; a judgement call (default: {MIN_SAMPLES})")
    parser.add_argument("--groups", type=int, default=GROUPS,
                        help=f"broad groups the topics are joined into, for presenting; a judgement call (default: {GROUPS})")
    parser.add_argument("--group-linkage", choices=sorted(GROUP_LINKAGES), default=GROUP_LINKAGE,
                        help=f"how topics are joined into groups; a judgement call (default: {GROUP_LINKAGE})")
    parser.add_argument("--top-words", type=int, default=TOP_WORDS,
                        help=f"top words kept per topic (default: {TOP_WORDS})")
    parser.add_argument("--seed", type=int, default=SEED, help=f"UMAP random_state (default: {SEED})")
    parser.add_argument("--none-share", type=float, default=NONE_SHARE,
                        help=f"share of category 'none' above which a topic is a 'candidate new theme'; "
                             f"a judgement call (default: {NONE_SHARE})")
    parser.add_argument("--use-saved", action="store_true",
                        help="read the saved topic files instead of recomputing (offline mode)")
    parser.add_argument("--topics", default="data/topics.csv", help="one row per topic (default: data/topics.csv)")
    parser.add_argument("--groups-out", default="data/topic_groups.csv",
                        help="one row per group (default: data/topic_groups.csv)")
    parser.add_argument("--item-topics", default="data/item_topics.csv",
                        help="one row per item (default: data/item_topics.csv)")
    parser.add_argument("--embeddings", default="data/embeddings.npy", help="saved vectors (default: data/embeddings.npy)")
    parser.add_argument("--embedding-ids", default="data/embeddings_ids.csv",
                        help="id, text_hash and model of each saved vector (default: data/embeddings_ids.csv)")
    args = parser.parse_args()
    if args.min_topic_size < 2:
        parser.error("--min-topic-size must be at least 2")
    if args.min_samples < 1:
        parser.error("--min-samples must be at least 1")
    if args.groups < 1:
        parser.error("--groups must be at least 1")
    if args.umap_neighbors < 2 or args.umap_components < 1 or args.top_words < 1:
        parser.error("--umap-neighbors must be at least 2, --umap-components and --top-words at least 1")
    if not 0 <= args.none_share <= 1:
        parser.error("--none-share must be between 0 and 1")

    items = pd.read_csv(args.inp, dtype=str, keep_default_na=False)

    if args.use_saved:
        topics = pd.read_csv(args.topics, dtype={"top_words": str}, keep_default_na=False)
        groups = pd.read_csv(args.groups_out, dtype={"top_words": str, "topics": str, "candidate_topics": str},
                             keep_default_na=False)
        item_topics = pd.read_csv(args.item_topics, dtype=str, keep_default_na=False)
        if set(item_topics["id"]) != set(items["id"]):
            raise SystemExit(f"{args.item_topics} does not cover the same item ids as {args.inp}: run without --use-saved")
        item_topics["topic"] = item_topics["topic"].astype(int)
        print_summary(item_topics, topics, groups, None, [args.topics, args.groups_out, args.item_topics],
                      "(read from saved files)", "Read")
        return

    # Embed title + text only (the tags never enter the model)
    texts = embedded_text(items).tolist()
    vectors, reused, new = embed(items, texts, args.model, args.embeddings, args.embedding_ids)

    # Cluster, then join the tags to the topics
    topic_per_item, words = fit_topics(texts, vectors, args.min_topic_size, args.min_samples, args.seed,
                                       args.umap_neighbors, args.umap_components, args.top_words)
    items["topic"] = topic_per_item
    items["topic_name"] = [topic_name(t, words[t]) for t in topic_per_item]
    topics = topic_table(items, vectors, words, args.none_share)
    items["label"] = items["topic"].map(dict(zip(topics["topic"], topics["label"])))

    # Second level: join similar topics into groups
    mapping = group_topics(items, vectors, args.groups, args.group_linkage)
    items["group"] = items["topic"].map(mapping)
    g_words = group_words(texts, items["group"].tolist(), args.top_words)
    items["group_name"] = items["group"].map(lambda g: group_name(g, g_words[g]))
    topics.insert(3, "group", topics["topic"].map(mapping))
    topics.insert(4, "group_name", topics["group"].map(lambda g: group_name(g, g_words[g])))
    groups = group_table(items, topics, g_words)

    # Save
    item_topics = items[["id", "topic", "topic_name", "label", "group", "group_name"] + JOINED_COLUMNS]
    for path, table in [(Path(args.topics), topics), (Path(args.groups_out), groups), (Path(args.item_topics), item_topics)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    note = f"embeddings reused: {reused}   newly embedded: {new}   model: {args.model}"
    print_summary(item_topics, topics, groups, args.none_share,
                  [args.topics, args.groups_out, args.item_topics, args.embeddings, args.embedding_ids], note, "Saved")


if __name__ == "__main__":
    main()
