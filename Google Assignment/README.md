# Google Assignment: is Google search losing ground to AI?

First assignment of the Strategic Scanning course (see [[Class 1 & 2#Assignment]]).

## The question

**Problem to show:** fewer people use classic Google search, maybe because they use AI tools instead, so Google earns less from search ads.

**Angle:** if that is really happening, the press should be writing about it. So I collected 3 years of news headlines about Google's search business and ad revenue. Then I cleaned them and checked what they actually say.

## What is in this folder

| File | What it is |
| --- | --- |
| `google_ad_news_all_steps.ipynb` | **The whole pipeline in one notebook**, one section per step. Start here to submit or review. |
| `*.py` | The 6 scripts, one per step (the originals; the notebook is a copy of them) |
| `to_notebooks.py` | Rebuilds the notebook from the scripts after a script changes |
| `data/` | Every CSV the steps read and write |
| `Assignment - Google ad news data process.md` | The detailed log: every try, every number, every decision. Read it for the "why" |

## The steps, in plain words

### 1. Collect headlines (`google_ad_news.py`)
I searched Google News for headlines about Google + search + ad revenue over the last 3 years. Google News returns only ~100 results per search, so the script runs one search per 30-day window and merges the results.

- First tries were too broad (any company's ad revenue) or too narrow (long queries got cut off by Google News).
- Final version: 4 short queries, each requiring "Google" in the headline. **1,337 articles.**
- Output: `data/google_ad_news.csv`

### 2. Get the real link and the sub-header (`fetch_summaries.py`)
Google News gives only a redirect link and the headline. This step opens each article and reads the short description the publisher wrote, which gives more text to work with.

- Sub-header found for 74% of articles. The rest were blocked by paywalls or bot protection.
- Output: `data/article_summaries.csv`

### 3. Clean the data (`clean_news.py`)
Most collected headlines were noise: how-to guides, Google Ads product news, SEO gossip, stock tips, other Google products.

- Drop sources that are not news (17 sites).
- Keep a headline only if it matches a topic (search business, ad revenue, earnings, publisher traffic) and no noise pattern.
- Group articles reporting the **same story** (similar words, published within 4 days) and keep one per story.
- Every rejected article is saved with its reason, so the filter can be checked.
- **648 kept, 689 rejected.** Outputs: `data/google_ad_news_clean.csv`, `data/google_ad_news_rejected.csv`

### 4. Get the opening paragraphs (`fetch_leads.py`)
Sub-headers are often too short ("That's a lot of overviews."), so I also fetched the first paragraphs of each article.

- The script got 420 of 648. A browser service (TinyFish, used by hand, not a script) got 620. The clean file currently uses the TinyFish version (`data/article_leads.csv`), and the script's version is in `data/article_leads_script.csv`. **Still to decide** which one to report.
- 38 sub-headers were written by Claude when the publisher's own text could not be read. They are marked (`description_source`) and count less (`description_weight`).

### 5. Check the articles against the problem (`analyze_news.py`)
The problem is really 2 claims: "search volume is down" and "ad revenue is down". For each article, the script looks for a subject ("searches", "ad revenue") near a direction word ("falls", "grows") and labels the article as supporting, opposing, or not taking a side.

| support | oppose | both | no claim | others |
| --- | --- | --- | --- | --- |
| 19 | 48 | 5 | 145 | 431 |

More articles **oppose** the problem than support it. Most take no side.
- Output: `data/matching_analysis_results.csv`

### 6. What do the other articles talk about? (`scan_topics.py`)
To find hidden trends and weak signals, this step counts the words used by the 576 articles that take no side, in 3 tables:
- **distinctive** terms (used much more than in the articles that take a side): "publishers", "traffic", "antitrust", "summaries"...
- **rising and falling** terms per year: "traffic", "publishers" and "ai mode" go up, while "monopoly" and "search engine" go down.
- **weak signals**: rare, recent terms from several sources: "paying publishers", "licensing deals", "ai payments"...
- Outputs: `data/topic_scan_distinctive.csv`, `data/topic_scan_rising.csv`, `data/topic_scan_weak_signals.csv`

## What I took away (my reading, not proven)

- The press talks less about Google losing users or revenue, and more about **websites losing traffic** because Google answers questions itself with AI. The damage may be on the publishers, not on Google.
- Antitrust was a big theme in 2024 and is fading.
- Possible new trend (low confidence, few articles): **Google paying publishers** for use of their content in AI answers.
- Possible legal risk (low confidence): EU action on AI summaries.

## Limits to remember

- Topics and noise are judged on the headline only, and the keyword rules were tuned by reading samples, not measured against a labelled set.
- More articles in recent years can just mean Google News keeps recent news better. Article counts don't measure the problem itself.
- The SEO trade press is a big share of the sources, so its vocabulary weighs a lot.
- The AI-only collection (`--ai`) was never cleaned.

## How to run it again

From this folder, with the project's virtual environment (`repo/` one level up):

```bash
cd "Google Assignment"
../repo/bin/python analyze_news.py        # any step, same as in the notebook
../repo/bin/python to_notebooks.py --force # rebuild the notebook after changing a script
```

To use the notebook, open it from this folder so that `data/...` points to the right files. Steps 1, 2 and 4 go to the web. Step 1 **replaces** the collected headlines with a new collection, so skip it to keep the saved data.
