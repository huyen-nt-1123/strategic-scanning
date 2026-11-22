Homework for [[Class 1 & 2#Assignment|Class 1 & 2 › Assignment]] (steps 2 and 3: collect data, then clean it up).
https://claude.ai/artifact/JkVWi9bVnUNPmjNV8WD3RL#slide-3dd33bdac6c2

**Problem to show:** usage of Google legacy search (google.com) is down, maybe because of AI -> ad revenue is down.
**Angle:** news headlines over the last 3 years. If the problem is real, the press should be writing about Google's search usage, its ad revenue and the traffic it sends to websites.

**Scripts**
- `google_ad_news.py` collects headlines from Google News -> `data/google_ad_news.csv`
- `fetch_summaries.py` fetches the real address and the sub-header of every article -> `data/article_summaries.csv`
- `clean_news.py` cleans them -> `data/google_ad_news_clean.csv` and `data/google_ad_news_rejected.csv`
- `fetch_leads.py` fetches the opening paragraphs of the kept articles -> `data/article_leads_script.csv`
- `data/article_leads.csv` (the leads now in the clean file) was not made by a script: fetched through TinyFish, see step 3
- `analyze_news.py` checks every article against the two claims -> `data/matching_analysis_results.csv`
- `scan_topics.py` counts the terms of the articles that take no side -> `data/topic_scan_distinctive.csv`, `data/topic_scan_rising.csv`, `data/topic_scan_weak_signals.csv`

### 1. Rough keyword search
Source: the Google News RSS search (`https://news.google.com/rss/search`), read with `requests` and saved with `pandas`. No API key needed.

**Windows from the start.** Google News returns at most ~100 results per search, so one search over 3 years would give a tiny, recent-biased sample. From the first version, the script cuts the 3 years into 30-day windows and runs one search per window with `after:` / `before:`: the limit applies per window and not to the whole period.

**First try: one generic query**
- `(Google OR Alphabet) ("ad revenue" OR "advertising revenue" OR "search revenue")`
- With `--ai`, this was added: `(AI OR ChatGPT OR Gemini OR "AI Overviews" OR "generative AI")`

**Result: 1,528 articles** (1,193 with `--ai`)

| Year | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- |
| Articles | 107 | 459 | 524 | 438 |
| Articles with `--ai` | 44 | 262 | 475 | 412 |

- Each window returned 33 to 51 articles, so the 100-result limit was not reached.
- Top sources: The Motley Fool (66), CNBC (34), Business Insider (32), ADWEEK (32), Reuters (31), Yahoo Finance (30).
- **Too generic.** The query only asks for Google or Alphabet somewhere and "ad revenue" somewhere, so it brings in any company's ad revenue and nothing ties it to search usage. Examples: "Meta's Endless Efforts to Boost Ad Revenue Growth: Introducing Game-Changing Reels Ad Features" (Dazeinfo), "Northeastern B.C. cities lose their last newspapers" (CBC).

**Second try: one long query with exact phrases**, to keep only Google's own revenue linked to search usage. An article had to match one phrase of each group:
- `("Google's ad revenue" OR "Google ad revenue" OR "Google's advertising revenue" OR "Google's search revenue" OR "Google search revenue")`
- `("search traffic" OR "search volume" OR "search queries" OR "search market share" OR "zero-click")`
- With `--ai`: `(AI OR ChatGPT OR "AI Overviews" OR Perplexity)`

**Result: 7 articles**, all from May 2025 to July 2026.
- The query was too long: Google News cut the end of it, which was the date limits and the AI group. Every window returned the same 7 recent articles.
- The exact phrases were too strict: Google News matches mostly on the headline and a short summary, not the full article. With the dates placed first, a test month returned 0.

What the two tries ran into:
- **Long queries get cut.** Google News drops whatever comes after roughly 32 words, and the date limits were the part that got dropped.
- **Most hits only mention Google in passing.**

### 2. Refining the query
- **Split the long query into 4 short ones**, each under the word limit:
	- `intitle:Google "ad revenue" search traffic`
	- `intitle:Google "search revenue" (traffic OR queries OR usage OR "market share")`
	- `intitle:Google search (traffic OR queries OR usage) (ads OR advertising) revenue`
	- `intitle:Google "search market share" revenue`
- **`intitle:Google`** keeps only headlines that name Google.
- **Date limits placed first** in the query (`after:` / `before:`, then the keywords), so they are never the part that gets cut.
- **Date check in the script**: articles outside the window are skipped, in case Google ignored the date limits.
- **Exact de-duplication**: same link, or same title + source.
- **`--ai` option** adds `(AI OR ChatGPT OR "AI Overviews")` to every query, to look at the AI angle separately (`data/google_ad_news_ai.csv`).

**Result without `--ai`: 1,337 articles**

| Query | Articles |
| --- | --- |
| `search (traffic OR queries OR usage) (ads OR advertising) revenue` | 867 |
| `"ad revenue" search traffic` | 263 |
| `"search revenue" (traffic OR ...)` | 149 |
| `"search market share" revenue` | 58 |

| Year | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- |
| Articles | 57 | 291 | 483 | 506 |

### 3. Data cleanup
**Why:** the loosest query brought in 867 of the 1,337 rows, and a lot of it is not about the problem. Reading a sample showed 5 kinds of noise:
- Advertiser how-tos and Google Ads product updates ("Google Ads simplifies enhanced conversions")
- SEO trade chatter ("Google Search Ranking Volatility July 24th")
- Evergreen explainer pages that get a new date on every update ("Google CEO Salary")
- Stock picks ("Should You Buy Alphabet Stock...")
- Other Google products (Pixel, Maps, Cloud on its own)

**Decision on scope:** keep articles about Google's search business, Google's ad revenue, Alphabet earnings, **and publisher traffic** (sites losing Google traffic is the other side of the same story).

**`clean_news.py` works in 3 steps:**
1. **Source blocklist.** 17 sources that are not news: explainer / statistics sites (FourWeekMBA, Statista, Semrush...), how-to guides, and Google's own blogs.
2. **Topic and noise patterns**, on the headline only. Every headline is tagged with the topics it matches (`search_business`, `ad_revenue`, `earnings`, `publisher_traffic`) and the noise it matches (`howto`, `ads_product`, `seo_chatter`, `stock_pick`, `other_product`). Kept = at least one topic and no noise.
3. **Same story.** Articles that report the same story are grouped, and only the first one is kept (details below).

**Nothing is deleted silently.** Rejected rows go to their own file with a `reason` column, so the filter can be checked.

**How the patterns were tuned:** run -> read a random sample of kept and rejected rows -> fix the pattern -> run again. 3 rounds:
- Round 1 (550 kept): topics too narrow. Real stories were rejected ("Google's AI answer engine is quietly strangling..."), and `google pay` rejected "Google pays publishers".
- Round 2 (627 kept): the word "results" in the earnings pattern let in "Google will stop summarizing recipes in search results"; "Google Discover" was treated as another product although it is a traffic source for publishers.
- Round 3 (744 kept): any headline with the word "search" counts as a topic, since every headline already names Google; the noise patterns do the removing.

#### Same story: one article per piece of news
**Why:** after round 3, the clean file still had the same news many times. Example: Google paying Apple 36% of its Safari search revenue (November 2023) was there 6 times (CNBC twice, BBC, Android Police, Ars Technica, Search Engine Land). The first duplicate check only compared the letters of two headlines (90% alike), so it found 7 re-posted headlines and nothing else: two outlets never word a story the same way.

**More text to compare: the sub-header.** Google News gives a redirect link and no summary. `fetch_summaries.py` finds the publisher's real address, opens the page and reads the description the publisher wrote for it (the `og:description` / `description` tag, the text shown in link previews).
- Real address found for all 1,337 articles.
- Sub-header found for 992 (74%). 309 pages blocked the script (paywalls and bot protection: Search Engine Land, Press Gazette, Reuters, WSJ, NYT...), 30 had no description, 6 did not load.
- 17 of them are the same text under several different headlines (the site's slogan, not the article), so `clean_news.py` ignores them. 975 usable sub-headers.
- From Europe, Google News answers with a cookie-consent page. The script sends the cookie that says the choice was made.

**How two articles are compared:** on the words they share, in the headline and the sub-header.
- Filler words and "Google" / "Alphabet" are dropped. Numbers are kept ("36%", "$100").
- A word counts more when few articles use it: "Penske" or "Offerwall" says more about the story than "search". Headline words count double.
- Similarity between two articles = cosine similarity of their weighted words (0 = nothing in common, 1 = same words).
- Same story = published within 4 days of each other and similarity of at least 0.25. The first article is kept. The others go to the rejected file with the reason `same story as: <headline kept>`.
- New column `story_size` in the clean file: how many articles reported that story. A way to see which news got the most coverage.

**Decision: different angles on the same event stay separate.** On an earnings day, "search revenue up", "network ad revenue down" and "cloud surges" are 3 stories, not 1. The network ad revenue decline is part of our problem and must not disappear into a general earnings article.

**How the 0.25 threshold was chosen:** by reading the groups at several thresholds, and the pairs just under the threshold.
- At 0.25, nearly every group is one piece of news.
- Between 0.19 and 0.25, about half of the pairs are a different angle or a follow-up comment (ex: "Google Revenue Soars to Record as AI Boom Lifts Cloud Business" and "Google Q3 Report: AI Mode, AI Overviews Lift Total Search Usage"). Going lower would merge angles.
- The sub-header helps with paraphrase: "Google sends a third of Safari ad revenue to Apple" (BBC) joins the "36%" articles only because of it.

**Result: 648 kept, 689 rejected** (744 kept before the same-story step)

| Reason for rejection | Articles |
| --- | --- |
| No topic matched | 382 |
| Same story as an earlier article | 101 |
| Blocked source | 95 |
| Noise: Google Ads product updates | 42 |
| Noise: how-to / guides / statistics | 35 |
| Noise: other Google product | 16 |
| Noise: SEO chatter | 15 |
| Noise: stock picks | 8 |

(A headline can match 2 kinds of noise, so the noise rows add up to more than the 111 articles rejected for noise.)

56 stories were reported by more than one article. The most covered (6 articles each): Google admits the open web is in "rapid decline", the first $100 billion quarter, Q1 2026 earnings (cloud surge), Google paying about 100 publishers for AI Overviews.

| Topic of kept articles (can have several) | Articles |
| --- | --- |
| search_business | 427 |
| publisher_traffic | 206 |
| earnings | 127 |
| ad_revenue | 70 |

| Year | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- |
| Kept | 25 | 135 | 252 | 236 |

#### Filling in the missing sub-headers
**Why:** 172 of the 648 kept articles still had no sub-header. 148 pages blocked the script (HTTP 401 / 403 / 406...), 14 had no description tag, 7 only had the site's slogan, 3 timed out or failed. Step 4 will read the sub-headers, so a quarter of the file would have been judged on the headline alone.

**How (11 October 2026, done by hand with Claude Code, not a script):** each article's real address was opened again, with a fallback every time a page stayed blocked.
1. **The live page, through a browser-style fetcher** (TinyFish) instead of `requests`: 127 pages opened.
2. **The Internet Archive copy** (`web.archive.org`) for the pages still blocked: 34 more (NYT, WSJ, Forbes, Search Engine Land, The Information...).
3. **The search-result snippet** for the last ones, which shows the start of the publisher's own text: mostly Reuters.

**Where the 172 descriptions come from:**

| Origin | Articles | `description_source` | `description_weight` |
| --- | --- | --- | --- |
| Publisher's own text, live page | 101 | `publisher` | 1.0 |
| Publisher's own text, Internet Archive copy | 33 | `publisher` | 1.0 |
| Written by Claude from the article text | 24 | `written_from_text` | 0.5 |
| Written by Claude from a search-result snippet | 14 | `written_from_snippet` | 0.25 |

- **Written from the article text (24):** the page opened but had no description, only the site's slogan (GIGAZINE, Xpert.Digital, Analytics India Magazine), or a teaser that says nothing about the content ("Hint: It involves better targeting"). One sentence that sticks to the figures and claims in the article.
- **Written from a snippet (14):** the article itself was never read. 5 Reuters, 2 Borneo Bulletin, 2 Search Engine Land, 1 each for WSJ, Medium, Moomoo, Seeking Alpha and Xpert.Digital.

**Decision: the 38 written descriptions are marked and count less.** They are not the publisher's words: the wording is Claude's, so it must not weigh as much as the press's own framing when step 4 counts words or themes.
- 2 new columns in the clean file, `description_source` and `description_weight`. The 476 sub-headers that were already there are `publisher`, 1.0. The file now has 610 at 1.0, 24 at 0.5, 14 at 0.25.
- The weights are a judgement call, not measured: half for a second-hand reading of the article, a quarter when only the snippet was seen.
- The same information is in the `status` column of `data/article_summaries.csv` (`ok`, `ok (Internet Archive copy ...)`, `summary written from article text`, `summary written from search-result snippet (page blocked)`). `clean_news.py` now reads it and writes the 2 columns itself.

> [!warning] `clean_news.py` was not run again
> The 172 descriptions were added to the clean file after the same-story step, so the 648 / 689 figures above still stand. Running the script again would use the new sub-headers in the same-story comparison: a trial run on a copy gave **650 kept** (99 same-story rejections instead of 101), and 3 descriptions were emptied again because they sit under several headlines in the full file.

#### More text per article: the opening paragraphs
**Why:** even when the sub-header is the publisher's, it is often too thin to work with. Median 23 words; 73 of the 648 are under 15 words, and many are teasers that say nothing ("We all knew ads would eventually come.", "That's a lot of overviews.").

**Where a longer text could come from (11 October 2026, tried with Claude Code):**

| Source | Result |
| --- | --- |
| Page metadata (`og:` / `twitter:` / other meta tags, via TinyFish `page_metadata`) | Nothing more: the same short description, no abstract field |
| TinyFish "highlights" (ranked extracts answering a question) | Not enabled on this account (HTTP 403) |
| Google News RSS | Headline and link only, already known from step 1 |
| **Article body** (TinyFish fetch, markdown) | **Works**: body returned for 634 of 648 pages |
| Internet Archive copy, read through the same fetcher | Works for the blocked pages that have a copy (32 of 34) |

**Decision: keep the article's opening paragraphs, word for word, in a new column `lead`.** No rewriting, so it stays the press's own framing and counts fully (unlike the 38 sub-headers written by Claude).

**How the lead is picked from the body:**
- Paragraphs of at least 15 words, minus page furniture (subscribe / cookie / affiliate notices, photo credits, the author's biography, newsletter greetings).
- Starts at the first paragraph when the opening shares words with the headline; if not, at the first of the next few paragraphs that does. Nothing is kept when no opening paragraph is about the headline (the page is a paywall notice, a home page or another article).
- Paragraphs are added until there are about 60 words, capped at 130 words. Repeated paragraphs are dropped.

**First pass, through TinyFish (not a Python script): 620 of 648 articles have a lead** (median 80 words; 591 from the live page, 29 from the Internet Archive copy). TinyFish is an outside service that opens each page in a real browser; it was called from the command line.
- 70 of the 73 articles with a sub-header under 15 words now have a lead.
- 23 of the 38 articles whose sub-header was written by Claude now have the publisher's own opening instead.
- 28 without a lead: 16 pages blocked, 7 with no readable text, 5 where the text was not about the headline (Reuters, The Information, Borneo Bulletin, Xpert.Digital...). 15 articles have neither a publisher sub-header nor a lead.
- Saved in `data/article_leads.csv` (`link`, `url`, `lead`, `lead_source`, `status`). `clean_news.py` reads it (`--leads`) and adds the `lead` column to the clean file; it does not use it to compare stories.

**Known flaws, from reading samples:** a few leads start with a photo caption or a newsletter intro, and a few start one or two paragraphs into the article. Not checked row by row.

#### The same step as a Python script: `fetch_leads.py`
**Why:** the assignment is about Python scripts, and the first pass depended on an outside service. Question: how many leads does a plain script get?

**How it works:** same shape as `fetch_summaries.py` (`requests`, the standard `HTMLParser`, a thread pool, results saved every 25 articles, can be stopped and started again).
1. Opens the article with `requests` and collects the text of its `<p>` tags, skipping `script`, `style`, `nav`, `header`, `footer`, `aside`, `form`, `figure`.
2. Picks the lead with the same rules as above.
3. If the page is blocked or gives no lead, asks the Internet Archive for its latest copy (`https://web.archive.org/web/2id_/<address>`).
4. Writes `link`, `url`, `lead`, `lead_source` (`page` / `archive`) and `status` (what went wrong, for each of the two tries).

**Result: 420 of 648 articles get a lead from the live page** (median 82 words), against 620 with TinyFish.

| Why the script got no lead (228 articles) | Articles |
| --- | --- |
| Page blocked (HTTP 403: 123, 401: 17, 406: 4, 402: 3, 429: 1, 451: 1) | 149 |
| Page opened, but no opening paragraph about the headline | 78 |
| Page did not load | 1 |

- **Blocked pages are the same publishers as in `fetch_summaries.py`:** Press Gazette (27), Search Engine Land (18), ADWEEK (9), WSJ (8), Reuters (6)...
- **The 78 pages that open without a lead:** the text is not in the HTML the script receives. It is added by JavaScript once a browser runs the page, or it is not in `<p>` tags. A browser sees it, `requests` does not.
- **Same article, two methods:** 416 articles have a lead from both. 309 start with the same words. In the other 107 the script usually starts one paragraph later or keeps a stray line of the page: a rendered page separates the article from the rest better than raw HTML. Letting the parser read `header` and `figure` blocks was tested and fixed 6 of them, so it was left as is.
- 204 articles have a lead only through TinyFish, 4 only through the script, 24 through neither.

**The Internet Archive fallback gave nothing in this run.**
- First run: 6 articles fetched at the same time -> the archive answered `429 Too Many Requests` or refused the connection for all 227 misses.
- Fix: archive requests now go one at a time, with a 4-second pause and one more try after 30 seconds.
- Second run: still `429` on every request, also on single test requests: the address was being throttled after the first run. A run that waited 60 seconds for each refusal had saved only 75 articles after 27 minutes, and was stopped.
- Fix: after 2 refusals in a row the script stops asking the archive for the rest of the run and says so. The whole run now takes a few minutes.
- So the archive part of the script is **written but not yet seen working**. To try it once the throttling is over: `python fetch_leads.py --out data/article_leads_script.csv --retry`. (With `curl`, earlier the same day, the same archive addresses gave 34 pages.)

> [!warning] Two lead files, and the clean file uses the one that is not from a script
> - `data/article_leads.csv`: TinyFish, 620 leads. This is what the `lead` column of the clean file holds, and what `clean_news.py --leads` reads by default.
> - `data/article_leads_script.csv`: `fetch_leads.py`, 420 leads.
> - **To decide:** keep the 620 and say in the report that they needed a browser service, or switch to the script's 420 (plus whatever the archive adds on a retry) so that every number comes from our own Python. To switch: `python clean_news.py --leads data/article_leads_script.csv` (this also redoes the same-story step, see the warning above).

### Limits to keep in mind
- **Topics and noise are judged on the headline only**, so a relevant article with a vague headline is rejected ("Google: From Symbiote to Parasite"), and the opposite happens too. The sub-headers are now in the file and could be used for this.
- **Same-story grouping is not perfect.** It is set on the careful side, so some repeats are still in: the Ars Technica article on the 36% story was not grouped (similarity 0.22), and the UK opt-out ruling of June 2026 ended up as 2 groups of 4. A few groups mix two angles (an earnings preview with the results article).
- **1 article in 4 had no sub-header when the same-story step ran** (blocked pages) and was compared on its headline alone, so its repeats are harder to catch. They have one now (see "Filling in the missing sub-headers"), but the grouping was not redone.
- **38 sub-headers were written by Claude**, not by the publisher (`description_source`). Use `description_weight` in step 4, or leave them out to check that a result does not depend on them.
- **Keyword patterns are a judgement call**, tuned by reading samples and not measured against a hand-labelled set. Some noise is still in the clean file and some good articles are in the rejected file.
- **Antitrust stories** are kept only when the headline also mentions search, ads or publishers.
- **Article counts are not a measure of the problem.** More articles in 2025-2026 can also mean Google News indexes recent news better.
- **The AI file** (`google_ad_news_ai.csv`) has not been cleaned yet: `python clean_news.py --in data/google_ad_news_ai.csv`

### For step 4 (analysis) - things already noticed
- `publisher_traffic` is the second biggest topic. The press talks more about **websites losing Google traffic** than about Google losing ad revenue. Possible shift of the problem: is Google's usage down, or is Google keeping the users and sending fewer of them on?
- Several kept headlines say the opposite of our problem ("AI is actually making Google search bigger", first $100 billion quarter). To check against Alphabet's reported search revenue.

### 4. What the articles that take no side talk about: `scan_topics.py`
**Why:** `analyze_news.py` checks every article against the two claims (search volume is down, ad revenue is down). Most articles take no side:

| Category | support | oppose | both | no claim | others |
| --- | --- | --- | --- | --- | --- |
| Articles | 19 | 48 | 5 | 145 | 431 |

576 of the 648 articles are `no claim` or `others`. If the press is writing about another problem or a new trend, it is in those. Question: what do they talk about?

**A plain word count is not enough.** The most common words would be "google", "search", "ai", "ads": the words the articles were collected on. So the script counts terms in 3 ways that leave those out.

**How it works:**
- A term = one word, or two words that follow each other ("search engine", "ai mode"). Lower-cased, final "'s" removed, common English words dropped. Nothing is done about plurals: "publisher" and "publishers" are two terms.
- A term counts once per article. Every figure is a number of articles, or a share (% of the articles of a group or of a year).
- **Scanned** = `others` + `no claim` (576). **Compared with** = `support` + `oppose` + `both` (72). Both are options (`--scan`, `--compare`).
- Reads the headline, the sub-header and the lead. 17,237 different terms in the scanned articles.

**3 tables:**
1. **Distinctive**: terms used by a bigger share of the scanned articles than of the articles that take a side. In at least 5 scanned articles.
2. **Rising**: share of the scanned articles using a term each year, sorted by the change from the first year to the last. The falling terms are at the bottom of the same file. 2023 is left out (25 scanned articles, under the minimum of 30), so the change is 2024 -> 2026.
3. **Weak signals**: terms in 2 to 5 articles, from at least 2 different sources, first seen in the last 12 months of the file (after 11 October 2025; the newest article is from 11 October 2026). "Recent" is counted from the newest article and not from today, so a second run gives the same file.

**Result: 1,519 distinctive terms, 1,450 terms with a share per year, 655 weak-signal candidates.** Two runs gave identical files.

| Distinctive term | Scanned articles | Share of scanned | Share of articles that take a side |
| --- | --- | --- | --- |
| search engine | 118 | 20.5% | 4.2% |
| media | 81 | 14.1% | 0% |
| publishers | 148 | 25.7% | 12.5% |
| traffic | 175 | 30.4% | 18.1% |
| antitrust | 61 | 10.6% | 1.4% |
| summaries | 62 | 10.8% | 2.8% |
| judge | 40 | 6.9% | 0% |
| default | 39 | 6.8% | 0% |

| Term | 2024 | 2025 | 2026 | Change |
| --- | --- | --- | --- | --- |
| traffic | 14.9% | 35.1% | 37.0% | +22.1 |
| publishers | 15.7% | 20.7% | 37.0% | +21.3 |
| publisher | 2.5% | 5.4% | 17.8% | +15.3 |
| ai mode | 0% | 21.6% | 14.4% | +14.4 |
| revenue | 24.8% | 23.0% | 37.5% | +12.7 |
| search traffic | 0% | 5.4% | 9.1% | +9.1 |
| monopoly | 13.2% | 6.8% | 3.4% | -9.8 |
| generative | 14.0% | 7.2% | 3.8% | -10.2 |
| search engine | 33.1% | 18.9% | 12.5% | -20.6 |
| giant | 28.1% | 15.8% | 5.8% | -22.3 |

**Weak-signal candidates, the most recent ones** (September - October 2026, 2 or 3 articles each): "payouts", "paying publishers", "ai payments", "licensing deals", "dismissal", "antitrust lawsuits", "union antitrust". Their example headlines: Google testing payments to publishers for AI answers, the Chegg and Penske lawsuits dismissed, EU regulators and AI summaries.

**What went wrong:**
- **Table 3 is noisy.** Many rows are two outlets reporting one story on the same day, so any pair of words they share comes out as a "signal" ("vary widely", "one-tenth", "without losing"). The table has to be read with the example headline next to it. Possible fix, not done: ask for a minimum number of days between the first and the last article.
- **Table 1 is compared with only 72 articles.** One article there is 1.4 points, so a small difference means little.
- **Plurals are separate terms**, so "publisher" and "publishers" each have their own row.
- **2026 stops in October.** Shares correct for the number of articles, but the stories of the last weeks weigh more in the 2026 column.
- **The pool was already filtered.** `clean_news.py` only kept articles matching its 4 topics, so a new subject can only show up next to search, ads or publishers. The rejected file has not been scanned.
- **Source bias.** The SEO trade press is among the biggest sources (Search Engine Roundtable 27, Press Gazette 27, Search Engine Journal 21, PPC Land 20), so its vocabulary ranks high. The `sources` column of table 3 is there to see it.

**The thresholds are judgement calls, not measured:** 5 articles for tables 1 and 2, 30 articles per period, 2 to 5 articles and 2 sources and 12 months for table 3. All are options (`--min-articles`, `--min-period-articles`, `--weak-min`, `--weak-max`, `--min-sources`, `--recent-months`).

**Reading of the result (not measured by the script, to check):**
- The articles that take no side are mostly about **publishers losing traffic to AI answers**, not about Google's own search volume or revenue. Same shift as noticed in step 3: the damage may be on the websites, not on Google.
- **Antitrust** was a big second theme in 2024 ("monopoly", "judge", "default") and is fading from the headlines.
- Possible new trend, low confidence (2 to 3 articles each, last weeks of the file only): **Google paying publishers for AI answers**, licensing deals.
- Possible legal risk for a PESTLE: **EU action on AI summaries**. Low confidence as well.

**Not done yet:** grouping the articles into themes automatically (clustering). It needs `scikit-learn`, the number of groups has to be chosen, and the names of the groups would be ours and not the script's. To try after this if the term lists are too scattered.
