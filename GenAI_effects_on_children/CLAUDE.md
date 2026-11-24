# GenAI and the new generation (final project)

Final project of Strategic Scanning (brief in `../Class 3.md` and `../Class 4.md`). The root `../CLAUDE.md` still applies in full: how to work with me, the venv, the script conventions. This file only adds what is specific to this project. Don't repeat the root rules here; if they need to change, change them there.

## The question

How does generative AI affect the lives of the generation born from 2018 onwards? Monitor every kind of source (academic, policy, news, community), and detect **trends, weak signals and risks** as they appear, with a pipeline that runs on its own every week and reports what is new.

- In 2026 these children are **0 to 8 years old**: early childhood, preschool, early primary. Keep the scope there; teenagers come up in the sources all the time and are noise for this project unless a rule says otherwise.
- **Source bias to keep visible:** these children produce almost no data themselves. Every source is adults (researchers, parents, teachers, companies, regulators) talking about them. Say so in the report; don't present what adults write as what children experience.

## Deliverables (submit before Thursday 26 November 2026)

To the class folder:

1. **One `.ipynb` notebook** with every script, built from the scripts by the notebook builder (never edited by hand). It must run top to bottom with no errors on the reviewer's machine: the build adds an **offline mode** that uses the committed data, so no web access, no API key and no re-embedding are needed. Check it with `jupyter nbconvert --execute` before calling it done.
2. **The scanned data** (`data/`).
3. **Word report** and **slides** explaining the process and the outcome. Slides are technical (exact queries, rules, full tables), see memory.
4. **The running pipeline and its explanation** (extra mark): what runs, when, what it sends back, how to rerun it.
5. **"Sell the value"**: the teacher wants the project sold to him. The report and slides need a section on who would use this weekly brief and what decision it helps them make (parents, schools, edtech and toy companies, regulators), what it costs to run, and what it caught that a one-off search would miss. Claims in that section come from the scan's own output.

## Sources

Every collector writes the **same columns**, so all later steps treat all sources the same way:

`id, source, source_type, weight, date, title, text, url, query`

Anything specific to one source (OpenAlex stores abstracts as a word index, PubMed answers in XML, Google News gives redirect links) is handled inside that source's collector and nowhere else.

| `source_type` | Source | Default weight | Notes |
| --- | --- | --- | --- |
| `peer_reviewed` | OpenAlex API, PubMed E-utilities | 1.0 | free; OpenAlex key read from `.env` (`OPENALEX_API_KEY`), never in the code or `data/` |
| `preprint` | arXiv API (cs.CY, cs.HC) | 0.7 | earliest academic signals |
| `policy` | Google News RSS restricted to institution sites (UNICEF, OECD, WHO, EU, FTC, UK ICO...) | 0.9 | |
| `news` | Google News RSS | 0.5 | same 30-day window method as `../Google Assignment/google_ad_news.py` |
| `community` | Reddit public JSON (parenting and teacher subreddits) | 0.3 | **dropped 22 Nov**: every request got HTTP 403 from this machine; the official API would need my Reddit account. Note the gap in the report (source bias) |
| `search_interest` | Google Trends | none | **optional**, unreliable library; same rule |

The weights are a judgement call (Class 4: academic journals are more credible than news, news more than blogs). They live in `config/source_weights.csv`, not in the code.

Collectors are **incremental**: a weekly run only adds items not already collected (matched by `id`), and every failed request is written to `data/run_log.csv`.

## Rules of this project

All vocabulary lists live in `config/*.csv` so I can edit them without touching code. Each rule is stated in the docstring of the script that applies it.

- **Relevance:** an item is kept if its text has an AI term AND a child/age term (both lists in `config/`). Rejected items go to `data/clean/rejected.csv` with the reason, as in `clean_news.py`.
- **Clinical use (decided 22 Nov):** AI used by clinicians or researchers to diagnose, treat or write medical information is out of scope; AI that children, parents or teachers use is in scope, health topics included. `clean.py` rejects an item whose **title** (not abstract) has a term of `config/clinical_terms.csv`, reason `clinical use: <term>`. "readability" is deliberately not in the list (it removed a parenting-chatbot study). PubMed's query has no `infant*` term, because that word mostly brought neonatal medicine (87 of 152 results).
- **Language (decided 24 Nov):** non-English items are out (they formed their own cluster). `clean.py` rule 4 detects the language of the text with `langdetect` (fixed seed) when it has at least 20 words (`--min-language-words`) and rejects others as `language: <code>`; headline-only items are not checked (detection mislabelled English headlines). First run: 89 rejected, all peer-reviewed.
- **Age band:** tag each item 0-2, 3-5, 6-8, "Gen Alpha", or generic "children" when no age is given. Items kept only through a school word (`school_generic` group of `config/child_terms.csv`: K-12, pupil*, classroom*, plural "schools", added 22 Nov) are tagged "school (age not given)": weaker evidence about 0-8, say so in the report.
- **Categories (known labels, set in advance)**, each mapped to a PESTLE letter: learning and school, language and cognitive development, socio-emotional and mental health, AI companions and toys, privacy and data, safety and harmful content, screen time and play, parenting, regulation and policy, inequality and access.
  - **Excluded phrases (decided 24 Nov):** `config/category_exclusions.csv` lists phrases that do not count for one category; `categorize.py` blanks them before matching and traces them in `excluded_phrases`. First use: "AI literacy", "digital literacy", "health literacy"... are not children's language development ("ai" or "digital" came before "literacy" in 350 of 533 items). Broad terms are checked with `check_terms.py` (neighbour words + fixed-seed sample) before changing a list.
- **Clustering (let the data group itself, to find what the categories miss): Embeddings + BERTopic**, chosen by me.
  - A small sentence model (`all-MiniLM-L6-v2`, about 90 MB, runs on CPU) turns each text into a vector of its meaning; UMAP reduces those vectors; HDBSCAN groups them; each topic is named by its top words.
  - Repeatable: fixed `random_state` for UMAP; embeddings saved to `data/` and reused for texts that have not changed; `--use-saved` reads the saved topics (used by the notebook's offline mode).
  - Options: `--model`, `--min-topic-size` (a judgement call).
  - Outliers (topic -1) are counted and reported, not dropped.
  - A topic whose items mostly have no known category is listed as **"candidate new theme"** (Class 3: a relevant item that fits no known category may be a new trend).
  - Risk: PyTorch is a big install and Python 3.14 is new. Install and test it first; if it fails, fall back to TF-IDF + NMF and tell me.
- **Trends:** counts per month, per category, per source type, weighted by source weight; rising terms versus a baseline period (window lengths are options).
- **Weak signal:** a term or topic that is still rare, is rising versus the baseline, and appears in at least 2 source types. All three thresholds are options.
- **Risk register:** the script measures the evidence for each risk (weighted count, growth, number of source types). **Impact and probability are my judgement**, entered in `config/risk_register.csv`; the script only draws the Impact x Probability matrix from them (Class 1 & 2: risk = impact x probability, then accept / avoid / mitigate / transfer).
- **New since last scan:** each run saves a snapshot in `data/snapshots/YYYY-MM-DD/`; the report compares with the previous snapshot and lists new items, new topics, new weak signals.
- **Analyst note (the only AI step in the pipeline):** `analyst_note.py` runs only if `ANTHROPIC_API_KEY` is set. It sends the measured report to Claude and saves the answer under the heading "AI conjecture, not measured". Its output is never read by any other step. The pipeline works the same without it.

## Pipeline

Planned; update this table as scripts are written (status column).

| Step | Script | Writes | Status |
| --- | --- | --- | --- |
| Collect news | `collect_news.py` | `data/raw/news.csv` | done 22 Nov |
| Collect academic | `collect_academic.py` | `data/raw/academic.csv` | done 22 Nov |
| Collect policy | `collect_policy.py` | `data/raw/policy.csv` | done 22 Nov |
| Compare candidate queries (tool, not in the weekly run) | `check_queries.py` | `data/query_check/per_query.csv`, `per_version.csv`, `samples.csv` | done 22 Nov |
| Clean all sources with one method | `clean.py` | `data/clean/items.csv`, `data/clean/rejected.csv` | first version 22 Nov (word lists to confirm) |
| Tag age band and categories | `categorize.py` | `data/clean/items_tagged.csv`, `data/clean/category_term_counts.csv` | first version 23 Nov (`config/categories.csv` to review) |
| Check how a category term is used (tool, not in the weekly run) | `check_terms.py` | `data/term_check/neighbours.csv`, `samples.csv` | done 24 Nov |
| Cluster | `cluster.py` | `data/topics.csv`, `data/topic_groups.csv`, `data/item_topics.csv`, `data/embeddings.npy`, `data/embeddings_ids.csv` | first version 23 Nov; two levels (`--groups`, default 10, ward) and `--min-topic-size` 20 on 24 Nov |
| Read a group or topic (tool, not in the weekly run) | `inspect_topics.py` | `data/inspect/<group or topic>_<n>.csv` | done 24 Nov |
| Trends and weak signals | `trends.py` | `data/trends_*.csv`, `data/weak_signals.csv` | to do |
| Risk register | `risks.py` | `data/risk_evidence.csv`, risk matrix chart | to do |
| Report and dashboard | `report.py` | `reports/YYYY-MM-DD.md`, `reports/dashboard.html` | to do |
| AI analyst note (optional) | `analyst_note.py` | `reports/YYYY-MM-DD_analyst_note.md` | to do |
| Run everything | `run_scan.py` | `data/run_log.csv`, snapshot | to do |
| Build the notebook | `to_notebooks.py` | `genai_children_all_steps.ipynb` | to do |

Reuse from `../Google Assignment/`: the RSS window method (`google_ad_news.py`), the rejected-file and same-story grouping mechanics (`clean_news.py`, not its Google-specific word lists), the rising/weak-signal counting (`scan_topics.py`), and the notebook builder (`to_notebooks.py`: copy it, point its `PIPELINE` at these scripts, add the offline mode).

## Schedule (GitHub Actions)

- `.github/workflows/genai_scan.yml` at the **repo root** (GitHub only reads workflows there): every week on a cron, plus a manual "Run workflow" button.
- It installs `requirements.txt`, runs `run_scan.py`, commits the new data and report, and opens a GitHub issue "Scan YYYY-MM-DD: N new items, N new weak signals" with the report summary. That issue is the "report sent back"; GitHub emails it to me.
- The dashboard is published with GitHub Pages.
- **Ask me before** creating the GitHub repo, adding secrets, or pushing anything. When we get there, decide whether the repo holds the whole vault or only this folder (my class notes are private).

## Layout

```
GenAI_effects_on_children/
  CLAUDE.md, README.md                  this file; plain-language summary
  Assignment - GenAI children process.md  log of every try, number and decision (same style as the Google one)
  collect_*.py, clean.py, ... run_scan.py one script per step
  config/                               word lists, source weights, risk register (I edit these)
  data/raw/  data/clean/  data/snapshots/YYYY-MM-DD/
  reports/                              weekly reports + dashboard.html
  deliverables/                         Word report, slides
  requirements.txt                      this project's packages
```

## Agents

- **Inside the pipeline: no autonomous agents.** The grade rests on scripts that run and give the same result twice; agents don't, cost money, and the teacher can't rerun them. The analyst note is the one AI call, and it is labelled.
- **For building it: Claude Code subagents** (next step after this file):
  - one subagent per collector, run in parallel, each given the column list above
  - `rules-reviewer` (`.claude/agents/rules-reviewer.md`): checks each new script against the root rules (no conjecture in code, options instead of constants, rejects traced, repeatable)
  - a research subagent for the literature part of the Word report, with citations checked by `check-citations`

## Timeline

- Mon 23 Nov: install and test the BERTopic stack; collectors; `clean.py`; first full collection.
- Tue 24 Nov: `categorize.py`, `cluster.py`, `trends.py`, `risks.py`, `run_scan.py`; GitHub Actions (after I OK the push).
- Wed 25 Nov: report and dashboard, notebook (built and executed), Word report, slides, log note.
- Cut first if late: Google Trends (Reddit already dropped), then the analyst note, then GitHub Pages (the dashboard file is still committed).
