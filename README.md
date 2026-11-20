# Study in Sweden Advisor

An LLM agent that helps international students choose **what** to study and **where** to study in Sweden.
It recommends real English-taught master's programmes and compares the cities they are taught in.

1RT730 Large Language Models and Societal Consequences of AI, Uppsala University, HT2026.

## Data

Programme data comes from Skolverket's **Susa-navet** open API (EMIL 3), which collects higher-education
offerings daily from UHR (the agency behind universityadmissions.se). Licence: CC0 1.0.
No website scraping is used.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: download and inspect the data

```bash
python src/fetch_susa.py --max-pages 1     # quick test
python src/fetch_susa.py                   # full download into data/raw/
python src/inspect_raw.py > data/inspect_report.txt
```

## Step 2: build the programme dataset

```bash
python src/build_programs.py > data/build_report.txt
```

Keeps English-taught master's programmes open to international applicants with an upcoming start,
and writes one record per offering to `data/programs.json`.

## City knowledge base

`knowledge/cities.json` holds hand-curated profiles of the 12 cities with the most programmes
(population from SCB, living costs from each university's own page, national figures from
Migrationsverket and Study in Sweden). Every figure has a source URL.

## Step 3: search index and agent

```bash
cp .env.example .env                 # then paste your Gemini API key into .env
python src/build_index.py            # embeds all programmes (downloads the model once)
python src/chat_cli.py               # chat with the agent in the terminal
python src/chat_cli.py --baseline    # same model without tools, for comparison
```

The agent (Gemini, function calling) has six tools: `search_programs`, `get_program_details`,
`get_city_info`, `compare_cities`, `list_cities`, `get_general_info`.

## Step 4: demo UI

```bash
python src/app.py          # http://127.0.0.1:7860
```

Chat on the left; on the right, which tools the agent called, latency, tokens and the sources used.

## Step 5: evaluation

```bash
python src/evaluate.py --retrieval-only   # search quality, no API calls
python src/evaluate.py                    # agent (RAG + tools) vs baseline (same model, no tools)
```

15 questions in `eval/questions.json` (10 retrieval, 2 city, 1 fact, 2 adversarial).
Results are saved in `eval/results/<timestamp>/` (summary.json, answers.md).

## Project structure

```
src/fetch_susa.py     download Susa-navet data (infos, events, providers)
src/inspect_raw.py    field distributions and samples, used to choose filters
src/build_programs.py filter + clean into data/programs.json
src/susa_utils.py     helpers for EMIL 3 JSON
src/retriever.py      multilingual embeddings + filtered semantic search
src/build_index.py    builds data/index/
src/tools.py          the agent's tools and their JSON schemas
src/agent.py          Gemini function-calling loop (+ baseline mode)
src/chat_cli.py       terminal chat
src/app.py            Gradio demo UI
src/evaluate.py       evaluation: retrieval hit@k, grounding of links and fees, latency
eval/questions.json   test questions with expected criteria
knowledge/cities.json curated city profiles
data/                 raw data and reports (not committed)
```
