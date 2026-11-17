# Running the automated evals

How to run the golden question sets against a
chatbot backend, and how to read what comes out.

The runner is a **client of the public HTTP contract** — it drives
`POST /chat/message` with `stream: false` exactly as the Gradio frontend and a
future .NET proxy do. It needs no backend changes and no Langfuse credentials.

## Step 0 — the eval data

The golden sets and the run records are deployment data, so they live in a
separate private eval-data repo. Ask the maintainers for its URL, and
clone it at `data/eval/`, which is gitignored like the rest of `data/`:

```bash
git clone <eval-data-repo-url> data/eval
```

| Variable | Default | Holds |
| -------- | ------- | ----- |
| `EVAL_DATASETS_DIR` | `data/eval/datasets` | `*-domande.csv`, the legend, `derived/` (rubrics, ID lists) |
| `EVAL_RESULTS_DIR` | `data/eval/results` | One directory per run |

Set them in `.env` only for a checkout somewhere else. Relative paths are taken
from the current directory, so run the tools from the repo root. Without the
checkout, the runner stops with a message saying where to get the data, and
`tests/unit/test_eval_golden_set.py` is skipped. The other eval tests run on the
synthetic set in `tests/fixtures/eval/`.

## Step 1 — a backend with the corpus in it

Nothing below is meaningful without a backend whose `documents` collection
actually contains the manuals the question set asks about (the set's `documento`
column names them). A fresh checkout has an empty `data/`, so pick one of:

| Target | How | Good for |
| ------ | --- | -------- |
| Local, restored corpus | `bash scripts/import_sample_manuals.sh` then `uv run sme-chatbot-api` | Iterating on the runner |
| Local, full DocSvc ingest | `uv run sme-ingest-docsvc` ([procedure](docsvc-ingestion.md)) | A representative `documents` run |
| Deployed instance | `--api-url http://deploy-host:8000` | The baseline that run comparisons use |

The runner calls `GET /health` first and aborts with a clear message if the
requested domain is not served; it records the backend's `version` and `domains`
into every artifact, so a run can always be traced back to the build that
produced it.

> **Note.** The deployed instance answers on **plain HTTP**, not HTTPS. Use
> `http://deploy-host:8000`. (`--insecure` exists for a TLS endpoint reached by
> IP, but it is not what this host needs.)

## Run it

```bash
uv run sme-chatbot-eval \
  --dataset data/eval/datasets/<set>-domande.csv \
  --api-url http://deploy-host:8000 \
  --domain documents \
  --run-name baseline
```

With no `--dataset`, every `$EVAL_DATASETS_DIR/*-domande.csv` is run. The run
directory goes under `$EVAL_RESULTS_DIR` unless `--out` says otherwise.

Expect roughly **30 s per question on average** (median 17 s, p95 about 90 s; the
tool-calling loop runs several rounds). That was measured on the 172-question set
with `qwen3.6:35b` on the deploy host, where a full run took **1 h 26 min**. It is
sequential by default on purpose: the LLM endpoint's rate limiting is
still unresolved.

### Useful flags

| Flag | Why |
| ---- | --- |
| `--only Q01,Q05-Q08` | Iterate on a few questions without a full run. `--only @FILE` reads IDs and ranges from a file (`#` comments), and one list spans all question sets: a set holding none of the IDs is skipped |
| `--dry-run` | Print the item list and the anchors each one will be scored on, and exit |
| `--no-write` | Run and print, but write no artifacts |
| `--repeat N` | Ask each question N times to see how stable the answers are (`Q01#1`, `Q01#2`) |
| `--no-rubric` | Ignore the derived rubric — see *What the rubric does* below |
| `--judge` | Resolve the undetermined items with an LLM judge |
| `--judge-model`, `--judge-temperature`, `--judge-timeout` | Judge model (default `$LLM_MODEL`), temperature (default 0, dropped if the model rejects it) and per-request timeout (default 60 s) |
| `--timeout`, `--max-retries` | Slow multi-hop items; 429s |
| `--page-offset`, `--page-tolerance` | Printed-page → PDF-page mapping (defaults 10 and 2) |

Exit code is `0` for a completed run **whatever the verdicts are**, and `1` only
when the run could not complete (backend unreachable, domain missing, dataset
unreadable). Verdicts deliberately do not set the exit code: thresholds are a
separate decision, and baking one in now would make the first red run look like
a broken tool.

## Which run to use

A full run takes about 1.5 hours, too slow to iterate on, so use the cheapest
tier that can answer the question:

| Tier | Command | Time | Use it for |
| ---- | ------- | ---- | ---------- |
| Retrieval only | `sme-chatbot-eval-retrieval` (below) | seconds | Anything that changes retrieval: fusion, chunking, the embedder |
| Smoke subset | `sme-chatbot-eval --only @data/eval/datasets/derived/smoke-ids.txt` | 13–17 min | A quick behaviour or latency check; every latency experiment so far |
| Full set | `sme-chatbot-eval` | about 1.5 h | Baselines and releases |

The **smoke subset** is 30 of the 172 questions: every one of the 27 categories,
all three question sets, the EN / RO / no-serial items, and three round-cap hitters
(10%, as in the full set). It is listed with the reason for each ID in
`datasets/derived/smoke-ids.txt` in the eval-data repo.
`datasets/derived/cap-hit-ids.txt` holds the 19 questions that hit the round cap,
for diagnosing the tool loop; it is not a benchmark.

A subset run says so: `report.md` states "Subset run" and `summary.json` records
`only` and `dataset_rows`. **Do not compare its percentages with a full run's.**
The subset deliberately over-represents hard categories and the machines outside the main set,
so compare it with the same questions from an earlier subset run (the header of
`smoke-ids.txt` records the baseline numbers for exactly these IDs).

**Warm the backend up first.** The first questions after a restart pay for loading
the model, the embedder and the Chroma store. With a warm OS page cache that is
about half a minute; after a long idle it was close to **6 minutes** for the first
question, and the first question touching each machine's part of the 9 GB store
can take minutes too. Send one question per question set before the run, with a
long timeout so the retry logic does not kick in:

```bash
uv run sme-chatbot-eval --no-write --timeout 900 --only Q01,Q117,Q143
```

## Reproducibility: how much noise to expect

The backend sends no sampling settings by default, so the model server's own
defaults apply. For qwen3.6 and qwen3.8 on Ollama that is temperature 1 with no
seed, and **two runs of the same code on the same questions then disagree**.
Measured on the smoke list:

| Setting | Model | Same verdict | Same rounds | Identical answer text |
| ------- | ----- | ------------ | ----------- | --------------------- |
| Server defaults (two runs) | qwen3.6:35b | 18 of 30 | 12 of 30 | not measured |
| `LLM_TEMPERATURE=0`, `LLM_SEED=42` (two repeats, main-set questions only) | qwen3.6:35b | 14 of 14 | 14 of 14 | 14 of 14 |
| `LLM_TEMPERATURE=0.2`, `LLM_SEED=42`, `LLM_MAX_TOKENS=4096` (two repeats) | qwen3.8:27b | 29 of 30 | 30 of 30 | 27 of 30 |

Notes on the table:

- **Do not use temperature 0 for these models.** The temperature-0 run on
  qwen3.6 stalled on one of the smaller sets: one request generated for 531 s without a
  visible answer. Greedy decoding can make a thinking model repeat itself, and
  nothing bounded the output.
- The rows use different models, so they show what each setting gives, not a clean
  comparison of the settings: qwen3.8 was not measured at the server defaults.
- At `0.2` with a seed, the two repeats of the smoke list differed in mean latency
  by 6% (29.2 s and 27.4 s) and had the same number of round-cap hits (3 each).
  This is **one pair of repeats**, so treat it as a rough floor: a difference
  inside about 10% in mean latency or a single verdict flip is within it, while
  anything smaller than that needs `--repeat`.

For experiments that compare two backends or two settings, start the backend with
the same values for both arms:

```bash
LLM_TEMPERATURE=0.2 LLM_SEED=42 LLM_MAX_TOKENS=4096 uv run sme-chatbot-api
```

`GET /config` reports them and every run record stores that response, so a run
always says what it ran with. `LLM_MAX_TOKENS` bounds one LLM call, reasoning
included; a call it cuts off is counted in the report as "LLM calls cut off by
max_tokens". That row should stay at 0: if it does not, the cap is truncating
answers and is too low.

"Answers forced by the tool-round limit" counts the questions whose turn used
up its tool rounds, so the answer came from the forced tools-less call.
It is counted from the run's own records, so older runs can be
compared too. A high count means the model keeps searching without settling
on an answer.

## Comparing a release with a baseline

A release gate asks one question: did the **code** change the answers? Any
other difference between the two runs makes the answer ambiguous. Sampling
noise is one (see above). Two more are easy to miss on a deployed host, because
they change without a deploy:

| What | Where to read it | Why it drifts |
| ---- | ---------------- | ------------- |
| Settings | `GET /config` (stored in every run record) | `.env` edited on the host |
| Corpus | chunk count of `docsvc` | a newer data archive extracted into `data/` |
| Retrieval mode | whether `lexical_fts.sqlite3` sits next to `chroma.sqlite3` | without it, `search_documents` is dense-only, and the backend logs `Lexical index … not found; search_documents is dense-only` on the first search |

Record all three before the run. Without them, a later run can't tell what it
is being compared with. Count the chunks read-only, straight from SQLite, so
nothing opens the store for writing:

```bash
python3 -c "import sqlite3,sys; c=sqlite3.connect('file:'+sys.argv[1]+'/chroma.sqlite3?mode=ro', uri=True); print(c.execute('select count(*) from embeddings').fetchone()[0])" data/chroma_docsvc_db
ls data/chroma_docsvc_db/lexical_fts.sqlite3
```

Example: the v0.8.0 baseline on the deploy host ran on 817,884
chunks with no lexical index, so retrieval was dense-only. By the time rc.2 was
deployed, the host served 850,727 chunks from a later archive, with a lexical
index, so retrieval was hybrid. An eval of the deployed backend would have mixed
three changes together.

### A side-by-side backend on the baseline's corpus

If the deployment has drifted, don't roll it back. Start a second backend from
the same image on another port, and point it at the baseline's corpus. Keep
that corpus as a copy under `data/` (on the deploy host it is
`data/chroma_docsvc_db.OLD`). From the deployment directory:

```bash
docker compose run -d --no-deps --name chatbot-api-eval -p 8001:8000 \
  -e CHROMA_DOCSVC_DIR=data/chroma_docsvc_db.OLD chatbot-api
curl -s localhost:8001/config      # same settings as the baseline?
```

- `docker compose run` parses `.env` the same way the deployment does, and
  `--no-deps` leaves the running services alone. `docker run --env-file`
  would not strip `.env`'s inline comments.
- The default `LEXICAL_INDEX_PATH` lives inside `CHROMA_DOCSVC_DIR`. A
  baseline corpus with no index therefore gives dense-only retrieval, as the
  baseline had. Check for the log line above after the warm-up.
- Check that both runs use the same `chromadb` version (`uv.lock` at both
  tags). A newer one may migrate the copy in place.
- The container carries the service's labels, so with the `observability`
  profile on, Alloy ships its logs too. They go under `job="chatbot"` with
  `container="chatbot-api-eval"`.

Point the runner at `--api-url http://localhost:8001`. Remove the container
when the run is done: `docker rm -f chatbot-api-eval`.

### Running on the host

A full run takes 1.5 h or more. Run it on the host inside `tmux`, so a dropped
VPN can't kill it. A deployment host has no checkout, so clone the repo next to
the deployment, take the runner from `main`, and run `uv sync`. The runner is
only an HTTP client, so a newer runner is fine against an older backend.

`--judge` reads `LLM_BASE_URL`, `LLM_API_TOKEN` and `LLM_MODEL` from a `.env`
in the checkout (see [The judge](#the-judge---judge)). Copy just those three
lines from the deployment's `.env`, so the token never goes through a terminal
or a chat:

```bash
(umask 077; grep -E '^LLM_(BASE_URL|MODEL|API_TOKEN)=' ../chatbotv2-test/.env \
  | sed -E 's/[[:space:]]+#.*$//' > .env)
```

## Retrieval only, no LLM

To check a retrieval change (hybrid fusion, chunking, embedding model) in
seconds rather than a full run, rank the gold page directly:

```bash
uv run sme-chatbot-eval-retrieval --dataset data/eval/datasets/<set>-domande.csv
```

It runs in-process against the local `CHROMA_DOCSVC_DIR` (and its lexical
index, if built). It sends each question verbatim, scoped to the item's
matricola, and prints the rank of the first result from the gold document
within `Pag. stampata + --page-offset ± --page-tolerance` under `dense`,
`lexical` and `hybrid`, plus recall@1/5/10/20. Items without a gold page
(refusals, scoping) are skipped. `--json PATH` also writes the ranks.

A hit here is not a correct answer: the LLM still has to read and use the
chunk. But a miss here usually means the answer never reaches the model.

## What you get

Four files under `$EVAL_RESULTS_DIR/<date>-<run-name>/`, described in
`results/README.md` in the eval-data repo. Commit them
**there**, not in this repo. They are dated and append-only — never edited after
the fact.

`results.csv` is shaped to paste straight back into `the golden set's .xlsx`:
the source columns in source order, with the workbook's yellow columns filled
and five columns appended (`Risposta ottenuta`, `Fonti citate`, `Latenza (ms)`,
`Round`, `TTFT (ms)`). `TTFT (ms)` is the backend-measured time to the first
visible token and is blank for runs recorded before the backend reported
`usage.timings`; `report.md` and `summary.json` likewise show `—` /
`null` for those runs, and add p50/p95 time to first token, mean setup / LLM /
tool time, and a latency-by-rounds table.

## Trend across runs

`sme-chatbot-eval-trend` turns every run under `$EVAL_RESULTS_DIR` into one
self-contained HTML page: the share of S / P / N / ? / ERROR per run as stacked
columns, p50 and p95 end-to-end latency and time to first token per run, and the
table behind both. Nothing is fetched at run time, so the page can be opened
locally or published as is.

```bash
uv run sme-chatbot-eval-trend --out eval-trend.html \
  --target p50_latency=10 --target p95_latency=30 --target p50_ttft=5   # the latency targets
uv run sme-chatbot-eval-trend --list-scopes                              # which question sets exist
uv run sme-chatbot-eval-trend --scope "all 172 questions" --markdown     # one scope, plus a table for an issue
```

- A run that wrote one directory per question set (`…-<set-a>`,
  `-<set-b>`, …) is one point; its percentiles are recomputed from
  the answers, with the same rule as `summary.json`.
- Each point has a **scope**, the question set it asked: `all N questions`, or
  the name of the `--only` list (`smoke-ids`, `cap-hit-ids`, ...). A dashed line
  marks every change of scope, and the time lines also break where the model
  changes. Compare runs only within one scope and on the same model; the page
  has a scope filter, and `--scope` keeps one on the command line.
- Counts are per answer, so a `--repeat 2` run counts each question twice.
- Runs recorded before the backend timings have no time to first token.

## How to read `report.md`

### Verdicts

The golden set's own vocabulary (`legenda`), plus two values the workbook does
not have:

| Verdict | Meaning |
| ------- | ------- |
| `S` | Correct — every expected value present, nothing forbidden, behavioural expectation met |
| `P` | Partial — some expected values present, or all of them but without a required qualification |
| `N` | Wrong — no expected value present, a forbidden value present, a required refusal that did not happen, or an empty answer |
| `?` | **Undetermined** — no value anchor and no rubric entry, so there is nothing to check mechanically |
| `ERROR` | Transport failure, timeout or HTTP 5xx. **Excluded from every percentage** — a failed request is not a wrong answer |

A summary line reading `S 14 · P 5 · N 3 · ? 14` is honest. One reading "58%
correct" after quietly dropping the undetermined items is not, which is why
percentages in `summary.json` state their denominator (`scored_items` = S+P+N).

### `≈` on pages

`Pag. stampata` is the page **printed on the paper**; `Citation.page` is the
**PDF** page, roughly ten pages later because of front matter. Page matching
therefore applies `--page-offset` with `--page-tolerance` and is always rendered
as approximate. **It never gates a verdict.** It becomes exact only if ingestion
ever stores `page_printed`.

### What `anchor_recall` is, and is not

It is **recall**: the fraction of the answer key's checkable values that appear
in the answer. It is *not* correctness — an answer containing both `47 Nm` and a
spurious `147 Nm` still scores 1.0. The rubric's `forbidden` column is the
counterweight, and the report prints every matched and missing anchor verbatim
so you can overrule the machine without opening the JSON.

Values are matched on word boundaries (`92` is not found inside `192`), with
decimal comma/point, thousands separators and unit spelling normalised
(`0,55 kW` ≡ `0.55 kW`, `9000 bph` ≡ `9 000 pezzi/ora`). A unit only
*disqualifies* a match when the answer states a different one — `8 ore` does not
satisfy a `7 bar` anchor, but a bare `61` still satisfies `61 Nm`.

Chapter and page cross-references in `Risposta attesa` (`11.5.3`, `p. 118`) and
the machine's own matricola are **not** anchors. Without that filter, five of
the 36 questions would be scored against their own cross-references.

### Session protocol — a stated deviation

Every question is sent as its own request with `history: []`. The API is
stateless, so each item is a fresh session by construction: the legend's
"Q18/Q19 and Q03/Q35/Q36 in separate sessions" is satisfied for free, and "run
Q34 last" becomes moot.

This is **harder than the manual protocol** for two items. Q10 ("*E*
l'orientatore tappi della 90001…") and Q19 ("Sulla 90001 la capsula è troppo
avvitata.") are phrased as conversational follow-ups. Both still carry the
matricola so they remain answerable, but a baseline may under-report them. The
report header says so on every run.

## What the rubric does

`datasets/derived/<set>-rubric.csv` (in the eval-data repo)
is a **derived** sidecar (the source CSVs stay untouched) that pins `required`
and `forbidden` patterns per question — the traps the golden set names in its
own `Comportamento atteso` column, such as "a bare `3` is a retrieval error, not
a head count" or "315 Nm is the typical wrong answer for Q07".

It matters for how much of a report is mechanical:

| | Items with nothing checkable (`?`) |
| --- | --- |
| With the rubric | 0 of 36 |
| `--no-rubric` | 10 of 36 |

Because that is a lot of leverage for one derived file, `report.md` states on
its face how many verdicts exist only because of it, and every rubric row must
quote the sentence in the source set it derives from. See
`datasets/derived/README.md` in the eval-data repo for the pattern
syntax and the authoring rule.

## The judge (`--judge`)

Opt-in. Uses the same `LLM_BASE_URL` / `LLM_API_TOKEN` / `LLM_MODEL` as the
backend (or `--judge-model`), and:

- resolves `?` items only — it never overwrites an `S`/`N` that has a concrete
  anchor behind it;
- never sees the deterministic verdict, so it cannot anchor on it;
- never sees the retrieved chunk text (groundedness is a decision for the
  Langfuse eval harness, not this tool);
- is recorded side by side with the deterministic verdict, and every
  disagreement is listed at the top of the report — that list is the human
  review queue.

It runs on the same model family as the chatbot under test, which is a known
weak oracle; `judge_model` is recorded in `results.json` so a later switch shows
up as a change rather than as a silent improvement. Judge runs are not
reproducible, and the report says so.

Before the first question the runner sends the judge one probe request. If the
probe fails (unreachable endpoint, bad credentials, unusable reply) the run
stops with exit code `1` and asks nothing, so a run cannot finish with no judge
verdicts and still look judged. Models that accept only their default
temperature (Azure `gpt-5` returns HTTP 400 for `temperature=0`) are handled on
the probe: the parameter is dropped for the rest of the run, and
`results.json` records `judge_temperature: null` (the server default) next to
`judge_temperature_requested`. `judge_calls` and `judge_errors` record how many
judgements were attempted and how many failed. When any failed, the report says
so under the judge note.

A judge prompt can be refused by the LLM provider's content filter. On Azure
this happened for Q26 and Q75: the golden expected answer about the
lubricant cartridge is flagged as "sexual" at severity "medium", a false
positive. The block is deterministic and does not depend on how the
prompt is framed, so the item is not retried. Its `judge_reason` reads
`judge error: blocked by the LLM provider's content filter (sexual:medium)`.
`judge_content_filtered` in `results.json` counts these blocks, and the
report's judge-errors line counts them separately from other failures. Only a
change to the deployment's content-filter policy makes these items judgeable.

The same filter can hit the chatbot itself, on the question or on the answer.
Either way the item is an `ERROR`, not a scored answer, and its `error` is
`HTTP 500:` followed by the backend's message. For a blocked question (Q169 on
2026-10-21) that is `The request was blocked by the LLM provider's
content filter (…)`; for an answer stopped partway, `The answer was stopped
partway by the LLM provider's content filter (…), so it is incomplete.` The
runner retries other HTTP 500s but not these: the block is deterministic, so
a retry would only repeat the whole chat round.

To judge answers a run has already captured, without asking the backend again,
combine the flags: `--rescore <results.json> --judge`. `--only` limits the
rescore to those questions. That writes a new run
directory in which the stored answers are judged again. Any earlier judge
verdict or judge error is replaced.

## Langfuse (optional)

With `LANGFUSE_ENABLED` set (see [`langfuse-setup.md`](langfuse-setup.md)) the
runner opens one span per question, injects a W3C `traceparent` into the
request, and writes the trace id into the `Trace Langfuse` column.

**Limitation:** until the backend adds a root span on the backend's non-streaming
path, the backend does not adopt that context — each LLM round still becomes its
own root trace. The column carries the *runner's* trace id, which is real and
useful for correlating by timestamp, but is not yet a link to one joined backend
trace. With `LANGFUSE_ENABLED` unset, no Langfuse or OpenTelemetry module is
imported at all.

## Tests

```bash
uv run pytest tests/unit/test_eval_dataset.py tests/unit/test_eval_scoring.py \
              tests/unit/test_eval_report.py tests/backend_api/test_eval_runner.py -v
```

No network, no LLM, no ChromaDB: the end-to-end suite drives the **real**
FastAPI app with `stream_chat` monkeypatched to a scripted fake.

## What this tool does not do

It does not decide whether the chatbot is good enough. It produces a reviewable
record. Thresholds, CI gates and fixing what it measures are separate
pieces of work.
