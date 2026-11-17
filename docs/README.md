# Documentation

Documentation for **sme-chatbot** — a chatbot PoC (FastAPI backend + thin
Gradio client) over CDO records, machine manuals, and the DocumentService
library. Start with the repo [`README.md`](../README.md) for setup and
[`AGENTS.md`](../AGENTS.md) for the codebase map.

## Layout

| Folder | What lives here | Maintenance rule |
| ------ | ---------------- | ----------------- |
| [`specs/`](specs/) | Specs and implementation plans whose contracts the code cites (docstrings in `src/`, `tests/` reference these paths) | **Paths are stable — do not move or rename files.** Update only via the supersession-note pattern below |
| [`adr/`](adr/) | Architecture Decision Records — decisions still in force, one per file | Add a record when a decision is made; supersede, never delete |
| [`procedures/`](procedures/) | Up-to-date how-tos you can follow today | Keep consistent with the codebase — fix these in the same PR as the change that invalidates them |
| [`diagrams/`](diagrams/) | Shared diagram sources (Mermaid) | Referenced from docs in all folders |

The eval golden sets and run records, once `eval/` here and `tests/eval-results/`,
moved to a separate private eval-data repo;
[`procedures/running-evals.md`](procedures/running-evals.md) says how to check it out.

Top-level exception: [`operating-manual.md`](operating-manual.md) stays at the docs
root — the `.claude/commands/` slash commands and `AGENTS.md` reference it by that
path.

## Conventions

- **Dated docs** are named `YYYY-MM-DD-short-slug.md` and are snapshots of that date;
  undated docs (procedures, ADRs) are living documents.
- **Supersession notes:** when code moves on from a dated doc, don't rewrite it — add
  one italic line at the top: *(X has been superseded by Y — see Z)*.
- **New document?** Procedure → `procedures/`. Proposal or analysis → a GitHub
  issue. Decision → `adr/`. Contract the code will cite → `specs/`. Finished with it →
  delete it.

## Most relevant documents

**How the system works today**

- [`specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md`](specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md)
  — backend/frontend split + DocumentService ingestion (implemented; §4 contracts are
  frozen and cited from code)
- [`adr/`](adr/README.md) — the standing architecture decisions in one page each

**Doing things**

- [`procedures/api-reference.md`](procedures/api-reference.md) — practical
  `GET /health` / `GET /config` / `POST /chat/message` reference (SSE event
  table, auth modes, how to regenerate from `/openapi.json`)
- [`procedures/docker-deployment.md`](procedures/docker-deployment.md) — run the
  chatbot stack with Docker Compose
- [`procedures/docsvc-ingestion.md`](procedures/docsvc-ingestion.md) — run the
  DocumentService ingestion pipeline, incl. `--backfill-tags`
- [`procedures/release-chatbot.md`](procedures/release-chatbot.md) — cut a chatbot
  release (tag → image + GitHub Release) and pull the published image
- [`procedures/viewing-sqlite-files.md`](procedures/viewing-sqlite-files.md)
  — tools for inspecting the project's SQLite files, incl. Chroma-specific workarounds
- [`procedures/langfuse-setup.md`](procedures/langfuse-setup.md)
  — obtain Langfuse credentials and enable LLM call tracing in the backend
- [`procedures/running-evals.md`](procedures/running-evals.md)
  — check out the private eval repo, run its golden question sets against a
  backend with `sme-chatbot-eval`, and read the `S`/`P`/`N`/`?` report it writes
  to `$EVAL_RESULTS_DIR`
- [`procedures/generate-pdf-from-markdown.md`](procedures/generate-pdf-from-markdown.md)
  — render any Markdown doc to an A4 PDF with `scripts/publish-md-pdf.sh`
- [`operating-manual.md`](operating-manual.md) — the working method for this project
  (backs the `/risk-map`, `/decompose`, `/verify-claim`, `/attack`, `/self-test`
  commands)

<!-- EOF -->
