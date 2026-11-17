# 0001. Backend owns all chat/RAG logic; frontends are thin HTTP/SSE clients

- **Status:** Accepted (implemented)
- **Date:** 2026-07-02 — backfilled 2026-07-07 from
  [`docs/specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md`](../specs/2026-07-02-backend-split-and-docsvc-ingestion-plan.md) §1

## Context

The PoC started as Gradio launchers (`chatbot_cdo.py`, `chatbot_manuals.py`) that
imported the RAG/chat logic directly. The production UI will be a React component in
the company's web portal, reached through a .NET portal proxy — it cannot import
Python code. Keeping chat logic inside a Gradio process meant every frontend would
need its own copy of the loop, tools, and auth.

## Decision

A FastAPI backend (`src/sme_chatbot/api`) owns all RAG/chat logic behind one
streaming contract: `POST /chat/message` (SSE) + `GET /health`. A single `domain`
field (`"sample_cdo" | "sample_manuals" | "documents"`) selects the system prompt and
toolset **server-side** (`core/domains.py` registry). The Gradio app
(`frontends/gradio_app.py`) is a thin HTTP/SSE client with no `sme_chatbot` imports —
it dogfoods the exact contract the future .NET portal proxy will consume. The core
loop (`core/chat.py`) is framework-agnostic and yields typed `ChatEvent`s.

## Consequences

- New frontends (React via .NET proxy, CLI, tests) are clients of one stable contract;
  the API shapes in `api/schemas.py` are frozen by the spec.
- Adding a chat domain is a registry entry + tools, not a new launcher script.
- Auth becomes a backend concern (`AUTH_MODE=dev|jwt`), pluggable per deployment.
- Cost: every UI interaction crosses an HTTP hop, and the SSE event contract must be
  versioned carefully — Gradio can no longer reach into Python internals.
