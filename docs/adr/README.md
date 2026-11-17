# Architecture Decision Records

Short, numbered records of architecture decisions that are **still in force**. One
decision per file, named `NNNN-short-slug.md`. A decision that is reversed is not
deleted — its status changes to *Superseded by NNNN* so the reasoning trail survives.

Format (keep it under a page):

```markdown
# NNNN. Title (imperative: "Split backend from frontends")

- **Status:** Accepted | Superseded by NNNN
- **Date:** YYYY-MM-DD (when decided, not when written down)

## Context
What situation forced a decision.

## Decision
What was decided, stated as fact.

## Consequences
What this makes easier, what it makes harder, what it commits us to.
```

The first records here were **backfilled on 2026-07-07** from decisions already made
and implemented (their sources are cited inline). New decisions should get an ADR at
the time they are made — a table row in a spec marked "do not re-litigate" is a good
sign it deserves one.

## Index

| ADR | Decision |
| ----- | ---------- |
| [0001](0001-backend-owns-chat-logic-frontends-are-thin-clients.md) | Backend owns all chat/RAG logic; frontends are thin HTTP/SSE clients |
| [0002](0002-read-only-sql-tool-with-two-defense-layers.md) | LLM SQL tool is read-only, enforced by two independent layers |
| [0003](0003-access-filtering-via-business-key-metadata-at-query-time.md) | Document access is filtered at query time via business-key chunk metadata |
