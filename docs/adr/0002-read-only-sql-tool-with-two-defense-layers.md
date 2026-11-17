# 0002. LLM SQL tool is read-only, enforced by two independent layers

- **Status:** Accepted (implemented)
- **Date:** predates 2026-07-02 (carried over from the original PoC tools) —
  backfilled 2026-07-07 from `src/sme_chatbot/tools/cdo_sql.py` and the coding rules
  in `AGENTS.md`

## Context

The `sample_cdo` domain lets the LLM write SQL against `data/cdo.db`
(`query_cdo_db` tool). LLM-generated SQL is untrusted input: a prompt-injected or
hallucinated statement must not be able to modify the database, and no single guard
should be the only thing standing in the way.

## Decision

Two independent layers, both mandatory:

1. **Statement sanitization** — `_is_safe()` accepts only `SELECT`/`WITH` statements
   and rejects write/DDL keywords before execution.
2. **Read-only connection** — SQLite is opened with URI mode `file:...?mode=ro`, so
   even a statement that slips past layer 1 cannot write.

Results are capped (200 rows) to bound context size. Neither layer may be weakened or
bypassed, even if the other seems sufficient.

## Consequences

- A bug in either layer is not exploitable on its own (defense in depth).
- The tool can never serve write use cases; any future write path must be a separate,
  deliberately designed tool — not a relaxation of this one.
- Legitimate read-only constructs that the regex is too blunt for (e.g. CTEs with
  unusual formatting) may need sanitizer fixes rather than workarounds.
