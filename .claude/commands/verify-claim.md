---
description: Verify a load-bearing claim by re-deriving it through an independent route, not by recognizing it as plausible (operating manual §4)
argument-hint: [the claim to verify; omit to verify the load-bearing claim of the current answer]
---

# Verify by re-deriving, not by recognizing

Apply §4 of `docs/operating-manual.md` ("Verify by re-deriving, not by recognizing") to: $ARGUMENTS
If no claim was given, first identify the single claim the current answer or plan rests on, state it, then verify that.

Procedure:

1. Ask: *if this claim were false, would the process that produced it have noticed?* If the honest answer is no, it is unverified regardless of how confident it feels.
2. Re-derive through a **different channel** than the one that produced it:
   - Recalled API/library behavior → read the installed source (`uv run python -c ...`, grep the venv) or run it.
   - A computed number → recompute a different way (e.g. for `data/cdo.db`, cross-check an aggregate with a second SQL grouping).
   - A summarized document → reopen it and find the exact sentence the summary rests on.
   - Schema or column assumptions → check `db/schema.sql` or `PRAGMA table_info`, never memory. Remember `"order"` is quoted and Italian column names (`matricola`, `cdo_esportata`, `codice_campione`) are exact.
3. Execution beats inspection: one run with a probe input outweighs ten minutes of reading what code "clearly does".
4. Probe the boundary, not the center — empty input, largest value, first/last element, NULLs in optional CDO columns. One edge probe out-verifies five typical-case confirmations.
5. If independent verification is impossible here (no access, no data), the claim does not silently pass — mark it **unverified** in the output, inline, at the sentence where it is used.

Output: the claim, the independent route used, what it showed, and the verdict — confirmed, refuted, or unverified-and-labeled.
