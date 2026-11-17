---
description: Break a hard problem into independently checkable pieces — split by oracle, not by topic (operating manual §2)
argument-hint: [the problem to decompose]
---

# Break the problem along verification lines

Apply §2 of `docs/operating-manual.md` ("Break the problem along verification lines") to: $ARGUMENTS

Procedure:

1. Do not split by topic or hypothesis. Split by **oracle**: for each candidate piece, name how it can be verified independently of the other pieces being right — a test that can run, a source that can be read, a query that can be executed, a computation that can be redone. A piece with no oracle is not a piece yet; cut differently.
2. Make dependencies explicit and one-directional. If piece B is only correct when piece A is, order A first and say so. Reject circular pieces ("the parser works because the tests pass; the tests are right because the output looked correct") — that is one piece wearing two costumes.
3. Extract the assumption layer: collect everything being taken as given into one explicit list before working the pieces. The list itself is a separate, inspectable piece.
4. Prefer cuts where a wrong sub-answer fails loudly — where it will contradict something else you'll compute rather than silently blend in.
5. Identify which single piece, if checked first, eliminates the most hypothesis space. Start there.

Output: the assumption list, then the pieces in verification order, each with its named oracle, and a note on which check runs first and why. Then execute the checks — do not stop at the decomposition unless the user asked only for a plan.
