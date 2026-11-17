---
description: Map where the real risk lives in the current task and allocate effort there (operating manual §3)
argument-hint: [optional: the task or plan to triage]
---

# Find where the risk actually lives

Apply §3 of `docs/operating-manual.md` ("Find where the risk actually lives") to the current task$ARGUMENTS.

Procedure:

1. List the distinct parts of the task (steps of a plan, claims of an answer, files of a change).
2. Score each part on three axes:
   - **Likelihood of being wrong** — novelty, ambiguity, distance from a source you can check.
   - **Cost if wrong** — does the conclusion rest on it? does an action get taken on it?
   - **Silence of failure** — will anything downstream catch the error, or does it just ship?
3. The product of the three is the effort map. Name the one or two parts that score highest — that is where re-derivation and double-checking go. Explicitly mark the parts that can run at reflex speed.
4. Distrust the parts that felt easiest: recalled version numbers, "standard" API signatures, arithmetic done in passing, dates. Fluency is not evidence of correctness.
5. If the task rests on one load-bearing claim (a number, a name, one alleged behavior), that claim is the risk center regardless of how small it looks.

Output: a short table or list — part, risk score rationale, effort decision — followed by one sentence naming the risk center. Then proceed with effort allocated accordingly. Do not review everything with equal care; uniform diligence is the failure this command exists to prevent.
