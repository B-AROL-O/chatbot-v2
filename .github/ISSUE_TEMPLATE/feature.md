---
name: Feature
about: New functionality, or a change to how something that works today should behave
title: ""
labels: []
type: Feature
---

<!--
A developer must be able to build this, and a tester verify it, from this issue alone.
Describe the need before the solution. Every section is filled; write "None" where it truly does not apply.
-->

## Summary

<!-- One or two sentences: what will exist that does not exist today, and for whom. -->

## Context

<!-- Where this lives today (page, screen, flow, role) and what led to the request. -->

## Problem

<!-- What the user cannot do, or does badly, today. Concrete situations, not abstractions. -->

## Ideal solution

<!-- The behavior as the user experiences it, step by step: what they see, what they click (exact labels), what happens. -->

## Scope

| | |
| --- | --- |
| **Users** | <!-- who sees it, who can use it, who must not --> |
| **Variants** | <!-- brands, customers, plants or tenants it applies to, and anything that differs between them; "None" if the product has no variants --> |
| **Areas** | <!-- e.g. frontend, backend, database, integrations, docs --> |

**Out of scope:** <!-- what this issue deliberately does not cover -->

## Business rules

<!-- Numbered so tests and review can reference them (BR-1, BR-2…). Permissions, validations, limits, defaults, ordering, what is saved, notifications sent. -->

1. **BR-1** —

## Edge cases

<!-- Each with its required behavior: empty state, no permission, concurrent edits, very long or special-character input, the record deleted meanwhile, network or server error, another variant, small screens. -->

| Case | Required behavior |
| --- | --- |
| | |

## Implementation plan

<!-- Grounded in the current code, area by area, with file paths. Write "No change" for an area this does not touch. A plan, not a commitment: the implementer may refine it. -->

| Area | Change |
| --- | --- |
| Data / storage | |
| Permissions | |
| Backend / API | |
| Frontend / UI | |
| Copy / translations | |
| Docs / user guide | |

## Acceptance tests

### Automated

<!-- Which tests are added and what each proves, tied to the business rules. -->

- [ ]

### Manual

<!-- Scripted checks: account and role, environment, steps, expected outcome. Cover the main path, at least one denied user, and the edge cases above. -->

| # | Account / role | Environment | Steps | Expected |
| --- | --- | --- | --- | --- |
| 1 | | | | |

## Open points

<!-- Decisions not yet taken, each with whoever has to take it. "None" if everything is decided. -->
