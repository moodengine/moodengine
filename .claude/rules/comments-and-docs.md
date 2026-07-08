---
paths:
  - "src/**/*.py"
  - "scripts/**/*.py"
---

# Comments & docstrings — self-contained, why-first

Every comment and docstring must be understandable by an external developer who has
**only this repo's source** — nothing else. That drives four hard rules:

1. **Never cite documents the reader can't see.** No `(spec 19)`, `(R&D spec 27)`,
   `Family-5`, internal roadmaps, or references to private planning notes — an external
   dev has none of them. If removing a citation would lose real information, write the
   1–2 sentence idea itself in its place. New code introduces zero such citations
   (a CI check greps for them).
2. **Name-drops complement, never replace, an explanation.** A technique or paper name
   (`Guo+'17`, `SASRec`, `Foote novelty`, `Platt scaling`, `MMR`, `Camelot wheel`) may
   appear only *next to* a plain-language statement of the idea. If the name is the only
   content, the comment is incomplete.
3. **Explain why, not what.** Comments carry constraints, invariants, trade-offs, and
   non-obvious causes ("recenter before softmax because CLAP's modality gap skews all
   cosines to one side") — never a paraphrase of the next line. Delete narration comments
   on sight.
4. **English only** — code, comments, docstrings.

**Docstring bar** (a strength of this codebase — keep it): a public function documents its
contract — array shapes, dtypes, degenerate cases (`n < 2`, empty input, NaN policy), side
effects if any, and the memory envelope when it is O(n²) or worse. A reader must be able to
call the function correctly without opening its body.
