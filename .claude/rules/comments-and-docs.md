---
paths:
  - "src/**/*.py"
  - "scripts/**/*.py"
  - "examples/**/*.py"
---

# Comments & docstrings — self-contained, why-first

Every comment and docstring must be understandable by an external developer who has **only this repo's source**. That drives four hard rules:

1. **Never cite internal documents.** No `(spec 19)`, `(R&D spec 27)`, `Family-5`, `plans/…`, roadmap references — external devs don't have them. When removing a citation would lose information, write the 1–2 sentence idea itself in its place. New code must not introduce any.
2. **Name-drops are complements, never explanations.** An academic or technique name (`Guo+'17`, `SASRec`, `Foote novelty`, `Platt scaling`, `MMR`, `Camelot wheel`) may appear only *next to* a plain-language explanation of the idea. If the name is the only content, the comment is incomplete.
3. **Explain why, not what.** Comments state constraints, invariants, trade-offs, and non-obvious causes ("recenter before softmax because CLAP's modality gap skews all cosines to one side") — never paraphrase the next line of code. Delete narration comments on sight.
4. **English only** — code, comments, docstrings. (French is fine in `plans/` and conversation.)

Docstring bar (keep it — it's a strength of this codebase): public functions document the contract — array shapes, dtypes, degenerate cases (`n < 2`, empty input, NaN policy), and side effects if any. A reader must be able to call the function correctly without opening its body.
