# Status — Mathematics for ML course

**Live:** https://tal-giladi.github.io/math-for-machine-learning-course/
**Repo:** https://github.com/tal-giladi/math-for-machine-learning-course

## What's built
- 12 modules, **39 lessons**, covering all 28 Parts of the brief, in dependency order.
- 3 reference sheets: math cheat sheet, PyTorch cheat sheet, glossary.
- Curriculum map with the two dependency chains (calculus/training and architecture).
- docsify static site; math rendered with **KaTeX auto-render** (works in body, display, and
  inside the prereq/callout HTML boxes).
- Every lesson: Prerequisites / You will learn / Why this matters for ML / six-pass
  (intuition → math → numeric example → ML → PyTorch → under the hood) / Next / Check yourself.

## Verification done
- Milestone 2→2→1 network (lesson 23), optimizers + Adam step + Muon orthogonalization
  (lessons 26–28), attention + LayerNorm/RMSNorm (lessons 32/34) — all worked numbers checked
  against PyTorch/numpy and matched.
- Modules 7 and 10 numbers spot-checked by hand.
- **0 broken internal links** across all pages.
- No `$` accidentally left inside code fences (would break rendering).

## Known cosmetic nit (not blocking)
- A couple of display matrices (e.g. the final attention output `O` in lesson 32) render their
  entries on one line instead of a 3×2 grid — values are all correct, just spacing. Fixable by
  adding `\\` row breaks in those `bmatrix` blocks if it bothers you.

## Deliberately NOT built (per brief)
- No auto-graded exercises / answer-checking / progress tracking. Self-checks are inline with
  collapsed answers. No separate `assessments/` quizzes — the brief said keep it static.
