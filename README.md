# Mathematics for Machine Learning and Deep Learning

**12 modules · 39 lessons · worked-by-hand numerical examples · PyTorch on every page · 3 reference sheets**

A comprehensive, static, bottom-up course for a programmer who can already write code but has
**never been taught the mathematics of neural networks**. It starts from what a number and a
variable are, and ends with you able to read the mathematics of a GPT-2 training loop — the
forward pass, `loss.backward()`, AdamW, Muon, LoRA — and know exactly what every symbol,
every tensor shape, and every PyTorch line means.

> The goal: go from *"I can program, but I don't understand the math behind neural networks"*
> to *"I can read the mathematics of GPT-2, understand backpropagation and optimization, and
> know what PyTorch is actually doing on the forward and backward pass."*

---

## How this course is built

Every major concept is taught in the same six passes, in order:

1. **Intuition** — what is happening, in plain English.
2. **Mathematics** — the equation, with *every symbol named*.
3. **Numerical example** — the same thing with actual small numbers, worked by hand.
4. **ML application** — why a neural network needs this.
5. **PyTorch** — the corresponding code.
6. **Under the hood** — what PyTorch (tensors, autograd, the optimizer) is actually doing.

Nothing is introduced before its prerequisites. Every page opens with **Prerequisites**,
**You will learn**, **Why this matters for ML**, and closes with **Next**. Math is rendered
with KaTeX so the notation looks the way it does in the papers.

<div class="callout key">
<p><strong>You do not need to know any mathematics to start.</strong> If you remember algebra,
great — skim Module 1. If you don't, Module 1 rebuilds it from numbers and variables.</p>
</div>

---

## The path

```text
Math language ─▶ Vectors ─▶ Matrices ─▶ Tensors ─▶ Functions
      │
      ▼
  Calculus ─▶ Partial derivatives ─▶ Chain rule ─▶ Gradients ─▶ Jacobians ─▶ Matrix calculus
      │
      ▼
  Linear layer ─▶ Loss functions ─▶ Backpropagation
      │
      ▼
  Gradient descent ─▶ SGD/Momentum/Adam/AdamW ─▶ Muon
      │
      ▼
  Probability ─▶ Softmax & LM math ─▶ Embeddings ─▶ Attention ─▶ Transformer ─▶ Normalization
      │
      ▼
  Numerical stability ─▶ Autograd ─▶ GPT-2 training end-to-end ─▶ Fine-tuning & LoRA ─▶ Read the code
```

See the full dependency graph on the **[Curriculum map](curriculum/course-outline.md)**.

---

## Modules

| # | Module | Covers |
|---|--------|--------|
| 1 | [The language of mathematics](lessons/module-01/lesson-01.md) | numbers, variables, functions, powers, logs, ∑, and the notation zoo |
| 2 | [Vectors](lessons/module-02/lesson-01.md) | lists/points/directions, dot product, norm, cosine, projection |
| 3 | [Matrices](lessons/module-03/lesson-01.md) | shapes, matmul, transpose, identity, inverse, rank, linear maps |
| 4 | [Tensors and shapes](lessons/module-04/lesson-01.md) | ND tensors, `(B,T,C)`, reshape/view/permute, broadcasting |
| 5 | [Functions and composition](lessons/module-05/lesson-01.md) | nested functions, computational graphs, the forward pass |
| 6 | [Calculus](lessons/module-06/lesson-01.md) | limits, the derivative, rules, exp/log/sigmoid/softmax, partials, gradient |
| 7 | [Chain rule & backprop foundations](lessons/module-07/lesson-01.md) | chain rule, gradients, Jacobians, matrix calculus |
| 8 | [Neural-network mathematics](lessons/module-08/lesson-01.md) | `z=Wx+b`, loss functions, backpropagation & autograd |
| 9 | [Optimization](lessons/module-09/lesson-01.md) | gradient descent, SGD→Adam→AdamW, Muon |
| 10 | [Probability & language modeling](lessons/module-10/lesson-01.md) | probability, entropy/KL, softmax LM math, embeddings |
| 11 | [Transformers](lessons/module-11/lesson-01.md) | attention, the Transformer block, LayerNorm & RMSNorm |
| 12 | [Training systems & the finish line](lessons/module-12/lesson-01.md) | numerical stability, autodiff, GPT-2 training, fine-tuning, reading a training loop |

**Reference:** [Math cheat sheet](assets/math-cheatsheet.md) · [PyTorch cheat sheet](assets/pytorch-cheatsheet.md) · [Glossary](assets/glossary.md)

---

## How to read it locally

This is a plain [docsify](https://docsify.js.org) site — no build step. From this folder:

```bash
python -m http.server 8080
```

Then open <http://localhost:8080>. Or just read the Markdown files directly in any editor;
the math is written so it is legible as source too.
