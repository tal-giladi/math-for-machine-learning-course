# Curriculum map

This page shows **what depends on what**. No lesson uses a concept that an earlier lesson has
not already built. Two threads run in parallel and meet at the Transformer.

## The two dependency chains

**The training / calculus chain** — how a network learns:

```text
Algebra & functions
      ↓
Vectors → Matrices → Tensors
      ↓
Composition (forward pass)
      ↓
Derivatives → Partial derivatives → Chain rule
      ↓
Gradients → Jacobians → Matrix calculus
      ↓
Linear layer (Wx+b) → Loss functions
      ↓
Backpropagation
      ↓
Gradient descent → SGD → Momentum → RMSProp → Adam → AdamW → Muon
```

**The architecture chain** — how a GPT is built:

```text
Vectors → Matrices → Linear algebra
      ↓
Tensors & shapes (B,T,C)
      ↓
Probability → Softmax → Cross-entropy
      ↓
Embeddings
      ↓
Q, K, V → Attention → Multi-head attention
      ↓
Normalization → Residuals → MLP → Transformer block
      ↓
GPT-2
```

The two chains join in **Module 12**, where a full GPT-2 training step is read end to end:
the architecture chain produces `logits`, the calculus chain turns the loss into gradients,
and the optimizer chain updates the parameters.

---

## Modules, objectives and brief coverage

Every "Part N" below refers to a section of the original course brief.

### Module 1 — The language of mathematics *(Part 1)*
Rebuild mathematical literacy from numbers up. After it you can read an equation aloud and say
what each symbol does.
- **01** numbers, variables, constants, expressions, equations
- **02** functions, `f(x)`, domain/range, input→output, why ML is "learning a function"
- **03** powers, roots, exponentials `e^x`, logarithms `ln`
- **04** summation `∑`, products `∏`, absolute value, inequalities, fractions, average rate of change
- **05** the notation zoo: `∂`, `∇`, `‖x‖`, subscripts/superscripts, indices, transpose, inverse, element-wise `⊙`

### Module 2 — Vectors *(Part 2)*
- **06** vector as list / point / direction; dimension; indexing; vector vs Python list vs tensor
- **07** addition, scalar multiplication, the dot product
- **08** norm / magnitude, unit vectors, normalization, cosine similarity, projection

### Module 3 — Matrices *(Part 3)*
- **09** rows, columns, shape, indexing, addition, scalar multiplication
- **10** matrix–vector and matrix–matrix multiplication, shapes at every step
- **11** transpose, identity, inverse, rank, linear transformations

### Module 4 — Tensors and shapes *(Part 4)*
- **12** scalar → vector → matrix → 3-D → 4-D → N-D; shape, dimension, axis
- **13** `(B,T,C)` and `(B,T,n_heads,head_dim)`; reshape, view, transpose, permute, broadcasting, slicing

### Module 5 — Functions and composition *(Part 5)*
- **14** one-input / multi-input / vector-valued functions, composition `f(g(x))`, computational graphs, the forward pass as a composition

### Module 6 — Calculus *(Parts 6 & 7)*
- **15** limits, continuity, slope, the derivative as local sensitivity
- **16** power / sum / product / quotient rules and the chain rule
- **17** derivatives of `eˣ`, `ln`, sigmoid, and the softmax building blocks
- **18** partial derivatives, holding variables constant, the gradient vector

### Module 7 — Chain rule & backprop foundations *(Parts 8–11)*
- **19** chain rule through a graph `x→a→b→y`, local derivatives multiplied backward
- **20** gradients: `∂L/∂x` vs `∇ₓL`, gradient wrt a vector / matrix / tensor
- **21** Jacobians: vector-valued derivatives, Jacobian shape, chain rule as matrix product
- **22** matrix calculus: derivative of a scalar wrt vector/matrix, VJP and JVP, `y=Wx+b`, useful identities

### Module 8 — Neural-network mathematics *(Parts 12–14)*
- **23** `z=Wx+b`, `a=activation(z)`, `L=loss(a,t)`; a full 2→2→1 network forward **and** backward by hand, then reproduced exactly in PyTorch — *the milestone lesson*
- **24** MSE, MAE, cross-entropy, NLL, BCE, softmax, log-softmax; the `logits→softmax→CE→loss` pipeline
- **25** backpropagation from first principles; `requires_grad`, `grad_fn`, `.grad`, leaf tensors, graph build/traversal, gradient accumulation, `zero_grad()`

### Module 9 — Optimization *(Parts 15–17)*
- **26** gradient descent `θ ← θ − η∇L(θ)`, learning rate, loss landscape, minima/saddles/overshoot
- **27** SGD, momentum, Nesterov, RMSProp, Adam, AdamW — with the state each stores and manual steps
- **28** Muon: why gradients have matrix structure, orthogonalization, Newton–Schulz, singular values, a full small-matrix example

### Module 10 — Probability & language modeling *(Parts 18–20)*
- **29** probability, random variables, distributions, expectation, variance, covariance, conditional probability, Bayes, entropy, cross-entropy, KL, maximum likelihood, NLL
- **30** `hidden → linear → logits → softmax → distribution`, vocabulary size, temperature, next-token prediction
- **31** embeddings: lookup, one-hot view, matrix-multiplication view, gradients through a lookup, why only used rows get gradients

### Module 11 — Transformers *(Parts 21–23)*
- **32** attention: Q, K, V, scaled dot products, causal mask, softmax, weighted sum, multi-head — a full `T=3, head_dim=2` example
- **33** the Transformer block: LayerNorm → QKV → attention → projection → residual → LayerNorm → MLP → residual; GELU; gradient flow through residuals
- **34** normalization: mean/variance/std, LayerNorm, RMSNorm, worked by hand, and why RMSNorm differs

### Module 12 — Training systems & the finish line *(Parts 24–28)*
- **35** floating point, precision, overflow/underflow, NaN/inf, log-sum-exp, stable softmax, epsilon, exploding/vanishing gradients
- **36** symbolic vs numerical vs automatic differentiation, forward- vs reverse-mode, VJPs, why training uses reverse mode
- **37** the whole GPT-2 training step in equations and shapes: tokens → embeddings → blocks → logits → cross-entropy → backprop → optimizer → updated parameters
- **38** fine-tuning: pretrained init, freezing, catastrophic forgetting, full vs parameter-efficient tuning, LoRA `W' = W + BA`
- **39** a real training loop read line by line: `zero_grad` → forward → `backward` → `step`

---

## Reference sheets
- **[Math cheat sheet](assets/math-cheatsheet.md)** — the most important formulas in one place.
- **[PyTorch cheat sheet](assets/pytorch-cheatsheet.md)** — each math concept mapped to the PyTorch that does it.
- **[Glossary](assets/glossary.md)** — every term, defined.
