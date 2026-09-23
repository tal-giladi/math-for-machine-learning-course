# 28 · Muon

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-09/lesson-01">26 · Gradient descent</a> and <a href="#/lessons/module-09/lesson-02">27 · SGD, momentum, RMSProp, Adam, AdamW</a>; matrix multiplication and the transpose from <a href="#/lessons/module-03/lesson-02">10 · Matrix multiplication</a>; and especially rank and transpose from <a href="#/lessons/module-03/lesson-03">11 · Transpose, identity, inverse, rank</a>. Gradients with respect to matrices from <a href="#/lessons/module-08/lesson-01">backpropagation</a>.</p>
<p><strong>You will learn:</strong> why a gradient with respect to a weight <em>matrix</em> has geometric structure that ordinary optimizers ignore, what the singular value decomposition (SVD) says about that structure, what it means to <strong>orthogonalize</strong> a gradient (replace it by the nearest rotation, $U V^\top$), the Newton–Schulz iteration that computes this with only matrix multiplications, and a complete calculator-sized example of a Muon update from forward pass to new weights.</p>
<p><strong>Why this matters for ML:</strong> Muon is a recent optimizer that trains the hidden weight matrices of Transformers faster than AdamW by treating each gradient as a geometric map rather than a flat bag of numbers. Understanding it means understanding what the <em>shape</em> of an update is, and why equalizing a matrix's singular values can help.</p>
</div>

## 1. What ordinary gradient descent does to a weight matrix

A linear layer holds a weight **matrix** $\mathbf{W}$: it maps an input vector to an output vector by $\mathbf{y} = \mathbf{W}\mathbf{x}$. During backpropagation we get a gradient $\mathbf{G} = \partial L / \partial \mathbf{W}$, which is *also* a matrix, the same shape as $\mathbf{W}$: entry $G_{ij}$ is $\partial L / \partial W_{ij}$, how the loss responds to nudging that one weight.

Every optimizer from the last two lessons then does the update **element by element**. Plain gradient descent:

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta\,\mathbf{G}.
$$

Adam is the same idea with each entry independently smoothed and scaled. In every case the matrix is treated as a *flat bag of numbers* — $2\times 3 = 6$ unrelated scalars that happen to be written in a grid. Each number is updated on its own; the grid layout is ignored. That is the assumption Muon questions.

## 2. Why the gradient has matrix *structure*

Here is the thing the flat-bag view throws away. $\mathbf{W}$ is not a random pile of numbers; it is a **linear map** from one space to another (from the layer's input space to its output space — see [lesson 10](lessons/module-03/lesson-02.md) for matrix–vector multiplication as a transformation). A map has geometry: it stretches some directions, shrinks others, and rotates. The gradient $\mathbf{G}$ is a matrix of the same shape, so it too describes a map, and its rows, columns, and — most importantly — its *singular directions* carry geometric meaning about how the loss wants the layer to change.

### A light recap of the SVD

Any matrix $\mathbf{G}$ can be broken into three pieces — the **singular value decomposition**:

$$
\mathbf{G} = \mathbf{U}\,\boldsymbol{\Sigma}\,\mathbf{V}^\top.
$$

- $\boldsymbol{\Sigma}$ (Sigma) — a diagonal matrix of **singular values** $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. These are non-negative **stretch factors**: how much the map scales along each of its principal directions.
- $\mathbf{U}$ and $\mathbf{V}$ — **orthonormal** matrices (their columns are perpendicular unit vectors). Geometrically they are pure **rotations** (or reflections): they change orientation but never stretch. $\mathbf{V}^\top$ rotates the input, $\boldsymbol{\Sigma}$ stretches along axes, $\mathbf{U}$ rotates the result.

So the SVD says *every* matrix is "rotate, then stretch each axis by its $\sigma$, then rotate." The rank of $\mathbf{G}$ (from [lesson 11](lessons/module-03/lesson-03.md)) is just the number of nonzero singular values. The singular values are exactly the geometric content the element-wise view cannot see.

## 3. Why element-wise updates may be suboptimal

Suppose a gradient has singular values $\sigma_1 = 10$ and $\sigma_2 = 0.1$. That means the loss is pushing the weight matrix to change enormously along one direction and barely at all along another — the update is **skewed**, dominated by a single direction. Element-wise optimizers (GD, and even Adam, which scales each *entry* but not each *singular direction*) will faithfully take a huge step along $\sigma_1$'s direction and a negligible one along $\sigma_2$'s.

Is that what we want? Often not. A very lopsided update pours almost all of the step into one direction of the weight space and lets the others stagnate. We might prefer an update that treats **all directions on an equal footing** — that moves the matrix meaningfully along *every* direction the loss cares about, instead of letting the largest singular value hog the step. That is the geometric wish Muon acts on.

<div class="callout key"><p>An element-wise update ignores that a weight gradient is a geometric map with singular directions. When one singular value dominates, the update's "shape" is skewed toward a single direction. Muon reshapes the update so all directions count equally.</p></div>

## 4. What Muon changes: orthogonalize the gradient

Muon's move is to replace the raw gradient $\mathbf{G}$ by its **orthogonalization** — the nearest orthogonal (or, for non-square matrices, semi-orthogonal) matrix. Using the SVD, that nearest matrix has a beautifully simple form: keep the rotations, throw away the stretches.

$$
\mathbf{O} = \mathbf{U}\,\mathbf{V}^\top \qquad(\text{drop } \boldsymbol{\Sigma}).
$$

Compare with $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$: we have replaced $\boldsymbol{\Sigma} = \mathrm{diag}(\sigma_1, \sigma_2, \dots)$ by the identity $\mathrm{diag}(1, 1, \dots)$. Every singular value is forced to $1$. The update direction keeps the *orientation* information in $\mathbf{U}$ and $\mathbf{V}$ (which directions the loss wants to move) but discards the lopsided *magnitudes* in $\boldsymbol{\Sigma}$ (how unevenly). The result is an update that pushes on all singular directions equally — exactly the "equal footing" we wanted. Muon then steps:

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta\,\mathbf{O}, \qquad \mathbf{O} = \mathbf{U}\mathbf{V}^\top.
$$

In words: take the gradient, extract its pure-rotation part, and step by that instead of by the raw gradient.

## 5. Newton–Schulz: orthogonalizing without an SVD

Computing an actual SVD of every weight matrix every step would be far too slow. Muon uses a cheaper trick that needs **only matrix multiplications** (which GPUs love): the **Newton–Schulz iteration**. It approximately turns $\mathbf{G}$ into $\mathbf{U}\mathbf{V}^\top$ without ever finding $\mathbf{U}$, $\boldsymbol{\Sigma}$, or $\mathbf{V}$ explicitly.

First **normalize** so all singular values are $\le 1$: divide by the Frobenius norm (the square root of the sum of all squared entries),

$$
\mathbf{X}_0 = \frac{\mathbf{G}}{\|\mathbf{G}\|_F}.
$$

Then iterate, about five times, the polynomial map

$$
\mathbf{X} \leftarrow a\,\mathbf{X} + b\,(\mathbf{X}\mathbf{X}^\top)\mathbf{X} + c\,(\mathbf{X}\mathbf{X}^\top)^2\mathbf{X},
$$

with the standard coefficients $a \approx 3.4445$, $b \approx -4.7750$, $c \approx 2.0315$.

- $\mathbf{X}\mathbf{X}^\top$ — a matrix product; the whole update is just matmuls, no decomposition.
- $a, b, c$ — fixed constants chosen so the map drives the singular values toward $1$ quickly.

Why it works, in one sentence: this update acts on $\mathbf{X}$'s singular values through the odd polynomial $p(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$ (the rotations $\mathbf{U}, \mathbf{V}$ pass through untouched), and the coefficients are tuned so that repeatedly applying $p$ pushes any $\sigma \in (0, 1]$ toward $1$. After $\sim 5$ iterations the singular values sit in a tight band around $1$, so $\mathbf{X} \approx \mathbf{U}\mathbf{V}^\top$ — the orthogonalized gradient — computed entirely with multiply-adds.

<div class="callout warn"><p>One Newton–Schulz iteration does <em>not</em> converge. A single step only nudges the singular values; it takes about five iterations to bring them close to $1$. Do not expect the orthogonal target after one pass.</p></div>

## 6. Singular values ↔ updates: the relationship

Collecting the idea into one statement:

- The **raw** gradient applies stretch $\sigma_i$ along its $i$-th singular direction — uneven, possibly dominated by one large $\sigma$.
- The **orthogonalized** gradient applies stretch $1$ along *every* singular direction — even, all directions equal.
- Newton–Schulz is the practical machine that turns the first into the second using only matmuls, by squashing the spread of singular values down to $\approx 1$.

So "Muon" = "step along $\mathbf{U}\mathbf{V}^\top$ instead of $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$," and Newton–Schulz is how it gets $\mathbf{U}\mathbf{V}^\top$ cheaply.

## 7. A complete calculator-sized example

Let us watch the whole pipeline on a tiny linear layer with a $2\times 2$ weight, using clean numbers you can check by hand.

### 7.1 Setup and forward pass

Weight starts as the identity, $\mathbf{W} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$. We have two training inputs and targets:

$$
\mathbf{x}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix},\ \mathbf{t}_1 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}; \qquad \mathbf{x}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix},\ \mathbf{t}_2 = \begin{bmatrix} -3 \\ 1 \end{bmatrix}.
$$

Forward pass $\mathbf{y} = \mathbf{W}\mathbf{x}$ (with $\mathbf{W}$ the identity, output equals input):

$$
\mathbf{y}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \qquad \mathbf{y}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}.
$$

### 7.2 Loss and gradient

Use the sum of squared errors, $L = \tfrac{1}{2}\big(\|\mathbf{y}_1 - \mathbf{t}_1\|^2 + \|\mathbf{y}_2 - \mathbf{t}_2\|^2\big)$. The residuals are

$$
\mathbf{r}_1 = \mathbf{y}_1 - \mathbf{t}_1 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \qquad \mathbf{r}_2 = \mathbf{y}_2 - \mathbf{t}_2 = \begin{bmatrix} 3 \\ 0 \end{bmatrix},
$$

so $L = \tfrac{1}{2}(1 + 9) = 5$. For this loss the gradient with respect to $\mathbf{W}$ is the sum of outer products $\mathbf{r}_i \mathbf{x}_i^\top$ (this is the matrix-calculus result for a linear layer; each residual "votes" through its input):

$$
\mathbf{G} = \mathbf{r}_1\mathbf{x}_1^\top + \mathbf{r}_2\mathbf{x}_2^\top
= \begin{bmatrix} 0 \\ 1 \end{bmatrix}\begin{bmatrix} 1 & 0 \end{bmatrix} + \begin{bmatrix} 3 \\ 0 \end{bmatrix}\begin{bmatrix} 0 & 1 \end{bmatrix}
= \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix} + \begin{bmatrix} 0 & 3 \\ 0 & 0 \end{bmatrix}
= \begin{bmatrix} 0 & 3 \\ 1 & 0 \end{bmatrix}.
$$

So $\mathbf{G} = \begin{bmatrix} 0 & 3 \\ 1 & 0 \end{bmatrix}$.

### 7.3 The SVD of this gradient (by hand)

$\mathbf{G}$ is "anti-diagonal." Its singular values come from $\mathbf{G}\mathbf{G}^\top = \begin{bmatrix} 9 & 0 \\ 0 & 1 \end{bmatrix}$, whose eigenvalues $9$ and $1$ give $\sigma_1 = 3$ and $\sigma_2 = 1$. Working through the singular vectors gives

$$
\mathbf{U} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},\quad \boldsymbol{\Sigma} = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix},\quad \mathbf{V}^\top = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}.
$$

Check: $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top = \boldsymbol{\Sigma}\mathbf{V}^\top = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 3 \\ 1 & 0 \end{bmatrix} = \mathbf{G}$. Good.

### 7.4 Orthogonalize

Drop $\boldsymbol{\Sigma}$ (set both singular values to $1$):

$$
\mathbf{O} = \mathbf{U}\mathbf{V}^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}.
$$

The raw gradient stretched its first singular direction three times as hard as the second ($\sigma_1 = 3$ vs $\sigma_2 = 1$); the orthogonalized $\mathbf{O}$ is a clean rotation/permutation that treats both directions equally.

### 7.5 The Muon update

Step with learning rate $\eta = 0.1$:

$$
\mathbf{W} \leftarrow \mathbf{W} - \eta\,\mathbf{O} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} - 0.1\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & -0.1 \\ -0.1 & 1 \end{bmatrix}.
$$

Contrast with plain gradient descent, which would have used the raw $\mathbf{G}$: $\mathbf{W} - 0.1\,\mathbf{G} = \begin{bmatrix} 1 & -0.3 \\ -0.1 & 1 \end{bmatrix}$. GD moved the top-right entry three times as far as the bottom-left (because $\sigma_1 = 3$ dominated); Muon moved them by the same amount. That is the whole difference, made concrete.

### 7.6 One Newton–Schulz iteration, numerically

To see the SVD-free machinery, run one iteration on the same $\mathbf{G} = \begin{bmatrix} 0 & 3 \\ 1 & 0 \end{bmatrix}$. Its Frobenius norm is $\|\mathbf{G}\|_F = \sqrt{0 + 9 + 1 + 0} = \sqrt{10} \approx 3.1623$, so

$$
\mathbf{X}_0 = \frac{\mathbf{G}}{\sqrt{10}} = \begin{bmatrix} 0 & 0.9487 \\ 0.3162 & 0 \end{bmatrix}.
$$

Its singular values are now $3/\sqrt{10} = 0.9487$ and $1/\sqrt{10} = 0.3162$ — a $3\!:\!1$ spread, both below $1$. Applying one step of $\mathbf{X} \leftarrow a\mathbf{X} + b(\mathbf{X}\mathbf{X}^\top)\mathbf{X} + c(\mathbf{X}\mathbf{X}^\top)^2\mathbf{X}$ with $a = 3.4445$, $b = -4.7750$, $c = 2.0315$ (the polynomial acts on each singular value as $p(\sigma) = a\sigma + b\sigma^3 + c\sigma^5$):

- $p(0.9487) = 3.4445(0.9487) - 4.7750(0.8538) + 2.0315(0.7684) \approx 0.7518.$
- $p(0.3162) = 3.4445(0.3162) - 4.7750(0.03162) + 2.0315(0.003162) \approx 0.9447.$

giving

$$
\mathbf{X}_1 \approx \begin{bmatrix} 0 & 0.7518 \\ 0.9447 & 0 \end{bmatrix}.
$$

Read the singular values: they went from $(0.9487,\ 0.3162)$ to $(0.7518,\ 0.9447)$. The iteration did **not** send each straight to $1$ — the larger one dipped, the smaller one shot up — but the *spread* collapsed from a $3\!:\!1$ ratio to about $1.26\!:\!1$. That flattening of the singular spectrum is exactly orthogonalization at work. Repeat the iteration about five times and both singular values settle into a tight band around $1$, so $\mathbf{X}$ approaches the orthogonal target $\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \mathbf{O}$ we found exactly via the SVD. One pass illustrates the direction of travel; five passes get there.

### 7.7 The trivial-but-instructive case

If instead $\mathbf{G} = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}$ (already diagonal), then $\mathbf{U} = \mathbf{V} = \mathbf{I}$ and $\boldsymbol{\Sigma} = \mathrm{diag}(3, 1)$, so orthogonalization gives $\mathbf{O} = \mathbf{U}\mathbf{V}^\top = \mathbf{I} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$. Plain GD would have stepped three times harder along the first axis than the second; Muon steps equally along both. Same lesson, even simpler numbers.

## 8. How this relates to a real PyTorch Muon optimizer

A production Muon implementation has three moving parts:

1. **Momentum on the raw gradient first.** Just like [lesson 27](lessons/module-09/lesson-02.md)'s momentum, Muon keeps a velocity buffer, $\mathbf{buf} \leftarrow \mu\,\mathbf{buf} + \mathbf{G}$, and orthogonalizes *that* smoothed gradient, not the noisy single-step $\mathbf{G}$.
2. **A `zeropower_via_newtonschulz5`-style function** that runs about five Newton–Schulz iterations on each 2-D weight's (momentum-smoothed) gradient to produce $\approx \mathbf{U}\mathbf{V}^\top$, using only matmuls, typically in `bfloat16` for speed.
3. **Muon only for hidden 2-D matrices.** The orthogonalization idea needs a genuine matrix. So Muon is applied to the hidden weight matrices (attention and MLP projections), while 1-D parameters and the input/output layers — token **embeddings** and the final classifier **head** — are trained with **AdamW** from [lesson 27](lessons/module-09/lesson-02.md). A real training script runs *two* optimizers side by side.

Here is a faithful, minimal version of the orthogonalization and the update:

```python
import torch

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    # G: a 2-D gradient of shape (out_features, in_features).
    # Returns an approximation of U V^T (the orthogonalized gradient).
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.clone().float()
    X = X / (X.norm() + eps)          # normalize by Frobenius norm -> singular values <= 1
    transpose = X.shape[0] > X.shape[1]
    if transpose:                     # keep the smaller dimension first (cheaper matmuls)
        X = X.T
    for _ in range(steps):
        A = X @ X.T                   # only matrix multiplications, no SVD
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    if transpose:
        X = X.T
    return X

# One Muon step on a single 2-D weight W with gradient W.grad:
mu, lr = 0.95, 0.1
buf = mu * buf + W.grad               # momentum on the raw gradient first
O   = zeropower_via_newtonschulz5(buf)  # approx U V^T, singular values ~1
with torch.no_grad():
    W -= lr * O                       # step by the orthogonalized update
```

Feed this the $\mathbf{G} = \begin{bmatrix} 0 & 3 \\ 1 & 0 \end{bmatrix}$ from section 7 (with `buf` equal to `G` and five iterations) and it returns a matrix very close to $\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$, reproducing the orthogonalization we did by hand with the SVD.

<div class="callout pt"><p>Muon in practice: momentum-smooth the gradient, run about five Newton–Schulz matmul iterations to orthogonalize each 2-D weight's update, and step by that. Embeddings, the final head, and all 1-D parameters stay on AdamW. Two optimizers, each where it fits best.</p></div>

## 9. An honest summary

Muon is not magic and does not replace AdamW everywhere. What it *does*, precisely: for the hidden weight *matrices* of a network, it replaces the raw (momentum-smoothed) gradient with its nearest orthogonal matrix $\mathbf{U}\mathbf{V}^\top$, computed cheaply by Newton–Schulz, so the update pushes on all singular directions equally instead of letting the largest singular value dominate. Empirically this trains those matrices faster per step and per unit compute than AdamW on several modern setups. It relies on the parameter being a real 2-D matrix with meaningful singular structure, which is why embeddings and heads are left to AdamW. Everything else in this lesson — SVD, orthogonalization, Newton–Schulz — is the machinery that makes that one idea practical.

## Check yourself

<details><summary>What does orthogonalizing a gradient $\mathbf{G} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ produce, and what happens to the singular values?</summary>

It produces $\mathbf{O} = \mathbf{U}\mathbf{V}^\top$: keep the rotations $\mathbf{U}$ and $\mathbf{V}$, drop the stretch matrix $\boldsymbol{\Sigma}$. Every singular value is set to $1$, so all singular directions are treated equally.

</details>

<details><summary>For $\mathbf{G} = \begin{bmatrix} 0 & 3 \\ 1 & 0 \end{bmatrix}$, what is the orthogonalized update $\mathbf{O}$, and the new weight if $\mathbf{W} = \mathbf{I}$ and $\eta = 0.1$?</summary>

$\mathbf{O} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$, so $\mathbf{W} \leftarrow \mathbf{I} - 0.1\mathbf{O} = \begin{bmatrix} 1 & -0.1 \\ -0.1 & 1 \end{bmatrix}$.

</details>

<details><summary>Why does Muon use the Newton–Schulz iteration instead of an actual SVD?</summary>

An SVD of every weight matrix every step is too slow. Newton–Schulz approximates $\mathbf{U}\mathbf{V}^\top$ using only matrix multiplications (which GPUs do very fast), driving the singular values toward $1$ in about five iterations without ever decomposing the matrix.

</details>

<details><summary>Why is Muon applied to hidden matrices but not to embeddings or the output head?</summary>

Orthogonalization is a statement about a matrix's singular structure, which is meaningful for genuine hidden weight matrices (attention/MLP projections). Embeddings and the final head behave differently (and 1-D parameters have no matrix structure at all), so those are trained with AdamW instead.

</details>

## Next

You now understand the full optimizer landscape used to train GPT-style models: gradient descent, the adaptive family up to AdamW, and Muon's geometric update. With optimization complete, the course turns to the probability and statistics that explain *what* a language model's output actually represents.

Continue to [the next module](lessons/module-10/lesson-01.md).
