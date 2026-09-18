# 08 · Norm, unit vectors, cosine, projection

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-02.md">07 · Adding, scaling, and the dot product</a> (you must be comfortable computing $\mathbf{x}\cdot\mathbf{w}$), and square roots / powers from <a href="../module-01/lesson-03.md">03 · Powers, roots and summation</a>.</p>
<p><strong>You will learn:</strong> How to measure a vector's length (the norm), how to shrink it to length 1 (normalization and unit vectors), how to measure the angle between two vectors (cosine similarity), and how to project one vector onto another — all worked by hand and checked in PyTorch.</p>
<p><strong>Why this matters for ML:</strong> Cosine similarity is how models compare embeddings and how attention decides what is "related." Normalization to a fixed length is the core idea inside LayerNorm and RMSNorm, which keep GPT training numerically stable. These are not abstractions — they are lines of Transformer code.</p>
</div>

## Part A — The norm (magnitude)

### 1. Intuition

The **norm** of a vector is its length: how long the arrow is. For the arrow from the origin to $[3, 4]$, the length is just the straight-line distance to that point — and you already know how to get it, because it is the Pythagorean theorem. Walk 3 across and 4 up; the hypotenuse is the length.

### 2. The mathematics

The standard length is the **Euclidean norm**, also called the **L2 norm**, written with double bars $\|\mathbf{x}\|$:

$$
\|\mathbf{x}\| = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}
$$

- $\|\mathbf{x}\|$ — a single non-negative scalar, the length of $\mathbf{x}$.
- $x_i^2$ — each component squared (so signs disappear; length is never negative).
- The square root turns "sum of squares" back into a length in the original units. This is exactly $\sqrt{\mathbf{x}\cdot\mathbf{x}}$ — the norm is the square root of the vector dotted with itself.

### 3. A tiny example

For $[3, 4]$ (the classic 3-4-5 triangle):

$$
\|[3,4]\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

For our running vectors:

$$
\|\mathbf{x}\| = \|[2,3]\| = \sqrt{2^2+3^2} = \sqrt{4+9} = \sqrt{13} \approx 3.6056
$$
$$
\|\mathbf{w}\| = \|[4,5]\| = \sqrt{4^2+5^2} = \sqrt{16+25} = \sqrt{41} \approx 6.4031
$$

We will reuse these two numbers throughout the lesson, so keep $\sqrt{13}$ and $\sqrt{41}$ handy.

### General $L_p$ norms (briefly)

The L2 norm is one member of a family. The general **$L_p$ norm** is

$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}.
$$

Two special cases show up constantly:

- $p = 2$ gives the ordinary Euclidean length above (the plain $\|\mathbf{x}\|$ with no subscript almost always means L2).
- $p = 1$ gives the **L1 norm**, $\|\mathbf{x}\|_1 = \sum_i |x_i|$ — just the sum of absolute values. For $\mathbf{x}=[2,3]$ that is $|2|+|3| = 5$. L1 measures "total travel along the axes" (city-block distance) and appears in sparsity-encouraging regularization.

For everything in the Transformer stack, though, "norm" means L2 unless stated otherwise.

## Part B — Unit vectors and normalization

### 1. Intuition

Sometimes you care only about a vector's *direction*, not its length — which way the arrow points, regardless of how long it is. A **unit vector** is an arrow of length exactly $1$ that captures pure direction. **Normalizing** a vector means shrinking (or growing) it to length 1 while keeping its direction fixed.

### 2. The mathematics

To normalize $\mathbf{x}$, divide it by its own norm:

$$
\hat{\mathbf{x}} = \frac{\mathbf{x}}{\|\mathbf{x}\|}
$$

- $\hat{\mathbf{x}}$ — the normalized vector (the "hat" conventionally marks a unit vector).
- Dividing a vector by a scalar means dividing every component by that scalar (scalar multiplication by $1/\|\mathbf{x}\|$).
- The result has norm $1$: $\|\hat{\mathbf{x}}\| = 1$, always (as long as $\mathbf{x}\neq\mathbf{0}$).

### 3. A tiny example

Normalize $\mathbf{x} = [2, 3]$, whose norm is $\sqrt{13} \approx 3.6056$:

$$
\hat{\mathbf{x}} = \frac{[2,3]}{\sqrt{13}} = \left[\frac{2}{\sqrt{13}},\; \frac{3}{\sqrt{13}}\right] \approx [0.5547,\; 0.8321]
$$

Check the length of the result:

$$
\|\hat{\mathbf{x}}\| = \sqrt{0.5547^2 + 0.8321^2} = \sqrt{0.3077 + 0.6923} = \sqrt{1.0000} = 1 \;\checkmark
$$

Same direction as $[2,3]$ — the ratio of the components is still $0.8321 / 0.5547 = 1.5 = 3/2$ — but now length exactly 1.

## Part C — Cosine similarity

### 1. Intuition

We want a single number that says "how similar in *direction* are these two vectors," ignoring how long they are. That is **cosine similarity**: the cosine of the angle $\theta$ between the arrows. It ranges from $+1$ (identical direction) through $0$ (perpendicular, unrelated) to $-1$ (exactly opposite).

### 2. The mathematics

Rearrange the geometric dot-product formula from lesson 07, $\mathbf{x}\cdot\mathbf{w} = \|\mathbf{x}\|\|\mathbf{w}\|\cos\theta$, solving for the cosine:

$$
\cos\theta = \frac{\mathbf{x}\cdot\mathbf{w}}{\|\mathbf{x}\|\,\|\mathbf{w}\|}
$$

- The numerator is the dot product — the raw alignment score.
- The denominator is the product of the two lengths — dividing by it *cancels out magnitude*, leaving pure direction. This is exactly the dot product of the two **normalized** vectors: $\cos\theta = \hat{\mathbf{x}}\cdot\hat{\mathbf{w}}$.

Why does this cancel magnitude? Because $\mathbf{x}\cdot\mathbf{w}$ scales with both lengths (double $\mathbf{x}$ and the dot product doubles); dividing by $\|\mathbf{x}\|\|\mathbf{w}\|$ divides that scaling right back out. What survives is only the angle.

### 3. A tiny example (with the actual angle)

Everything we need is already computed: $\mathbf{x}\cdot\mathbf{w} = 23$, $\|\mathbf{x}\| = \sqrt{13}$, $\|\mathbf{w}\| = \sqrt{41}$. So

$$
\cos\theta = \frac{23}{\sqrt{13}\,\sqrt{41}} = \frac{23}{\sqrt{533}} = \frac{23}{23.0868} \approx 0.9962
$$

(Note $\sqrt{13}\cdot\sqrt{41} = \sqrt{13\cdot 41} = \sqrt{533} \approx 23.0868$.) A cosine of $0.9962$ is extremely close to $1$, so these two arrows are almost — but not quite — pointing the same way. The actual angle is

$$
\theta = \arccos(0.9962) \approx 4.97^\circ \approx 0.0867 \text{ radians}.
$$

That matches the eyeball picture from lesson 06: $[2,3]$ and $[4,5]$ both head up-and-to-the-right, differing by only about five degrees.

<div class="callout key"><p>Cosine similarity = dot product of the two <em>normalized</em> vectors. It measures direction alone, on a fixed scale from $-1$ to $+1$, no matter how long the original vectors are. That scale-invariance is exactly why it is the default way to compare embeddings.</p></div>

## Part D — Projection

### 1. Intuition

The **projection** of $\mathbf{x}$ onto $\mathbf{w}$ answers: "if I shine a light straight down onto the line through $\mathbf{w}$, what shadow does $\mathbf{x}$ cast on that line?" It is the part of $\mathbf{x}$ that lies along $\mathbf{w}$'s direction — how far along $\mathbf{w}$ you get by dropping $\mathbf{x}$ perpendicularly onto it.

### 2. The mathematics

The **scalar projection** (the signed length of the shadow) is

$$
\text{comp}_{\mathbf{w}}\mathbf{x} = \frac{\mathbf{x}\cdot\mathbf{w}}{\|\mathbf{w}\|},
$$

and the **vector projection** (the shadow as an actual vector pointing along $\mathbf{w}$) is that length times the unit vector $\hat{\mathbf{w}} = \mathbf{w}/\|\mathbf{w}\|$:

$$
\text{proj}_{\mathbf{w}}\mathbf{x} = \frac{\mathbf{x}\cdot\mathbf{w}}{\|\mathbf{w}\|^2}\,\mathbf{w}.
$$

- Numerator $\mathbf{x}\cdot\mathbf{w}$ — the alignment between the two.
- Dividing by $\|\mathbf{w}\|^2$ (the scalar projection uses $\|\mathbf{w}\|$; the vector projection uses $\|\mathbf{w}\|^2$ because it then re-multiplies by the full $\mathbf{w}$, contributing one more factor of $\|\mathbf{w}\|$).

### 3. A tiny example

Project $\mathbf{x} = [2,3]$ onto $\mathbf{w} = [4,5]$. We have $\mathbf{x}\cdot\mathbf{w} = 23$ and $\|\mathbf{w}\|^2 = 41$ (no square root needed here — nice).

Scalar projection:

$$
\frac{23}{\sqrt{41}} = \frac{23}{6.4031} \approx 3.5920
$$

Vector projection:

$$
\text{proj}_{\mathbf{w}}\mathbf{x} = \frac{23}{41}\,[4,5] = \left[\frac{92}{41},\; \frac{115}{41}\right] \approx [2.2439,\; 2.8049]
$$

That resulting vector points in exactly $\mathbf{w}$'s direction (it is a positive multiple, $\tfrac{23}{41}\approx 0.561$, of $\mathbf{w}$) and represents "the part of $\mathbf{x}$ that lies along $\mathbf{w}$."

## 4. Why ML needs all of this

- **Cosine similarity compares embeddings.** Two words with similar meanings get embedding vectors pointing in nearly the same direction, so their cosine similarity is near $1$ — regardless of the vectors' raw lengths. Search, retrieval, and "which token is most related" comparisons run on cosine similarity for exactly this scale-invariance.
- **Attention is (scaled) dot products of directions.** An attention score is a query dotted with a key. Because a dot product is $\|\mathbf{q}\|\|\mathbf{k}\|\cos\theta$, it is largely a directional-agreement measurement — the projection/cosine geometry from this lesson is the geometry of "how much does this token attend to that one."
- **Normalization stabilizes training.** **LayerNorm** and **RMSNorm** — applied at nearly every step inside a Transformer block — rescale each hidden-state vector to a controlled magnitude before the next layer sees it. That is the same normalize-by-a-norm move as $\mathbf{x}/\|\mathbf{x}\|$, and it keeps activations and gradients from blowing up or vanishing as they pass through dozens of layers. Unit-length (or fixed-length) inputs make the numerics well-behaved, which is why fixed magnitude matters for stability. We devote a full later module to LayerNorm vs RMSNorm; this lesson is the geometric seed of both.

<div class="callout key"><p>Norm → magnitude. Normalize → keep direction, fix length to 1. Cosine → compare direction on a $[-1,1]$ scale. These three moves reappear as embedding comparison, attention scoring, and LayerNorm/RMSNorm throughout GPT.</p></div>

## 5. In PyTorch

Every hand calculation above, verified in code:

```python
import torch
import torch.nn.functional as F

x = torch.tensor([2., 3.])
w = torch.tensor([4., 5.])

# --- Norms (Part A) ---
print(torch.linalg.norm(x))          # tensor(3.6056)   = sqrt(13)
print(torch.linalg.norm(w))          # tensor(6.4031)   = sqrt(41)
print(torch.linalg.norm(torch.tensor([3., 4.])))   # tensor(5.)
# L1 norm
print(torch.linalg.norm(x, ord=1))   # tensor(5.)       = |2| + |3|

# --- Normalization / unit vector (Part B) ---
xhat = F.normalize(x, dim=0)         # divide by its L2 norm
print(xhat)                          # tensor([0.5547, 0.8321])
print(torch.linalg.norm(xhat))       # tensor(1.0000)   length is now 1

# --- Cosine similarity (Part C) ---
cos = F.cosine_similarity(x, w, dim=0)
print(cos)                           # tensor(0.9962)
angle_rad = torch.acos(cos)
print(angle_rad)                     # tensor(0.0867)   radians
print(angle_rad * 180 / torch.pi)    # tensor(4.9697)   degrees

# --- Projection (Part D) ---
scalar_proj = torch.dot(x, w) / torch.linalg.norm(w)
print(scalar_proj)                   # tensor(3.5920)
vec_proj = (torch.dot(x, w) / torch.dot(w, w)) * w   # ||w||^2 = w·w
print(vec_proj)                      # tensor([2.2439, 2.8049])
```

Every printed number matches the by-hand result: norms $3.6056$ and $6.4031$, cosine $0.9962$, angle $4.97^\circ$, scalar projection $3.5920$, vector projection $[2.2439, 2.8049]$.

<div class="callout pt"><p><code>F.normalize</code> and <code>F.cosine_similarity</code> take a <code>dim</code> argument telling them which axis is the "vector" axis. For a plain 1-D tensor that is <code>dim=0</code>. In real GPT code the hidden-state vectors live along the last axis of a <code>(B, T, C)</code> tensor, so you pass <code>dim=-1</code> to normalize each token independently — the same operation, applied across a whole batch at once.</p></div>

## 6. Under the hood

Two small but important numerical points, both of which resurface in the training-stability module:

- `torch.linalg.norm(x)` computes $\sqrt{\mathbf{x}\cdot\mathbf{x}}$: it squares each element, sums into one accumulator, then takes a single square root. It is a *reduction* — many numbers in, one number out — just like the dot product, and it collapses the vector axis you point it at.
- `F.cosine_similarity` does **not** literally divide by zero when a vector is tiny. It adds a small $\varepsilon$ (epsilon) in the denominator, computing $\dfrac{\mathbf{x}\cdot\mathbf{w}}{\max(\|\mathbf{x}\|\|\mathbf{w}\|,\ \varepsilon)}$, to stay finite. That epsilon guard is the same trick LayerNorm and RMSNorm use — divide by "the norm, but never quite zero" — and it is your first sight of *numerical stability* engineering, a theme we return to in force later.

## Check yourself

<details><summary>Compute $\|[3,4]\|$ and $\|[2,3]\|$.</summary>

$\|[3,4]\| = \sqrt{9+16} = \sqrt{25} = 5$ exactly. $\|[2,3]\| = \sqrt{4+9} = \sqrt{13} \approx 3.6056$.

</details>

<details><summary>The cosine similarity of $\mathbf{x}=[2,3]$ and $\mathbf{w}=[4,5]$ is about $0.9962$. What does that number say about the two vectors?</summary>

They point in almost the same direction — the angle between them is only about $4.97^\circ$. A value near $+1$ means near-identical direction; $0$ would mean perpendicular; $-1$ would mean opposite.

</details>

<details><summary>Why divide by $\|\mathbf{w}\|^2$ (not $\|\mathbf{w}\|$) in the <em>vector</em> projection, but by $\|\mathbf{w}\|$ in the <em>scalar</em> projection?</summary>

The scalar projection is a length: $\dfrac{\mathbf{x}\cdot\mathbf{w}}{\|\mathbf{w}\|}$. The vector projection multiplies that length by the unit vector $\mathbf{w}/\|\mathbf{w}\|$, contributing a second $\|\mathbf{w}\|$ in the denominator — giving $\dfrac{\mathbf{x}\cdot\mathbf{w}}{\|\mathbf{w}\|^2}\mathbf{w}$.

</details>

<details><summary>What is the relationship between normalizing a vector and cosine similarity?</summary>

Cosine similarity is exactly the dot product of the two normalized vectors: $\cos\theta = \hat{\mathbf{x}}\cdot\hat{\mathbf{w}}$ where $\hat{\mathbf{x}} = \mathbf{x}/\|\mathbf{x}\|$. Normalizing removes magnitude, so the dot product of unit vectors measures pure direction.

</details>

## Next

You have finished vectors: what they are, how to combine them, and how to measure their lengths and angles. Every one of these operations was really about *one* vector at a time or a pair. The next module stacks vectors into a grid — a **matrix** — and shows how matrix-vector multiplication is just a batch of dot products, which is the exact shape of a neural-network linear layer.

→ [Module 3 · Matrices](../module-03/lesson-01.md)
