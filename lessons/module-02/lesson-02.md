# 07 · Adding, scaling, and the dot product

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-02/lesson-01">06 · What a vector is</a> (components, dimension, the tensor-vs-list distinction), and summation notation $\sum$ from <a href="#/lessons/module-01/lesson-05">05 · Reading mathematical notation</a>.</p>
<p><strong>You will learn:</strong> The three operations that turn vectors from static lists into a working algebra — component-wise addition, scalar multiplication, and the <strong>dot product</strong> — with the geometry, worked arithmetic, and the exact PyTorch calls for each.</p>
<p><strong>Why this matters for ML:</strong> A single neuron computes a dot product of its inputs and its weights. Every entry of every matrix multiplication in a Transformer is a dot product. This one operation, repeated billions of times, is what a neural network mostly <em>does</em>.</p>
</div>

## Part A — Vector addition

### 1. Intuition

If a vector is an arrow, adding two vectors means *following one arrow, then the other*. Walk along $\mathbf{x}$, and from wherever you land, walk along $\mathbf{w}$. The single arrow from your start to your finish is $\mathbf{x} + \mathbf{w}$. This is the **tip-to-tail** picture: put the tail of the second arrow on the tip of the first, and the sum runs from the very beginning to the very end.

### 2. The mathematics

Addition is **component-wise**: line the two vectors up and add matching entries.

$$
\mathbf{x} + \mathbf{w} = [x_1 + w_1,\; x_2 + w_2,\; \dots,\; x_n + w_n]
$$

- $\mathbf{x}, \mathbf{w} \in \mathbb{R}^n$ — both vectors must have the **same dimension** $n$. You cannot add a 2-vector to a 3-vector; there is no matching partner for the extra entry.
- The result is again in $\mathbb{R}^n$ — same shape in, same shape out.

### 3. A tiny example

With $\mathbf{x} = [2, 3]$ and $\mathbf{w} = [4, 5]$:

$$
\mathbf{x} + \mathbf{w} = [2+4,\; 3+5] = [6, 8]
$$

Geometrically: start at the origin, go to $[2,3]$, then step by $[4,5]$ more (4 right, 5 up), landing at $[6,8]$. The straight arrow from $(0,0)$ to $(6,8)$ is the sum.

## Part B — Scalar multiplication

### 1. Intuition

Multiplying a vector by a single number (a **scalar**) *stretches, shrinks, or flips* the arrow without changing the line it lies on. Multiply by $2$ and the arrow doubles in length, same direction. Multiply by $\tfrac12$ and it halves. Multiply by $-1$ and it flips to point the opposite way.

### 2. The mathematics

Multiply every component by the scalar $c$:

$$
c\,\mathbf{x} = [c\,x_1,\; c\,x_2,\; \dots,\; c\,x_n]
$$

- $c$ — a scalar (a single number).
- $\mathbf{x} \in \mathbb{R}^n$ — the vector; the result stays in $\mathbb{R}^n$.

### 3. A tiny example

$$
2\,\mathbf{x} = 2\cdot[2,3] = [4, 6], \qquad -1\cdot\mathbf{x} = [-2, -3], \qquad \tfrac12\,\mathbf{x} = [1, 1.5]
$$

The arrow for $2\mathbf{x}$ points the exact same way as $\mathbf{x}$ but is twice as long; $-\mathbf{x}$ has the same length but points backward. Direction is preserved (or exactly reversed); only length and sign change.

<div class="callout key"><p>Addition combines two arrows tip-to-tail; scalar multiplication rescales one arrow along its own line. Together they are the entire "linear" in "linear algebra": any combination $c_1\mathbf{x} + c_2\mathbf{w}$ is a <em>linear combination</em>, and linear layers are built entirely from these.</p></div>

## Part C — The dot product

This is the heart of the lesson and one of the most important ideas in the whole course, so we give it the full six-pass treatment.

### 1. Intuition

The **dot product** takes two vectors and returns a single number that measures **how much they point in the same direction**. Big positive number → they largely agree (point the same way). Zero → they are perpendicular, completely unrelated. Big negative → they point opposite ways. Think of it as an "agreement score" or "alignment score" between two arrows.

Two names for the same thing you will see in code and papers: **dot product** and **inner product**. They mean the same operation here.

### 2. The mathematics

Multiply matching components, then add up all those products:

$$
\mathbf{x} \cdot \mathbf{w} \;=\; \sum_{i=1}^{n} x_i\, w_i \;=\; x_1 w_1 + x_2 w_2 + \dots + x_n w_n
$$

- $\mathbf{x}, \mathbf{w} \in \mathbb{R}^n$ — same dimension required (again, entries must pair up).
- $x_i w_i$ — the product of the $i$-th entries; the big $\sum$ says "sum this over every $i$ from $1$ to $n$."
- The output is a **single scalar**, not a vector. Two vectors go in; one number comes out. This is why the dot product is sometimes written to *collapse* a dimension.

There is a second, geometric formula for the exact same number:

$$
\mathbf{x} \cdot \mathbf{w} \;=\; \|\mathbf{x}\|\,\|\mathbf{w}\|\cos\theta
$$

where $\|\mathbf{x}\|$ is the length (magnitude) of $\mathbf{x}$, $\|\mathbf{w}\|$ the length of $\mathbf{w}$, and $\theta$ the angle between the two arrows. We will *derive and use* this form fully in the next lesson (norms and cosine). For now, absorb the qualitative message it carries.

### 3. A tiny example (do this by hand)

With $\mathbf{x} = [2, 3]$ and $\mathbf{w} = [4, 5]$:

$$
\mathbf{x} \cdot \mathbf{w} = (2)(4) + (3)(5) = 8 + 15 = 23
$$

Step by step: pair the first entries $2$ and $4$, multiply → $8$. Pair the second entries $3$ and $5$, multiply → $15$. Add the two products → $23$. That is the whole operation. A positive result of $23$ tells us these two arrows largely agree in direction — which fits the picture from lesson 06, where both pointed up-and-to-the-right.

### 4. Why ML needs it: the dot product is what a neuron computes

Here is the single most important connection in this module. A **neuron** takes a vector of inputs $\mathbf{x}$ and a vector of learned **weights** $\mathbf{w}$, and its core computation is exactly a dot product plus a bias $b$:

$$
z = \mathbf{w} \cdot \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b
$$

Each weight $w_i$ says "how much do I care about input $x_i$, and in which direction?" The dot product blends all the inputs into one number $z$ according to those weights. That number is the neuron's **pre-activation** (we use the symbol $z$ for it course-wide).

Now scale this up. A **linear layer** is just many neurons side by side, so its output is many dot products — one per neuron. And a **matrix multiplication**, the operation that dominates every Transformer, is defined so that *each single entry of the result is one dot product* of a row with a column. Attention scores? Query dotted with key. The final projection to vocabulary logits? Hidden state dotted with each row of the output matrix. When people say "neural networks are mostly matmuls," they are really saying **neural networks are mostly dot products**.

<div class="callout key"><p>A neuron computes $\mathbf{w}\cdot\mathbf{x}+b$. Every entry of every matrix multiply is a dot product. Learn this one operation cold and you have learned the atomic step of forward propagation.</p></div>

### 5. In PyTorch

All three operations, on our running vectors:

```python
import torch

x = torch.tensor([2., 3.])
w = torch.tensor([4., 5.])

# Addition — component-wise
print(x + w)          # tensor([6., 8.])

# Scalar multiplication
print(2 * x)          # tensor([4., 6.])
print(-1 * x)         # tensor([-2., -3.])

# Dot product — three equivalent ways
print(torch.dot(x, w))     # tensor(23.)
print(x @ w)               # tensor(23.)   <- the @ operator
print((x * w).sum())       # tensor(23.)   <- elementwise product, then sum
```

Three things worth pinning down:

- `x * w` is **not** the dot product. It is the *element-wise* product $[x_1 w_1, x_2 w_2] = [8, 15]$, a vector — the star `*` always means "multiply matching entries." The dot product is that vector *summed*: `(x * w).sum()` → `23`. This is literally the definition $\sum_i x_i w_i$ written in code.
- `torch.dot` works only for 1-D vectors. For matrices and batched tensors you use `@` (matrix multiply), which generalizes the dot product — but for two vectors, `x @ w` gives the same scalar `23`.
- `2 * x` multiplies every entry by 2 without you writing a loop; that is *broadcasting*, previewed below.

<div class="callout pt"><p>Read <code>@</code> as "matrix multiply / dot product," and reserve <code>*</code> for "element-wise." Confusing the two is one of the most common shape bugs in ML code — <code>*</code> silently returns a vector where you expected a scalar.</p></div>

### 6. Under the hood

**Scalar multiplication and broadcasting.** When you write `2 * x`, the scalar `2` has no shape and `x` has shape `(2,)`. PyTorch *broadcasts* the scalar: it conceptually stretches `2` to match every element of `x`, then multiplies element-wise. No new memory is allocated for a `[2, 2]` vector; the scalar is simply reused against each element as the loop runs over the contiguous buffer. Broadcasting is the general rule that lets tensors of compatible-but-different shapes combine, and we will formalize it in the tensors module — but scalar-times-vector is its simplest case, and you have now seen it.

**The dot product as two primitive steps.** `torch.dot(x, w)` is, mechanically, exactly what `(x * w).sum()` spells out:

1. **Element-wise multiply** the two contiguous float buffers, producing the intermediate $[8, 15]$.
2. **Reduce** (sum) that intermediate down to the single scalar $23$.

On a real machine these two steps are usually fused into one pass — a *multiply-accumulate* loop that keeps a running total and never materializes the full intermediate vector. This "multiply matching numbers and accumulate" is the exact hardware operation that GPUs and tensor cores are built to do at massive scale, which is precisely why dot products (and the matmuls made of them) are so fast on modern accelerators. The mathematics ($\sum_i x_i w_i$), the PyTorch (`x @ w`), and the silicon (multiply-accumulate) are the same idea at three levels.

## Check yourself

<details><summary>Compute $\mathbf{x}\cdot\mathbf{w}$ for $\mathbf{x}=[2,3]$, $\mathbf{w}=[4,5]$, and say what the sign of the result tells you.</summary>

$(2)(4)+(3)(5)=8+15=23$. A positive dot product means the two vectors largely point in the same direction (the angle between them is less than $90^\circ$).

</details>

<details><summary>In PyTorch, what is the difference between <code>x * w</code> and <code>x @ w</code> for two 1-D tensors?</summary>

`x * w` is the element-wise product — a vector, `tensor([8., 15.])`. `x @ w` (equivalently `torch.dot(x, w)`) is the dot product — the scalar `tensor(23.)`, i.e. the element-wise product summed.

</details>

<details><summary>Why is the dot product called "the operation a neuron computes"?</summary>

A neuron's pre-activation is $z=\mathbf{w}\cdot\mathbf{x}+b$: it dots the input vector with its weight vector (then adds a bias). Stacking many neurons gives a matrix multiply, and every entry of that matmul is itself a dot product — so the dot product is the atomic computation of a linear layer.

</details>

## Next

You now have addition, scaling, and the dot product — but we kept invoking a vector's *length* $\|\mathbf{x}\|$ and the *angle* $\theta$ between vectors without defining them. The next lesson makes both precise: norms, unit vectors, cosine similarity, and projection — and closes the loop on the geometric formula $\mathbf{x}\cdot\mathbf{w}=\|\mathbf{x}\|\|\mathbf{w}\|\cos\theta$.

→ [08 · Norm, unit vectors, cosine, projection](lessons/module-02/lesson-03.md)
