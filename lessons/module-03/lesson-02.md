# 10 · Matrix multiplication

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-01.md">09 · Matrices, shapes, and indexing</a> (rows, columns, shape, the (row, col) index order) and <a href="../module-02/lesson-02.md">07 · Adding, scaling, and the dot product</a> — you must be comfortable with the dot product of two vectors, because that is the atom that matrix multiplication is built from.</p>
<p><strong>You will learn:</strong> the matrix–vector product (each output entry is a dot product), then the general matrix–matrix product, the shape rule that governs both, why the order matters, and how every one of these maps onto a line of PyTorch and onto a neural-network layer.</p>
<p><strong>Why this matters for ML:</strong> this is <em>the</em> operation. A linear layer is a matrix multiply. Attention is matrix multiplies. A Transformer is, by compute, almost entirely matrix multiplies. The rule that inner dimensions must match is the reason tensor shapes dominate every hour you will spend reading Transformer code.</p>
</div>

## 1. Intuition: a machine built out of dot products

You already know the **dot product** of two vectors of the same length: multiply them position-by-position and add up the results. For $\mathbf{a} = \begin{bmatrix} 1 & 2 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$, the dot product is $1 \cdot 5 + 2 \cdot 6 = 17$. One dot product turns two vectors into a single number.

Matrix multiplication is nothing more than **a whole grid of dot products, computed in a disciplined order**. Every single number in the output is the dot product of one row taken from the left matrix and one column taken from the right matrix. That is the entire secret. If you can compute a dot product, you can compute any matrix product — the only new skill is bookkeeping: keeping track of *which* row pairs with *which* column to land in *which* output position.

So do not think of matrix multiplication as a mysterious new formula. Think of it as: "for each output cell, find the row and the column that meet there, and dot them together." We will build up to the general case starting from the simplest useful version — a matrix times a vector.

<div class="callout key"><p>Every entry of a matrix product is a dot product: the dot product of a <em>row</em> from the left matrix with a <em>column</em> from the right matrix.</p></div>

## 2. Matrix–vector product

### 2.1 The mathematics

The simplest matrix product is a matrix times a column vector. Take a matrix $A$ of shape $(m, n)$ and a vector $\mathbf{x}$ of length $n$. Their product $A\mathbf{x}$ is a new vector of length $m$. The rule for each entry:

$$
(A\mathbf{x})_i = \sum_{k=1}^{n} A_{ik}\, x_k
$$

Naming every symbol:

- $A$ — the matrix, shape $(m, n)$: $m$ rows, $n$ columns.
- $\mathbf{x}$ — the input vector, length $n$ (it must have exactly as many entries as $A$ has columns).
- $(A\mathbf{x})_i$ — the $i$-th entry of the output vector.
- $i$ — the output position, running from $1$ to $m$ (one per row of $A$).
- $k$ — the summation index, running across all $n$ columns of $A$ and all $n$ entries of $\mathbf{x}$.
- $\sum_{k=1}^{n}$ — "add up over $k$ from 1 to $n$": the dot-product sum.

Read the formula in words: **the $i$-th output number is the dot product of row $i$ of $A$ with the whole vector $\mathbf{x}$.** There is one output number per row, so the output has $m$ entries.

The shape bookkeeping is:

$$
(m, n) \cdot (n,) \;\longrightarrow\; (m,)
$$

The $n$ appears twice on the left — the columns of $A$ and the length of $\mathbf{x}$ — and they must be equal, because a dot product only works between two vectors of the same length. Those two matching $n$'s "cancel," and what survives is the $m$: the number of rows, which is the length of the answer.

### 2.2 A tiny numerical example, worked fully

Let us use the example from the course brief:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad \mathbf{x} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}
$$

$A$ is $(2, 2)$ and $\mathbf{x}$ has length $2$. The columns of $A$ (two of them) match the length of $\mathbf{x}$ (two), so the product is defined, and the answer will have length $2$ (the number of rows of $A$).

**Output entry 1** = dot product of *row 1 of $A$* with $\mathbf{x}$. Row 1 of $A$ is $\begin{bmatrix} 1 & 2 \end{bmatrix}$:

$$
(A\mathbf{x})_1 = 1 \cdot 5 + 2 \cdot 6 = 5 + 12 = 17
$$

**Output entry 2** = dot product of *row 2 of $A$* with $\mathbf{x}$. Row 2 of $A$ is $\begin{bmatrix} 3 & 4 \end{bmatrix}$:

$$
(A\mathbf{x})_2 = 3 \cdot 5 + 4 \cdot 6 = 15 + 24 = 39
$$

Stack the two results into the output vector:

$$
A\mathbf{x} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 6 \\ 3 \cdot 5 + 4 \cdot 6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}
$$

Every output number was one dot product of one row with $\mathbf{x}$. That is the whole operation, and it is worth doing a second time on scratch paper until the "row dot vector" motion feels automatic, because the matrix–matrix case in the next section is just this repeated.

<div class="callout key"><p>Matrix–vector product: output entry $i$ = (row $i$ of $A$) · $\mathbf{x}$. Shape: $(m, n) \cdot (n,) \to (m,)$. The two $n$'s must match; the surviving $m$ is the length of the answer.</p></div>

## 3. Matrix–matrix product

### 3.1 The mathematics

Now multiply two matrices. Take $A$ of shape $(m, n)$ and $B$ of shape $(n, p)$. Their product $C = AB$ has shape $(m, p)$. Each entry:

$$
C_{ij} = \sum_{k=1}^{n} A_{ik}\, B_{kj}
$$

Naming every symbol:

- $A$ — left matrix, shape $(m, n)$.
- $B$ — right matrix, shape $(n, p)$.
- $C = AB$ — the product, shape $(m, p)$.
- $C_{ij}$ — the entry of $C$ in row $i$, column $j$.
- $i$ — output row index, $1 \ldots m$ (chooses a **row of $A$**).
- $j$ — output column index, $1 \ldots p$ (chooses a **column of $B$**).
- $k$ — the summation index, $1 \ldots n$, running across the columns of $A$ and down the rows of $B$ at the same time.

In words: **the entry in row $i$, column $j$ of the product is the dot product of row $i$ of $A$ with column $j$ of $B$.** Compared to the matrix–vector case, the only new thing is that now there are several columns to dot against, one per column of $B$ — so the output is a full grid rather than a single column.

### 3.2 The shape rule — the most important rule in the course

Here is the shape story, and you should read it twice:

$$
(m, n) \cdot (n, p) \;\longrightarrow\; (m, p)
$$

Write the two shapes next to each other: $(m, \underline{n}) \cdot (\underline{n}, p)$. The two **inner** numbers — the $n$ from $A$'s columns and the $n$ from $B$'s rows — must be **equal**. They are the length of the dot products, so they have to agree, exactly as in the vector case. When they match, they cancel out, and the two **outer** numbers, $m$ and $p$, become the shape of the result.

- **Inner dimensions must match** ($A$'s columns = $B$'s rows), or the product is simply undefined.
- **Outer dimensions survive** and give the output shape $(m, p)$.

A quick way to check any product before computing it: write the shapes side by side, confirm the inner pair matches, and read off the outer pair as the answer's shape. For example $(4, 768) \cdot (768, 3072) \to (4, 3072)$: the 768's match, so it works, and the result is $(4, 3072)$. But $(4, 768) \cdot (3072, 768)$ fails immediately — the inner numbers $768$ and $3072$ disagree — and no arithmetic is even worth attempting.

<div class="callout warn"><p>Before you multiply, line the shapes up: $(m, n) \cdot (n, p)$. If the two inner numbers are not equal, the product does not exist — full stop. This single check will catch the large majority of shape bugs you hit in real model code.</p></div>

### 3.3 A tiny numerical example, worked fully

Multiply two $2 \times 2$ matrices:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

Shapes: $A$ is $(2, 2)$, $B$ is $(2, 2)$. Inner numbers are both $2$ — they match — so the product exists and is $(2, 2)$: four numbers to compute. For each, we take a row of $A$ and a column of $B$.

The rows of $A$ are $\begin{bmatrix} 1 & 2 \end{bmatrix}$ (row 1) and $\begin{bmatrix} 3 & 4 \end{bmatrix}$ (row 2). The columns of $B$ are $\begin{bmatrix} 5 \\ 7 \end{bmatrix}$ (column 1) and $\begin{bmatrix} 6 \\ 8 \end{bmatrix}$ (column 2).

**$C_{11}$** = row 1 of $A$ · column 1 of $B$:
$$
C_{11} = 1 \cdot 5 + 2 \cdot 7 = 5 + 14 = 19
$$

**$C_{12}$** = row 1 of $A$ · column 2 of $B$:
$$
C_{12} = 1 \cdot 6 + 2 \cdot 8 = 6 + 16 = 22
$$

**$C_{21}$** = row 2 of $A$ · column 1 of $B$:
$$
C_{21} = 3 \cdot 5 + 4 \cdot 7 = 15 + 28 = 43
$$

**$C_{22}$** = row 2 of $A$ · column 2 of $B$:
$$
C_{22} = 3 \cdot 6 + 4 \cdot 8 = 18 + 32 = 50
$$

Assemble the four numbers into the grid, each in its (row, col) slot:

$$
AB = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

Trace one entry back to convince yourself the bookkeeping is right: $C_{21} = 43$ sits in row 2, column 1, and indeed it came from **row 2** of $A$ and **column 1** of $B$. The first index of the output picks the row of $A$; the second index picks the column of $B$. That is the pattern for every entry, in every matrix product, of any size.

### 3.4 Order matters: matrix multiplication is not commutative

For ordinary numbers, $5 \cdot 6 = 6 \cdot 5$. For matrices, $AB$ and $BA$ are, in general, **not equal** — and may not even both exist. This property is called **non-commutativity**, and it is one of the biggest differences between matrix algebra and the number algebra you are used to.

Let us actually compute $BA$ with the same two matrices and see it differ. Now the rows come from $B$ and the columns come from $A$. Rows of $B$: $\begin{bmatrix} 5 & 6 \end{bmatrix}$, $\begin{bmatrix} 7 & 8 \end{bmatrix}$. Columns of $A$: $\begin{bmatrix} 1 \\ 3 \end{bmatrix}$, $\begin{bmatrix} 2 \\ 4 \end{bmatrix}$.

$$
\begin{aligned}
(BA)_{11} &= 5 \cdot 1 + 6 \cdot 3 = 5 + 18 = 23 \\
(BA)_{12} &= 5 \cdot 2 + 6 \cdot 4 = 10 + 24 = 34 \\
(BA)_{21} &= 7 \cdot 1 + 8 \cdot 3 = 7 + 24 = 31 \\
(BA)_{22} &= 7 \cdot 2 + 8 \cdot 4 = 14 + 32 = 46
\end{aligned}
$$

$$
BA = \begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix} \;\neq\; \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix} = AB
$$

Same two matrices, opposite order, completely different result. The lesson is concrete and permanent: **the order of a matrix product is part of its meaning.** In neural-network code you can never "swap the operands to make the shapes work" — swapping changes the computation, and usually breaks the shape rule too. When shapes force a particular order, that order is telling you something real about which quantity is transforming which.

<div class="callout warn"><p>$AB \neq BA$ in general. Order is meaning, not a formality. Do not reorder a matrix product to fix a shape error — reordering computes a different thing (and often will not even be a legal product).</p></div>

## 4. Why ML needs it: the linear layer is a matrix multiply

Here is the single most important connection in this module. A **linear layer** — the most common building block in every neural network, including every one inside GPT-2 — computes exactly this:

$$
\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

- $\mathbf{x}$ — the input vector, length $C_\text{in}$ (the incoming number of features/channels).
- $\mathbf{W}$ — the **weight matrix**, shape $(C_\text{out}, C_\text{in})$. These are learned parameters.
- $\mathbf{W}\mathbf{x}$ — a matrix–vector product, exactly the operation from section 2. Its output has length $C_\text{out}$.
- $\mathbf{b}$ — the **bias vector**, length $C_\text{out}$, added on at the end (ordinary vector addition from lesson 07).
- $\mathbf{y}$ — the output vector, length $C_\text{out}$.

So a linear layer is: take the input, multiply by a matrix, add a bias. The matrix multiply is where all the mixing happens — each output number is a dot product of one row of $\mathbf{W}$ with the whole input, meaning every output feature is a weighted blend of every input feature. That is what "fully connected" means, expressed as one matrix product.

When you process a whole **batch** at once, $\mathbf{x}$ becomes a matrix $X$ of shape $(B, C_\text{in})$ — $B$ example-vectors stacked as rows, from lesson 09. The layer then computes $X \mathbf{W}^\top$ (the transpose $\mathbf{W}^\top$, which you will meet next lesson, just lines the shapes up), a matrix–matrix product of shape $(B, C_\text{in}) \cdot (C_\text{in}, C_\text{out}) \to (B, C_\text{out})$: every example transformed, in one operation, on parallel hardware.

Zoom out and the whole model is this pattern repeated. Attention computes scores as a matrix product $QK^\top$, then multiplies by $V$ — more matrix products (you will do this by hand in module 11). The MLP inside each block is two linear layers, so two more matrix products. A GPT-2 forward pass is, by fraction of compute spent, overwhelmingly matrix multiplication. When people say "training a Transformer is expensive," they mean it does an enormous number of these.

And now the shape rule from section 3.2 stops being abstract. The reason Transformer code is a constant stream of shape-juggling — `reshape`, `transpose`, `view`, splitting a $C$-dimensional vector into $n_h$ heads of size $d_h$ with $C = n_h \cdot d_h$ — is that every matrix product demands its inner dimensions line up. Most bugs in hand-written model code are shape-mismatch errors, and the fix is always the same discipline: write the shapes down, check the inner pair matches. The $n$ that must match is the thing your mental energy goes toward.

<div class="callout key"><p>A linear layer is $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$: a matrix–vector product plus a bias. GPT-2 is mostly matrix multiplies. The "inner dimensions must match" rule is why tensor shapes dominate Transformer code.</p></div>

## 5. In PyTorch

The operator for matrix multiplication in PyTorch (and NumPy) is `@`. Let us reproduce both hand results.

**Matrix–vector**, matching section 2.2:

```python
import torch

A = torch.tensor([[1, 2],
                  [3, 4]])
x = torch.tensor([5, 6])

print(A @ x)            # tensor([17, 39])   -> matches [1*5+2*6, 3*5+4*6]
print((A @ x).shape)    # torch.Size([2])    -> (2,2)@(2,) -> (2,)
```

The result `[17, 39]` is exactly the vector we computed by hand, and the shape `(2,)` is exactly what the rule $(2,2)\cdot(2,) \to (2,)$ predicted.

**Matrix–matrix**, matching section 3.3:

```python
B = torch.tensor([[5, 6],
                  [7, 8]])

print(A @ B)            # tensor([[19, 22],
                        #         [43, 50]])   -> matches our hand calculation
print(torch.matmul(A, B))   # same thing: matmul is the function form of @
print((A @ B).shape)    # torch.Size([2, 2])
```

`A @ B` and `torch.matmul(A, B)` are the same operation — `@` is just the readable operator form of `torch.matmul`. The output matches our hand-computed $\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$ entry for entry.

Non-commutativity, confirmed by the machine:

```python
print(B @ A)            # tensor([[23, 34],
                        #         [31, 46]])   -> different from A @ B
print(torch.equal(A @ B, B @ A))   # False
```

And the shape rule, enforced with a clear error:

```python
C = torch.randn(3, 5)
D = torch.randn(4, 2)
# C @ D  -> RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x5 and 4x2)
```

The message spells out the exact mismatch: $5 \neq 4$, so the inner dimensions do not agree. When you see this error in real training code — and you will, often — read the two shapes it prints and find which inner pair failed to match.

<div class="callout warn"><p>Do not confuse <code>@</code> with <code>*</code>. In PyTorch <code>A @ B</code> is matrix multiplication (dot products of rows and columns), while <code>A * B</code> is <em>element-wise</em> multiplication (multiply entries in the same position, same-shape required). They are completely different operations and both are common in ML code — mixing them up is a classic bug.</p></div>

## 6. What PyTorch is doing under the hood

Two things are worth knowing about what happens when you type `A @ B`.

**It dispatches to a specialized kernel.** PyTorch does not run a naive triple `for` loop over $i$, $j$, $k$ in Python — that would be far too slow. Instead `@` hands the job to a heavily optimized library routine: **BLAS** (Basic Linear Algebra Subprograms) on the CPU, and **cuBLAS** or similar tuned kernels on the GPU. These are decades-refined implementations that tile the matrices to fit CPU caches or GPU shared memory, use vectorized instructions, and run thousands of the underlying multiply-add operations in parallel. The mathematical result is identical to the dot-product-per-entry definition you learned here; only the execution is clever. This is *why* stacking examples into a batch matters: one big matrix multiply keeps that hardware saturated, whereas many small matrix–vector products leave it idle.

**Batched matmul — a preview.** In real Transformer code, `@` rarely acts on plain 2-D matrices. It acts on tensors with extra leading dimensions, like $(B, T, C)$. When you write `A @ B` on higher-dimensional tensors, PyTorch treats all the leading axes as a **batch** and performs an independent matrix multiplication for each combination of those leading indices, applying the ordinary 2-D rule to the *last two* axes only:

```python
Q = torch.randn(2, 3, 4)     # think (batch=2, rows=3, inner=4)
K = torch.randn(2, 4, 5)     # think (batch=2, inner=4, cols=5)
print((Q @ K).shape)         # torch.Size([2, 3, 5])
```

Here the leading `2` is carried along untouched, and for each of the 2 batch elements PyTorch does a $(3, 4) \cdot (4, 5) \to (3, 5)$ product — the same inner-dimensions-must-match rule you now know, applied to the last two axes. This "matmul over the last two dimensions, broadcast over the rest" behavior is exactly what makes the attention expression `q @ k.transpose(-2, -1)` work across a whole batch of sequences and heads at once. You will meet it in full in module 4 (tensor shapes) and put it to work in module 11 (attention). For now, the takeaway is that everything you learned about the 2-D product is the kernel of the batched version; the extra dimensions just ride along on the outside.

## Check yourself

<details><summary>What shape results from multiplying a $(3, 4)$ matrix by a $(4, 2)$ matrix? What about $(4, 2)$ by $(3, 4)$?</summary>

$(3, 4) \cdot (4, 2)$: the inner numbers ($4$ and $4$) match, so the product exists and has the outer shape $(3, 2)$.

$(4, 2) \cdot (3, 4)$: the inner numbers are $2$ and $3$ — they do not match — so this product is undefined. Order matters.

</details>

<details><summary>For $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $\mathbf{x} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$, which dot product gives the second entry of $A\mathbf{x}$, and what is it?</summary>

The second entry is the dot product of **row 2** of $A$ with $\mathbf{x}$: $\begin{bmatrix} 3 & 4 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 6 \end{bmatrix} = 3 \cdot 5 + 4 \cdot 6 = 15 + 24 = 39$.

</details>

<details><summary>In PyTorch, what is the difference between <code>A @ B</code> and <code>A * B</code>?</summary>

`A @ B` is matrix multiplication: each output entry is a dot product of a row of `A` with a column of `B`, and the inner dimensions must match. `A * B` is element-wise multiplication: it multiplies the numbers in matching positions and requires the shapes to be the same (or broadcastable). They generally give different results and are different operations.

</details>

<details><summary>Why does a Transformer forward pass involve so much shape manipulation?</summary>

Because it is built almost entirely from matrix multiplies, and every matrix multiply requires its inner dimensions to line up. Operations like reshaping $(B, T, C)$ tensors and splitting $C$ into $n_h$ heads of size $d_h$ exist to arrange the tensors so their inner dimensions match before each `@`.

</details>

## Next

You now have the central operation. The remaining pieces of matrix vocabulary — transpose (which lines up shapes for exactly these products), the identity matrix, the inverse, and rank — round out what you need before moving on to general tensors.

Continue to [11 · Transpose, identity, inverse, rank](lesson-03.md).
