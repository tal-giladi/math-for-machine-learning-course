# 11 · Transpose, identity, inverse, rank

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-03/lesson-01">09 · Matrices, shapes, and indexing</a> and <a href="#/lessons/module-03/lesson-02">10 · Matrix multiplication</a>. You need the (row, col) index order and the shape rule for matrix products.</p>
<p><strong>You will learn:</strong> the transpose (flipping rows and columns), the identity matrix (the "1" of matrices), the inverse (the "reciprocal" of a matrix) with a fully worked $2\times2$ example, and rank (how many real dimensions a matrix keeps). You will also see the geometric picture of a matrix as a transformation of space.</p>
<p><strong>Why this matters for ML:</strong> transpose lines up shapes for the matrix products in attention and backprop. The identity is the reference point for the inverse and for residual connections. Rank is the entire idea behind LoRA fine-tuning. The geometric view is how you will reason about what a weight matrix actually does.</p>
</div>

## 1. Transpose

### 1.1 Intuition and mathematics

The **transpose** of a matrix flips it over its main diagonal: rows become columns and columns become rows. We write the transpose of $A$ as $A^\top$ (read "A transpose"). The defining rule, entry by entry:

$$
(A^\top)_{ij} = A_{ji}
$$

- $A$ — the original matrix, shape $(m, n)$.
- $A^\top$ — its transpose, shape $(n, m)$: the shape is reversed.
- The rule $(A^\top)_{ij} = A_{ji}$ says the entry at (row $i$, col $j$) of the transpose equals the entry at (row $j$, col $i$) of the original — the two indices swap.

So row 1 of $A$ becomes column 1 of $A^\top$, row 2 becomes column 2, and so on. A $(2, 3)$ matrix transposes into a $(3, 2)$ matrix.

### 1.2 Worked on a 2×3

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \qquad \text{shape } (2, 3)
$$

Take each row of $A$ and stand it up as a column of $A^\top$. Row 1 $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$ becomes the first column; row 2 $\begin{bmatrix} 4 & 5 & 6 \end{bmatrix}$ becomes the second column:

$$
A^\top = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix} \qquad \text{shape } (3, 2)
$$

Check one entry against the rule: $(A^\top)_{31} = A_{13} = 3$. The entry in row 3, column 1 of the transpose is the entry in row 1, column 3 of the original. Correct.

### 1.3 The reversing property

One property of the transpose comes up constantly in backpropagation, so it is worth stating now:

$$
(AB)^\top = B^\top A^\top
$$

Transposing a product reverses the order of the factors. Notice this is also the only thing that keeps the shapes legal: if $A$ is $(m, n)$ and $B$ is $(n, p)$, then $AB$ is $(m, p)$ and $(AB)^\top$ is $(p, m)$. On the right, $B^\top$ is $(p, n)$ and $A^\top$ is $(n, m)$, so $B^\top A^\top$ is $(p, n) \cdot (n, m) \to (p, m)$ — matching. The order-flip is not arbitrary; it is exactly what makes the inner dimensions meet. (The same reversal happens with inverses, coming up in section 3.)

<div class="callout key"><p>Transpose swaps rows and columns: $(A^\top)_{ij} = A_{ji}$, and the shape $(m,n)$ becomes $(n,m)$. Transposing a product reverses it: $(AB)^\top = B^\top A^\top$.</p></div>

## 2. The identity matrix

### 2.1 Intuition and mathematics

Among numbers, $1$ is special: multiplying by it changes nothing. Among matrices, the object that plays this role is the **identity matrix**, written $I$. It is a square matrix with $1$s down the main diagonal (top-left to bottom-right) and $0$s everywhere else:

$$
I_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \qquad
I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

The subscript just names the size; $I_2$ is the $2 \times 2$ identity. Its defining property is that it leaves any vector or matrix unchanged under multiplication:

$$
I\mathbf{x} = \mathbf{x}, \qquad IA = A, \qquad AI = A
$$

(as long as the shapes are compatible for the product).

### 2.2 A quick check

Multiply $I_2$ by $\mathbf{x} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$ using the matrix–vector rule from lesson 10:

$$
I_2 \mathbf{x} = \begin{bmatrix} 1 \cdot 5 + 0 \cdot 6 \\ 0 \cdot 5 + 1 \cdot 6 \end{bmatrix} = \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \mathbf{x}
$$

Each output entry picks out exactly one input entry (the one where the $1$ sits) and zeroes the rest, so the vector comes out untouched. The identity is the "do nothing" transformation, and it is the reference point we need to define the inverse.

## 3. The inverse

### 3.1 Intuition and mathematics

For a nonzero number $a$, its **reciprocal** $a^{-1} = 1/a$ is the number that multiplies back to $1$: $a \cdot a^{-1} = 1$. The matrix analogue is the **inverse**. The inverse of a square matrix $A$, written $A^{-1}$, is the matrix that multiplies back to the identity:

$$
A A^{-1} = I \qquad \text{and} \qquad A^{-1} A = I
$$

If $A$ is a transformation of space (section 5), then $A^{-1}$ is the transformation that undoes it — applying $A$ and then $A^{-1}$ returns every vector to where it started, which is exactly what multiplying to $I$ means.

**When does it exist?** Not every matrix has an inverse. Only **square** matrices can, and even then only if the matrix does not collapse space — if it does not squash two dimensions into one (that is the rank story in section 4). A square matrix with an inverse is called **invertible** or **non-singular**; one without is **singular**. For a $2 \times 2$ matrix there is a clean test and a clean formula, which we do next.

### 3.2 The 2×2 formula, worked fully

For a $2 \times 2$ matrix

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix},
$$

define the **determinant** $\det(A) = ad - bc$ (a single number). The inverse exists exactly when $\det(A) \neq 0$, and then:

$$
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

In words: swap the two diagonal entries $a$ and $d$, negate the off-diagonal entries $b$ and $c$, and divide every entry by the determinant. If $\det(A) = 0$ you would be dividing by zero — which is precisely why a zero determinant means "no inverse."

Let us do a concrete one. Take:

$$
A = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}
$$

**Step 1 — determinant.** Here $a=4,\, b=7,\, c=2,\, d=6$:

$$
\det(A) = ad - bc = 4 \cdot 6 - 7 \cdot 2 = 24 - 14 = 10
$$

It is nonzero, so $A$ is invertible.

**Step 2 — rearrange.** Swap $a$ and $d$, negate $b$ and $c$:

$$
\begin{bmatrix} d & -b \\ -c & a \end{bmatrix} = \begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix}
$$

**Step 3 — divide by the determinant** ($10$):

$$
A^{-1} = \frac{1}{10} \begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix} = \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix}
$$

**Step 4 — verify** by multiplying $A A^{-1}$ and checking we get $I$. Using the matrix–matrix rule from lesson 10:

$$
\begin{aligned}
(A A^{-1})_{11} &= 4 \cdot 0.6 + 7 \cdot (-0.2) = 2.4 - 1.4 = 1 \\
(A A^{-1})_{12} &= 4 \cdot (-0.7) + 7 \cdot 0.4 = -2.8 + 2.8 = 0 \\
(A A^{-1})_{21} &= 2 \cdot 0.6 + 6 \cdot (-0.2) = 1.2 - 1.2 = 0 \\
(A A^{-1})_{22} &= 2 \cdot (-0.7) + 6 \cdot 0.4 = -1.4 + 2.4 = 1
\end{aligned}
$$

$$
A A^{-1} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I_2 \quad\checkmark
$$

The product is exactly the identity, so our inverse is correct.

### 3.3 Why we rarely invert matrices in deep learning

You will almost never see a matrix inverse in neural-network training code, and it is worth understanding why, because it says something about how numerical computing actually works.

- **Cost.** Inverting an $n \times n$ matrix costs on the order of $n^3$ operations. Weight matrices in GPT-2 have dimensions in the hundreds and thousands; inverting them would be enormously expensive and is almost never needed.
- **Stability.** Computing an explicit inverse and then multiplying by it amplifies floating-point error. When a matrix is close to singular (determinant near zero), the inverse contains huge numbers and the result is numerically unreliable — a theme you will meet again in module 12 on numerical issues.
- **You usually do not need the inverse itself.** In the rare cases the math calls for $A^{-1}\mathbf{b}$ (as in classical least-squares), what you actually want is the *vector* $\mathbf{x}$ solving $A\mathbf{x} = \mathbf{b}$. Solving that system directly (`torch.linalg.solve`) is cheaper and more stable than forming $A^{-1}$ and multiplying. The rule of thumb across scientific computing: **solve, don't invert.**

More fundamentally, training does not undo transformations — it *learns* them by gradient descent. The inverse is a beautiful idea and the right mental model for "undoing a matrix," but in practice it stays mostly in the theory.

<div class="callout warn"><p>You now know how to invert a $2\times2$ by hand, but in deep learning you rarely should. Inversion is costly ($n^3$) and numerically fragile; when you need $A^{-1}\mathbf{b}$, solve the system $A\mathbf{x}=\mathbf{b}$ instead. Solve, don't invert.</p></div>

## 4. Rank

### 4.1 Intuition

The **rank** of a matrix is the number of genuinely independent rows — equivalently, the number of genuinely independent columns (remarkably, these two counts are always equal). "Independent" means a row (or column) that cannot be built by scaling and adding the others; it carries information the others do not.

The picture to hold: rank is **how many real dimensions the matrix keeps** when it transforms space. A matrix takes input vectors and spits out output vectors (section 5). If its rank is full, the outputs can point in as many independent directions as the inputs did — no dimension is lost. If its rank is low, the matrix squashes its inputs into a smaller number of dimensions, collapsing some directions to nothing.

### 4.2 A tiny example

Consider:

$$
M = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
$$

Look at the rows: $\begin{bmatrix} 1 & 2 \end{bmatrix}$ and $\begin{bmatrix} 2 & 4 \end{bmatrix}$. The second row is exactly $2$ times the first — it adds no new direction. So although there are two rows, only *one* is independent: the rank is $1$, not $2$. Its determinant confirms the collapse: $\det(M) = 1 \cdot 4 - 2 \cdot 2 = 0$, so $M$ is singular and has no inverse. A $2\times2$ matrix that "should" span a 2-D plane but only reaches a 1-D line is **rank-deficient**.

By contrast, the matrix $A = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}$ from section 3 had nonzero determinant; its two rows point in genuinely different directions, so its rank is $2$. A square matrix is invertible exactly when it is **full rank** — when it keeps all its dimensions and collapses nothing.

### 4.3 Full-rank vs low-rank, and the seed of LoRA

- **Full rank:** rank equals the smaller of the two shape numbers, $\min(m, n)$. The matrix loses no dimensions it could keep.
- **Low rank:** rank is much smaller than $\min(m, n)$. Such a matrix, even if it is physically large (many rows and columns), carries only a little independent information — it can be reconstructed from a few independent directions.

This last idea is the entire foundation of **LoRA** (Low-Rank Adaptation), the parameter-efficient fine-tuning method you will study in module 12. The insight LoRA exploits is that the *update* you need to make to a giant pretrained weight matrix $\mathbf{W}$ during fine-tuning is often *low-rank* — it does not need all $\min(m, n)$ independent directions, just a handful. So instead of learning a full update matrix, LoRA writes the update as a product of two thin matrices $B A$ (with a small inner dimension $r$, the rank), giving the new weight $\mathbf{W}' = \mathbf{W} + BA$. Because $r$ is tiny, $B$ and $A$ together have far fewer numbers than a full-size update, so fine-tuning becomes dramatically cheaper. Flag this in your memory now: **rank is the quantity LoRA turns down to save parameters.** You have just met the reason it works.

<div class="callout key"><p>Rank = number of independent rows (= independent columns) = how many dimensions the matrix keeps. Full rank keeps everything and is invertible; low rank collapses dimensions. Low-rank updates are the mathematical seed of LoRA fine-tuning.</p></div>

## 5. The linear-transformation view

Everything above has a single unifying picture: **a matrix is a transformation of space.** Multiplying a vector $\mathbf{x}$ by a matrix $A$ moves it to a new vector $A\mathbf{x}$. Do this to every point in space at once and $A$ rotates, scales, shears, or flattens the whole space in one consistent way. Straight lines stay straight and the origin stays put — this well-behaved kind of transformation is what "linear" means.

The most useful fact in this picture: **the columns of $A$ are exactly where the basis vectors land.** Start with the standard axis vectors $\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (pointing along the x-axis) and $\mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ (along the y-axis). Multiply $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ by each:

$$
A\mathbf{e}_1 = \begin{bmatrix} 1 \\ 3 \end{bmatrix} \quad (\text{column 1 of } A), \qquad
A\mathbf{e}_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix} \quad (\text{column 2 of } A)
$$

The first axis vector lands on the first column of $A$; the second lands on the second column. That is why we sometimes read a matrix as a "list of column vectors" (lesson 09): each column tells you where one input axis is sent, and once you know where the axes go, the transformation of every other vector is fixed. This also makes the rank picture visual: if both columns point in different directions, the two axes spread out to fill the plane (full rank); if the two columns are parallel — as in the rank-1 $M$ above — both axes collapse onto the same line, and the plane is flattened. A weight matrix in a neural network is, geometrically, one of these transformations, learned so that the space it produces makes the network's job easier.

## 6. In PyTorch

```python
import torch

A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

# --- transpose ---
print(A.T)                 # tensor([[1., 4.],
                           #         [2., 5.],
                           #         [3., 6.]])   -> shape (3, 2)
print(A.transpose(0, 1))   # same result: swap axis 0 and axis 1

# --- identity ---
print(torch.eye(3))        # tensor([[1., 0., 0.],
                           #         [0., 1., 0.],
                           #         [0., 0., 1.]])
```

`A.T` is the transpose; `A.transpose(0, 1)` does the same by naming the two axes to swap (that explicit form is what you use on higher-dimensional tensors, e.g. `k.transpose(-2, -1)` in attention). `torch.eye(n)` builds the $n \times n$ identity.

```python
# --- inverse (matches the hand calculation in section 3.2) ---
B = torch.tensor([[4., 7.],
                  [2., 6.]])
print(torch.linalg.inv(B))     # tensor([[ 0.6000, -0.7000],
                               #         [-0.2000,  0.4000]])
print(B @ torch.linalg.inv(B)) # tensor([[1., 0.],
                               #         [0., 1.]])   -> the identity, as we proved
```

The inverse PyTorch returns is exactly the $\begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix}$ we computed by hand, and multiplying it back gives $I$. (Note `inv` requires floating-point input — hence the `.` on every number — which hints at the numerical fragility discussed in section 3.3.)

```python
# --- rank ---
full = torch.tensor([[4., 7.],
                     [2., 6.]])
low  = torch.tensor([[1., 2.],
                     [2., 4.]])
print(torch.linalg.matrix_rank(full))   # tensor(2)   -> full rank, invertible
print(torch.linalg.matrix_rank(low))    # tensor(1)   -> row 2 = 2 * row 1, collapses
```

`torch.linalg.matrix_rank` confirms our hand analysis: the first matrix keeps both dimensions (rank 2), the second collapses to one (rank 1).

<div class="callout pt"><p><code>A.T</code> / <code>A.transpose(0,1)</code> transpose; <code>torch.eye(n)</code> is the identity; <code>torch.linalg.inv</code> inverts (float input, rarely needed in training); <code>torch.linalg.matrix_rank</code> reports the rank. In real model code you will see <code>.transpose(-2, -1)</code> far more than <code>inv</code>.</p></div>

## Check yourself

<details><summary>What is the transpose of $\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$, and what shape does it have?</summary>

$\begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$, shape $(3, 2)$. Each row of the original becomes a column of the transpose.

</details>

<details><summary>Find the inverse of $\begin{bmatrix} 2 & 0 \\ 0 & 4 \end{bmatrix}$.</summary>

Determinant $= 2 \cdot 4 - 0 \cdot 0 = 8$. Applying the formula: swap the diagonal ($4, 2$), negate the off-diagonal (still $0, 0$), divide by $8$: $\begin{bmatrix} 4/8 & 0 \\ 0 & 2/8 \end{bmatrix} = \begin{bmatrix} 0.5 & 0 \\ 0 & 0.25 \end{bmatrix}$. (For a diagonal matrix the inverse just reciprocates each diagonal entry, which matches.)

</details>

<details><summary>What is the rank of $\begin{bmatrix} 3 & 6 \\ 1 & 2 \end{bmatrix}$? Is it invertible?</summary>

Row 1 is $3$ times row 2, so only one row is independent: rank $1$. The determinant is $3 \cdot 2 - 6 \cdot 1 = 0$, confirming it is singular — **not** invertible.

</details>

<details><summary>Why is LoRA called a "low-rank" method?</summary>

Because it approximates the fine-tuning update to a large weight matrix as a product of two thin matrices $BA$ whose inner dimension (the rank $r$) is small. A low-rank update carries far fewer independent numbers than a full update, so it needs far fewer trainable parameters while still capturing the directions that matter.

</details>

## Next

You now command the core vocabulary of linear algebra: shapes and indexing, the matrix product, and transpose/identity/inverse/rank. The next module generalizes matrices to **tensors** of any number of dimensions — the $(B, T, C)$ and $(B, T, n_h, d_h)$ objects that fill real Transformer code — and shows why that code constantly reshapes them.

Continue to [12 · From scalar to N-D tensor](lessons/module-04/lesson-01.md).
