# 09 · Matrices, shapes, and indexing

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-02/lesson-01">06 · What a vector is</a> and <a href="#/lessons/module-02/lesson-02">07 · Adding, scaling, and the dot product</a>. You should be comfortable with a vector as a list of numbers, and with adding two vectors and scaling one by a number.</p>
<p><strong>You will learn:</strong> what a matrix is (three ways to see it), how we describe its shape, how to name a single entry with two indices, how to add matrices and multiply them by a scalar, and how all of this looks in PyTorch.</p>
<p><strong>Why this matters for ML:</strong> the weights of every linear layer in GPT-2 are matrices. A batch of token vectors is a matrix. Before you can understand what a linear layer or an attention head <em>does</em>, you need to be fluent in reading and indexing matrices.</p>
</div>

## 1. Intuition: three ways to see a matrix

A **matrix** is a rectangular grid of numbers. That is the whole idea. Here is one:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

It has two horizontal **rows** ($\begin{bmatrix} 1 & 2 \end{bmatrix}$ and $\begin{bmatrix} 3 & 4 \end{bmatrix}$) and two vertical **columns** ($\begin{bmatrix} 1 \\ 3 \end{bmatrix}$ and $\begin{bmatrix} 2 \\ 4 \end{bmatrix}$).

As a programmer, the fastest way in is to think of a matrix as a **2-D array**: `int[,] a = { {1,2}, {3,4} };` in C#, or a list of lists. But in machine learning it pays to hold three mental pictures at once, and to switch between them freely:

1. **A grid of numbers.** Just a table with rows and columns. Good for talking about a single entry.
2. **A list of row vectors.** $A$ is two vectors stacked on top of each other: the top row $\begin{bmatrix} 1 & 2 \end{bmatrix}$ and the bottom row $\begin{bmatrix} 3 & 4 \end{bmatrix}$. This picture is the one to use for a batch of examples: each row is one example.
3. **A list of column vectors.** $A$ is two vectors placed side by side: $\begin{bmatrix} 1 \\ 3 \end{bmatrix}$ and $\begin{bmatrix} 2 \\ 4 \end{bmatrix}$. This picture is the one to use when you think of a matrix as a transformation, because the columns tell you where the input axes get sent (you will see that in lesson 11).

None of these pictures is "the real one." They are the same 2-D array, described from different angles. Part of becoming fluent is being able to say "look at this matrix as a stack of rows" or "look at it as a set of columns" on demand.

<div class="callout key"><p>A matrix is a 2-D array of numbers. See it as a grid, as a stack of row vectors, or as a set of column vectors — whichever makes the current problem easiest.</p></div>

## 2. The mathematics: shape, indices, and the (row, col) convention

### Shape

The **shape** of a matrix is the pair (number of rows, number of columns). We write it as $(m, n)$, where:

- $m$ = the number of rows (how tall the grid is),
- $n$ = the number of columns (how wide the grid is).

The matrix $A$ above is $(2, 2)$: two rows, two columns. A matrix with 2 rows and 3 columns is a $(2, 3)$ matrix and we say it is "2 by 3":

$$
M = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \qquad \text{shape } (2, 3)
$$

**Rows always come first.** This is a convention you must burn into memory, because it never changes: whenever you see a shape, the first number is rows and the second is columns. "A 2×3 matrix" means 2 rows, 3 columns, and never the reverse. This "rows first, then columns" habit is called the **row-major** convention, and PyTorch, NumPy, and C# 2-D arrays all follow it.

A matrix with a single row, like $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$, has shape $(1, 3)$ and is called a **row vector**. A matrix with a single column, like $\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$, has shape $(3, 1)$ and is a **column vector**. So a plain vector is just a matrix that is one cell wide or one cell tall — a useful thing to notice, because it means everything we learn about matrices also applies to vectors.

### Naming a single entry

To point at one number inside the grid we use two subscripts. The entry in **row $i$, column $j$** of matrix $A$ is written:

$$
A_{ij}
$$

- $A$ — the matrix (bold or capital-letter name).
- $i$ — the **row** index: which row, counting from the top.
- $j$ — the **column** index: which column, counting from the left.

The order is **(row, column)**, always. $A_{ij}$ means "row $i$, column $j$" — never "column $i$, row $j$". If you remember one thing from this lesson, remember that the first index walks down the rows and the second index walks across the columns.

For our $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, reading in 1-based math notation:

- $A_{11} = 1$ (row 1, column 1),
- $A_{12} = 2$ (row 1, column 2),
- $A_{21} = 3$ (row 2, column 1),
- $A_{22} = 4$ (row 2, column 2).

### 1-based math versus 0-based code

Here is a trap that catches every programmer. **Mathematical notation is 1-based**: the first row is row 1, and $A_{11}$ is the top-left number. **Code is 0-based**: the first row is row 0, and `A[0, 0]` is the top-left number.

So the same top-left entry is written $A_{11}$ on paper and `A[0, 0]` in PyTorch. The math entry $A_{ij}$ corresponds to the code entry `A[i-1, j-1]`. This is not a deep idea, but it is a constant source of off-by-one confusion, so whenever you translate a formula into code, consciously shift each index down by one.

| Entry (top-left first) | Math (1-based) | PyTorch (0-based) | Value |
|---|---|---|---|
| top-left | $A_{11}$ | `A[0, 0]` | 1 |
| top-right | $A_{12}$ | `A[0, 1]` | 2 |
| bottom-left | $A_{21}$ | `A[1, 0]` | 3 |
| bottom-right | $A_{22}$ | `A[1, 1]` | 4 |

In both worlds the order is the same: **(row, col)**. Only the starting number differs.

<div class="callout warn"><p>Two things that trip people up: the index order is always (row, column), and math counts from 1 while code counts from 0. The paper entry $A_{ij}$ is the code entry <code>A[i-1, j-1]</code>.</p></div>

## 3. A tiny numerical example: addition and scalar multiplication

Two operations on matrices are as simple as they look. Both work entry by entry.

### Matrix addition (component-wise, same shape required)

To add two matrices you add the numbers that sit in the same position. Formally, if $C = A + B$, then for every row $i$ and column $j$:

$$
C_{ij} = A_{ij} + B_{ij}
$$

This only makes sense if the two matrices have **exactly the same shape** — the same number of rows and the same number of columns. If the shapes differ, there is no matching position for some entries, and the addition is undefined. (PyTorch will sometimes stretch a smaller shape to fit via *broadcasting*, which you will meet in module 4; for now, treat "same shape required" as the rule.)

Let us add two $2 \times 2$ matrices:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

Work each of the four positions on its own:

- row 1, col 1: $1 + 5 = 6$
- row 1, col 2: $2 + 6 = 8$
- row 2, col 1: $3 + 7 = 10$
- row 2, col 2: $4 + 8 = 12$

$$
A + B = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}
$$

That is all matrix addition is: line up the two grids and add the overlapping cells.

### Scalar multiplication

To multiply a matrix by a single number (a **scalar**), you multiply *every* entry by that number. If $D = c A$ for a scalar $c$, then:

$$
D_{ij} = c \cdot A_{ij}
$$

With $c = 2$ and the same $A$:

- row 1, col 1: $2 \cdot 1 = 2$
- row 1, col 2: $2 \cdot 2 = 4$
- row 2, col 1: $2 \cdot 3 = 6$
- row 2, col 2: $2 \cdot 4 = 8$

$$
2A = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix}
$$

Notice these two operations never mix rows with columns or touch positions that are not aligned. They are exactly the vector operations from lesson 07, just arranged in a grid instead of a line. The genuinely new and interesting operation — matrix multiplication — is the whole of the next lesson, and it is nothing like these.

## 4. Why ML needs it: weights, batches, and shapes

Here is the payoff. In a neural network, a **weight matrix** $\mathbf{W}$ is the thing that turns a vector of one size into a vector of a different size.

Suppose an input vector $\mathbf{x}$ has $C$ numbers in it (in GPT-2, $C$ — the number of channels or features, also called $d_\text{model}$ — is 768). A linear layer might need to produce an output vector with, say, 3072 numbers (GPT-2's MLP does exactly this). The object that carries out that "768 numbers in, 3072 numbers out" mapping is a weight matrix of shape $(3072, 768)$. You will see in the next lesson *how* the mapping is computed; for now the point is only that **the map is stored as a matrix, and its shape records the input and output sizes**.

That single fact — a matrix maps a vector of one dimension to a vector of another dimension — is why matrices are everywhere in deep learning. Every linear layer is a weight matrix. GPT-2 is, to a first approximation, a large stack of weight matrices with a few nonlinear functions sprinkled between them.

The second reason matrices dominate is **batching**. You rarely push one example through a network at a time; that would waste the parallel hardware. Instead you stack many vectors into a matrix and process them together. If each example is a vector of length $C$ and you have a batch of $B$ examples, you stack them as $B$ rows:

$$
X = \begin{bmatrix} \text{— example 1 —} \\ \text{— example 2 —} \\ \vdots \\ \text{— example } B \text{ —} \end{bmatrix} \qquad \text{shape } (B, C)
$$

This is the "list of row vectors" picture from section 1, and it is the standard layout in PyTorch: the **first axis is the batch**, and each row is one example. When you later read a Transformer tensor of shape $(B, T, C)$ — batch, time/sequence position, channels — you are looking at this same idea with one extra axis for the position of each token in the sequence.

<div class="callout key"><p>Weight matrix: maps a vector of one size to a vector of another size — its shape $(\text{out}, \text{in})$ records both. Batch: many example-vectors stacked as the rows of a matrix, shape $(B, C)$.</p></div>

## 5. In PyTorch

In PyTorch a matrix is a 2-dimensional **tensor**. You build one from a list of lists — each inner list is one row.

```python
import torch

A = torch.tensor([[1, 2],
                  [3, 4]])
B = torch.tensor([[5, 6],
                  [7, 8]])

print(A.shape)      # torch.Size([2, 2])   -> (rows, cols)
print(A.ndim)       # 2                     -> it is 2-dimensional
```

`A.shape` reports `(rows, cols)` — the same rows-first convention as the math. Indexing uses **0-based (row, col)**:

```python
print(A[0, 1])      # tensor(2)   -> row 0, column 1  (== math A_{12})
print(A[1, 0])      # tensor(3)   -> row 1, column 0  (== math A_{21})

print(A[0])         # tensor([1, 2])   -> the whole first row (a row vector)
print(A[:, 0])      # tensor([1, 3])   -> the whole first column (all rows, column 0)
```

Read `A[0]` as "row 0" and `A[:, 0]` as "every row, column 0" — the colon means "all of this axis." That `:` in the first slot is how you pull out a column.

Addition and scalar multiplication are the operators you would expect, and they match the hand results from section 3 exactly:

```python
print(A + B)        # tensor([[ 6,  8],
                    #         [10, 12]])
print(2 * A)        # tensor([[2, 4],
                    #         [6, 8]])
```

If you try to add matrices of genuinely incompatible shapes, PyTorch raises a `RuntimeError` about the sizes not matching — the "same shape required" rule, enforced.

<div class="callout pt"><p><code>A.shape</code> is <code>(rows, cols)</code>. Index as <code>A[row, col]</code>, 0-based. <code>A[i]</code> is row <code>i</code>; <code>A[:, j]</code> is column <code>j</code>. <code>+</code> and <code>*</code> (by a scalar) work entry-by-entry, just like the math.</p></div>

## 6. What PyTorch is doing under the hood

It is worth knowing, once, that a 2-D tensor is not actually stored as a grid in memory. Computer memory is one long line of addresses — a flat 1-D array. PyTorch stores the four numbers of $A$ **contiguously in row-major order**, meaning it lays down row 0 first, then row 1:

```text
storage:   [ 1, 2, 3, 4 ]
            └─row 0─┘ └─row 1─┘
```

To make that flat line behave like a $(2, 2)$ grid, the tensor carries two small pieces of bookkeeping:

- its **shape** `(2, 2)` — how many rows and columns to pretend it has, and
- its **stride** `(2, 1)` — how many storage slots to jump to move one step along each axis.

The stride `(2, 1)` says: to move down one **row**, skip forward 2 slots in storage; to move across one **column**, skip forward 1 slot. So the entry `A[i, j]` lives at storage position $i \cdot 2 + j \cdot 1$. Check it: `A[1, 0]` is at position $1 \cdot 2 + 0 = 2$, and storage slot 2 holds the number 3 — which is indeed $A_{21}$. This matches what we computed by hand.

Why care? Because this shape-plus-stride trick is exactly what lets PyTorch reshape, slice, and transpose tensors *without copying the numbers* — it just hands you a new (shape, stride) view over the same storage. That is the machinery behind `view`, `reshape`, and `transpose`, which are central to Transformer code and which you will study in module 4. For now, simply hold the picture: a matrix is a flat block of numbers plus a shape and a stride that tell you how to read it as a grid.

## Check yourself

<details><summary>What is the shape of $\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$, and what is the entry $A_{23}$?</summary>

The shape is $(2, 3)$ — 2 rows, 3 columns. The entry $A_{23}$ is row 2, column 3, which is $6$. In PyTorch that same number is `A[1, 2]` (shift both indices down by one).

</details>

<details><summary>In PyTorch, what does <code>A[:, 1]</code> give you for <code>A = torch.tensor([[1,2],[3,4]])</code>?</summary>

`tensor([2, 4])` — the second column (column index 1): all rows, column 1. The `:` means "every row."

</details>

<details><summary>Can you add a $(2, 3)$ matrix to a $(3, 2)$ matrix?</summary>

No. Matrix addition is component-wise and requires the *same* shape. $(2, 3)$ and $(3, 2)$ have the same six numbers' worth of entries but different layouts, so positions do not line up and the sum is undefined. (You could add a $(2,3)$ to another $(2,3)$.)

</details>

## Next

You can now read a matrix, name any entry, and combine matrices with addition and scaling. The one operation we deliberately skipped — multiplying matrices together — is the engine of every neural network, and it deserves a lesson of its own.

Continue to [10 · Matrix multiplication](lessons/module-03/lesson-02.md).
