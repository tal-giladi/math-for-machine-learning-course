# 12 · From scalar to N-D tensor

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="../module-03/lesson-01.md">09 · Matrices, shapes, and indexing</a> and <a href="../module-03/lesson-02.md">10 · Matrix multiplication</a>. You need the idea of a matrix as a grid of numbers with a $(m, n)$ shape and $(\text{row}, \text{col})$ indexing.</p>
<p><strong>You will learn:</strong> the "ladder" from a single number up to an arbitrary N-dimensional tensor, and four words used precisely on every rung — <em>shape</em>, <em>dimension/rank</em>, <em>axis</em>, and <em>size</em>. You will index down the ladder ($A[i]$, $A[i,j]$, $A[i,j,k]$) and see how a tensor is stored as one flat 1-D array plus a shape and strides.</p>
<p><strong>Why this matters for ML:</strong> every quantity in GPT-2 — inputs, weights, activations, gradients — is a tensor. If you can read a shape like $(2,3,4)$ and know exactly how many numbers it holds and how to reach any one of them, the rest of the course (attention, backprop, optimizers) is bookkeeping you can follow.</p>
</div>

## 1. Intuition: one idea, many shapes

You already know two kinds of number-container from the last two modules: a **vector** (a 1-D list of numbers, module 2) and a **matrix** (a 2-D grid, module 3). A **tensor** is the single word that covers both of those *and* everything above and below them. The one-sentence definition to memorize:

<div class="callout key"><p>A tensor is just an N-dimensional array of numbers. A scalar, a vector, and a matrix are the N = 0, 1, and 2 cases of the exact same thing.</p></div>

If you come from C#, the mental model is nested arrays: a scalar is `float`, a vector is `float[]`, a matrix is `float[][]`, a 3-D tensor is `float[][][]`, and so on. PyTorch does not actually nest arrays like that (section 8 shows what it really does), but the *shape* of your thinking is right: adding a dimension means wrapping the previous thing in one more layer of "a list of...".

Let us climb the ladder one rung at a time.

## 2. The ladder

### 2.1 Scalar — 0-D, shape `()`

A **scalar** is a single number, standing alone: $3$, $-0.5$, $\pi$. It has **zero** dimensions — there is no index you could supply, because there is nothing to choose between. Its shape is the empty tuple `()`, which has length $0$.

In prose we write scalars in italics: $x = 3$. Loss values, learning rates, and single probabilities are scalars.

### 2.2 Vector — 1-D, shape `(n,)`

A **vector** is a 1-D list of $n$ numbers laid in a line. One index picks out one number. We met these in module 2:

$$
\mathbf{v} = \begin{bmatrix} 5 & 2 & 8 \end{bmatrix} \qquad \text{shape } (3,)
$$

The shape is written `(3,)` — a tuple of length $1$ whose single entry is the number of elements. (The trailing comma is Python's way of writing a one-element tuple; `(3)` would just be the number $3$.) We write vectors in bold: $\mathbf{v}$. A single token's embedding is a vector.

### 2.3 Matrix — 2-D, shape `(m, n)`

A **matrix** is a 2-D grid with $m$ rows and $n$ columns, from module 3. Two indices — row then column — pick out one number:

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \qquad \text{shape } (2, 3)
$$

Two rows, three columns, so shape $(2, 3)$. We write matrices in bold capitals: $\mathbf{A}$. A weight matrix of a linear layer is a matrix.

### 2.4 3-D tensor — shape `(d_0, d_1, d_2)`

Stack several matrices of the same shape and you get a **3-D tensor**: think of a small deck of cards where every card is a matrix. Picture two $(3, 4)$ matrices stacked:

$$
\text{shape } (2, 3, 4) \;=\; 2 \text{ matrices, each } 3 \text{ rows} \times 4 \text{ columns}
$$

Now it takes **three** indices to name one number: which matrix, then which row, then which column. In deep learning the most common 3-D shape is $(B, T, C)$ — batch, time, channels — the star of the next lesson.

### 2.5 4-D tensor — shape `(d_0, d_1, d_2, d_3)`

Stack several 3-D tensors and you get a **4-D tensor**: a list of decks. Four indices name one number. GPT attention uses the 4-D shape $(B, T, n_h, d_h)$ — batch, time, heads, head-dimension — and image models use $(N, C, H, W)$ — images, channels, height, width. You cannot easily *draw* four dimensions, and that is fine: you stop visualizing and start trusting the shape tuple.

### 2.6 Arbitrary N-D

There is nothing special that stops at 4. An **N-dimensional tensor** has a shape that is a tuple of $N$ positive integers, and it takes $N$ indices to reach a single number. The ladder is uniform: each new dimension is one more integer in the shape tuple and one more index you must supply.

<div class="callout key"><p>Rung = number of dimensions. Scalar $0$, vector $1$, matrix $2$, then $3, 4, \dots, N$. Each rung adds one integer to the shape tuple and one required index.</p></div>

## 3. Four words, defined precisely

These four terms are used constantly and often sloppily. Here they are, exactly, for a tensor whose shape is the tuple $(d_0, d_1, \dots, d_{N-1})$.

- **Shape** — the tuple listing the size along each dimension, e.g. $(2, 3, 4)$. It is the single most important fact about a tensor.
- **Dimension** (also called **rank**, or **ndim**) — the *number of axes*, which is the length of the shape tuple: $N = \texttt{len(shape)}$. A matrix has dimension/rank $2$. (Warning: this "rank" — number of axes — is a different meaning from the linear-algebra "rank" of lesson 11, which counted independent rows. In tensor code, rank almost always means number of axes.)
- **Axis** — one particular index *position*, named by its number starting at $0$. In shape $(2, 3, 4)$: axis $0$ has size $2$, axis $1$ has size $3$, axis $2$ has size $4$. "Sum along axis 1" means collapse the middle index.
- **Size** — the *total number of elements*, equal to the product of all the shape entries: $\prod_{i=0}^{N-1} d_i$. For $(2, 3, 4)$ the size is $2 \cdot 3 \cdot 4 = 24$.

The symbol $\prod$ (capital Greek pi) means "multiply these together," just as $\sum$ (from module 1) means "add these together." So $\prod_{i=0}^{N-1} d_i = d_0 \cdot d_1 \cdots d_{N-1}$.

<div class="callout warn"><p>Two different meanings of "rank" collide here. Linear-algebra rank (lesson 11) = number of independent rows. Tensor rank = number of axes = $\texttt{len(shape)}$. This lesson always means the second.</p></div>

## 4. Worked: shapes, element counts, and indexing

Let us make every rung concrete with the same running examples. First, the **shape and element count** of one tensor from each rung:

| Object | Shape | Dimension (rank) | Size (elements) |
|---|---|---|---|
| scalar | `()` | 0 | $1$ (the empty product) |
| vector | `(3,)` | 1 | $3$ |
| matrix | `(2, 3)` | 2 | $2 \cdot 3 = 6$ |
| 3-D tensor | `(2, 3, 4)` | 3 | $2 \cdot 3 \cdot 4 = 24$ |

Two things worth pausing on. The scalar has size $1$: it holds exactly one number, and the product of *no* factors is defined to be $1$ (just as an empty sum is $0$). And the size grows fast — each new dimension multiplies the count, which is why real tensors hold millions of numbers.

Now **indexing down the ladder.** Indexing means supplying integers to pick out a piece; each integer you supply peels off one dimension from the *left* of the shape.

Take the 3-D tensor $A$ with shape $(2, 3, 4)$. Suppose its two $(3,4)$ matrices are

$$
A[0] = \begin{bmatrix} 0 & 1 & 2 & 3 \\ 4 & 5 & 6 & 7 \\ 8 & 9 & 10 & 11 \end{bmatrix}, \qquad
A[1] = \begin{bmatrix} 12 & 13 & 14 & 15 \\ 16 & 17 & 18 & 19 \\ 20 & 21 & 22 & 23 \end{bmatrix}
$$

(the numbers $0$ through $23$ laid out in order — we will see in section 8 that this is exactly how PyTorch stores them).

- **One index, $A[i]$.** $A[1]$ selects the second matrix. You supplied one integer, so one dimension (axis $0$) is gone; what remains has shape $(3, 4)$ — a matrix.
- **Two indices, $A[i, j]$.** $A[1, 2]$ selects matrix $1$, then row $2$ of it: the vector $\begin{bmatrix} 20 & 21 & 22 & 23 \end{bmatrix}$, shape $(4,)$. Two integers peeled off two dimensions.
- **Three indices, $A[i, j, k]$.** $A[1, 2, 3]$ selects matrix $1$, row $2$, column $3$: the single number $23$, shape `()` — a scalar. All three dimensions are gone.

The pattern is exact: on an $N$-dimensional tensor, supplying $k$ indices leaves a tensor of dimension $N - k$. Supplying all $N$ gives you one scalar.

## 5. Why ML needs the whole ladder

Every object in a neural network lives somewhere on this ladder, and knowing the rung tells you what the numbers mean:

- **Scalar** — the training **loss** $L$, the learning rate $\eta$, a single probability.
- **Vector** — one token's **embedding**, a **bias** $\mathbf{b}$, the logits for one position.
- **Matrix** — a **weight** matrix $\mathbf{W}$ of a linear layer; a batch of vectors.
- **3-D and 4-D** — batches of sequences of vectors, the shapes that flow through a Transformer.

One axis you will see everywhere deserves an early name. The **batch axis** (almost always axis $0$) counts *independent examples processed together*. Instead of pushing one sentence through GPT-2 at a time, you stack, say, $B = 32$ sentences and process them in parallel; the batch axis is how they ride along without interfering. Two more axes you will meet in full next lesson: the **sequence** (or **time**) axis counts token positions within one example, and the **feature** (or **channel**) axis counts the numbers that describe one position. Hold those three names — batch, sequence, feature — they are the $(B, T, C)$ of lesson 13.

## 6. In PyTorch

PyTorch's core object *is* the tensor, so the whole ladder is one class, `torch.Tensor`. Here is the full tour — creating tensors at several rungs and asking each one about the four words from section 3.

```python
import torch

# --- climb the ladder ---
s = torch.tensor(3.)             # scalar, 0-D
v = torch.tensor([5., 2., 8.])   # vector, 1-D, shape (3,)
M = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])  # matrix, 2-D, shape (2, 3)
A = torch.zeros(2, 3, 4)          # 3-D tensor of zeros, shape (2, 3, 4)
```

Now interrogate them. Every tensor answers the same four questions:

```python
print(s.shape, s.ndim, s.numel())   # torch.Size([])      0   1
print(v.shape, v.ndim, v.numel())   # torch.Size([3])     1   3
print(M.shape, M.ndim, M.numel())   # torch.Size([2, 3])  2   6
print(A.shape, A.ndim, A.numel())   # torch.Size([2, 3, 4]) 3  24
```

- `.shape` is the shape tuple (printed as `torch.Size([...])`, which behaves like a tuple).
- `.ndim` is the dimension/rank — the length of the shape, i.e. how many axes.
- `.numel()` is the size — the *num*ber of *el*ements, the product of the shape. Note `s.numel()` is `1`, matching the "empty product" from section 4.

Every tensor also has a **dtype** — the numeric type of the elements it stores:

```python
print(s.dtype)                   # torch.float32   (the default for float literals)
print(torch.zeros(2, 3, 4).dtype)  # torch.float32
print(torch.tensor([1, 2, 3]).dtype)  # torch.int64  (integer literals -> integer tensor)
```

`float32` (32-bit floating point) is the default and the workhorse of training; you saw in lesson 11 why float is needed, and module 12 covers precision in depth. Indexing works exactly as in section 4:

```python
B = torch.arange(24).reshape(2, 3, 4)  # 0..23 laid into shape (2, 3, 4)
print(B[1].shape)        # torch.Size([3, 4])   one index  -> matrix
print(B[1, 2].shape)     # torch.Size([4])      two indices -> vector
print(B[1, 2, 3])        # tensor(23)           three indices -> scalar
print(B[1, 2, 3].item()) # 23                   .item() pulls the Python number out
```

The printed values match the hand example in section 4 exactly: `B[1, 2, 3]` is `23`. `torch.arange(24)` makes the vector $0, 1, \dots, 23$, and `.reshape(2, 3, 4)` folds it into the 3-D shape (the subject of the next lesson).

<div class="callout pt"><p><code>.shape</code> (the shape tuple), <code>.ndim</code> (number of axes = rank), <code>.numel()</code> (total elements = size), and <code>.dtype</code> (element type) are the four questions you will ask a tensor thousands of times. When shapes go wrong in your own code, print all four first.</p></div>

## 7. From modules 2–3 to here

Nothing you learned before is lost — it is absorbed. A dot product, matrix–vector product, and matrix–matrix product are all special cases of tensor operations. The vector operations of module 2 act on 1-D tensors; the matrix operations of module 3 act on 2-D tensors; the same functions in PyTorch simply accept more dimensions and apply the rule along the last one or two axes. When lesson 13 broadcasts a bias vector across a $(B, T, C)$ activation, it is reusing module 2's vectors and module 3's grids inside a bigger box. The ladder is the unifying structure the whole course has been building toward.

## 8. What PyTorch is doing under the hood

Here is the fact that makes tensors click, and it is *not* nested arrays. A PyTorch tensor is **one flat, contiguous 1-D block of numbers in memory**, plus two small pieces of metadata that tell PyTorch how to *interpret* that flat block as an N-D grid:

1. the **shape** — the size along each axis;
2. the **strides** — how many elements to step forward in the flat block to move one step along each axis.

The elements are stored in **row-major order** (also called C-order): the last axis varies fastest. For our $(2, 3, 4)$ tensor, that is exactly why the numbers $0..23$ landed the way they did — you walk the last axis (columns) first, then the middle (rows), then the first (which matrix).

**The linear-index formula.** To find where element $[i, j]$ of a $(2, 3)$ matrix physically sits in the flat block, you compute a single offset. For row-major $(2, 3)$:

$$
\text{offset} = i \cdot 3 + j
$$

The $3$ here is the **stride of axis 0**: moving one row forward ($i \to i+1$) skips $3$ elements, because each row has $3$ columns. The stride of axis 1 is $1$: moving one column forward is one step in memory. Let us work a full example on

$$
M = \begin{bmatrix} 10 & 11 & 12 \\ 13 & 14 & 15 \end{bmatrix} \qquad \text{shape } (2, 3),
$$

whose flat storage is $[\,10, 11, 12, 13, 14, 15\,]$ (positions $0$ through $5$):

- $M[0, 0]$: offset $= 0 \cdot 3 + 0 = 0$ → flat position $0$ → value $10$. ✓
- $M[0, 2]$: offset $= 0 \cdot 3 + 2 = 2$ → position $2$ → value $12$. ✓
- $M[1, 0]$: offset $= 1 \cdot 3 + 0 = 3$ → position $3$ → value $13$. ✓
- $M[1, 2]$: offset $= 1 \cdot 3 + 2 = 5$ → position $5$ → value $15$. ✓

Every index you write is silently turned into one of these offsets. The general row-major rule for shape $(d_0, d_1, \dots, d_{N-1})$ is: the stride of an axis is the product of all the sizes to its right, and

$$
\text{offset} = \sum_{a=0}^{N-1} i_a \cdot \text{stride}_a .
$$

You can see the strides directly in PyTorch:

```python
M = torch.arange(10, 16).reshape(2, 3)  # [[10,11,12],[13,14,15]]
print(M.stride())   # (3, 1)   -> step 3 in memory per row, 1 per column
print(M[1, 2].item())  # 15     offset 1*3 + 2 = 5 -> the 6th stored number
```

Why does this matter? Because the reshape/view/transpose operations in the next lesson mostly *do not move the numbers at all* — they only rewrite the shape and strides, which is why they are fast, and also why some of them come with a "contiguity" catch. You now have the mental model to understand that catch when it arrives.

<div class="callout key"><p>A tensor = one flat 1-D array + a shape + strides. Row-major means the last axis varies fastest; the offset of $[i,j]$ in a $(2,3)$ tensor is $i \cdot 3 + j$. Most shape operations rewrite only the metadata, not the data.</p></div>

## Check yourself

<details><summary>A tensor has shape $(4, 1, 5)$. What are its dimension (rank) and its size?</summary>

Dimension/rank $= 3$ (three axes, the length of the shape tuple). Size $= 4 \cdot 1 \cdot 5 = 20$ elements. An axis of size $1$ still counts as an axis.

</details>

<details><summary>On a tensor of shape $(2, 3, 4)$, what shape does $A[0, 1]$ have, and how many numbers does it contain?</summary>

You supplied two indices, peeling off two dimensions, leaving shape $(4,)$ — a vector of $4$ numbers.

</details>

<details><summary>The flat storage of a $(2, 3)$ tensor is $[7, 8, 9, 1, 2, 3]$. Which flat position holds element $[1, 1]$, and what is its value?</summary>

Offset $= i \cdot 3 + j = 1 \cdot 3 + 1 = 4$. Flat position $4$ (counting from $0$) holds $2$. So $M[1, 1] = 2$.

</details>

<details><summary>Why is a scalar's <code>numel()</code> equal to $1$ when its shape <code>()</code> is empty?</summary>

Size is the product of the shape entries, and the product of an empty list of factors is defined to be $1$ (the multiplicative identity), just as an empty sum is $0$. A scalar does hold exactly one number, so this matches.

</details>

## Next

You can now read any shape, count its elements, index into it, and picture how it sits in memory. The next lesson puts this to work on the single most important shape in this whole course — the $(B, T, C)$ activation that flows through every GPT layer — and the reshape, view, transpose, permute, and broadcast operations that Transformer code uses on it constantly.

Continue to [13 · (B, T, C), reshape, view, permute, broadcast](lesson-02.md).
