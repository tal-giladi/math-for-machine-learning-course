# 06 · What a vector is

<div class="prereq">
<p><strong>Prerequisites:</strong> Numbers, variables and expressions (<a href="../module-01/lesson-01.md">01 · Numbers and variables</a>), and the notation for subscripts, indices and superscripts (<a href="../module-01/lesson-05.md">05 · Reading mathematical notation</a>).</p>
<p><strong>You will learn:</strong> The three complementary ways to read a vector — an ordered list of numbers, a point in space, and an arrow with direction and magnitude — plus dimension, indexing, and the difference between a mathematical vector, a Python list, and a PyTorch tensor.</p>
<p><strong>Why this matters for ML:</strong> Everything a GPT touches — a token, a word, a hidden state — is a vector. Before you can understand attention or a linear layer, you need to be completely fluent in what a vector <em>is</em> and how PyTorch stores one.</p>
</div>

## 1. Intuition: three pictures of the same thing

A **vector** is one of those words that means slightly different things to a physicist, a mathematician, and a programmer. In machine learning you need all three pictures at once, because different parts of the code lean on different ones. The good news: they are three views of a single object. Let us build them with one running example:

$$
\mathbf{x} = [2, 3], \qquad \mathbf{w} = [4, 5]
$$

We write vectors in **bold** lowercase ($\mathbf{x}$, $\mathbf{w}$) to distinguish them from plain scalars like $x$, which are single numbers. The three ways to read $\mathbf{x} = [2, 3]$ are:

**View 1 — an ordered list of numbers.** A vector is a finite, *ordered* sequence of numbers. "Ordered" is the key word: $[2, 3]$ and $[3, 2]$ are different vectors, exactly the way `arr[0]` and `arr[1]` are different slots in an array. Each number is a **component** (also called an *entry* or *element*). This is the picture a programmer already has: a vector is basically a fixed-length array of numbers.

**View 2 — a point in space.** Take a sheet of graph paper. The vector $[2, 3]$ names the point you reach by going $2$ units along the horizontal axis and $3$ units up the vertical axis. Two numbers → a point on a 2-D plane. Three numbers → a point in 3-D space. This is why we say a dataset of houses, each described by (size, bedrooms, price), "lives in 3-D space": each house is one point.

**View 3 — an arrow with direction and magnitude.** Draw an arrow from the origin $(0,0)$ to that point $[2,3]$. Now the vector has a **direction** (which way the arrow points) and a **magnitude** (how long the arrow is). This is the picture you want when you think about "which way does this push?" — for example, a gradient is an arrow pointing in the direction of steepest increase.

<div class="callout key"><p>The same object $[2,3]$ is simultaneously a list of two numbers, a point on the plane, and an arrow from the origin to that point. Fluency means switching between these views without friction.</p></div>

## 2. The mathematics: components, dimension, indexing

In general a vector in $n$-dimensional space is written

$$
\mathbf{x} = [x_1, x_2, \dots, x_n]
$$

- $\mathbf{x}$ — the vector as a whole (bold).
- $x_i$ — the $i$-th **component**, a single scalar (not bold, italic). The subscript $i$ is the **index**.
- $n$ — the **dimension**: how many components the vector has. Our $\mathbf{x} = [2,3]$ has dimension $n = 2$.

The set of all vectors with $n$ real-number components is written $\mathbb{R}^n$ (read "R-n" or "real coordinate space of dimension $n$"). So $\mathbf{x} \in \mathbb{R}^2$ means "$\mathbf{x}$ is a vector of two real numbers." The symbol $\in$ means "is an element of."

### Indexing: math counts from 1, code counts from 0

This trips people up constantly, so let us be explicit. In mathematics the components are numbered starting at $1$:

$$
\mathbf{x} = [x_1, x_2], \quad x_1 = 2, \quad x_2 = 3
$$

In most programming languages — including Python and therefore PyTorch — indexing starts at $0$:

```text
x[0] = 2
x[1] = 3
```

So the mathematician's $x_1$ is the programmer's `x[0]`. When you translate a formula like $\sum_{i=1}^{n} x_i$ into code, the loop runs `for i in range(n)` and reads `x[i]`, silently shifting the index by one. Keep this offset in the back of your mind; a great many off-by-one bugs in ML code come from forgetting it.

<div class="callout warn"><p>Math is 1-indexed ($x_1$ is the first entry). Python, NumPy and PyTorch are 0-indexed (<code>x[0]</code> is the first entry). Same entry, different label.</p></div>

### Column vectors vs row vectors

On paper, a vector can be laid out as a **column**

$$
\mathbf{x} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}
$$

or as a **row**

$$
\mathbf{x}^\top = \begin{bmatrix} 2 & 3 \end{bmatrix}.
$$

By convention, "a vector" in linear algebra usually means a **column** vector, and the small superscript $\top$ (read "transpose") flips a column into a row and vice versa. We will not need transpose in earnest until matrices, but notice it now: $\mathbf{x}^\top$ is the same numbers, laid on their side. The distinction matters the moment you multiply vectors by matrices, because the shapes have to line up — that is the whole story of the next module.

## 3. A tiny example: naming the parts

Take $\mathbf{w} = [4, 5]$.

- Dimension: $n = 2$, so $\mathbf{w} \in \mathbb{R}^2$.
- Components: $w_1 = 4$ and $w_2 = 5$ (math, 1-indexed); equivalently `w[0] = 4`, `w[1] = 5` (code, 0-indexed).
- As a point: go $4$ right, $5$ up.
- As an arrow: it points up-and-to-the-right, slightly steeper than $\mathbf{x} = [2,3]$ because its rise-over-run $5/4 = 1.25$ is larger than $\mathbf{x}$'s $3/2 = 1.5$… wait — check that: $3/2 = 1.5 > 1.25$, so in fact $\mathbf{x}$ is the steeper arrow. This is exactly the kind of thing the *arrow picture* lets you sanity-check by eye, and we will make "how different are these two directions" precise with the dot product and cosine in the next two lessons.

## 4. Why ML needs vectors: everything is a vector

Here is the punchline that makes this whole module matter. **In a neural network, essentially every quantity you can name is a vector (or a stack of vectors).**

- A **word** or **token** is turned into a vector of numbers called an *embedding*. The model cannot do arithmetic on the string `"cat"`, so `"cat"` becomes something like $[0.12, -0.44, \dots]$.
- A **hidden state** inside GPT — the running representation of a token as it flows through the layers — is a vector. In GPT notation the number of components is called the **channel** or **feature** dimension $C$ (also written $d_\text{model}$). So a single token's hidden state is a vector in $\mathbb{R}^C$. For GPT-2 small, $C = 768$: every token, at every layer, is a point in 768-dimensional space.
- The **weights** of a single neuron form a vector, one weight per input.

When you read the shape `(B, T, C)` in Transformer code, the $C$ at the end is precisely "each position holds a $C$-dimensional vector." Batch $B$ and sequence length $T$ just say *how many* such vectors we are processing at once. Master the vector, and that intimidating 3-D shape becomes "a grid of vectors."

<div class="callout key"><p>A GPT hidden state is a vector in $\mathbb{R}^C$. Understanding a vector as a point/arrow in $C$-dimensional space is literally understanding what a token "is" inside the model.</p></div>

## 5. In PyTorch: mathematical vector vs Python list vs tensor

You will meet three things that all *look* like $[2, 3]$ but behave very differently. Getting the distinctions straight now saves enormous confusion later.

**A mathematical vector** is an abstract object — an element of $\mathbb{R}^n$. It has no "type", no memory layout; it is a pure idea you reason about on paper.

**A Python list** `[2, 3]` is a general-purpose container. It can hold anything (`[2, "cat", 3.0, None]`), each element is a full Python object with its own type, and `+` means *concatenation*, not addition:

```python
a = [2, 3]
b = [4, 5]
print(a + b)   # [2, 3, 4, 5]   <- glued together, NOT [6, 8]
print(2 * a)   # [2, 3, 2, 3]   <- repeated, NOT [4, 6]
```

That is almost never the math you want. Lists are flexible but slow and mathematically wrong for vector arithmetic.

**A PyTorch tensor** is the object you actually compute with. A 1-D tensor *is* the machine's representation of a mathematical vector:

```python
import torch

x = torch.tensor([2., 3.])   # note the dots: these are floats
w = torch.tensor([4., 5.])

print(x)          # tensor([2., 3.])
print(x.shape)    # torch.Size([2])   <- one axis, length 2
print(x.dtype)    # torch.float32
print(x[0])       # tensor(2.)        <- 0-indexed, like all Python
print(x + w)      # tensor([6., 8.])  <- REAL component-wise addition
```

Three things separate a tensor from a Python list:

| | Python list | PyTorch tensor |
|---|---|---|
| element type | any mix of objects | one fixed numeric `dtype` (e.g. `float32`) |
| `+` operator | concatenation | element-wise addition (real vector math) |
| speed | slow Python loop | one vectorized operation on a contiguous buffer |
| lives on | CPU, in Python heap | CPU **or GPU**, and can track gradients |

Two tensor features have no analog in a plain list and will dominate later lessons: a tensor can live on a **device** (`x.to('cuda')` moves it to the GPU) and it can carry **gradient** machinery (`requires_grad=True` tells PyTorch to remember how to differentiate through it). Those two — device and grad — are why deep learning uses tensors and not lists.

<div class="callout warn"><p>Wrote <code>[2., 3.]</code> with decimal points on purpose. <code>torch.tensor([2, 3])</code> infers an <em>integer</em> dtype (<code>int64</code>), and integer tensors cannot hold gradients and silently truncate division. For anything that will be trained, use floats.</p></div>

## 6. Under the hood: what a 1-D tensor actually is

When you write `torch.tensor([2., 3.])`, PyTorch does *not* store two boxed Python numbers. It allocates a small block of raw memory holding the bytes of two 32-bit floating-point numbers, laid out **contiguously** — back to back, in order. Alongside that buffer it stores a little metadata:

- **shape** — here `(2,)`, meaning "one axis of length 2."
- **dtype** — `torch.float32`, so each element is 4 bytes.
- **stride** — how many elements to step in the buffer to move one position along each axis; for this vector it is `(1,)`, i.e. "the next component is the next float in memory."
- **device** — where the buffer physically lives (CPU RAM or a GPU).

So a mathematical vector $\mathbf{x} \in \mathbb{R}^2$ becomes, concretely, *a contiguous run of floats + a shape tag*. The shape is what lets PyTorch know $[2., 3.]$ is a length-2 vector rather than, say, a $2\times 1$ matrix — those hold identical numbers but carry different shape metadata and therefore behave differently under multiplication. Because the data is contiguous and typed, an operation like `x + w` can run as a single tight loop (or a single GPU kernel) over the raw bytes, with no per-element Python overhead. That is the entire performance argument for tensors in one sentence: **one type, packed memory, one operation over the whole buffer.**

We will lean on this "buffer + shape metadata" mental model constantly — reshaping, transposing and viewing tensors (Part 4) are almost all just *relabeling the shape/stride* while leaving that underlying buffer untouched.

## Check yourself

<details><summary>In math, which component of $\mathbf{x}=[2,3]$ is $x_2$, and what is the corresponding PyTorch expression?</summary>

$x_2 = 3$ (math is 1-indexed). In PyTorch it is `x[1]` (code is 0-indexed) — the same entry, labelled one lower.

</details>

<details><summary>Why does <code>[2, 3] + [4, 5]</code> give <code>[2, 3, 4, 5]</code> in plain Python but <code>tensor([6., 8.])</code> in PyTorch?</summary>

For a Python list, `+` means concatenation — glue the two lists end to end. For a PyTorch tensor, `+` is overloaded to mean element-wise (component-wise) addition, which is the real vector-addition operation from mathematics.

</details>

<details><summary>A GPT-2 small hidden state is a vector in $\mathbb{R}^C$ with $C=768$. What does that tell you about one token at one layer?</summary>

At that layer the token is represented by 768 real numbers — a single point (or arrow) in 768-dimensional space. Everything the model "knows" about that token at that moment is encoded in those 768 components.

</details>

## Next

You can now name a vector's parts and store one as a tensor. Next we make vectors *do* something: adding them, scaling them, and the single most important operation in a neural network — the **dot product**.

→ [07 · Adding, scaling, and the dot product](lesson-02.md)
