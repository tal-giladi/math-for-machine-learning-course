# 13 · (B, T, C), reshape, view, permute, broadcast

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-04/lesson-01">12 · From scalar to N-D tensor</a>. You need shape, dimension/rank, axis, size, indexing, and the flat-storage + strides picture from that lesson. A reminder of matrix shapes from <a href="#/lessons/module-03/lesson-01">module 3</a> helps too.</p>
<p><strong>You will learn:</strong> the canonical GPT activation shape $(B, T, C)$ and what one element means; how and why attention splits $C$ into $(n_h, d_h)$; and the six shape operations Transformer code lives on — <em>reshape</em>, <em>view</em>, <em>transpose</em>, <em>permute</em>, <em>broadcasting</em>, and <em>slicing</em> — each with a tiny worked example and its precise rule.</p>
<p><strong>Why this matters for ML:</strong> this is the most important lesson in the module. Real Transformer code reshapes tensors on nearly every line, and if you cannot follow the shapes you cannot read attention. Everything here is reused directly in the attention lessons of module 9.</p>
</div>

## 1. The canonical shape: (B, T, C)

Almost every activation flowing through GPT-2 is a **3-D tensor** with shape $(B, T, C)$. These three letters are the vocabulary of Transformer code, so we define them precisely and keep them for the rest of the course.

- $B$ — the **batch** size: the number of *independent sequences* processed together. Different batch entries never interact; batching just lets the GPU work on many examples at once (lesson 12, section 5).
- $T$ — the **time** or **sequence length**: the number of *token positions* in each sequence. Token $0$ is the first word-piece, token $1$ the next, and so on. (The letter $T$ is for "time," because the positions run left to right like a timeline.)
- $C$ — the **channels** or **features**, also written $d_\text{model}$: the size of the hidden vector that represents *one token* at *one position*. In GPT-2 small, $C = 768$.

Read the shape as nesting from the outside in: a batch of $B$ sequences, each sequence a run of $T$ tokens, each token a vector of $C$ numbers. Let us make it tiny and concrete with

$$
B = 2, \quad T = 4, \quad C = 6 \qquad \Rightarrow \qquad \text{shape } (2, 4, 6).
$$

That is $2 \cdot 4 \cdot 6 = 48$ numbers total. A single element $x[b, t, c]$ names exactly one of them:

<div class="callout key"><p>$x[b, t, c]$ = feature number $c$, of the token at position $t$, in sequence $b$ of the batch. Fix $b$ and $t$ and let $c$ range: you get the length-$C$ hidden vector for one token. That vector is what every layer transforms.</p></div>

So $x[1, 3]$ (leaving $c$ free) is the $C$-vector for the *last* token ($t=3$) of the *second* sequence ($b=1$), and $x[1, 3, 0]$ is that token's first feature. This is precisely the "supply $k$ indices, peel off $k$ dimensions" rule from lesson 12.

## 2. Splitting C into heads: (B, T, n_h, d_h)

Attention does not look at all $C$ features as one undivided block. It cuts the $C$-vector of each token into several equal **heads**, and each head does its own attention over a smaller slice of features. That turns the 3-D $(B, T, C)$ into a 4-D shape:

$$
(B, T, C) \;\longrightarrow\; (B, T, n_h, d_h), \qquad C = n_h \cdot d_h .
$$

- $n_h$ — the number of **heads**.
- $d_h$ — the **head dimension**: how many features each head gets.
- The constraint $C = n_h \cdot d_h$ is what makes the split exact — no features left over.

With our numbers, take $C = 6 = 2 \cdot 3$, i.e. $n_h = 2$ heads of $d_h = 3$ features each. The token vector $\begin{bmatrix} c_0 & c_1 & c_2 & c_3 & c_4 & c_5 \end{bmatrix}$ is cut into head $0 = \begin{bmatrix} c_0 & c_1 & c_2 \end{bmatrix}$ and head $1 = \begin{bmatrix} c_3 & c_4 & c_5 \end{bmatrix}$. The full activation goes from shape $(2, 4, 6)$ to $(2, 4, 2, 3)$ — and crucially the *count of numbers is unchanged*: $2 \cdot 4 \cdot 2 \cdot 3 = 48$, same as $2 \cdot 4 \cdot 6$. Nothing is added or removed; the same 48 numbers are just re-grouped.

**Why bother?** Each head learns to attend to a different kind of relationship (one head might track subject–verb agreement, another nearby punctuation). Giving each head its own $d_h$-slice lets them specialize and run *in parallel*. Section 8 shows the reshape-then-permute dance that exposes the head axis for this parallel computation and then merges it back — the single most common reason Transformer code changes shapes.

## 3. reshape — same data, new shape

**Rule:** `reshape(new_shape)` reinterprets the *same elements* under a new shape. The only requirement is that the total element count matches: the product of `new_shape` must equal the current size (`numel`). It never changes, adds, or drops a number — only the grouping.

Our $(2, 4, 6)$ tensor holds $48$ elements, so it can be reshaped into any shape whose sizes multiply to $48$:

$$
(2, 4, 6) \;\to\; (2, 24) \;\to\; (48,) \;\to\; (8, 6) \;\to\; (2, 4, 2, 3)
$$

because $48 = 2\cdot24 = 48 = 8\cdot6 = 2\cdot4\cdot2\cdot3$. A reshape to $(2, 5, 6)$ would fail: $2\cdot5\cdot6 = 60 \neq 48$. A handy trick is to write $-1$ for *one* axis and let PyTorch solve for it: `reshape(2, -1)` on our tensor fills the $-1$ with $24$, since $48 / 2 = 24$.

Because storage is row-major (lesson 12, section 8), reshape reads the flat block in order and re-cuts it. The split into heads in section 2 is exactly `reshape(2, 4, 2, 3)`: the last axis of $6$ is chopped into $2$ groups of $3$, in order, which is why head $0$ gets features $c_0, c_1, c_2$.

## 4. view — reshape that needs contiguous memory

**Rule:** `view(new_shape)` does the same job as reshape — same element count, new grouping — but with a stricter contract: it *never copies data*. It only rewrites the shape and strides on the existing flat block. That is only possible when the current tensor is laid out compatibly (in practice, when it is **contiguous** — stored in plain row-major order with no axes swapped). If it is not, `view` raises an error instead of silently copying.

On a freshly created tensor everything is contiguous, so `view` and `reshape` behave identically:

```python
x = torch.arange(48).reshape(2, 4, 6)
x.view(2, 24)      # works: (2, 24), shares storage with x
x.view(2, 4, 2, 3) # works: the head split, shares storage
```

The difference only shows up after an operation like `transpose` (next section) leaves the tensor non-contiguous. Then `view` fails and `reshape` succeeds (by copying). Section 9 works that case in full.

<div class="callout warn"><p><code>view</code> = "give me this shape without moving data, or error." <code>reshape</code> = "give me this shape, copying if you must." Both need the element counts to match. Prefer <code>reshape</code> when unsure; use <code>view</code> when you want a guarantee that no copy happened.</p></div>

## 5. transpose(d0, d1) — swap two axes

**Rule:** `transpose(d0, d1)` swaps exactly *two* named axes and leaves all the others in place. The sizes at those two positions trade places in the shape.

On $(2, 4, 6)$, `transpose(1, 2)` swaps axis $1$ (size $4$) with axis $2$ (size $6$):

$$
(2, 4, 6) \xrightarrow{\ \text{transpose}(1,2)\ } (2, 6, 4).
$$

The element that was at $[b, t, c]$ is now found at $[b, c, t]$ — same number, new address. Importantly, transpose does **not** move any data in memory; it just swaps the two strides. That is what makes the result *non-contiguous*: the flat block is still in the old order, but the shape now says to read it in a different one. (This is the setup for the `view` error in section 9.)

Do not confuse it with the reshape from $(2,4,6)$ to $(2,6,4)$ — those have the same shape but different *contents*. Reshape re-cuts the flat block in storage order; transpose genuinely re-routes which axis is which. They give different tensors.

## 6. permute — reorder all axes at once

**Rule:** `permute(order)` is the general axis-reorderer: you list where every axis should go, as a full permutation of $0, 1, \dots, N-1$. `transpose` is just the special case that swaps two of them.

The classic Transformer use is on the 4-D head tensor. Take $(B, T, n_h, d_h) = (2, 4, 2, 3)$ and apply `permute(0, 2, 1, 3)`. Reading the argument: new axis $0$ = old axis $0$, new axis $1$ = old axis $2$, new axis $2$ = old axis $1$, new axis $3$ = old axis $3$. So axes $1$ and $2$ swap:

$$
(2, 4, 2, 3) \xrightarrow{\ \text{permute}(0,2,1,3)\ } (2, 2, 4, 3) \;=\; (B, n_h, T, d_h).
$$

This moves the head axis $n_h$ up next to the batch axis $B$. Now both "independent, parallel" axes ($B$ and $n_h$) sit in front, and the last two axes $(T, d_h)$ form the little matrices attention actually multiplies. Like transpose, permute only reorders strides — no data moves, and the result is generally non-contiguous.

## 7. Broadcasting — stretching without copying

**The problem:** you often want to combine tensors of *different but compatible* shapes — for example, add one bias vector of length $C$ to every token in a $(B, T, C)$ activation. Broadcasting is the rule that makes this work without you manually replicating the small tensor.

**The alignment-from-the-right rule.** Line the two shapes up by their *last* axis and compare pairs, moving leftward. For each pair, they are compatible if they are equal, or one of them is $1$. A missing axis on the shorter shape counts as $1$. A size-$1$ (or missing) axis is *stretched* to match the other. If any pair is neither equal nor has a $1$, broadcasting fails.

**Worked example A — bias across $(B, T, C)$.** Add a length-$C$ vector to a $(B, T, C)$ tensor, with $(2, 4, 6) + (6,)$:

```
    2  4  6      <- x, shape (B, T, C)
          6      <- bias, shape (6,)   [treated as (1, 1, 6)]
   -----------
    2  4  6      <- result
```

Aligning from the right, $6$ matches $C=6$; the two missing axes of the bias count as $1$ and stretch to $2$ and $4$. So the same six bias numbers are added to *every* one of the $2 \cdot 4 = 8$ token vectors. A small concrete version, $(2,3) + (3,)$:

$$
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} + \begin{bmatrix} 10 & 20 & 30 \end{bmatrix}
= \begin{bmatrix} 1{+}10 & 2{+}20 & 3{+}30 \\ 4{+}10 & 5{+}20 & 6{+}30 \end{bmatrix}
= \begin{bmatrix} 11 & 22 & 33 \\ 14 & 25 & 36 \end{bmatrix}.
$$

The vector $\begin{bmatrix} 10 & 20 & 30 \end{bmatrix}$ is reused on both rows — exactly what "stretched across the batch" means.

**Worked example B — $(T, 1) + (1, C)$.** This shape pattern shows both operands stretching at once. Take $(4, 1) + (1, 6)$:

```
    4  1
    1  6
   ------
    4  6
```

Column $vs$ $1 \to$ the first operand's axis-1 (size $1$) stretches to $6$; the second operand's axis-0 (size $1$) stretches to $4$. Result $(4, 6)$, where element $[i, j]$ = first$[i] +$ second$[j]$. With numbers,

$$
\begin{bmatrix} 10 \\ 20 \\ 30 \\ 40 \end{bmatrix} + \begin{bmatrix} 1 & 2 & 3 & 4 & 5 & 6 \end{bmatrix}
$$

gives a $(4, 6)$ grid whose top-left entry is $10 + 1 = 11$ and bottom-right is $40 + 6 = 46$. This "outer" broadcast is how position–feature grids and attention masks get built.

<div class="callout key"><p>Broadcasting rule: align shapes from the right; each pair must be equal or contain a $1$; size-$1$ (or missing) axes stretch to fit. It lets a bias, a mask, or a scale meet a big activation with no manual copying.</p></div>

## 8. Indexing and slicing

Indexing (from lesson 12) peels off dimensions; **slicing** with `:` keeps an axis but selects a range along it. The colon means "all of this axis." Two patterns appear constantly in Transformer code, on our $x$ of shape $(2, 4, 6)$:

- **`x[0]`** — the **first sequence**. One index on axis $0$ peels it off, leaving shape $(4, 6)$: all $4$ tokens, all $6$ features, of batch entry $0$.
- **`x[:, -1, :]`** — the **last token of every sequence**. The `:` on axis $0$ keeps all $B$ sequences; `-1` on axis $1$ picks the last position ($t = 3$, since negative indices count from the end); `:` on axis $2$ keeps all $C$ features. The result has shape $(2, 6)$ — for each of the $2$ sequences, the $6$-feature vector of its final token. This exact slice is how a language model grabs the representation it uses to predict the *next* token.

Slicing, like transpose, does not copy — it produces a view onto the same storage with adjusted shape, strides, and a starting offset.

## 9. In PyTorch

Here is the whole lesson as runnable code, with every shape printed as a comment.

```python
import torch

# --- canonical activation (B, T, C) = (2, 4, 6) ---
x = torch.arange(2 * 4 * 6).reshape(2, 4, 6)
print(x.shape)                 # torch.Size([2, 4, 6])
print(x[1, 3, 0].item())       # 42   -> b*24 + t*6 + c = 1*24 + 3*6 + 0

# --- split C=6 into n_h=2 heads of d_h=3 ---
xh = x.view(2, 4, 2, 3)        # (B, T, n_h, d_h)
print(xh.shape)                # torch.Size([2, 4, 2, 3])

# --- reshape: same 48 numbers, new grouping ---
print(x.reshape(2, 24).shape)  # torch.Size([2, 24])
print(x.reshape(2, -1).shape)  # torch.Size([2, 24])   (-1 solved as 48/2)

# --- transpose two axes: (B, T, C) -> (B, C, T) ---
print(x.transpose(1, 2).shape) # torch.Size([2, 6, 4])

# --- permute: expose the head axis, (B, T, n_h, d_h) -> (B, n_h, T, d_h) ---
print(xh.permute(0, 2, 1, 3).shape)  # torch.Size([2, 2, 4, 3])

# --- broadcasting: add a length-C bias to every token, no copy ---
bias = torch.arange(6)         # shape (6,)
print((x + bias).shape)        # torch.Size([2, 4, 6])

# --- slicing: last token of every sequence ---
print(x[:, -1, :].shape)       # torch.Size([2, 6])
print(x[0].shape)              # torch.Size([4, 6])   first sequence
```

The number `x[1, 3, 0]` is `42`, matching the row-major offset $1\cdot24 + 3\cdot6 + 0 = 42$ from lesson 12 — a good sanity check that you can predict a value from a shape.

<div class="callout pt"><p>The head-attention pattern in real GPT code is: <code>x.view(B, T, n_h, d_h).transpose(1, 2)</code> to get <code>(B, n_h, T, d_h)</code>, run attention on the last two axes for all $B \cdot n_h$ heads at once, then <code>.transpose(1, 2).contiguous().view(B, T, C)</code> to merge the heads back. Every reshape you just learned is in that one line.</p></div>

## 10. Under the hood: view vs reshape, contiguity, and free broadcasts

Everything in this lesson rests on the lesson-12 fact that a tensor is *one flat block of numbers* plus a *shape* and *strides*. Three consequences finish the picture.

**Why `view` can fail where `reshape` succeeds.** `view` refuses to copy: it can only give you the new shape if the new shape can be described by *some* strides over the *existing* flat block, in its current order. A contiguous tensor (plain row-major) can be re-viewed into any compatible shape. But `transpose` and `permute` swap strides without moving data, so the flat block is no longer in the order the new shape would need. Watch it break and then get fixed:

```python
x = torch.arange(48).reshape(2, 4, 6)   # contiguous
y = x.transpose(1, 2)                    # (2, 6, 4), NON-contiguous (strides swapped)

# y.view(2, 24)          # RuntimeError: view size is not compatible
                         #   with input tensor's size and stride ... use .reshape(...)

print(y.reshape(2, 24).shape)            # torch.Size([2, 24])   works: reshape copies
print(y.contiguous().view(2, 24).shape)  # torch.Size([2, 24])   works: copy first, then view
```

`reshape` inspects the tensor: if a no-copy view is possible it returns one, otherwise it silently makes a contiguous copy and views that. `y.contiguous()` does the copy explicitly — it lays the numbers back into row-major order — after which `view` is happy. This is exactly why real attention code writes `.transpose(1, 2).contiguous().view(...)` when merging heads: the transpose left the tensor non-contiguous, so a copy is forced before the final `view`.

**Broadcasting copies nothing either.** When you added the length-$6$ bias to the $(2,4,6)$ activation, PyTorch did *not* build $8$ physical copies of the bias. It set the bias's stride to $0$ along the stretched axes, so every read of a "stretched" position just re-reads the same underlying number. Broadcasting is a reading trick, not a memory expansion — which is why it is cheap even when the stretched-to shape is huge.

**The takeaway.** `reshape`, `view`, `transpose`, `permute`, and slicing are, in the common case, *metadata edits* — they rewrite shape and strides and cost almost nothing. The one time data actually moves is when a copy is forced: a non-contiguous `reshape`, an explicit `.contiguous()`, or writing into a broadcasted result. Knowing which operations are free and which copy is what lets you read Transformer code without fear of the shape churn.

## Check yourself

<details><summary>For $x$ of shape $(B, T, C) = (2, 4, 6)$, what does $x[:, 0, :]$ give, and what is its shape?</summary>

The `:` keeps all $B$ sequences, the `0` selects the first token position ($t=0$), the `:` keeps all $C$ features. Result: the first-token vector of every sequence, shape $(2, 6)$.

</details>

<details><summary>You have a tensor of shape $(2, 4, 6)$ and want $(B, T, n_h, d_h)$ with $n_h = 3$ heads. What is $d_h$, and is the split legal?</summary>

The constraint is $C = n_h \cdot d_h$, so $d_h = C / n_h = 6 / 3 = 2$. It is legal because $6$ divides evenly by $3$. The new shape is $(2, 4, 3, 2)$, still $48$ elements.

</details>

<details><summary>Do the shapes $(2, 4, 6)$ and $(4, 6)$ broadcast? What about $(2, 4, 6)$ and $(4, 1)$?</summary>

$(2,4,6)$ and $(4,6)$: align from the right — $6$ vs $6$ ✓, $4$ vs $4$ ✓, and the missing leading axis counts as $1$ and stretches to $2$. Yes, result $(2,4,6)$. $(2,4,6)$ and $(4,1)$: from the right, $6$ vs $1$ (stretch to $6$) ✓, $4$ vs $4$ ✓, missing axis $\to 1 \to 2$ ✓. Yes, result $(2,4,6)$.

</details>

<details><summary>Why does <code>x.transpose(1, 2).view(2, 24)</code> raise an error, and what are two ways to fix it?</summary>

`transpose` swaps strides without moving data, so the tensor is non-contiguous and its flat storage is not in the order the new shape needs; `view` refuses to copy, so it errors. Fix it with `.reshape(2, 24)` (copies when necessary) or `.contiguous().view(2, 24)` (copy explicitly, then view).

</details>

## Next

You can now read $(B, T, C)$, split it into heads, and follow every reshape, transpose, permute, broadcast, and slice that Transformer code throws at a tensor. That is the shape machinery attention runs on. The next module returns to functions and composition — how these tensor operations chain together into the layers of a network — which is the bridge to the calculus of training.

Continue to [Module 5 · Functions and composition](lessons/module-05/lesson-01.md).
