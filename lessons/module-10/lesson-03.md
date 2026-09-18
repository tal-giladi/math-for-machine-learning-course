# 31 · Embeddings

<div class="prereq">
<p><strong>Prerequisites:</strong> matrix–vector multiplication from <a href="../module-03/lesson-02.md">10 · Matrix multiplication</a>; the $(V, C)$ weight convention and the LM head from <a href="lesson-02.md">30 · Softmax and language-modeling math</a>; how a selected element passes gradient (backprop) from <a href="../module-08/lesson-03.md">25 · Backpropagation and autograd</a>.</p>
<p><strong>You will learn:</strong> the mathematics of an embedding lookup — a token ID indexing into an embedding matrix $\mathbf{E}$ of shape $(V, C)$; the one-hot interpretation that reveals the lookup to be a matrix multiplication; why gradients through a lookup are <strong>sparse</strong> (only the selected rows get a gradient); GPT-2's token embedding <code>wte</code> plus positional embedding <code>wpe</code>; the difference between a static embedding, a contextual representation, and a final hidden state; and weight tying with the LM head.</p>
<p><strong>Why this matters for ML:</strong> embeddings are how discrete tokens enter a network that only does arithmetic on real numbers. The embedding matrix is usually the single largest parameter tensor in a GPT-2 model, and understanding that a lookup is a differentiable matmul is what makes its gradient behavior — and weight tying — make sense.</p>
</div>

## 1. The problem: tokens are integers, networks need vectors

A tokenizer turns text into a sequence of integer IDs — "cat" might be token $0$, "dog" token $1$, "fish" token $2$. But a neural network multiplies matrices and adds biases; it cannot do arithmetic on "the integer $1$" as if the number itself were meaningful. Token $1$ is not "one more than" token $0$ in any useful sense; the IDs are arbitrary labels.

So the first thing every language model does is convert each integer ID into a **vector** of $C$ learnable numbers — its **embedding**. Now "cat" is not the integer $0$ but a point $[e_{0,1}, \dots, e_{0,C}]$ in a $C$-dimensional space, and the network can measure distances, add, project, and learn relationships between tokens geometrically.

## 2. The embedding matrix and the lookup

### 2.1 The mathematics

All the token vectors are stacked into one **embedding matrix** $\mathbf{E}$ of shape $(V, C)$:

$$
\mathbf{E} \in \mathbb{R}^{V \times C}, \qquad \text{row } t = \text{the embedding of token } t.
$$

- $V$ — vocabulary size (number of distinct tokens).
- $C$ — embedding dimension (the model width $d_\text{model}$); each token's vector has $C$ components.
- $\mathbf{E}[t]$ — row $t$, the $C$-vector for token $t$.

The **embedding lookup** for a token ID $t$ is simply "return row $t$":

$$
\text{embed}(t) = \mathbf{E}[t].
$$

That is it. No multiplication in the obvious sense — you index into a table. For a whole sequence of IDs $[t_0, t_1, \dots, t_{T-1}]$ you gather $T$ rows, producing a $(T, C)$ matrix; with a batch, $(B, T, C)$ — the standard GPT input shape.

### 2.2 A tiny worked example

$V = 3$, $C = 2$:

```text
E = [[1, 2],      # row 0: "cat"
     [3, 4],      # row 1: "dog"
     [5, 6]]      # shape (V, C) = (3, 2)
```

Looking up token ID $1$ ("dog") returns row $1$:

$$
\text{embed}(1) = \mathbf{E}[1] = [3, 4].
$$

Looking up the sequence $[1, 0]$ ("dog", "cat") returns

$$
\begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}, \quad \text{shape } (T, C) = (2, 2).
$$

## 3. The one-hot interpretation: a lookup is a matmul

### 3.1 The idea

Indexing feels like a different kind of operation from the matrix multiplications everywhere else in the network. But a lookup *is* a matrix multiplication in disguise — and seeing this is the key to understanding its gradient.

Represent token $t$ as a **one-hot vector**: a length-$V$ row that is $1$ in position $t$ and $0$ everywhere else. Write it $\text{onehot}(t)$. Then:

$$
\mathbf{E}[t] = \text{onehot}(t)\,\mathbf{E}.
$$

- $\text{onehot}(t)$ — shape $(1, V)$.
- $\mathbf{E}$ — shape $(V, C)$.
- The product has shape $(1, V) \times (V, C) = (1, C)$ — exactly one embedding row.

Why does this work? Matrix multiplication makes each output entry a dot product of the one-hot row with a column of $\mathbf{E}$. Since the one-hot has a single $1$ at position $t$, that dot product picks out exactly the $t$-th entry of the column — i.e. row $t$ of $\mathbf{E}$. The zeros kill every other row.

### 3.2 The worked example, as a matmul

Token ID $1$ with the $\mathbf{E}$ above. The one-hot for index $1$ is $[0, 1, 0]$:

$$
[0,\ 1,\ 0] \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} = 0\cdot[1,2] + 1\cdot[3,4] + 0\cdot[5,6] = [3, 4].
$$

Identical to the direct lookup $\mathbf{E}[1] = [3, 4]$ ✓. The lookup and the one-hot matmul are the same function; the lookup is just the efficient way to compute it (you never actually build the one-hot or do the multiply — you jump straight to row $t$).

<div class="callout key"><p>An embedding lookup $\mathbf{E}[t]$ equals the matmul $\text{onehot}(t)\,\mathbf{E}$. The one-hot's single $1$ selects row $t$ and its zeros discard every other row. This is why a lookup is a legitimate, differentiable layer and not a special case.</p></div>

## 4. Gradients through the lookup: why they are sparse

### 4.1 The reasoning

Because the lookup is the matmul $\text{onehot}(t)\,\mathbf{E}$, we can differentiate it with the matrix-multiply gradient rule you already know. In a matmul $\mathbf{y} = \mathbf{x}\,\mathbf{E}$, the gradient with respect to $\mathbf{E}$ is $\partial L / \partial \mathbf{E} = \mathbf{x}^\top (\partial L / \partial \mathbf{y})$. Here $\mathbf{x} = \text{onehot}(t)$, so

$$
\frac{\partial L}{\partial \mathbf{E}} = \text{onehot}(t)^\top \, \frac{\partial L}{\partial \mathbf{y}}.
$$

$\text{onehot}(t)^\top$ is a $(V, 1)$ column that is $1$ at row $t$ and $0$ elsewhere. Multiplying it by the upstream gradient $\partial L/\partial \mathbf{y}$ (shape $(1, C)$) places that gradient into **row $t$ only** and leaves every other row of $\partial L/\partial \mathbf{E}$ at exactly zero.

In words: only the embedding rows for tokens that actually appeared in the batch receive a gradient. The vector for a token you did not use this step is untouched — its gradient is a row of zeros, so the optimizer does not move it. This is exactly the phrase "only the selected rows receive gradients," and it is *sparsity*: with a vocabulary of $50257$ but a batch touching only a few thousand distinct tokens, the vast majority of embedding-gradient rows are zero.

### 4.2 A tiny worked gradient

Use $\mathbf{E}$ from section 2.2, look up token $1$, and let the loss be the sum of the embedding's components: $L = \mathbf{E}[1]_0 + \mathbf{E}[1]_1$. Then $\partial L / \partial \mathbf{E}[1] = [1, 1]$ (each component contributes $1$ to the sum), and the gradient for every other row is $[0, 0]$ because those rows never entered $L$. So

$$
\frac{\partial L}{\partial \mathbf{E}} = \begin{bmatrix} 0 & 0 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}.
$$

Only row $1$ — the token we used — is nonzero.

## 5. In PyTorch

`nn.Embedding(V, C)` holds the matrix $\mathbf{E}$ in its `.weight` (shape $(V, C)$) and does the lookup when you call it on integer IDs. The one-hot matmul gives the same result, and the gradient is sparse.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

V, C = 3, 2
emb = nn.Embedding(V, C)
with torch.no_grad():
    emb.weight.copy_(torch.tensor([[1., 2.],
                                   [3., 4.],
                                   [5., 6.]]))

ids = torch.tensor([1])            # token "dog"
print(emb(ids))                    # tensor([[3., 4.]])   the lookup

# same thing as a one-hot matmul
onehot = F.one_hot(ids, num_classes=V).float()   # tensor([[0., 1., 0.]])
print(onehot @ emb.weight)         # tensor([[3., 4.]])   identical

# gradients are sparse: only used rows are nonzero
emb.zero_grad()
loss = emb(torch.tensor([1])).sum()   # L = 3 + 4 = 7
loss.backward()
print(emb.weight.grad)
# tensor([[0., 0.],
#         [1., 1.],
#         [0., 0.]])   only row 1 (the token we used) got a gradient
```

The printed `.weight.grad` matches section 4.2 exactly: row $1$ is $[1, 1]$, the other rows are zero.

<div class="callout pt"><p><code>nn.Embedding</code> is <em>not</em> <code>nn.Linear</code> with one-hot inputs, even though the math is the same. It skips building the one-hot and skips the full matmul, jumping straight to the rows — far faster and far less memory when $V$ is $50000$. Under the hood its backward is a scatter-add, not a dense matmul (section 7).</p></div>

## 6. Embeddings in GPT-2: token + position

### 6.1 Two embedding tables

A raw token embedding says *what* a token is but not *where* it sits. Self-attention (module 11) has no built-in sense of order, so GPT-2 adds a second, learned **positional embedding**. Two tables:

- **`wte`** — word/token embedding, shape $(V, C) = (50257, 768)$ for GPT-2 small. Row $t$ is token $t$'s vector.
- **`wpe`** — positional embedding, shape $(T_\text{max}, C) = (1024, 768)$. Row $p$ is the vector for position $p$ in the sequence.

The Transformer's input for the token at position $p$ with ID $t$ is their **sum**:

$$
\mathbf{x}_p = \mathbf{E}_\text{wte}[t] + \mathbf{E}_\text{wpe}[p].
$$

Both are $C$-vectors, so the sum is a $C$-vector — same shape, now carrying both "which token" and "which position." For a batch this is a $(B, T, C)$ tensor, the input to the first Transformer block.

### 6.2 Three things people call "the vector for a token"

It is easy to conflate three distinct representations that all live in $\mathbb{R}^C$. Keep them separate:

- **Token embedding lookup** — $\mathbf{E}_\text{wte}[t]$. *Static*: it depends only on the token ID, not on the surrounding words. "bank" has one fixed embedding whether it is a river bank or a money bank.
- **Contextual representation** — the vector after one or more Transformer blocks have run. Attention has *mixed in* information from the other positions, so "bank" now differs depending on its neighbours. There is one such vector per layer.
- **Final hidden state** — the contextual representation at the very top of the stack, $\mathbf{h}$ from [lesson 30](lesson-02.md). This is the vector fed to the LM head to produce next-token logits.

The journey of a token is: ID $\to$ static embedding (+ position) $\to$ contextual representations through the layers $\to$ final hidden state $\to$ logits $\to$ distribution.

## 7. Weight tying: the embedding is the LM head

Recall from [lesson 30](lesson-02.md) that the LM head weight is shape $(V, C)$ — one $C$-vector per token — and the token embedding $\mathbf{E}_\text{wte}$ is *also* $(V, C)$ — one $C$-vector per token. GPT-2 **ties** them: the same matrix serves as the input embedding (row $t$ read on the way in) and the output projection (row $t$ dotted against the hidden state on the way out, since the LM head computes $\mathbf{h}\,\mathbf{E}_\text{wte}^\top$).

This makes sense: the vector that *means* "cat" and the vector used to *score* "cat" should be the same object. Practically, tying removes a $50257 \times 768 \approx 38.6$ million-parameter matrix — a large fraction of GPT-2 small's total — and often improves quality. It also means a gradient reaches $\mathbf{E}_\text{wte}$ from two directions each step: the sparse lookup path (section 4) and the dense LM-head path (every row gets a gradient from the softmax). The row for a token used as input *and* scored as output accumulates both.

## 8. What PyTorch is doing under the hood

**A lookup is a differentiable gather.** In the forward pass, `nn.Embedding` performs a `gather`/`index_select`: it copies rows $t_0, \dots$ out of `.weight` into the output. No one-hot, no matmul — just pointer arithmetic into the table. The one-hot-matmul identity of section 3 is the *mathematical* meaning; the gather is the *implementation*.

**Backward is a scatter-add.** When `backward()` runs, autograd takes the upstream gradient for each output row and *adds* it into the corresponding row of `.weight.grad`, leaving all other rows zero. "Add," not "assign," matters: if the same token ID appears several times in the batch, each occurrence contributes a gradient and they **accumulate** into that one row — the scatter-add sums them. This is the concrete mechanism behind "only selected rows receive gradients," and it is why `nn.Embedding` can declare `sparse=True` to store the gradient as a sparse tensor and let optimizers update only the touched rows.

**It is a leaf parameter like any other.** `emb.weight` has `requires_grad=True`, appears in `model.parameters()`, and is updated by the optimizer exactly like a linear layer's weight. The only thing special is the *shape* of its gradient — mostly zeros — and the efficient gather/scatter kernels that avoid ever forming the one-hot. Everything you learned about `requires_grad`, `grad_fn`, `.grad`, and `zero_grad()` in [lesson 25](../module-08/lesson-03.md) applies unchanged.

## Check yourself

<details><summary>An embedding matrix is $(V, C) = (50257, 768)$. You look up a batch of shape $(B, T) = (4, 16)$ token IDs. What shape comes out?</summary>

$(B, T, C) = (4, 16, 768)$. Each of the $4 \times 16 = 64$ IDs is replaced by its $768$-dimensional row; the ID axes are preserved and a new feature axis of size $C$ is appended.

</details>

<details><summary>Show that looking up token 2 in $\mathbf{E} = [[1,2],[3,4],[5,6]]$ via a one-hot equals the direct row.</summary>

$\text{onehot}(2) = [0, 0, 1]$, and $[0,0,1]\,\mathbf{E} = 0\cdot[1,2] + 0\cdot[3,4] + 1\cdot[5,6] = [5, 6]$, which is exactly $\mathbf{E}[2]$.

</details>

<details><summary>In one training step your batch uses token IDs $\{5, 5, 9\}$ out of a vocabulary of $50257$. Which rows of the embedding gradient are nonzero, and what is special about row 5?</summary>

Only rows $5$ and $9$ are nonzero; the other $50255$ rows are exactly zero (sparse gradient). Row $5$ received contributions from *both* occurrences of token $5$, scatter-added together, so its gradient is the sum of the two upstream gradients.

</details>

<details><summary>Why is a token's static embedding not the same as its final hidden state?</summary>

The static embedding $\mathbf{E}_\text{wte}[t]$ depends only on the token ID. The final hidden state is that embedding after every Transformer block has mixed in information from the surrounding tokens via attention, so it is *contextual* — the same word in different sentences produces different final hidden states, and that hidden state (not the static embedding) is what the LM head turns into next-token logits.

</details>

## Next

You have now traced a token all the way in (ID $\to$ embedding $+$ position) and, from the previous two lessons, all the way out (hidden state $\to$ logits $\to$ softmax $\to$ distribution $\to$ loss). The one black box remaining between them is what turns a static embedding into a contextual representation: **attention**. The next module builds attention from scratch — queries, keys, values, dot-product scores, scaling, causal masking, and the weighted sum — with tiny matrices worked by hand.

Continue to [../module-11/lesson-01.md](../module-11/lesson-01.md).
