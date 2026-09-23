# 32 · Attention mathematics

<div class="prereq">
<p><strong>Prerequisites:</strong> matrix multiplication and transpose from <a href="#/lessons/module-03/lesson-01">module 3</a>; the $(B, T, C)$ shape and the reshape/transpose/permute machinery from <a href="#/lessons/module-04/lesson-02">13 · (B, T, C), reshape, view, permute, broadcast</a>; softmax as a map from scores to a probability distribution from <a href="#/lessons/module-10/lesson-01">the probability module</a>; the dot product from <a href="#/lessons/module-02/lesson-02">module 2</a>.</p>
<p><strong>You will learn:</strong> what queries, keys, and values are; how a single attention score is a dot product measuring relatedness; why we scale by $1/\sqrt{d_h}$; how a causal mask forbids looking at the future; how softmax turns scores into weights and the output becomes a weighted average of values — all worked by hand on a $T=3$, $d_h=2$ example with exact numbers. Then how <strong>multi-head</strong> attention splits the channel dimension, runs several attentions in parallel, and concatenates the results.</p>
<p><strong>Why this matters for ML:</strong> attention is the one operation that makes a Transformer a Transformer. It is how each token gathers information from the other tokens. Every GPT-2 layer is attention plus a small feed-forward network, so once you can compute attention by hand and read its PyTorch form line by line, the rest of the architecture is assembly.</p>
</div>

## 1. Intuition: tokens looking things up from each other

A Transformer represents each token position as a vector of $C$ numbers (module 4). By itself that vector only knows about *one* token. To predict the next word, a position needs to pull in information from the other positions — the subject a verb agrees with, the noun a pronoun refers to, the opening bracket a closing one must match. **Attention** is the mechanism for that pulling-in.

The cleanest analogy for a programmer is a soft dictionary lookup. In a normal dictionary you look up one exact key and get back its one value. Attention does a *fuzzy, weighted* lookup: every position asks a question, compares it against what every position offers, and gets back a blend of everyone's contents, weighted by how well each one matched.

Three roles make this precise, one vector per role, per position:

- **Query** ($\mathbf{q}$) — *what I am looking for.* The current position's question.
- **Key** ($\mathbf{k}$) — *what I offer.* An advertisement each position publishes describing what it is about.
- **Value** ($\mathbf{v}$) — *what I pass on.* The actual content a position hands over if it is attended to.

A position compares its query to every key to decide *how much* to listen to each other position, then takes that weighted blend of their values. That is the whole idea. The rest of this lesson turns each step into arithmetic.

## 2. The setup: three positions, head dimension two

We will use a sequence of $T = 3$ positions and a head dimension of $d_h = 2$ (two numbers per query/key/value vector). In real code $\mathbf{q}$, $\mathbf{k}$, $\mathbf{v}$ are produced by multiplying the token vectors by learned weight matrices (that is the QKV projection of [the next lesson](lessons/module-11/lesson-02.md)); here we skip that and simply hand you the resulting matrices, so we can focus on attention itself.

$$
Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \qquad
K = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix}, \qquad
V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}.
$$

Each matrix has shape $(T, d_h) = (3, 2)$. Read the rows as positions: row $1$ is the query/key/value of the first token, row $2$ of the second, row $3$ of the third. So the query of position $2$ is $\mathbf{q}_2 = \begin{bmatrix} 0 & 1 \end{bmatrix}$, the key of position $3$ is $\mathbf{k}_3 = \begin{bmatrix} 0 & 1 \end{bmatrix}$, and so on.

<div class="callout key"><p>Attention is defined by one formula, which the rest of the lesson unpacks term by term:</p>
<p>$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}} + M\right) V.$$</p>
<p>Here $QK^\top$ are the raw scores, $\sqrt{d_h}$ is the scale, $M$ is the causal mask, softmax turns each row into weights, and multiplying by $V$ takes the weighted average.</p></div>

## 3. Step one — scores are dot products

The first thing attention computes is, for every pair of positions $(i, j)$, a **score** measuring how much position $i$'s query matches position $j$'s key. That match is a **dot product**: multiply the two vectors component-wise and add up. From module 2, the dot product is large and positive when two vectors point the same way, near zero when they are unrelated (perpendicular), and negative when they point oppositely. So a score is exactly a numerical answer to "how related is my question to what you offer?"

The score for query $i$ against key $j$ is

$$
s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j = \sum_{m=1}^{d_h} q_{im}\, k_{jm}.
$$

Stacking all $T \times T$ of these into a matrix is one matrix multiplication: $S = Q K^\top$. The transpose $K^\top$ turns $K$'s rows (keys) into columns, so that row $i$, column $j$ of the product is exactly $\mathbf{q}_i \cdot \mathbf{k}_j$. The result has shape $(T, T) = (3, 3)$: one score for every (query position, key position) pair.

Let us compute all nine by hand. Recall $\mathbf{q}_1 = [1,0]$, $\mathbf{q}_2 = [0,1]$, $\mathbf{q}_3 = [1,1]$ and $\mathbf{k}_1 = [1,0]$, $\mathbf{k}_2 = [1,1]$, $\mathbf{k}_3 = [0,1]$.

- Row 1 ($\mathbf{q}_1 = [1,0]$): $\;[1,0]\cdot[1,0]=1$, $\;[1,0]\cdot[1,1]=1$, $\;[1,0]\cdot[0,1]=0$.
- Row 2 ($\mathbf{q}_2 = [0,1]$): $\;[0,1]\cdot[1,0]=0$, $\;[0,1]\cdot[1,1]=1$, $\;[0,1]\cdot[0,1]=1$.
- Row 3 ($\mathbf{q}_3 = [1,1]$): $\;[1,1]\cdot[1,0]=1$, $\;[1,1]\cdot[1,1]=2$, $\;[1,1]\cdot[0,1]=1$.

$$
S = Q K^\top = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \\ 1 & 2 & 1 \end{bmatrix}.
$$

The largest entry, $s_{32} = 2$, says position $3$'s query $[1,1]$ aligns most strongly with position $2$'s key $[1,1]$ — they point in exactly the same direction, so the dot product is maximal. That is attention noticing a strong match.

## 4. Step two — scale by $1/\sqrt{d_h}$

We do not feed the raw scores into softmax. We first divide every score by $\sqrt{d_h}$. Here $d_h = 2$, so

$$
\text{scale} = \frac{1}{\sqrt{d_h}} = \frac{1}{\sqrt{2}} \approx 0.7071.
$$

Multiplying $S$ by $0.7071$:

$$
\frac{S}{\sqrt 2} = \begin{bmatrix} 0.7071 & 0.7071 & 0 \\ 0 & 0.7071 & 0.7071 \\ 0.7071 & 1.4142 & 0.7071 \end{bmatrix}.
$$

**Why divide at all?** A dot product of two $d_h$-dimensional vectors is a sum of $d_h$ products. As $d_h$ grows, that sum tends to grow too (more terms piled up), so raw scores in a real model with $d_h = 64$ can be large. Softmax applied to large-magnitude inputs **saturates**: it puts almost all the weight on the single biggest score and near-zero on the rest, which behaves like a hard pick rather than a soft blend. A near-one-hot softmax also has a nearly flat gradient, so learning stalls. Dividing by $\sqrt{d_h}$ keeps the scores at a moderate size regardless of $d_h$, so softmax stays soft and its gradient stays healthy. We will treat softmax saturation and numerical stability carefully in [module 12](lessons/module-12/lesson-01.md); for now, read the scale as "keep the numbers going into softmax reasonable."

## 5. Step three — the causal mask: no peeking at the future

GPT is trained to predict the *next* token. At position $i$ it must produce its output using only positions $1, 2, \dots, i$ — the tokens it has already seen. If it were allowed to attend to positions after $i$, it would be looking at the very future it is supposed to predict, and training would be a cheat that fails at generation time.

We enforce this with a **causal mask** $M$: before softmax, set every "future" score (column $j > i$) to $-\infty$. We keep entries with $j \le i$ and mask entries with $j > i$:

$$
M = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}.
$$

Adding $M$ to the scaled scores leaves the allowed (lower-triangular, including diagonal) entries untouched and drives the forbidden ones to $-\infty$:

$$
\frac{S}{\sqrt 2} + M = \begin{bmatrix} 0.7071 & -\infty & -\infty \\ 0 & 0.7071 & -\infty \\ 0.7071 & 1.4142 & 0.7071 \end{bmatrix}.
$$

So the three rows are, exactly:

- row 1: $[\,0.7071,\ -\infty,\ -\infty\,]$ — position 1 may attend only to itself.
- row 2: $[\,0,\ 0.7071,\ -\infty\,]$ — position 2 may attend to positions 1 and 2.
- row 3: $[\,0.7071,\ 1.4142,\ 0.7071\,]$ — position 3 may attend to all three.

Why $-\infty$ and not, say, $0$? Because the next step is softmax, and $\exp(-\infty) = 0$. A masked score therefore contributes *exactly zero* weight — the future position is not down-weighted, it is switched off entirely.

## 6. Step four — softmax turns each row into weights

Softmax (from the probability module) converts a row of scores into a set of non-negative weights that sum to $1$: an actual probability distribution over "how much to attend to each position." For a row $\mathbf{r}$ with entries $r_j$,

$$
\mathrm{softmax}(\mathbf{r})_j = \frac{\exp(r_j)}{\sum_{k} \exp(r_k)},
$$

where the sum runs only over the unmasked entries (the masked ones contribute $\exp(-\infty)=0$). We apply softmax **independently to each row**, because each row is one query deciding how to distribute its attention.

**Row 1** has a single live entry ($0.7071$), the other two masked. With only one option, softmax must put all the weight there:

$$
\text{row 1 weights} = [\,1.0000,\ 0,\ 0\,].
$$

**Row 2**, live entries $[0, 0.7071]$. Exponentiate: $\exp(0) = 1$, $\exp(0.7071) \approx 2.0281$. Their sum is $3.0281$. Divide:

$$
\frac{1}{3.0281} \approx 0.3302, \qquad \frac{2.0281}{3.0281} \approx 0.6698.
$$

$$
\text{row 2 weights} = [\,0.3302,\ 0.6698,\ 0\,].
$$

**Row 3**, live entries $[0.7071, 1.4142, 0.7071]$. Exponentiate: $\exp(0.7071) \approx 2.0281$, $\exp(1.4142) \approx 4.1133$, $\exp(0.7071) \approx 2.0281$. Sum $= 8.1695$. Divide each:

$$
\frac{2.0281}{8.1695} \approx 0.2483, \quad \frac{4.1133}{8.1695} \approx 0.5035, \quad \frac{2.0281}{8.1695} \approx 0.2483.
$$

$$
\text{row 3 weights} = [\,0.2483,\ 0.5035,\ 0.2483\,].
$$

Collecting the three rows gives the **attention weight matrix**:

$$
A = \begin{bmatrix} 1.0000 & 0 & 0 \\ 0.3302 & 0.6698 & 0 \\ 0.2483 & 0.5035 & 0.2483 \end{bmatrix}.
$$

Two sanity checks. Every row sums to $1$ ($0.3302 + 0.6698 = 1$; $0.2483 + 0.5035 + 0.2483 = 1.0001$, off only by rounding). And the matrix is lower-triangular — all the upper-right entries are exactly zero, which is the causal mask doing its job: no position attends to a later one. In row 3, the middle position gets the most weight ($0.5035$) because it had the highest score ($1.4142$), exactly the strong $[1,1]\cdot[1,1]$ match we spotted in section 3.

## 7. Step five — output is a weighted average of values

Now each query has a set of weights saying how much to listen to each position. The output for that query is the corresponding **weighted sum of the value vectors** — literally $\sum_j A_{ij}\, \mathbf{v}_j$. Stacking all queries, this is one more matrix multiplication:

$$
O = A V, \qquad \text{shape } (T, T) \times (T, d_h) = (3, 3) \times (3, 2) = (3, 2).
$$

The output has the same shape as the input $V$: one $d_h$-vector per position, now enriched with information blended from the positions it attended to. With $\mathbf{v}_1 = [1,0]$, $\mathbf{v}_2 = [0,1]$, $\mathbf{v}_3 = [1,1]$:

**Row 1:** weights $[1, 0, 0]$, so the output is just $\mathbf{v}_1$:

$$
1.0000\,[1,0] = [\,1.0000,\ 0.0000\,].
$$

**Row 2:** weights $[0.3302, 0.6698, 0]$:

$$
0.3302\,[1,0] + 0.6698\,[0,1] = [\,0.3302,\ 0.6698\,].
$$

**Row 3:** weights $[0.2483, 0.5035, 0.2483]$:

$$
0.2483\,[1,0] + 0.5035\,[0,1] + 0.2483\,[1,1] = [\,0.2483 + 0.2483,\ 0.5035 + 0.2483\,] = [\,0.4965,\ 0.7517\,].
$$

The complete attention output is

$$
O = \begin{bmatrix} 1.0000 & 0.0000 \\ 0.3302 & 0.6698 \\ 0.4965 & 0.7517 \end{bmatrix}.
$$

That is a full scaled dot-product attention, worked end to end. Trace the whole pipeline once more in your head: **scores** ($QK^\top$) → **scale** ($/\sqrt{d_h}$) → **mask** ($+M$) → **weights** (softmax per row) → **output** ($AV$). Every arrow is one of the numbers above.

## 8. Multi-head attention

A single attention lets each position gather one blended view of the others. But a token often relates to different positions for different *reasons* at once — one relationship is grammatical agreement, another is topical, another is positional. Forcing all of that through a single set of $Q, K, V$ vectors is limiting. **Multi-head attention** runs several attentions in parallel, each on its own slice of the features, so each "head" can specialize in a different kind of relationship.

### 8.1 Splitting the channels into heads

Recall from module 4 that a token's full hidden vector has $C$ channels, and $C = n_h \cdot d_h$ where $n_h$ is the number of heads and $d_h$ the per-head dimension. Multi-head attention takes the full $(B, T, C)$ activation, produces $Q, K, V$ each of shape $(B, T, C)$, and then **splits** the channel axis into $n_h$ groups of $d_h$:

$$
(B, T, C) \;\xrightarrow{\text{reshape}}\; (B, T, n_h, d_h) \;\xrightarrow{\text{transpose}}\; (B, n_h, T, d_h).
$$

The reshape re-groups the $C$ channels into $n_h \times d_h$ without moving data; the transpose swaps the $T$ and $n_h$ axes so the head axis sits next to the batch axis. This is exactly the `view(B, T, n_h, d_h).transpose(1, 2)` pattern you learned in [module 4, lesson 2](lessons/module-04/lesson-02.md). After it, the last two axes of the tensor are $(T, d_h)$ — precisely the little $Q, K, V$ matrices attention operates on — and there are $B \cdot n_h$ of them, one per (sequence, head).

Each head then runs the *entire* section 3–7 computation on its own $(T, d_h)$ slices, completely independently: its own scores, its own scaled-and-masked matrix, its own softmax, its own weighted sum of *its* values. Because the heads sit on separate channel slices and on a separate axis, all $n_h$ of them compute at once as a batched matrix multiply — no Python loop over heads.

### 8.2 Concatenate and project

Each head produces an output of shape $(B, n_h, T, d_h)$. To hand a single $C$-vector per position back to the rest of the network, we reverse the split — transpose the head axis back down and merge $n_h$ and $d_h$ into $C$ again:

$$
(B, n_h, T, d_h) \;\xrightarrow{\text{transpose}}\; (B, T, n_h, d_h) \;\xrightarrow{\text{reshape}}\; (B, T, C).
$$

Merging the head axis back into the channel axis is **concatenation**: for each position we lay the $n_h$ head outputs (each $d_h$ long) end to end into one $C$-long vector. Finally we multiply by a learned **output projection** matrix $W_O$ of shape $(C, C)$:

$$
\text{MultiHead} = \mathrm{Concat}(\text{head}_1, \dots, \text{head}_{n_h})\, W_O.
$$

$W_O$ lets the model mix information across heads — without it, the head outputs would just sit in disjoint channel slices, unable to combine. Conceptually: the heads gather relationships in parallel in their own subspaces, and $W_O$ blends those independent findings into the final result.

<div class="callout key"><p>Multi-head attention = split $C$ into $n_h$ independent subspaces of size $d_h$, run scaled dot-product attention in each, concatenate the outputs back to width $C$, then mix them with $W_O$. Same math as one head, done $n_h$ times in parallel on different feature slices.</p></div>

## 9. In PyTorch

Here is the single-head computation of sections 3–7, reproducing every number exactly.

```python
import torch
import torch.nn.functional as F

Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])   # (T, d_h) = (3, 2)
K = torch.tensor([[1., 0.], [1., 1.], [0., 1.]])
V = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])

T, d_h = Q.shape                                    # 3, 2

# 1) scores = Q K^T ; 2) scale by 1/sqrt(d_h)
scores = Q @ K.transpose(-2, -1) / d_h**0.5
# tensor([[0.7071, 0.7071, 0.0000],
#         [0.0000, 0.7071, 0.7071],
#         [0.7071, 1.4142, 0.7071]])

# 3) causal mask: keep j <= i, set the future to -inf
mask = torch.tril(torch.ones(T, T))                 # lower-triangular ones
scores = scores.masked_fill(mask == 0, float('-inf'))
# tensor([[0.7071,   -inf,   -inf],
#         [0.0000, 0.7071,   -inf],
#         [0.7071, 1.4142, 0.7071]])

# 4) softmax over the last axis (each row -> weights summing to 1)
weights = F.softmax(scores, dim=-1)
# tensor([[1.0000, 0.0000, 0.0000],
#         [0.3302, 0.6698, 0.0000],
#         [0.2483, 0.5035, 0.2483]])

# 5) output = weights @ V
out = weights @ V
# tensor([[1.0000, 0.0000],
#         [0.3302, 0.6698],
#         [0.4965, 0.7517]])
```

Line for line: `Q @ K.transpose(-2, -1)` is the score matrix $QK^\top$ of section 3 (the `-2, -1` swap the last two axes, so it works unchanged when there are extra leading batch and head axes); dividing by `d_h**0.5` is the $1/\sqrt{d_h}$ scale of section 4; `masked_fill(mask == 0, float('-inf'))` writes $-\infty$ into the future positions of section 5; `F.softmax(..., dim=-1)` is the per-row softmax of section 6 (`dim=-1` means "normalize along each row"); and `weights @ V` is the weighted sum of section 7. The printed tensors match the hand computation to four decimals.

<div class="callout pt"><p>The three lines <code>scores = Q @ K.transpose(-2, -1) / d_h**0.5</code>, <code>F.softmax(scores, dim=-1)</code>, and <code>weights @ V</code> are the entire attention formula. If you can read those three lines you can read the attention core of any GPT implementation.</p></div>

The multi-head version wraps the same three operations in the split-and-merge of section 8:

```python
B, T, C, n_h = 2, 3, 8, 4          # C = n_h * d_h -> d_h = 2
d_h = C // n_h

x = torch.randn(B, T, C)
# a real block makes Q,K,V from x via a Linear; here take them = x for shape demo
Q = K = Vv = x                      # each (B, T, C)

def split_heads(t):                 # (B, T, C) -> (B, n_h, T, d_h)
    return t.view(B, T, n_h, d_h).transpose(1, 2)

q, k, v = split_heads(Q), split_heads(K), split_heads(Vv)   # (B, n_h, T, d_h)

scores = q @ k.transpose(-2, -1) / d_h**0.5                 # (B, n_h, T, T)
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float('-inf'))
weights = F.softmax(scores, dim=-1)                         # (B, n_h, T, T)
heads = weights @ v                                         # (B, n_h, T, d_h)

merged = heads.transpose(1, 2).contiguous().view(B, T, C)   # concat heads -> (B, T, C)
W_O = torch.randn(C, C)
out = merged @ W_O                                          # (B, T, C)
print(out.shape)                    # torch.Size([2, 3, 8])
```

Notice the shapes: the mask $(T, T)$ **broadcasts** across the leading $(B, n_h)$ axes, so one small triangular matrix masks every sequence and every head at once. The output is back to $(B, T, C)$ — the same shape as the input, which is what lets attention slot into a stack of layers.

## 10. What PyTorch is doing under the hood

Three things are worth making explicit.

**Batched matrix multiply.** `q @ k.transpose(-2, -1)` on 4-D tensors is not one big matmul; it is $B \cdot n_h$ independent $(T, d_h) \times (d_h, T)$ matmuls, one per (sequence, head), dispatched together to the GPU. The `@` operator treats all axes *before* the last two as batch axes and loops over them in parallel. This is why the head axis is placed next to the batch axis in section 8: it turns "run attention per head" into a single batched operation with no Python loop.

**The mask is added, not multiplied, before softmax.** Setting scores to $-\infty$ and letting $\exp(-\infty) = 0$ removes future positions cleanly inside softmax, and it keeps the surviving weights a correct probability distribution (they still sum to $1$). This is exactly why the mask lives *before* softmax rather than being applied to the weights afterward — zeroing weights after the fact would break the normalization.

**Materializing the $T \times T$ matrix is the expensive part.** The scores and weights are each $(B, n_h, T, T)$: their size grows with $T^2$. For long sequences that matrix dominates both memory and time. Production kernels called **flash attention** compute the same output *without* ever storing the full $T \times T$ matrix — they stream over it in tiles and accumulate the weighted sum on the fly. The mathematics is identical to what you computed by hand; only the bookkeeping changes. We return to this in [module 12](lessons/module-12/lesson-01.md).

The gradients flow back through exactly these operations — the matmuls, the softmax, the masked fills — using the vector–Jacobian products from [module 7](lessons/module-07/lesson-01.md). Softmax has a known Jacobian (module 6), and autograd already carries backward rules for `@`, `softmax`, and `masked_fill`, so `loss.backward()` differentiates the whole attention block with no extra work from you.

## Check yourself

<details><summary>Why do we scale the scores by $1/\sqrt{d_h}$ before softmax, and what goes wrong without it?</summary>

A dot product over $d_h$ dimensions grows in magnitude as $d_h$ grows. Large scores make softmax saturate — nearly all weight on one position, a near-one-hot distribution — which turns the soft blend into a hard pick and flattens the gradient, stalling learning. Dividing by $\sqrt{d_h}$ keeps the score magnitudes moderate regardless of head size.

</details>

<details><summary>In the worked example, why is row 1 of the weight matrix $[1, 0, 0]$?</summary>

Position 1 is the first token, so the causal mask forbids it from attending to positions 2 and 3 (they are in its future). Only one entry survives softmax, so it must receive all the weight: $[1, 0, 0]$.

</details>

<details><summary>Why is the mask applied by adding $-\infty$ before softmax rather than by zeroing weights after softmax?</summary>

Because $\exp(-\infty) = 0$, a $-\infty$ score contributes exactly zero to both the softmax numerator and its normalizing sum, so the surviving weights still sum to $1$ — a valid distribution. Zeroing weights *after* softmax would leave them summing to less than $1$, breaking the weighted-average interpretation.

</details>

<details><summary>With $C = 768$ and $n_h = 12$ heads, what is $d_h$, and what shape is the scores tensor for a batch of $B$ sequences of length $T$?</summary>

$d_h = C / n_h = 768 / 12 = 64$. The scores tensor is $(B, n_h, T, T) = (B, 12, T, T)$ — one $T \times T$ score matrix per sequence per head.

</details>

## Next

You can now compute attention by hand and read its three-line PyTorch core, and you understand how multiple heads split the channels to attend in parallel. But attention never stands alone: in a real GPT it is wrapped in normalization, a residual connection, and a feed-forward network. The next lesson assembles exactly that — the Transformer block — around the attention you just built.

Continue to [33 · The Transformer block](lessons/module-11/lesson-02.md).
