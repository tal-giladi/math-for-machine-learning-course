# 30 · Softmax and language-modeling math

<div class="prereq">
<p><strong>Prerequisites:</strong> the categorical distribution and cross-entropy from <a href="lesson-01.md">29 · Probability and statistics for ML</a>; softmax and its gradient from <a href="../module-06/lesson-03.md">17 · Derivatives of exp, log, sigmoid, softmax</a>; matrix–vector multiplication and the $(V, C)$ weight convention from <a href="../module-03/lesson-02.md">10 · Matrix multiplication</a> and <a href="../module-08/lesson-01.md">23 · The linear layer, forward and backward by hand</a>.</p>
<p><strong>You will learn:</strong> exactly how a single hidden vector becomes a probability distribution over the vocabulary — the LM head matmul (weight shape $(V, C)$) that produces one logit per token, the softmax that normalizes logits into probabilities, the <strong>temperature</strong> knob that sharpens or flattens them, cross-entropy for the next token, and the two ways to turn the distribution back into a token: <strong>argmax</strong> and <strong>sampling</strong>.</p>
<p><strong>Why this matters for ML:</strong> this is the last few centimetres of a GPT-2 forward pass — the part that turns the Transformer's abstract hidden state into an actual prediction and an actual loss. Every generated token you have ever seen from a language model came out of the softmax you are about to compute by hand.</p>
</div>

## 1. The picture: from hidden state to distribution

At the top of a Transformer, each position has a **hidden state** $\mathbf{h}$ — a vector of $C$ numbers ($C$ is the model width, a.k.a. $d_\text{model}$; $768$ for GPT-2 small). It is a dense, learned summary of "everything the model understands about what should come next here." But $\mathbf{h}$ is not yet a prediction — it is $C$ numbers, and we need $V$ numbers, one score per vocabulary token.

The full pipeline of this lesson:

```text
hidden state h   (C numbers)
      |  linear projection: the LM head, weight shape (V, C)
      v
logits z         (V numbers, one per token, any real value)
      |  softmax
      v
probabilities p  (V numbers in (0,1), summing to 1)
      |  argmax  (greedy)   OR   sample  (multinomial)
      v
next token
```

Everything else in the model exists to produce a good $\mathbf{h}$. This lesson is what happens to $\mathbf{h}$ after that.

## 2. The LM head: a matmul that makes one logit per word

### 2.1 The shape story

The **language-model head** is a single linear layer with no activation. It maps the $C$-dimensional hidden state to $V$ **logits**. Its weight matrix $\mathbf{W}_\text{head}$ has shape $(V, C)$ — one row per vocabulary token, each row a $C$-length vector living in the same space as $\mathbf{h}$:

$$
\mathbf{z} = \mathbf{h}\,\mathbf{W}_\text{head}^\top + \mathbf{b}.
$$

- $\mathbf{h}$ — hidden state, shape $(C,)$ (think of it as a $1 \times C$ row).
- $\mathbf{W}_\text{head}$ — LM head weights, shape $(V, C)$. Row $j$ is the "template" for token $j$.
- $\mathbf{W}_\text{head}^\top$ — its transpose, shape $(C, V)$.
- $\mathbf{b}$ — an optional bias, shape $(V,)$; GPT-2's LM head has no bias, so we take $\mathbf{b} = \mathbf{0}$.
- $\mathbf{z}$ — the logits, shape $(V,)$.

Why does this produce one logit per word? Because logit $j$ is the **dot product** of the hidden state with token $j$'s row:

$$
z_j = \mathbf{h} \cdot \mathbf{W}_\text{head}[j] = \sum_{k=1}^{C} h_k \, W_{\text{head},\,j,k}.
$$

A dot product measures alignment (you saw this as similarity back in module 2). So the logit for a token is high when the hidden state points in the same direction as that token's row. The LM head asks, for every token in the vocabulary, "how well does the current hidden state match you?" — and the answer is that token's logit.

<div class="callout key"><p>The LM head weight shape is $(V, C)$: one $C$-dimensional row per vocabulary token. The matmul $\mathbf{h}\,\mathbf{W}_\text{head}^\top$ dots the hidden state against every row at once, producing $V$ logits — one alignment score per possible next token.</p></div>

### 2.2 A tiny worked example, by hand

Vocabulary of three tokens and a width of two:

```text
vocabulary = ["cat", "dog", "fish"]     V = 3
hidden size C = 2
h        = [2, 1]
W_head   = [[1,  0],      # row 0: "cat"
            [0,  1],      # row 1: "dog"
            [1, -2]]      # row 2: "fish"     shape (V, C) = (3, 2)
```

Compute each logit as a dot product of $\mathbf{h} = [2, 1]$ with a row:

$$
z_\text{cat}  = 2\cdot 1 + 1\cdot 0 = 2,
$$
$$
z_\text{dog}  = 2\cdot 0 + 1\cdot 1 = 1,
$$
$$
z_\text{fish} = 2\cdot 1 + 1\cdot(-2) = 2 - 2 = 0.
$$

So $\mathbf{z} = [2, 1, 0]$. (If there were a bias $\mathbf{b} = [b_\text{cat}, b_\text{dog}, b_\text{fish}]$, you would simply add it component-wise; with $\mathbf{b} = \mathbf{0}$ nothing changes.) The hidden state aligns best with the "cat" row, so "cat" gets the largest logit. These raw scores can be any real number — positive, negative, zero — and they are not yet probabilities.

## 3. Softmax: logits to a probability distribution

### 3.1 The mathematics

Softmax turns the logit vector into a categorical distribution — positive numbers summing to $1$:

$$
p_i = \text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}.
$$

- $z_i$ — the logit for token $i$.
- $e^{z_i}$ — exponentiate, making every value strictly positive and amplifying larger logits.
- $\sum_{j} e^{z_j}$ — the **normalizer** (partition function): the sum of all exponentials, the single denominator shared by every token.
- $p_i$ — the resulting probability of token $i$, in $(0, 1)$; the $V$ of them sum to $1$.

The denominator is what couples the tokens together: raising one logit steals probability mass from all the others, because the total is fixed at $1$. That is the "soft" competition softmax implements.

### 3.2 The worked example continued

Take $\mathbf{z} = [2, 1, 0]$ from section 2.2. Exponentiate:

$$
e^{2} = 7.3891, \qquad e^{1} = 2.7183, \qquad e^{0} = 1.0000.
$$

Normalizer: $7.3891 + 2.7183 + 1.0000 = 11.1073$. Divide each:

$$
\mathbf{p} = \left[\frac{7.3891}{11.1073},\ \frac{2.7183}{11.1073},\ \frac{1.0000}{11.1073}\right] = [0.6652,\ 0.2447,\ 0.0900].
$$

They sum to $1.0000$ ✓. The model assigns "cat" a $66.5\%$ chance, "dog" $24.5\%$, "fish" $9.0\%$ — the ordering of the logits is preserved, but now they are a genuine distribution over the vocabulary. This is the categorical distribution from [lesson 29](lesson-01.md), produced on demand from a hidden state.

## 4. Temperature: sharpening and flattening

### 4.1 The idea

**Temperature** $\tau$ is a single positive number you divide the logits by *before* softmax:

$$
p_i = \text{softmax}(\mathbf{z}/\tau)_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}.
$$

- $\tau = 1$ — no change; the ordinary softmax.
- $\tau < 1$ — divides by a small number, *magnifying* the gaps between logits, so the distribution **sharpens** toward the top token (more confident, more deterministic).
- $\tau > 1$ — divides by a large number, *shrinking* the gaps, so the distribution **flattens** toward uniform (more random, more diverse).

Temperature does not change which token is largest — only how peaked the distribution is. It is the main dial for controlling how "creative" versus "safe" generated text is.

### 4.2 Worked numerics on $[2, 1, 0]$

Start from $\mathbf{z} = [2, 1, 0]$.

**$\tau = 0.5$ (sharpen).** Divide logits by $0.5$: $\mathbf{z}/\tau = [4, 2, 0]$.

$$
e^{4} = 54.598, \quad e^{2} = 7.389, \quad e^{0} = 1.000, \quad \text{sum} = 62.987.
$$
$$
\mathbf{p} = [0.8668,\ 0.1173,\ 0.0159].
$$

The top token climbed from $0.6652$ to $0.8668$ — sharper, more confident.

**$\tau = 2$ (flatten).** Divide logits by $2$: $\mathbf{z}/\tau = [1, 0.5, 0]$.

$$
e^{1} = 2.7183, \quad e^{0.5} = 1.6487, \quad e^{0} = 1.0000, \quad \text{sum} = 5.3670.
$$
$$
\mathbf{p} = [0.5065,\ 0.3072,\ 0.1863].
$$

The top token dropped from $0.6652$ to $0.5065$, and the tail rose — flatter, more random. As $\tau \to \infty$ the distribution approaches uniform $[\tfrac13, \tfrac13, \tfrac13]$; as $\tau \to 0$ it approaches the one-hot $[1, 0, 0]$ (pure argmax).

<div class="callout key"><p>Temperature $\tau$ rescales logits before softmax. $\tau<1$ sharpens toward the top token (here the peak rose to $0.87$); $\tau>1$ flattens toward uniform (the peak fell to $0.51$). It changes confidence, never the ranking.</p></div>

## 5. Cross-entropy for the next token

During **training** you do not sample — you compare the distribution to the truth. The true next token is one specific index $c$; the loss is the cross-entropy against that one-hot target, which (from [lesson 29](lesson-01.md) and [lesson 24](../module-08/lesson-02.md)) is just the negative log probability of the correct token:

$$
L = -\log p_c.
$$

Suppose the true next token is "dog" (index $1$). From section 3.2, $p_\text{dog} = 0.2447$, so

$$
L = -\ln(0.2447) = 1.4076 \text{ nats.}
$$

Had the truth been "cat" (the model's favourite, $p = 0.6652$), the loss would be $-\ln(0.6652) = 0.4076$ — much smaller, because the model was already confident in the right answer. This single number, averaged over every position in every sequence in the batch, is the GPT-2 training loss. Its gradient with respect to the logits is the clean $\mathbf{p} - \mathbf{y}$ from [lesson 24](../module-08/lesson-02.md) — here $[0.6652, 0.2447, 0.0900] - [0, 1, 0] = [0.6652, -0.7553, 0.0900]$ — the seed of the entire backward pass.

## 6. Next-token prediction: argmax vs sampling

At **generation** time you have the distribution $\mathbf{p}$ and must commit to one token. Two strategies:

**Argmax (greedy).** Take the single most probable token: $\hat{c} = \arg\max_i p_i$. For $\mathbf{p} = [0.6652, 0.2447, 0.0900]$ that is index $0$, "cat," every time. Deterministic, repeatable — and notoriously prone to dull, looping text, because it never explores.

**Sampling.** Draw a token *randomly according to* $\mathbf{p}$: "cat" $66.5\%$ of the time, "dog" $24.5\%$, "fish" $9.0\%$. This is exactly `Categorical.sample()` from [lesson 29](lesson-01.md). Sampling produces varied, natural text, and it is where temperature earns its keep: raise $\tau$ for more diversity, lower it for more focus. Argmax is the $\tau \to 0$ limit of sampling.

## 7. In PyTorch

The LM head is `nn.Linear(C, V)` — note the argument order `(in_features, out_features)` gives it a weight of shape $(V, C)$, exactly our convention.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

C, V = 2, 3
h = torch.tensor([[2.0, 1.0]])          # (1, C) one position

head = nn.Linear(C, V, bias=False)      # weight shape (V, C) = (3, 2)
with torch.no_grad():                   # set our hand-picked weights
    head.weight.copy_(torch.tensor([[1., 0.],
                                     [0., 1.],
                                     [1., -2.]]))

logits = head(h)                        # h @ weight.T  ->  (1, V)
print(logits)                           # tensor([[2., 1., 0.]])

probs = F.softmax(logits, dim=-1)
print(probs)                            # tensor([[0.6652, 0.2447, 0.0900]])

# temperature
print(F.softmax(logits / 0.5, dim=-1))  # tensor([[0.8668, 0.1173, 0.0159]])
print(F.softmax(logits / 2.0, dim=-1))  # tensor([[0.5065, 0.3072, 0.1863]])

# training loss for true token "dog" (index 1): pass RAW logits
target = torch.tensor([1])
print(F.cross_entropy(logits, target))  # tensor(1.4076)

# generation: argmax vs sampling
print(torch.argmax(probs, dim=-1))      # tensor([0])  -> "cat", greedy
print(torch.multinomial(probs[0], num_samples=1))  # a random index per p
```

<div class="callout pt"><p><code>nn.Linear(C, V)</code> stores <code>weight</code> of shape $(V, C)$ and computes <code>h @ weight.T</code> — so you never transpose by hand; the layer does it. And feed <code>F.cross_entropy</code> the <strong>raw logits</strong>, never the softmax output — it applies log-softmax itself (see the gotcha in <a href="../module-08/lesson-02.md">lesson 24</a>).</p></div>

## 8. Weight tying (a forward reference)

The LM head's weight is $(V, C)$: one $C$-vector per token. The **token embedding** matrix (next lesson) is *also* $(V, C)$: one $C$-vector per token. GPT-2 exploits this by **tying** them — using the same matrix for both the input embedding lookup and the output LM head. This halves a large chunk of the parameters and reflects a real symmetry: the vector that represents "cat" going in is the same vector used to score "cat" coming out. We build embeddings from scratch and return to weight tying in [lesson 31](lesson-03.md).

## 9. What PyTorch is doing under the hood

**Logits are one matmul.** For a real forward pass the hidden states have shape $(B, T, C)$ — batch $B$, sequence length $T$, width $C$. The LM head multiplies by $\mathbf{W}_\text{head}^\top$ of shape $(C, V)$ to give logits of shape $(B, T, V)$: a full distribution at every one of the $B \cdot T$ positions, computed in a single batched matmul on the GPU. For GPT-2 small that final matmul is $(B, T, 768) \times (768, 50257)$ — the widest matrix in the model, which is why weight tying (sharing it with the embedding) matters for memory.

**Softmax and cross-entropy are fused for stability.** As covered in [lesson 24](../module-08/lesson-02.md), `F.cross_entropy` does not materialize the probabilities. It computes log-softmax with the max-subtraction trick — subtract $\max_j z_j$ from every logit before exponentiating, so the largest exponential is $e^0 = 1$ and nothing overflows — then indexes the log-probability of the true token and negates it. The math is identical to "softmax, then $\log$, then pick," but it never forms the fragile intermediate. The full numerical-stability treatment (log-sum-exp) is [module 12](../module-12/lesson-01.md).

**`grad_fn` remembers the softmax.** When you call `F.softmax` or `F.cross_entropy` on tensors that require grad, autograd records the operation so that `backward()` can apply the $\mathbf{p} - \mathbf{y}$ gradient to the logits and, through the LM head matmul, on to $\mathbf{W}_\text{head}$ and the hidden state. Sampling (`torch.multinomial`, `argmax`) is used only at generation time and is *not* differentiated — you never backprop through a sampled token in ordinary language-model training.

## Check yourself

<details><summary>The LM head weight has shape $(V, C)$. If $C = 768$ and $V = 50257$, what shape are the logits for a batch of hidden states of shape $(B, T, C)$?</summary>

$(B, T, V) = (B, T, 50257)$. The matmul is $(B, T, C) \times (C, V)$; the $C$ dimensions contract and $V$ takes its place, giving one logit per vocabulary token at every position.

</details>

<details><summary>You raise the temperature from $\tau = 1$ to $\tau = 2$. Does the argmax token change? Does the distribution get more or less confident?</summary>

The argmax does not change — temperature rescales all logits by the same factor, so the ordering is preserved. The distribution gets *less* confident (flatter): with $\mathbf{z} = [2,1,0]$ the top probability fell from $0.6652$ to $0.5065$.

</details>

<details><summary>Given logits $[2, 1, 0]$ and true token index 0 ("cat"), what is the cross-entropy loss?</summary>

$p_\text{cat} = 0.6652$, so $L = -\ln(0.6652) = 0.4076$ nats. Small, because the correct token was already the model's top choice.

</details>

<details><summary>Why pass raw logits to <code>F.cross_entropy</code> instead of softmax probabilities?</summary>

`cross_entropy` applies log-softmax internally (for numerical stability). If you softmax first and pass that in, it takes log-softmax of already-normalized numbers — a wrong, too-small loss. Always hand it the raw logits straight from the LM head.

</details>

## Next

You can now turn a hidden state into logits, logits into a distribution, and a distribution into either a loss (training) or a token (generation). The one piece still opaque is where that hidden state's journey *begins*: how a token ID becomes a vector in the first place. The next lesson opens the embedding matrix — the mirror image of the LM head — and shows why a lookup is secretly a one-hot matmul, and why only the rows you use receive gradients.

Continue to [31 · Embeddings](lesson-03.md).
