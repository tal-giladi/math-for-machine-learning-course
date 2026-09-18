# 37 · Training a GPT-2-style model end to end

<div class="prereq">
<p><strong>Prerequisites:</strong> this lesson synthesizes the whole course. You need embeddings from <a href="../module-10/lesson-03.md">31 · Embeddings</a>, attention from <a href="../module-11/lesson-01.md">32 · Attention mathematics</a>, the Transformer block from <a href="../module-11/lesson-02.md">33 · The Transformer block</a>, LayerNorm from <a href="../module-11/lesson-03.md">34 · Normalization</a>, cross-entropy from <a href="../module-08/lesson-02.md">24 · Loss functions</a>, backprop from <a href="../module-08/lesson-03.md">25 · Backpropagation and autograd</a>, and AdamW/Muon from <a href="../module-09/lesson-02.md">27 · AdamW</a> and <a href="../module-09/lesson-03.md">28 · Muon</a>.</p>
<p><strong>You will learn:</strong> the entire GPT-2 training step as one pipeline — every tensor's <strong>shape</strong>, the <strong>math</strong> each stage performs, and the <strong>classification</strong> of every quantity as a parameter, activation, gradient, or optimizer state. You will count the parameters of a concrete toy model by hand and tell the whole memory story.</p>
<p><strong>Why this matters for ML:</strong> after this lesson you can open any real GPT-2 implementation, point at any tensor, and say exactly what it is, what shape it has, which of the four kinds of quantity it is, and where it sits in the forward-backward-update cycle. This is the payoff the whole course was building toward.</p>
</div>

## 1. The four kinds of quantity

Before any tensor appears, fix the vocabulary that organizes everything. Every number in a training step is exactly one of four kinds:

- **Parameter** — a weight the model *learns*. Created once, persists for the whole run, updated every step. Collectively called $\theta$.
- **Activation** — an intermediate value computed *during the forward pass* from the input and the parameters. Recomputed every step, and (the ones needed for the backward pass) kept in memory until `backward()` consumes them.
- **Gradient** — one number per parameter, $\partial L/\partial\theta_i$, produced by `backward()` and stored in each parameter's `.grad`. Same shape as the parameter it belongs to.
- **Optimizer state** — extra bookkeeping the optimizer keeps *per parameter* between steps: for AdamW the first and second moments $m$ and $v$; for Muon a momentum buffer. Same shape as the parameter (or matrix) it tracks.

Keep these four labels in mind; the entire lesson is an exercise in tagging each tensor with one of them.

## 2. The pipeline, top to bottom

Here is the whole training step as a chain of stages. The rest of the lesson walks it stage by stage, but this is the map:

```text
tokens (B,T) ints
   │  token + positional embedding lookup + add
   ▼
x (B,T,C)
   │  N transformer blocks, each (B,T,C) -> (B,T,C)
   ▼
x (B,T,C)
   │  final LayerNorm
   ▼
x (B,T,C)
   │  LM head, a linear map C -> V
   ▼
logits (B,T,V)
   │  cross-entropy vs targets (inputs shifted by one)
   ▼
loss  (a single scalar)
   │  loss.backward()  -> fills every param.grad
   ▼
gradients (one per parameter)
   │  optimizer.step()  -> AdamW on most params, Muon on 2-D hidden matrices
   ▼
updated parameters
   │  repeat
```

Throughout, the GPT dimension names from the course hold: batch $B$, sequence length $T$, channels/features $C$ (= $d_\text{model}$), vocabulary $V$, heads $n_h$, head dimension $d_h$ with $C = n_h\,d_h$.

We will make every "(B,T,C)" concrete with the toy config **$V=1000$, $C=64$, $T=8$, $n_\text{layers}=4$, $n_h=4$** (so $d_h = C/n_h = 16$), and a batch of $B=2$ sequences.

## 3. Stage 1 — tokens

Training data is text turned into integer token IDs. One batch is a tensor of shape $(B, T)$ containing integers in $[0, V)$:

$$
\text{tokens} \in \{0,1,\ldots,V-1\}^{B\times T}.
$$

- **Shape:** $(B, T) = (2, 8)$.
- **Math:** none yet — these are indices, not values to compute with.
- **Classification:** *input data.* Not a parameter (you do not train the data), not an activation (nothing computed it). Its dtype is integer, and it never carries a gradient.

The targets are the same tensor **shifted left by one**: the label at position $t$ is the token that actually came at position $t+1$. That single shift is what makes this "predict the next token." Targets also have shape $(B, T)$.

## 4. Stage 2 — embeddings

Two lookup tables turn each integer into a $C$-dimensional vector, from [lesson 31](../module-10/lesson-03.md).

- **Token embedding** `wte`: a matrix of shape $(V, C)$. Row $i$ is the learned vector for token ID $i$. The lookup `wte[tokens]` gathers one row per token, producing shape $(B, T, C)$.
- **Positional embedding** `wpe`: a matrix of shape $(T_\text{max}, C)$. Row $t$ is the learned vector for *position* $t$. The lookup `wpe[0:T]` gives shape $(T, C)$.

The two are added (the position vector broadcasts across the batch):

$$
x_{b,t,:} = \text{wte}[\text{tokens}_{b,t}] + \text{wpe}[t], \qquad x \in \mathbb{R}^{B\times T\times C}.
$$

| tensor | shape | kind |
|---|---|---|
| `wte` | $(V, C) = (1000, 64)$ | **parameter** |
| `wpe` | $(T_\text{max}, C) = (8, 64)$ | **parameter** |
| `x` (embedded input) | $(B, T, C) = (2, 8, 64)$ | **activation** |

The embedding tables are parameters (learned). Their output $x$ is the first activation. Note the gradient behavior from [lesson 31](../module-10/lesson-03.md): only the *rows* of `wte` for tokens that actually appeared get a nonzero gradient this step.

## 5. Stage 3 — the transformer blocks

$x$ now flows through $N = n_\text{layers}$ identical-in-shape blocks. **Every block maps $(B,T,C)\to(B,T,C)$** — the shape is invariant, which is what lets you stack them. Inside each block (pre-LayerNorm GPT-2 style, from [lesson 33](../module-11/lesson-02.md)):

$$
x \leftarrow x + \text{Attn}(\text{LN}_1(x)), \qquad x \leftarrow x + \text{MLP}(\text{LN}_2(x)).
$$

The two `x +` are the **residual connections** ([lesson 33](../module-11/lesson-02.md)): they give the backward pass a multiply-by-one path so gradients reach early layers undiminished ([lesson 35](lesson-01.md)). Each block owns these parameters:

| tensor | shape | kind |
|---|---|---|
| `ln1.weight` $\gamma$, `ln1.bias` $\beta$ | $(C,)$ each | **parameter** |
| `attn.qkv.weight` | $(3C, C) = (192, 64)$ | **parameter** |
| `attn.qkv.bias` | $(3C,) = (192,)$ | **parameter** |
| `attn.proj.weight`, `attn.proj.bias` | $(C, C)$, $(C,)$ | **parameter** |
| `ln2.weight` $\gamma$, `ln2.bias` $\beta$ | $(C,)$ each | **parameter** |
| `mlp.fc.weight` | $(4C, C) = (256, 64)$ | **parameter** |
| `mlp.fc.bias` | $(4C,) = (256,)$ | **parameter** |
| `mlp.proj.weight` | $(C, 4C) = (64, 256)$ | **parameter** |
| `mlp.proj.bias` | $(C,) = (64,)$ | **parameter** |

**The math inside attention**, briefly (full derivation in [lesson 32](../module-11/lesson-01.md)). The QKV linear maps $\text{LN}_1(x)$ of shape $(B,T,C)$ to $(B,T,3C)$, split into $Q, K, V$ each $(B,T,C)$, then reshaped to $(B, n_h, T, d_h)$. Attention per head:

$$
\text{Attn}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + \text{mask}\right)V,
$$

where $QK^\top$ has shape $(B, n_h, T, T)$ — the attention scores, one $T\times T$ matrix per head — the $\sqrt{d_h}$ scaling keeps the scores at a sane magnitude, and the causal `mask` sets future positions to $-\infty$ so position $t$ attends only to $\le t$. The result is reshaped back to $(B,T,C)$ and passed through `attn.proj`.

**The math inside the MLP:** two linear layers with a GELU in between, widening to $4C$ and back:

$$
\text{MLP}(x) = \text{GELU}(x\,W_\text{fc}^\top + b_\text{fc})\,W_\text{proj}^\top + b_\text{proj}, \qquad (B,T,C)\to(B,T,4C)\to(B,T,C).
$$

Every intermediate here — $\text{LN}_1(x)$, $Q$, $K$, $V$, the scores, the softmax weights, the GELU output — is an **activation**. PyTorch keeps the ones needed for the backward pass alive in memory until `backward()` runs; this is the bulk of training memory and it scales with $B$ and $T$.

## 6. Stage 4 — final LayerNorm and the LM head

After the last block, one more LayerNorm ($\gamma,\beta$ of shape $(C,)$, **parameters**) normalizes the final hidden states, still $(B,T,C)$.

Then the **LM head** projects each $C$-dimensional hidden vector to a $V$-dimensional vector of **logits** — one raw score per vocabulary token:

$$
\text{logits} = x\,W_\text{head}^\top, \qquad (B,T,C)\times(C,V) \to (B,T,V).
$$

- **`W_head`** has shape $(V, C)$ (PyTorch stores a `Linear`'s weight as (out, in), applied as $xW^\top$). Classification: **parameter**.
- **`logits`** has shape $(B,T,V) = (2,8,1000)$. Classification: **activation**. Each of the $B\times T$ positions now holds $V$ scores predicting the *next* token.

A standard GPT-2 economy: **weight tying** sets $W_\text{head} = \text{wte}$ — the same $(V,C)$ matrix serves as both the input embedding and the output projection. This saves $V\cdot C$ parameters and ties the "meaning" of a token on the way in to its score on the way out. We assume tying below.

## 7. Stage 5 — the loss

The logits become a scalar loss via cross-entropy against the shifted targets, from [lesson 24](../module-08/lesson-02.md). Conceptually: softmax each length-$V$ logit vector into a probability distribution, then take the negative log-probability the model assigned to the *true* next token, and average over all $B\times T$ positions:

$$
L = -\frac{1}{B\,T}\sum_{b=1}^{B}\sum_{t=1}^{T} \log \text{softmax}(\text{logits}_{b,t,:})_{\;y_{b,t}},
$$

where $y_{b,t}$ is the target token index at that position. In practice `F.cross_entropy` fuses the softmax and log with the log-sum-exp trick for stability ([lesson 35](lesson-01.md)) and takes raw logits.

- **Shape:** $L$ is a single scalar, shape $()$.
- **Math:** softmax $\to$ negative log-likelihood $\to$ mean over $B\cdot T = 16$ positions.
- **Classification:** **activation** (the final one) — a scalar computed from parameters and data. It is the single output whose gradient we want with respect to every parameter, which is exactly the shape reverse-mode AD ([lesson 36](lesson-02.md)) handles in one backward pass.

## 8. Stage 6 — backward: filling the gradients

`loss.backward()` runs reverse-mode automatic differentiation over the graph recorded during stages 2–5. It seeds $\partial L/\partial L = 1$ at the loss and propagates backward through the head, the final LayerNorm, all $N$ blocks (through both residual branches), and the embeddings, depositing $\partial L/\partial\theta_i$ into each parameter's `.grad`.

The single most important fact about gradients for the memory story:

<div class="callout key"><p>Every parameter gets exactly <strong>one gradient of the same shape</strong>. If <code>wte</code> is $(1000,64)$, then <code>wte.grad</code> is $(1000,64)$. The set of all gradients is therefore the same total size as the set of all parameters — a full second copy of the model.</p></div>

- **Shapes:** each `param.grad` matches its `param` exactly.
- **Math:** a chain of vector-Jacobian products ([lesson 36](lesson-02.md)), never full Jacobians.
- **Classification:** **gradient.** These live in `.grad` until the optimizer consumes them and `zero_grad()` clears them.

The activations stored in stage 5 are consumed here and then freed — which is why the graph is gone after `backward()` unless you asked to retain it ([lesson 25](../module-08/lesson-03.md)).

## 9. Stage 7 — the optimizer step

`optimizer.step()` turns gradients into parameter updates. A modern GPT-2-style setup runs **two optimizers side by side** ([lesson 28](../module-09/lesson-03.md)):

- **AdamW** on most parameters — embeddings, all biases, LayerNorm $\gamma/\beta$, the head. For each such parameter it keeps first and second moment estimates $m$ and $v$ (each the same shape as the parameter) and updates $\theta \leftarrow \theta - \eta\big(\hat m/(\sqrt{\hat v}+\varepsilon) + \lambda\theta\big)$.
- **Muon** on the 2-D hidden weight *matrices* — the QKV, attention-proj, and MLP weights. It keeps a momentum buffer per matrix, orthogonalizes the (momentum-smoothed) gradient via Newton–Schulz, and steps by that.

| tensor | shape | kind |
|---|---|---|
| AdamW $m$, $v$ (per param) | same as the parameter | **optimizer state** |
| Muon momentum buffer (per matrix) | same as the matrix | **optimizer state** |

- **Math:** the AdamW / Muon update rules from module 9.
- **Classification:** **optimizer state** — persists *between* steps (unlike gradients, which are recomputed each step), and is the reason AdamW needs roughly two extra full-model-sized buffers.

After the step, `optimizer.zero_grad()` clears every `.grad` (recall from [lesson 25](../module-08/lesson-03.md) that `backward()` *accumulates*, so you must reset), and the loop repeats with the next batch.

## 10. Counting the parameters of the toy model

Now make it fully concrete by counting every parameter of the $V=1000,\ C=64,\ T=8,\ n_\text{layers}=4,\ n_h=4$ model (weights tied). A `Linear` mapping $\text{in}\to\text{out}$ has $\text{out}\times\text{in}$ weights plus $\text{out}$ biases; a LayerNorm has $2C$ ($\gamma$ and $\beta$).

**Embeddings:**

$$
\text{wte} = V\cdot C = 1000\cdot 64 = 64{,}000, \qquad \text{wpe} = T\cdot C = 8\cdot 64 = 512.
$$

**One transformer block:**

$$
\begin{aligned}
\text{LN}_1 &= 2C = 128,\\
\text{QKV} &= (3C)\cdot C + 3C = 3\cdot64\cdot64 + 192 = 12{,}288 + 192 = 12{,}480,\\
\text{attn proj} &= C\cdot C + C = 4096 + 64 = 4{,}160,\\
\text{LN}_2 &= 2C = 128,\\
\text{MLP fc} &= (4C)\cdot C + 4C = 16{,}384 + 256 = 16{,}640,\\
\text{MLP proj} &= C\cdot(4C) + C = 16{,}384 + 64 = 16{,}448.
\end{aligned}
$$

Block total: $128 + 12{,}480 + 4{,}160 + 128 + 16{,}640 + 16{,}448 = 49{,}984$. Four blocks: $4 \times 49{,}984 = 199{,}936$.

**Final LayerNorm:** $2C = 128$. **LM head:** $0$ new parameters (tied to `wte`).

$$
\text{Total} = 64{,}000 + 512 + 199{,}936 + 128 = \boxed{264{,}576} \approx 0.26\text{ million}.
$$

(Untied, the head would add another $V\cdot C = 64{,}000$, giving $328{,}576$.) Real GPT-2 small uses $V=50{,}257,\ C=768,\ T=1024,\ n_\text{layers}=12,\ n_h=12$ and lands at $\approx 124$ million by exactly this arithmetic — the same formula, bigger numbers.

## 11. The memory story

Now you can say precisely what training memory holds. For $P$ parameters, a full AdamW training step keeps, at minimum:

- **Parameters:** $P$ numbers.
- **Gradients:** $P$ numbers (one per parameter, section 8).
- **Optimizer state:** $2P$ numbers for AdamW ($m$ and $v$).
- **Activations:** everything stage 3–5 stored for the backward pass — this piece scales with $B\times T$ (and the model depth/width), *not* just $P$, and is often the largest term for long sequences or big batches.

The first three are fixed at $4P$ numbers regardless of batch size. In float32 (4 bytes) that is $16P$ bytes. For the toy model, $16 \times 264{,}576 \approx 4.23$ MB before activations. For GPT-2 small, $16 \times 124\text{M} \approx 2$ GB before a single activation — which is why the moment counts matter, why mixed precision ([lesson 35](lesson-01.md)) is used to shrink them, and why "the optimizer states are twice the size of the model" is a sentence every practitioner internalizes.

<div class="callout warn"><p>AdamW roughly <em>triples</em> the model's memory footprint before activations: parameters + gradients + two moment buffers = four full copies. When someone says a model "doesn't fit for training but fits for inference," this is why — inference needs only the parameters; training needs all four.</p></div>

## 12. The whole model in PyTorch

A compact but complete forward pass returning `(logits, loss)`, with shapes as comments. This is deliberately close to real GPT-2 code so you can map it onto the stages above.

```python
import torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, C, n_h):
        super().__init__()
        self.ln1  = nn.LayerNorm(C)
        self.qkv  = nn.Linear(C, 3 * C)     # (C) -> (3C)
        self.proj = nn.Linear(C, C)
        self.ln2  = nn.LayerNorm(C)
        self.fc   = nn.Linear(C, 4 * C)     # widen
        self.out  = nn.Linear(4 * C, C)     # back to C
        self.n_h  = n_h

    def forward(self, x):                   # x: (B, T, C)
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)            # each (B, T, C)
        # reshape to heads: (B, n_h, T, d_h)
        q = q.view(B, T, self.n_h, C // self.n_h).transpose(1, 2)
        k = k.view(B, T, self.n_h, C // self.n_h).transpose(1, 2)
        v = v.view(B, T, self.n_h, C // self.n_h).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, n_h, T, d_h)
        a = a.transpose(1, 2).reshape(B, T, C)           # back to (B, T, C)
        x = x + self.proj(a)                             # residual 1
        x = x + self.out(F.gelu(self.fc(self.ln2(x))))   # residual 2 (MLP)
        return x                                         # (B, T, C)

class GPT(nn.Module):
    def __init__(self, V, C, T, n_layers, n_h):
        super().__init__()
        self.wte = nn.Embedding(V, C)                    # (V, C) parameter
        self.wpe = nn.Embedding(T, C)                    # (T, C) parameter
        self.blocks = nn.ModuleList(Block(C, n_h) for _ in range(n_layers))
        self.lnf = nn.LayerNorm(C)
        self.head = nn.Linear(C, V, bias=False)          # (V, C)
        self.head.weight = self.wte.weight               # weight tying

    def forward(self, tokens, targets=None):             # tokens: (B, T) ints
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)             # (B, T, C) activation
        for blk in self.blocks:
            x = blk(x)                                   # (B, T, C) each
        x = self.lnf(x)
        logits = self.head(x)                            # (B, T, V) activation
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),        # (B*T, V)
                targets.view(-1))                        # (B*T,)   scalar loss
        return logits, loss
```

And the four-line training step — the subject of the [next lesson](lesson-05.md), previewed here so you see where it plugs in:

```python
optimizer.zero_grad()                 # clear last step's .grad
logits, loss = model(x, y)            # forward: activations + scalar loss
loss.backward()                       # reverse-mode AD: fills every param.grad
optimizer.step()                      # AdamW/Muon: update params from .grad + state
```

## 13. Point at any tensor and name it

That is the whole training step. Every quantity in a real GPT-2 implementation is now nameable. Run down the checklist:

- `tokens`, `targets` $(B,T)$ ints — **input data**, no gradient.
- `wte` $(V,C)$, `wpe` $(T,C)$, every block's LN/QKV/proj/MLP weights and biases, final LN, head — **parameters**.
- `x` $(B,T,C)$, $Q/K/V$, attention scores $(B,n_h,T,T)$, GELU outputs, `logits` $(B,T,V)$, `loss` $()$ — **activations**.
- every `param.grad`, same shape as its parameter — **gradients** from `backward()`.
- AdamW's $m,v$ and Muon's momentum buffer, same shape as the parameter they track — **optimizer state**.

If you can attach one of those four labels and one shape to every tensor you see in GPT-2 source, this lesson has done its job.

## Check yourself

<details><summary>A block's `mlp.fc.weight` has shape $(4C, C)$. Classify it, and give the shape of its gradient and its AdamW optimizer state.</summary>

It is a **parameter**, shape $(4C, C) = (256, 64)$. Its gradient `mlp.fc.weight.grad` is **gradient**, same shape $(256,64)$. AdamW keeps $m$ and $v$ as **optimizer state**, each also $(256,64)$. Parameter, gradient, and each moment all share the parameter's shape.

</details>

<details><summary>Why do targets equal the inputs shifted left by one?</summary>

Because the task is next-token prediction: at position $t$ the model sees tokens $\le t$ and must predict the token at $t+1$. Shifting the input left by one makes the label at position $t$ be the actual next token, so the same tensor serves as both input (positions $0..T-1$) and target (positions $1..T$).

</details>

<details><summary>The toy model has 264,576 parameters. Roughly how much memory (float32) do parameters + gradients + AdamW state need, before activations?</summary>

Four full-model copies: parameters ($P$) + gradients ($P$) + AdamW $m$ and $v$ ($2P$) $= 4P$ numbers $= 16P$ bytes in float32. $16 \times 264{,}576 \approx 4.23$ MB. Activations are extra and scale with $B\times T$.

</details>

<details><summary>Which parameters go to Muon and which to AdamW, and why?</summary>

Muon takes the 2-D hidden weight matrices (QKV, attention proj, MLP fc/proj) because orthogonalization is a statement about a matrix's singular structure. AdamW takes everything else — embeddings, the head, LayerNorm $\gamma/\beta$, and all 1-D biases — which have no meaningful matrix geometry ([lesson 28](../module-09/lesson-03.md)).

</details>

## Next

You have assembled the whole model and named every tensor. The final lesson zooms into the four-line training step itself and reads each line — `zero_grad`, forward, `backward`, `step` — as a precise mathematical statement, then annotates a fully realistic loop.

Continue to [38 · Fine-tuning and LoRA](lesson-04.md).
