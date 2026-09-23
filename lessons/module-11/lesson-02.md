# 33 · The Transformer block

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-11/lesson-01">32 · Attention mathematics</a> (scaled dot-product and multi-head attention); the $(B, T, C)$ shape and reshape/transpose machinery from <a href="#/lessons/module-04/lesson-02">13 · (B, T, C), reshape, view, permute, broadcast</a>; the branching/fan-out rule for gradients from <a href="#/lessons/module-07/lesson-01">19 · The chain rule and computational graphs</a>; matrix–vector multiplication from <a href="#/lessons/module-03/lesson-01">module 3</a>. LayerNorm gets its own full treatment in the <a href="#/lessons/module-11/lesson-03">next lesson</a>.</p>
<p><strong>You will learn:</strong> how to assemble a modern <strong>pre-norm Transformer block</strong> — LayerNorm, QKV projection, attention, output projection, residual, LayerNorm, MLP, residual — naming every parameter matrix and its shape; why residual connections create a gradient highway (with the derivative shown); what GELU is and how it differs from ReLU; and the rough parameter count of a block in terms of $C$.</p>
<p><strong>Why this matters for ML:</strong> a GPT-2 model is almost nothing but a stack of identical Transformer blocks. Learn one block and you have learned the whole trunk of the network. The residual and normalization structure here is also exactly what makes deep stacks trainable at all — the subject of <a href="#/lessons/module-12/lesson-01">module 12</a>.</p>
</div>

## 1. Intuition: one block, two sublayers

A Transformer block does two jobs, in order. First it lets each token **mix information with other tokens** — that is attention, from the previous lesson. Then it lets each token **think about that information on its own** — a small two-layer neural network, called the MLP or feed-forward network, applied to each position independently. Everything else in the block (the normalization and the residual connections) exists to make those two sublayers trainable when you stack dozens of them.

The programmer's mental model: a block is a function from $(B, T, C)$ to $(B, T, C)$ — same shape in, same shape out. Because the shape is preserved, blocks are **stackable**: the output of block $\ell$ is a legal input to block $\ell+1$. GPT-2 small stacks $12$ of them; GPT-2 large stacks $36$. A block never changes what a tensor *looks like*, only what it *contains*.

## 2. The pre-norm block, end to end

Modern GPTs use the **pre-norm** arrangement: normalize *before* each sublayer, and add the sublayer's result back onto its input. Writing $x$ for the $(B, T, C)$ input to the block, the two sublayers are:

$$
x' = x + \mathrm{Attention}\big(\mathrm{LayerNorm}_1(x)\big),
$$
$$
y = x' + \mathrm{MLP}\big(\mathrm{LayerNorm}_2(x')\big).
$$

Read it as a pipeline, which is how the code reads too:

```text
x ─┬─────────────────────────────────────────────▶ (+) ─┬───────────────────────────────────▶ (+) ─▶ y
   │                                                 ▲    │                                     ▲
   └─▶ LayerNorm1 ─▶ QKV ─▶ attention ─▶ proj W_O ───┘    └─▶ LayerNorm2 ─▶ MLP (Linear,GELU,Linear) ─┘
```

The top line carrying $x$ straight through to each $(+)$ is the **residual stream** — the untouched copy of the input that each sublayer adds onto. The two branches dropping down are the two sublayers. We now walk each component and name its parameters. Throughout, $C$ is the channel width ($d_\text{model}$); GPT-2 small has $C = 768$.

## 3. LayerNorm

Each sublayer's input first passes through a **LayerNorm**, which rescales each token's $C$-vector to have mean $0$ and variance $1$, then applies a learned scale $\gamma$ and shift $\beta$ (both length $C$). The full mechanics — mean, variance, the $\varepsilon$ term, the backward pass — are the whole of the [next lesson](lessons/module-11/lesson-03.md); here you only need three facts:

- It operates **per token, over the $C$ feature axis**, independently for each $(b, t)$ position.
- It has two parameter vectors, $\gamma$ and $\beta$, each of shape $(C,)$ — a total of $2C$ parameters, negligible next to the big matrices below.
- Its output has the same shape as its input, $(B, T, C)$.

Its purpose in the block is stability: it keeps the numbers entering attention and the MLP at a consistent scale no matter how deep the stack is, which stops activations from drifting to huge or tiny magnitudes as they pass through many blocks. Pre-norm puts it *before* each sublayer so the sublayer always receives well-scaled input.

## 4. The QKV projection

Attention needs a query, key, and value vector for every token. These are produced from the (normalized) token vector by a single **linear layer** that maps width $C$ to width $3C$:

$$
[\,Q \mid K \mid V\,] = \mathrm{LayerNorm}_1(x)\, W_{qkv} + b_{qkv},
$$

with $W_{qkv}$ of shape $(C, 3C)$ and bias $b_{qkv}$ of shape $(3C,)$. For each token the layer outputs $3C$ numbers, which we slice into three $C$-vectors: the first $C$ are that token's query, the next $C$ its key, the last $C$ its value. (Producing all three with one $C \times 3C$ matrix instead of three separate $C \times C$ matrices is just an efficiency: one matmul rather than three.) Each of $Q, K, V$ is then reshaped into $(B, n_h, T, d_h)$ and fed to the attention of [lesson 1](lessons/module-11/lesson-01.md).

Parameter cost: the matrix $W_{qkv}$ has $C \cdot 3C = 3C^2$ entries.

## 5. Attention and the output projection

Attention itself (lesson 1) has **no parameters of its own** — it is pure arithmetic on $Q, K, V$: scores, scale, mask, softmax, weighted sum. It returns a $(B, T, C)$ tensor (the concatenated heads). That result is then passed through the **output projection**, a linear layer mapping $C \to C$:

$$
\text{attn\_out} = \mathrm{Attention}(Q, K, V)\, W_O + b_O,
$$

with $W_O$ of shape $(C, C)$ and bias $(C,)$. As explained in lesson 1, $W_O$ mixes information across the heads and produces the sublayer's final contribution. Parameter cost: $C^2$.

So the whole attention sublayer's weight cost is $3C^2$ (QKV) $+\ C^2$ (output) $= 4C^2$, plus small biases.

## 6. The residual connection and why gradients flow through it

After the attention sublayer computes its contribution $f(x) = \mathrm{Attention}(\mathrm{LayerNorm}_1(x))\,W_O$, we do not replace $x$ with it. We **add** it:

$$
x' = x + f(x).
$$

This is a **residual** (or skip) connection, and it is the single most important structural trick for training deep networks. Here is the intuition and then the mathematics.

**Intuition.** The sublayer only has to learn a *correction* to add to the input, not to reproduce the whole input from scratch. If a sublayer has nothing useful to add, it can output near zero and the input passes through essentially unchanged. Stacking many such "input plus a small correction" steps is far more stable than stacking many "completely transform the input" steps.

**The gradient highway.** Consider how a gradient flows backward through $x' = x + f(x)$. Differentiating with respect to $x$:

$$
\frac{\partial x'}{\partial x} = \frac{\partial}{\partial x}\big(x + f(x)\big) = 1 + f'(x).
$$

The key is that leading **$1$**. During backpropagation, the gradient arriving at $x'$ is passed to $x$ multiplied by $(1 + f'(x))$. Even if the sublayer's own derivative $f'(x)$ is tiny — which happens when a layer is near-saturated or barely contributing — the gradient still reaches $x$ essentially undiminished, because of the $+1$. Without the residual, the backward factor would be just $f'(x)$, and a chain of $36$ tiny factors multiplied together would shrink toward zero: the **vanishing gradient** problem, where early layers stop learning. The residual guarantees each block contributes an additive $1$ to the backward path, so gradients flow from the loss all the way back to the earliest block. This connection to vanishing and exploding gradients in deep stacks is the theme of [module 12](lessons/module-12/lesson-01.md).

<div class="callout key"><p>A residual $x' = x + f(x)$ has derivative $1 + f'(x)$. The $+1$ is a gradient highway: it carries the gradient back through the block undiminished even when the sublayer's own gradient $f'(x)$ is near zero. This is what makes a stack of dozens of blocks trainable.</p></div>

The second sublayer uses a residual in exactly the same way: $y = x' + \mathrm{MLP}(\mathrm{LayerNorm}_2(x'))$.

## 7. The MLP (feed-forward network)

After attention has mixed information across positions, the **MLP** processes each position on its own. It is a two-layer network with a wider hidden layer: it expands the width from $C$ up to $4C$, applies a nonlinear activation, and projects back down to $C$:

$$
\mathrm{MLP}(z) = \mathrm{GELU}(z\, W_1 + b_1)\, W_2 + b_2,
$$

with shapes

- $W_1$: $(C, 4C)$, bias $b_1$: $(4C,)$ — the **up-projection** to the hidden width.
- $W_2$: $(4C, C)$, bias $b_2$: $(C,)$ — the **down-projection** back to $C$.

The **$4\times$ expansion** is a GPT-2 convention: the hidden layer is four times as wide as the model, giving the network room to compute richer per-token features before compressing back. The activation in the middle is what makes the MLP nonlinear — without it, two stacked linear layers would collapse into a single linear layer and could compute nothing a single matrix could not.

Parameter cost: $W_1$ has $C \cdot 4C = 4C^2$ entries and $W_2$ has $4C \cdot C = 4C^2$, for $8C^2$ total.

## 8. GELU

GPT-2's MLP uses **GELU** (Gaussian Error Linear Unit) as its activation, not the more familiar ReLU. Compare the two.

**ReLU** is the simplest nonlinearity: it passes positive inputs unchanged and clamps negatives to zero.

$$
\mathrm{ReLU}(x) = \max(0, x).
$$

It has a hard corner at $x = 0$ and a derivative that is exactly $0$ for all negative inputs — so a unit stuck in the negative region receives no gradient at all (a "dead" unit).

**GELU** is a smooth version. It weights the input by the probability, under a standard normal distribution, that a value falls below $x$ — written $\Phi(x)$, the normal cumulative distribution function:

$$
\mathrm{GELU}(x) = x \cdot \Phi(x).
$$

A practical closed form (the tanh approximation GPT-2 actually uses) is

$$
\mathrm{GELU}(x) \approx 0.5\,x\left(1 + \tanh\!\left[\sqrt{\tfrac{2}{\pi}}\,\big(x + 0.044715\,x^3\big)\right]\right).
$$

The important differences from ReLU: GELU is **smooth** (no corner — differentiable everywhere), and it lets a *little* signal through for small negative inputs rather than hard-zeroing them, so units are less prone to dying. A few values make the shape concrete:

| $x$ | $\mathrm{ReLU}(x)$ | $\mathrm{GELU}(x)$ |
| --- | --- | --- |
| $-2$ | $0$ | $\approx -0.0455$ |
| $-1$ | $0$ | $\approx -0.1587$ |
| $0$ | $0$ | $0$ |
| $1$ | $1$ | $\approx 0.8413$ |
| $2$ | $2$ | $\approx 1.9545$ |

At $x = 1$, GELU passes $0.8413$ (a bit less than ReLU's full $1$); at $x = -1$ it passes a small negative $-0.1587$ where ReLU passes nothing; and for large $|x|$ it converges to ReLU. Smoothness gives it a well-defined, nonzero derivative almost everywhere, which autograd differentiates automatically.

## 9. Parameter count of a block

Adding up the weight matrices (ignoring the small bias and LayerNorm vectors, which total roughly $6C$):

| Component | Matrix | Parameters |
| --- | --- | --- |
| QKV projection | $W_{qkv}$: $(C, 3C)$ | $3C^2$ |
| Output projection | $W_O$: $(C, C)$ | $C^2$ |
| MLP up | $W_1$: $(C, 4C)$ | $4C^2$ |
| MLP down | $W_2$: $(4C, C)$ | $4C^2$ |

So the attention sublayer holds about $4C^2$ parameters and the MLP about $8C^2$, giving roughly $\mathbf{12C^2}$ parameters per block. For GPT-2 small, $C = 768$, so $12 \cdot 768^2 \approx 7.1$ million parameters per block, times $12$ blocks $\approx 85$ million — the bulk of GPT-2's $124$ million (the rest is the token and position embeddings). Notice the MLP is *twice* the size of attention; most of a Transformer's weights live in its feed-forward layers.

## 10. Pre-norm vs post-norm

The original 2017 Transformer put the LayerNorm *after* each sublayer, on the residual sum — **post-norm**:

$$
x' = \mathrm{LayerNorm}\big(x + \mathrm{Attention}(x)\big). \quad(\text{post-norm})
$$

Modern GPTs instead put it *before* the sublayer, inside the branch — **pre-norm**, which is what sections 2–7 used:

$$
x' = x + \mathrm{Attention}\big(\mathrm{LayerNorm}(x)\big). \quad(\text{pre-norm})
$$

The difference matters for the gradient highway of section 6. In pre-norm the residual addition is "clean" — nothing is applied to the summed stream itself — so the $+1$ path runs unobstructed from the loss to the very first block. In post-norm a LayerNorm sits *on* the residual sum, so the clean highway is interrupted at every block, which makes very deep post-norm stacks harder to train (they often need a careful learning-rate warmup). Pre-norm is more robust to depth, so essentially all modern GPT-style models use it.

## 11. In PyTorch

A compact block as an `nn.Module`, wiring together exactly the components above. Attention is written inline as the three lines from lesson 1.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, C, n_h):
        super().__init__()
        self.n_h = n_h
        self.ln1 = nn.LayerNorm(C)
        self.qkv = nn.Linear(C, 3 * C)      # W_qkv: (C, 3C)
        self.proj = nn.Linear(C, C)         # W_O:   (C, C)
        self.ln2 = nn.LayerNorm(C)
        self.mlp = nn.Sequential(
            nn.Linear(C, 4 * C),            # W_1: (C, 4C)
            nn.GELU(),
            nn.Linear(4 * C, C),            # W_2: (4C, C)
        )

    def attention(self, x):                 # x: (B, T, C)
        B, T, C = x.shape
        d_h = C // self.n_h
        q, k, v = self.qkv(x).split(C, dim=-1)             # each (B, T, C)
        # (B, T, C) -> (B, n_h, T, d_h)
        q = q.view(B, T, self.n_h, d_h).transpose(1, 2)
        k = k.view(B, T, self.n_h, d_h).transpose(1, 2)
        v = v.view(B, T, self.n_h, d_h).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / d_h**0.5        # (B, n_h, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(scores, dim=-1)
        out = w @ v                                        # (B, n_h, T, d_h)
        out = out.transpose(1, 2).contiguous().view(B, T, C)   # concat heads
        return self.proj(out)

    def forward(self, x):                   # x: (B, T, C)
        x = x + self.attention(self.ln1(x))     # sublayer 1 + residual
        x = x + self.mlp(self.ln2(x))           # sublayer 2 + residual
        return x

B, T, C, n_h = 2, 5, 8, 4
block = Block(C, n_h)
x = torch.randn(B, T, C)
y = block(x)
print(y.shape)          # torch.Size([2, 5, 8])  -> same shape in and out
```

The output shape equals the input shape, $(B, T, C)$, so `block` composes with itself: `nn.Sequential(*[Block(C, n_h) for _ in range(12)])` is essentially the GPT-2 trunk. The `forward` reads as the two equations of section 2 — `x = x + attention(ln1(x))` then `x = x + mlp(ln2(x))` — with the residual `x +` sitting in front of each sublayer.

<div class="callout pt"><p>The two lines <code>x = x + self.attention(self.ln1(x))</code> and <code>x = x + self.mlp(self.ln2(x))</code> ARE the pre-norm block. Everything else in the module is the definition of the two sublayers. The <code>x +</code> in front is the residual; the <code>ln</code> inside is pre-norm.</p></div>

## 12. What PyTorch is doing under the hood

**The residual splits the gradient two ways.** In the forward pass, `x = x + self.attention(self.ln1(x))` uses `x` in two places: it flows straight into the sum, and it flows into the attention branch. That is a **fan-out**, and by the branching rule from [lesson 19](lessons/module-07/lesson-01.md), the gradient of `x` on the backward pass is the *sum* of the two contributions coming back: one arriving directly through the addition (with local derivative $1$ — the highway), and one arriving through the whole attention sublayer (with the sublayer's own derivative). Autograd implements the `+` with a backward rule that simply copies the incoming gradient to *both* inputs, which is precisely the $1 + f'(x)$ of section 6 realized as graph arithmetic.

**GELU's derivative is handled automatically.** `nn.GELU()` registers a backward rule for its smooth activation, so when `loss.backward()` runs, autograd multiplies the upstream gradient by GELU's local derivative at each hidden unit — no manual formula needed. Because GELU is smooth, that derivative is defined and nonzero almost everywhere, unlike ReLU's flat-zero region for negatives.

**Every matmul and LayerNorm carries a `grad_fn`.** Each `nn.Linear`, the attention matmuls, the softmax, and both LayerNorms record themselves on the computation graph during `forward`. When you later call `.backward()` on the loss, autograd walks that graph from the loss back to every parameter, depositing $\partial L/\partial \theta$ in each parameter's `.grad`. The residual additions ensure that walk reaches even the first block's parameters with a healthy gradient. LayerNorm's own (more involved) backward pass is the subject of the next lesson.

## Check yourself

<details><summary>Write the two equations of a pre-norm Transformer block, given input $x$.</summary>

$x' = x + \mathrm{Attention}(\mathrm{LayerNorm}_1(x))$, then $y = x' + \mathrm{MLP}(\mathrm{LayerNorm}_2(x'))$. Normalize inside each branch, and add the branch's result back onto the residual stream.

</details>

<details><summary>What is the derivative of $x' = x + f(x)$ with respect to $x$, and why does the leading term matter?</summary>

$\frac{\partial x'}{\partial x} = 1 + f'(x)$. The $+1$ is a gradient highway: it lets the gradient pass back through the block undiminished even when the sublayer derivative $f'(x)$ is near zero, preventing vanishing gradients in a deep stack.

</details>

<details><summary>Roughly how many parameters does one block have in terms of $C$, and how are they split between attention and the MLP?</summary>

About $12C^2$: attention holds $\approx 4C^2$ ($3C^2$ for QKV plus $C^2$ for the output projection), and the MLP holds $\approx 8C^2$ ($4C^2$ up plus $4C^2$ down). The MLP is twice the size of attention.

</details>

<details><summary>Name two ways GELU differs from ReLU.</summary>

GELU is smooth (differentiable everywhere, no hard corner at $0$), and it passes a small nonzero signal for small negative inputs rather than clamping all negatives to exactly zero — so it avoids ReLU's flat-zero derivative region and the "dead unit" problem.

</details>

## Next

You can now assemble and read a full Transformer block: two sublayers, each normalized, wrapped in a residual, mapping $(B, T, C)$ to $(B, T, C)$ so blocks stack. The one component we deferred is the LayerNorm itself — its mean/variance arithmetic, its backward pass, and its cheaper cousin RMSNorm. That is the next lesson.

Continue to [34 · Normalization: LayerNorm & RMSNorm](lessons/module-11/lesson-03.md).
