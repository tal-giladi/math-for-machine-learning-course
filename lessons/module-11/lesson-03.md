# 34 · Normalization: LayerNorm & RMSNorm

<div class="prereq">
<p><strong>Prerequisites:</strong> mean, variance, and standard deviation from <a href="#/lessons/module-10/lesson-01">the probability & statistics module</a>; the Transformer block from <a href="#/lessons/module-11/lesson-02">33 · The Transformer block</a>, which used LayerNorm as a black box; element-wise operations and the feature axis $C$ from <a href="#/lessons/module-04/lesson-02">module 4</a>; the fan-out/branching gradient rule from <a href="#/lessons/module-07/lesson-01">module 7</a>.</p>
<p><strong>You will learn:</strong> exactly what LayerNorm computes — mean, variance, standardize, then a learned scale and shift — worked by hand on $x = [1, 2, 3]$; what the $\varepsilon$ term is for; what RMSNorm does differently (no mean subtraction, no bias) and why it still works; which axis normalization runs over; and how each appears in PyTorch, matching the hand numbers.</p>
<p><strong>Why this matters for ML:</strong> normalization is what keeps a deep Transformer's activations at a usable scale from the first block to the last. LayerNorm is in every GPT-2 block; RMSNorm is its leaner replacement in LLaMA and most recent LLMs. Both are direct applications of the mean/variance statistics you already know.</p>
</div>

## 1. Recap: mean, variance, standard deviation

From [the statistics module](lessons/module-10/lesson-01.md), for a list of numbers $x = (x_1, \dots, x_n)$:

- The **mean** $\mu$ is the average: $\mu = \frac{1}{n}\sum_{i=1}^{n} x_i$.
- The **variance** $\sigma^2$ is the average squared distance from the mean: $\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2$. (This is the *population*, or *biased*, variance — divide by $n$, not $n-1$. Normalization layers use this version.)
- The **standard deviation** $\sigma$ is $\sqrt{\sigma^2}$: the typical distance of a value from the mean, in the original units.

To **standardize** a value is to measure it in units of standard deviation from the mean: $\hat{x}_i = (x_i - \mu)/\sigma$. The standardized list has mean $0$ and variance $1$ by construction. That single operation is the heart of both normalization layers in this lesson.

## 2. LayerNorm: intuition

As a vector passes through block after block of a Transformer, its numbers can drift — growing large in some layers, tiny in others. Downstream layers then receive input at unpredictable scales, which makes training unstable and slow. **LayerNorm** fixes this by re-centering and re-scaling each token's feature vector back to a standard scale (mean $0$, variance $1$) before it enters a sublayer, then letting the network *learn* how much to stretch and shift it via two parameters. The analogy: no matter how loud or quiet the signal arriving from the previous layer, LayerNorm resets the volume to a known level, and a learned knob decides the final volume from there.

Crucially, LayerNorm normalizes **each token independently, over its own $C$ features** — it does not look at other tokens or other sequences. That independence is what distinguishes it from BatchNorm (section 8).

## 3. LayerNorm: the mathematics

For one token's feature vector $x \in \mathbb{R}^C$, LayerNorm computes:

$$
\mu = \frac{1}{C}\sum_{i=1}^{C} x_i, \qquad
\sigma^2 = \frac{1}{C}\sum_{i=1}^{C} (x_i - \mu)^2,
$$
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad
y_i = \gamma_i\,\hat{x}_i + \beta_i.
$$

Naming every symbol and its shape:

- $x$ — the input feature vector for one token, shape $(C,)$.
- $\mu$ — the mean of those $C$ features, a single scalar.
- $\sigma^2$ — the (population/biased) variance of the $C$ features, a scalar.
- $\varepsilon$ — a tiny constant (default $10^{-5}$) added inside the square root so we never divide by zero when the variance is near zero; it also keeps the operation numerically stable.
- $\hat{x}$ — the standardized vector, shape $(C,)$, with mean $\approx 0$ and variance $\approx 1$.
- $\gamma$ — a learned **scale** (gain) vector, shape $(C,)$, applied element-wise ($\odot$).
- $\beta$ — a learned **shift** (bias) vector, shape $(C,)$.
- $y$ — the output, shape $(C,)$, same shape as the input.

The $\gamma$ and $\beta$ matter: standardizing throws away the vector's overall scale and offset, but sometimes the network *wants* a particular scale. Giving each feature its own learnable $\gamma_i$ and $\beta_i$ lets the model recover any scale/shift it finds useful — including undoing the normalization entirely if that were best. There are $2C$ learned parameters total. Initialized at $\gamma = 1, \beta = 0$, LayerNorm starts as pure standardization.

## 4. LayerNorm: a tiny numerical example

Take $x = [1, 2, 3]$ (so $C = 3$), with the defaults $\gamma = [1,1,1]$, $\beta = [0,0,0]$, $\varepsilon = 10^{-5}$.

**Mean:**

$$
\mu = \frac{1 + 2 + 3}{3} = \frac{6}{3} = 2.
$$

**Variance** (subtract the mean, square, average):

$$
\sigma^2 = \frac{(1-2)^2 + (2-2)^2 + (3-2)^2}{3} = \frac{1 + 0 + 1}{3} = \frac{2}{3} \approx 0.6667.
$$

**Standard deviation** (with $\varepsilon$ folded in):

$$
\sqrt{\sigma^2 + \varepsilon} = \sqrt{0.6667 + 0.00001} = \sqrt{0.66671} \approx 0.8165.
$$

**Standardize** each component, $\hat{x}_i = (x_i - 2)/0.8165$:

$$
\hat{x}_1 = \frac{1-2}{0.8165} = -1.2247, \quad
\hat{x}_2 = \frac{2-2}{0.8165} = 0, \quad
\hat{x}_3 = \frac{3-2}{0.8165} = 1.2247.
$$

$$
\hat{x} = [\,-1.2247,\ 0,\ 1.2247\,].
$$

**Scale and shift:** with $\gamma = 1$ and $\beta = 0$, $y = \hat{x}$ unchanged. Check the result: its mean is $(-1.2247 + 0 + 1.2247)/3 = 0$ and its variance is $((-1.2247)^2 + 0 + 1.2247^2)/3 = (1.5 + 1.5)/3 = 1$. LayerNorm delivered exactly mean $0$, variance $1$, as promised.

## 5. RMSNorm: the leaner alternative

**RMSNorm** (root-mean-square normalization) is a simplification of LayerNorm used by LLaMA and most recent LLMs. It makes two changes: it does **not subtract the mean** (no centering), and it has **no shift parameter $\beta$**. It divides by the root-mean-square of the features instead of the standard deviation:

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{C}\sum_{i=1}^{C} x_i^2 + \varepsilon}, \qquad
y_i = \gamma_i \cdot \frac{x_i}{\mathrm{RMS}(x)}.
$$

The only statistic it needs is the mean of the *squares* — note there is no "$x_i - \mu$" anywhere. It keeps the learned scale $\gamma$ (shape $(C,)$) but drops $\beta$, so it has $C$ parameters instead of LayerNorm's $2C$.

**Worked example**, same $x = [1, 2, 3]$, $\gamma = [1,1,1]$, $\varepsilon = 10^{-5}$:

**Mean of squares:**

$$
\frac{1^2 + 2^2 + 3^2}{3} = \frac{1 + 4 + 9}{3} = \frac{14}{3} \approx 4.6667.
$$

**Root-mean-square:**

$$
\mathrm{RMS}(x) = \sqrt{4.6667 + 0.00001} \approx \sqrt{4.6667} \approx 2.1602.
$$

**Divide each component** by $2.1602$:

$$
y_1 = \frac{1}{2.1602} \approx 0.4629, \quad
y_2 = \frac{2}{2.1602} \approx 0.9258, \quad
y_3 = \frac{3}{2.1602} \approx 1.3887.
$$

$$
y = [\,0.4629,\ 0.9258,\ 1.3887\,].
$$

## 6. The difference, precisely — and why RMSNorm still works

Compare the two outputs for $x = [1, 2, 3]$:

| | LayerNorm | RMSNorm |
| --- | --- | --- |
| Subtracts mean? | yes ($\mu = 2$) | no |
| Divides by | $\sqrt{\sigma^2 + \varepsilon} = 0.8165$ | $\sqrt{\overline{x^2} + \varepsilon} = 2.1602$ |
| Learned params | $\gamma$ and $\beta$ ($2C$) | $\gamma$ only ($C$) |
| Output | $[-1.2247, 0, 1.2247]$ | $[0.4629, 0.9258, 1.3887]$ |
| Output mean | $0$ (centered) | not forced to $0$ |

RMSNorm skips **centering** (mean subtraction) and skips the **bias** $\beta$. What it keeps is the part that matters most: rescaling the vector to a controlled magnitude. The insight behind RMSNorm is that the *scale* of the activations is what drifts and destabilizes deep training; the *mean offset* turns out to matter much less. Dropping the mean saves computing $\mu$ and the subtraction across all $C$ features, and dropping $\beta$ saves parameters and a backward path. The result is cheaper — fewer operations, fewer parameters, one less reduction over $C$ — while giving up almost nothing in quality, which is why modern large models adopted it. RMSNorm is essentially "LayerNorm without the parts that were not pulling their weight."

## 7. The axis: normalize over C, per token

Both layers normalize **over the feature axis $C$**, independently for each position. On a $(B, T, C)$ activation, that means: for every one of the $B \cdot T$ tokens, take its own $C$-vector, compute that vector's own mean/variance (LayerNorm) or mean-of-squares (RMSNorm), and normalize it. The statistics of token $(b, t)$ never mix with those of any other token or any other sequence. So there are $B \cdot T$ independent normalizations per layer, each over $C$ numbers. The same $\gamma$ (and $\beta$) vectors of shape $(C,)$ are shared across all tokens — they are per-*feature*, not per-token.

<div class="callout key"><p>LayerNorm and RMSNorm both normalize each token's $C$-feature vector on its own, over the $C$ axis, using only that token's own numbers. LayerNorm centers (subtract $\mu$) and rescales (divide by $\sigma$), with learned $\gamma, \beta$. RMSNorm only rescales (divide by RMS), with learned $\gamma$. Same axis, same per-token independence; RMSNorm just drops the centering and the bias.</p></div>

## 8. In PyTorch

`nn.LayerNorm(C)` is the built-in; here it is next to a manual computation that reproduces the section-4 numbers exactly.

```python
import torch
import torch.nn as nn

x = torch.tensor([1., 2., 3.])          # (C,) = (3,)

# --- built-in LayerNorm over the last dim of size C=3 ---
ln = nn.LayerNorm(3)                     # gamma=1, beta=0, eps=1e-5 by default
print(ln(x))     # tensor([-1.2247,  0.0000,  1.2247], grad_fn=<NativeLayerNormBackward0>)

# --- manual LayerNorm, matching the by-hand steps ---
mu = x.mean()                            # 2.0
var = x.var(unbiased=False)              # 0.6667  (population variance, divide by C)
xhat = (x - mu) / torch.sqrt(var + 1e-5) # [-1.2247, 0.0000, 1.2247]
y = 1.0 * xhat + 0.0                     # gamma=1, beta=0
print(y)         # tensor([-1.2247,  0.0000,  1.2247])
```

The critical detail is `unbiased=False`: LayerNorm uses the population variance (divide by $C$), so the manual version must too, or the numbers will not match. Now RMSNorm, which has no built-in in older PyTorch, written manually to match section 5:

```python
# --- manual RMSNorm ---
ms = (x**2).mean()                       # 4.6667  (mean of squares, no centering)
rms = torch.sqrt(ms + 1e-5)              # 2.1602
y_rms = 1.0 * (x / rms)                  # gamma=1, no beta
print(y_rms)     # tensor([0.4629, 0.9258, 1.3887])
```

Both snippets print exactly the hand-computed vectors. For a real $(B, T, C)$ activation, `nn.LayerNorm(C)` applied to the last axis does all $B \cdot T$ per-token normalizations at once; the manual versions would use `x.mean(dim=-1, keepdim=True)` and `x.var(dim=-1, keepdim=True, unbiased=False)` so the reductions run over the $C$ axis only.

<div class="callout pt"><p><code>nn.LayerNorm(C)</code> normalizes over the last dimension of size $C$ and holds two learnable vectors, <code>weight</code> ($\gamma$) and <code>bias</code> ($\beta$), each shape $(C,)$. RMSNorm (e.g. <code>nn.RMSNorm</code> in recent PyTorch) holds only <code>weight</code> ($\gamma$) — no bias — and divides by the root-mean-square instead of the standard deviation.</p></div>

## 9. What PyTorch is doing under the hood

**The backward pass couples all $C$ features.** In a plain element-wise layer, the gradient of output $y_i$ depends only on input $x_i$. LayerNorm is different: because $\mu$ and $\sigma$ are computed from *all* $C$ features, every output $y_i$ depends on *every* input $x_j$. So during backprop, the gradient with respect to one input $x_j$ gathers contributions routed through the shared $\mu$ and $\sigma^2$ from all $C$ outputs — a fan-out through the shared statistics, summed exactly as the branching rule of [module 7](lessons/module-07/lesson-01.md) prescribes. This is why LayerNorm's backward formula is more involved than a simple activation's: it must account for how nudging one feature shifts the mean and variance that scale *all* the features. Autograd's `NativeLayerNormBackward` implements this coupled derivative for you; `loss.backward()` computes gradients for $x$, $\gamma$, and $\beta$ automatically. RMSNorm's backward is similar but simpler — it couples through only the shared RMS, with no mean term and no $\beta$ gradient.

**No running statistics.** LayerNorm and RMSNorm compute their statistics fresh from the current input every forward pass, and they behave identically in training and inference. This is unlike **BatchNorm**, which normalizes over the *batch* axis and must maintain running averages of the mean and variance to use at inference — a dependence on batch composition that LayerNorm's per-token design deliberately avoids, which is one reason Transformers use LayerNorm rather than BatchNorm.

## Check yourself

<details><summary>For $x = [1, 2, 3]$, what are $\mu$, $\sigma^2$ (population), and the LayerNorm output with $\gamma=1, \beta=0$?</summary>

$\mu = 2$; $\sigma^2 = \frac{(1{-}2)^2 + (2{-}2)^2 + (3{-}2)^2}{3} = \frac{2}{3} \approx 0.6667$; dividing $x-\mu$ by $\sqrt{0.6667 + 10^{-5}} \approx 0.8165$ gives $[-1.2247, 0, 1.2247]$.

</details>

<details><summary>Name the two things RMSNorm drops relative to LayerNorm, and give RMSNorm's output for $x = [1,2,3]$.</summary>

It drops mean subtraction (no centering) and the shift parameter $\beta$. With $\mathrm{RMS} = \sqrt{14/3 + \varepsilon} \approx 2.1602$, the output is $x / 2.1602 = [0.4629, 0.9258, 1.3887]$.

</details>

<details><summary>Which axis does LayerNorm normalize over in a $(B, T, C)$ tensor, and how many independent normalizations are there?</summary>

Over the feature axis $C$, independently for each token. There are $B \cdot T$ independent normalizations, each computed from one token's own $C$ numbers; $\gamma$ and $\beta$ (shape $(C,)$) are shared across all tokens.

</details>

<details><summary>Why is LayerNorm's backward pass not element-wise, unlike a plain activation?</summary>

Because $\mu$ and $\sigma^2$ are computed from all $C$ features, every output depends on every input. The gradient with respect to one input therefore collects contributions from all $C$ outputs, routed through the shared mean and variance — the features are coupled, so the backward pass mixes them.

</details>

## Next

You now understand every component of a Transformer block down to its arithmetic: attention, the residual, the MLP with GELU, and both flavors of normalization. What you have not yet seen is *why* all this careful scaling and residual structure is necessary — what actually goes wrong in deep networks without it. That is the numerical-stability story: saturating softmax, vanishing and exploding gradients, and the tricks (stable softmax, the $\varepsilon$ terms you just met, gradient clipping) that keep training alive.

Continue to [Module 12 · Numerical stability](lessons/module-12/lesson-01.md).
