# 35 · Numerical issues and stability

<div class="prereq">
<p><strong>Prerequisites:</strong> exponentials and logarithms from <a href="#/lessons/module-01/lesson-03">03 · Powers, roots, exponentials, logarithms</a>; softmax from <a href="#/lessons/module-08/lesson-02">24 · Loss functions</a>; normalization from <a href="#/lessons/module-11/lesson-03">34 · Normalization</a>; Adam and AdamW from <a href="#/lessons/module-09/lesson-02">27 · SGD, momentum, RMSProp, Adam, AdamW</a>; and residual connections from <a href="#/lessons/module-11/lesson-02">33 · The Transformer block</a>.</p>
<p><strong>You will learn:</strong> how real numbers are actually stored (float32, float16, bfloat16 — bits, range, precision), where the arithmetic breaks (rounding, overflow, underflow, <code>NaN</code>, infinity), and the exact tricks production code uses to stay stable — the log-sum-exp identity, stable softmax, epsilon terms, and gradient clipping — with worked numbers for each.</p>
<p><strong>Why this matters for ML:</strong> a single <code>NaN</code> anywhere in a GPT-2 forward pass poisons the whole model and destroys a training run. Every stabilizing trick in the Transformer — subtracting the max before softmax, the $\varepsilon$ inside LayerNorm, clipping the gradient — exists to keep finite-precision arithmetic from blowing up. This lesson is where the mathematics meets the hardware.</p>
</div>

## 1. Why a math course suddenly cares about bits

Everywhere else in this course a number is a number: $0.1$ is exactly one tenth, $e^{1000}$ is a specific (enormous) value, and $\log$ of a positive number always exists. A computer does not have that luxury. It stores each number in a fixed, small number of bits, so most real numbers are stored *approximately*, the largest and smallest representable values are *bounded*, and some operations produce not-a-number results that then contaminate everything they touch.

For an experienced programmer this is the familiar territory of IEEE-754 floating point — but ML pushes it far harder than typical application code, because a network multiplies and sums millions of numbers, exponentiates logits, takes logarithms of probabilities, and divides by standard deviations, all in low precision. This lesson is the catalogue of what goes wrong and the specific mathematics used to prevent it.

## 2. How a floating-point number is stored

A floating-point number is stored like scientific notation in binary: a **sign**, a set of **exponent** bits (how big), and a set of **mantissa** (a.k.a. fraction/significand) bits (the precise digits). The value is roughly

$$
(-1)^{\text{sign}} \times 1.\text{mantissa} \times 2^{\text{exponent}}.
$$

The split of bits between exponent and mantissa is the whole story: **exponent bits buy range** (how large or small a number you can name), **mantissa bits buy precision** (how finely you can distinguish nearby numbers). The three formats used in deep learning spend their bits very differently.

| format | total bits | sign | exponent | mantissa | approx max | approx precision |
|---|---|---|---|---|---|---|
| float32 (FP32) | 32 | 1 | 8 | 23 | $3.4\times10^{38}$ | ~7 decimal digits |
| float16 (FP16, "half") | 16 | 1 | 5 | 10 | $65504$ | ~3 decimal digits |
| bfloat16 (BF16) | 16 | 1 | 8 | 7 | $3.4\times10^{38}$ | ~2–3 decimal digits |

The key insight for training: **bfloat16 keeps float32's 8 exponent bits**, so it has the *same enormous range* as float32 — it can represent numbers up to about $3.4\times10^{38}$ and will not overflow where float32 would not. It pays for that by having only 7 mantissa bits, so it is *less precise* than float16. Float16, by contrast, has more precision (10 mantissa bits) but only 5 exponent bits, so its maximum value is a mere $65504$ — and activations in a large model overflow that limit easily. That single difference is why modern large-model training almost always chooses bfloat16 over float16: range matters more than precision when values can grow large, and the network tolerates a little noise in the low bits.

<div class="callout key"><p>Exponent bits = range; mantissa bits = precision. bfloat16 spends its 16 bits like float32's exponent (same range, no easy overflow) but with far fewer mantissa bits (coarser precision). float16 does the opposite: finer precision, tiny range, easy overflow.</p></div>

## 3. Precision and rounding: 0.1 is a lie

The number $0.1$ has no exact representation in binary floating point, exactly as $\tfrac{1}{3}$ has no exact finite decimal. In binary, $0.1$ is the repeating fraction $0.0001100110011\ldots_2$, which gets cut off at the mantissa's last bit and rounded. The stored value is very slightly off:

```python
print(0.1 + 0.2 == 0.3)   # False
print(0.1 + 0.2)          # 0.30000000000000004
```

The gap between $1.0$ and the next representable number above it is called **machine epsilon**, written $\varepsilon_\text{mach}$. It is the smallest number you can add to $1.0$ and still change it:

- float32: $\varepsilon_\text{mach} = 2^{-23} \approx 1.19\times10^{-7}$.
- float16: $\varepsilon_\text{mach} = 2^{-10} \approx 9.77\times10^{-4}$.
- bfloat16: $\varepsilon_\text{mach} = 2^{-7} \approx 7.81\times10^{-3}$.

Read the bfloat16 line carefully: adding anything smaller than about $0.008$ to $1.0$ in bfloat16 *does nothing at all* — it rounds straight back to $1.0$. This is why accumulations (summing many small numbers, averaging over a long sequence) are done in float32 even when the data is bfloat16: a running sum in bfloat16 stops growing once the total dwarfs the increments.

## 4. Overflow: $e^{1000} = \infty$

Every format has a largest representable value. Produce something bigger and it becomes **infinity** (`inf`), a special bit pattern meaning "too big to store." The exponential function reaches it almost instantly:

```python
import torch
print(torch.tensor(1000.).exp())   # tensor(inf)
```

$e^{1000} \approx 1.97\times10^{434}$, far past float32's ceiling of $3.4\times10^{38}$, so it saturates to `inf`. Logits of size $1000$ are not exotic — they show up in an untrained or diverging model — and the naive softmax exponentiates them directly, which is precisely the disaster section 7 fixes.

## 5. Underflow: $e^{-1000} = 0$

The mirror image. Every format has a smallest positive value; anything closer to zero **underflows** to exactly $0$:

```python
print(torch.tensor(-1000.).exp())   # tensor(0.)
```

$e^{-1000} \approx 5\times10^{-435}$ rounds to $0$. Underflow is quieter than overflow but just as dangerous when you then take a logarithm: $\log(0) = -\infty$, and a probability that underflowed to $0$ turns a cross-entropy loss into $-\infty$, then into `NaN` at the next step.

## 6. NaN and infinity: how one bad number poisons everything

`inf` is a value with a sign and sensible arithmetic ($1/\text{inf}=0$, $\text{inf}+1=\text{inf}$). **`NaN`** ("not a number") is different: it is the result of an operation with no meaningful answer, and it is *contagious* — nearly every arithmetic operation involving a `NaN` produces a `NaN`.

The operations that manufacture `NaN`:

- $0/0$ — no defined value.
- $\infty - \infty$ — indeterminate.
- $0 \times \infty$ — indeterminate.
- $\log(0) = -\infty$, then e.g. $-\infty + \infty$ downstream $\to$ `NaN`.
- $\sqrt{-1}$ on reals.

The reason this matters so much in ML is the poisoning property. A network is one giant sum of products. If a single activation becomes `NaN`, every sum it participates in becomes `NaN`, every downstream layer becomes `NaN`, the loss becomes `NaN`, and `loss.backward()` fills *every* parameter's `.grad` with `NaN`. One optimizer step later, every weight in the model is `NaN`, and the run is dead — a model that took hours to train is destroyed by one bad division.

```python
x = torch.tensor([1.0, float('inf'), 3.0])
print(x - x)                 # tensor([0., nan, 0.])  -> inf - inf = nan
print((x - x).sum())         # tensor(nan)            -> one nan poisons the sum
print(torch.isnan(x - x))    # tensor([False, True, False])  -> how you detect it
```

<div class="callout warn"><p>A single <code>NaN</code> anywhere spreads to the loss and then, through <code>backward()</code>, to every parameter's gradient — the next optimizer step turns the whole model to <code>NaN</code>. When a run "explodes to NaN," find the <em>first</em> operation that produced it (often an <code>exp</code>, a <code>log</code>, or a divide by a near-zero standard deviation); the rest is just contamination.</p></div>

## 7. Fix 1 — the log-sum-exp trick

Many ML quantities have the shape $\log \sum_i e^{x_i}$ (it is the denominator of a log-softmax, and the normalizer of cross-entropy). Computed naively, the $e^{x_i}$ overflow for large $x_i$. The **log-sum-exp identity** rewrites it so the exponentials never blow up. Let $m = \max_i x_i$. Then

$$
\log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m}.
$$

**Why it is exact.** Factor $e^m$ out of the sum: $\sum_i e^{x_i} = e^m \sum_i e^{x_i - m}$. Take the log, and $\log(e^m \cdot S) = m + \log S$. No approximation — it is algebra. The payoff: every exponent $x_i - m$ is $\le 0$, so every $e^{x_i - m} \le 1$. No overflow, ever, and the largest term is exactly $1$ so no underflow of the dominant term either.

**Worked example.** Take $x = [1000, 1001, 1002]$. Naively, $e^{1000}, e^{1001}, e^{1002}$ all overflow to `inf`, their sum is `inf`, and $\log(\infty) = \infty$ — a useless answer. With the trick, $m = 1002$:

$$
x_i - m = [-2,\ -1,\ 0], \qquad e^{-2}=0.135335,\ e^{-1}=0.367879,\ e^{0}=1,
$$

$$
\sum_i e^{x_i-m} = 0.135335 + 0.367879 + 1 = 1.503215,
$$

$$
\log\sum_i e^{x_i} = 1002 + \log(1.503215) = 1002 + 0.407605 = 1002.407605.
$$

A clean, finite answer where the naive route gave `inf`.

## 8. Fix 2 — stable softmax (subtract the max)

Softmax (from [lesson 24](lessons/module-08/lesson-02.md)) is $\text{softmax}(x)_i = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$. The same overflow lurks in the exponentials. The fix uses a property softmax already has: **shift invariance**. Subtract any constant $c$ from every logit and the output is unchanged, because the constant factors out of numerator and denominator:

$$
\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{-c}e^{x_i}}{e^{-c}\sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

Choose $c = m = \max_i x_i$. Then every exponent is $\le 0$ and every $e^{x_i - m} \le 1$ — no overflow. So

$$
\text{softmax}([1000, 1001, 1002]) = \text{softmax}([-2, -1, 0]).
$$

**Worked example.** Using the exponentials from section 7, $\sum = 1.503215$:

$$
\text{softmax} = \left[\frac{0.135335}{1.503215},\ \frac{0.367879}{1.503215},\ \frac{1}{1.503215}\right] = [0.090031,\ 0.244728,\ 0.665241].
$$

These sum to $1$, as probabilities must, and the naive computation on the raw $[1000,1001,1002]$ would have produced $\tfrac{\infty}{\infty} = $ `NaN`.

```python
import torch

x = torch.tensor([1000., 1001., 1002.])

# naive: overflows
naive = x.exp() / x.exp().sum()
print(naive)                       # tensor([nan, nan, nan])

# stable: subtract the max first
stable = (x - x.max()).exp()
stable = stable / stable.sum()
print(stable)                      # tensor([0.0900, 0.2447, 0.6652])

print(torch.softmax(x, dim=0))     # tensor([0.0900, 0.2447, 0.6652])  -- torch does the stable one
```

`torch.softmax` (and `F.log_softmax`) already subtract the max internally, so you get the stable version for free — but knowing *why* it subtracts the max is what lets you read the source and trust it.

## 9. Fix 3 — epsilon terms: never divide by zero, never log zero

The most common stabilizer in a Transformer is a tiny constant $\varepsilon$ (epsilon, typically $10^{-5}$ to $10^{-8}$) added inside a square root or a logarithm so the argument can never be exactly $0$. Three you have already met or will meet in real GPT-2 code:

- **LayerNorm / RMSNorm** ([lesson 34](lessons/module-11/lesson-03.md)) divide by a standard deviation: $\hat{x} = \dfrac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}}$. If a feature vector happened to be constant, $\sigma^2 = 0$ and you would divide by zero; the $+\varepsilon$ (here $\approx 10^{-5}$) makes the denominator strictly positive.
- **Adam / AdamW** ([lesson 27](lessons/module-09/lesson-02.md)) divide the update by $\sqrt{\hat{v}} + \varepsilon$, where $\hat{v}$ is the second-moment estimate. Early in training $\hat{v}$ can be essentially $0$; the $\varepsilon$ (default $10^{-8}$) keeps the step finite.
- **Log-probabilities** sometimes compute $\log(p + \varepsilon)$ so that a probability that underflowed to $0$ gives a large-but-finite negative number instead of $-\infty$.

The rule of thumb: whenever code divides by a quantity that *could* be zero, or takes the log of one, there is almost always an $\varepsilon$ nearby, and it is there for exactly this reason.

## 10. Fix 4 — gradient explosion and clipping

**Gradient explosion** is a numerical failure mode of the *backward* pass. In a deep network the chain rule multiplies many local derivatives together; if their product grows, the gradient's magnitude can blow up to `inf` or `NaN`, and a single huge optimizer step throws the weights into nonsense. You see it as a loss that suddenly spikes to `inf`/`NaN` after training fine for a while.

The standard fix is **gradient clipping by norm**. Treat all the parameter gradients as one long vector $\mathbf{g}$, measure its Euclidean norm $\|\mathbf{g}\|$ (the "total gradient size," from [lesson 08](lessons/module-02/lesson-03.md)), and if it exceeds a chosen threshold $\tau$, scale the *whole* gradient down so its norm equals $\tau$ — preserving its *direction* but capping its *length*:

$$
\mathbf{g} \leftarrow \mathbf{g}\cdot\frac{\tau}{\|\mathbf{g}\|} \quad\text{if } \|\mathbf{g}\| > \tau, \qquad \text{otherwise leave } \mathbf{g} \text{ unchanged.}
$$

**Worked example.** Suppose $\mathbf{g} = [6, 8]$, so $\|\mathbf{g}\| = \sqrt{36 + 64} = \sqrt{100} = 10$, and the threshold is $\tau = 1$. Since $10 > 1$, scale by $\tfrac{1}{10}$: $\mathbf{g} \leftarrow [0.6, 0.8]$, whose norm is $\sqrt{0.36 + 0.64} = 1$. Same direction, length capped at $1$.

```python
import torch

loss.backward()                                       # fills every .grad
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()                                      # step on the clipped gradient
```

`clip_grad_norm_` computes the norm across *all* parameters jointly and rescales in place, exactly the formula above. It is a near-universal line in Transformer training loops, right between `backward()` and `step()`.

## 11. Fix 5 — vanishing gradients

The opposite failure: when the multiplied local derivatives are each smaller than $1$, their product shrinks toward $0$ as it travels back through many layers, and the early layers receive **vanishing gradients** — essentially no learning signal. This is not `NaN`; it is silent stagnation.

The Transformer's three main defenses, all met earlier in the course:

- **Residual connections** ([lesson 33](lessons/module-11/lesson-02.md)): a block computes $x + f(x)$, so the backward pass has a path that multiplies by $1$ (the derivative of the $x$ term), letting gradient flow to early layers undiminished.
- **Normalization** (LayerNorm/RMSNorm, [lesson 34](lessons/module-11/lesson-03.md)): keeps activations at a controlled scale, so local derivatives stay near $1$ rather than drifting toward $0$ or blowing up.
- **Careful weight initialization**: scaling initial weights so that signal variance is preserved layer to layer, keeping the derivative products near $1$ from the very first step.

Explosion and vanishing are two ends of one phenomenon — the product of many local derivatives is unstable unless the architecture is built to hold it near $1$ — which is exactly what residuals and normalization do.

## 12. Under the hood: why `F.cross_entropy` is safer than softmax-then-log

Put the whole lesson to work on one line of real GPT-2 code. The mathematically obvious way to get the loss is: logits $\to$ softmax $\to$ probabilities $\to$ take the log of the true class's probability $\to$ negate. But that route exponentiates (overflow risk), then takes a logarithm of a possibly-underflowed probability (`log(0)` = $-\infty$ risk) — two hazards stacked.

PyTorch's `F.cross_entropy` (and `F.log_softmax`) instead compute $\log\text{-softmax}$ *directly* with the log-sum-exp trick, never forming the raw probabilities:

$$
\log\text{softmax}(x)_i = x_i - \Big(m + \log\sum_j e^{x_j - m}\Big),\qquad m=\max_j x_j.
$$

Every exponent is $\le 0$ (no overflow), and there is no explicit division and no $\log$ of a small probability (no underflow-to-$-\infty$). This is why you should feed **raw logits** to `F.cross_entropy` rather than softmax probabilities: it fuses the softmax and the log into one numerically stable operation.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[1000., 1001., 1002.]])   # huge logits, would overflow if exp'd
target = torch.tensor([2])                         # correct class = index 2

# stable: give it raw logits
print(F.cross_entropy(logits, target))            # tensor(0.4076)  -- finite, correct

# unstable hand-rolled version overflows:
probs = logits.exp() / logits.exp().sum()          # tensor([[nan, nan, nan]])
print(-torch.log(probs[0, target]))                # tensor(nan)
```

The stable loss is $-\log(0.665241) = 0.407605$ — the same log-sum-exp number from section 7, now recognizable as the cross-entropy of these logits against class $2$. The naive route gives `NaN`. Mixed-precision training (`torch.autocast`, bfloat16) leans on all of this: matmuls run in bf16 for speed, but the loss and other reductions are computed in float32 precisely so these stabilizers have the range and precision they need.

<div class="callout pt"><p>Feed <strong>raw logits</strong> to <code>F.cross_entropy</code> / <code>F.log_softmax</code>, never pre-softmaxed probabilities. Those functions fuse softmax and log with the log-sum-exp trick, so they never overflow on large logits or take <code>log(0)</code> on underflowed probabilities. Doing it by hand in two steps re-introduces both hazards.</p></div>

## Check yourself

<details><summary>Why does large-model training usually prefer bfloat16 over float16, given float16 is more precise?</summary>

bfloat16 keeps float32's 8 exponent bits, so it has the same huge range (max $\approx 3.4\times10^{38}$) and does not overflow where float32 would not. float16 has only 5 exponent bits, so its max is $65504$ — large activations overflow to `inf` easily. In training, range matters more than the extra mantissa precision, so bfloat16 wins.

</details>

<details><summary>Compute $\log\sum_i e^{x_i}$ for $x=[1000,1001,1002]$ with the log-sum-exp trick.</summary>

$m=1002$, so exponents are $[-2,-1,0]$: $e^{-2}+e^{-1}+e^{0}=0.135335+0.367879+1=1.503215$. Then $1002+\log(1.503215)=1002+0.407605=1002.407605$. The naive version overflows to `inf`.

</details>

<details><summary>You clip gradients to max-norm $\tau=1$ and the current gradient is $\mathbf{g}=[6,8]$. What is the clipped gradient?</summary>

$\|\mathbf{g}\|=\sqrt{36+64}=10>1$, so scale by $\tfrac{1}{10}$: $\mathbf{g}\leftarrow[0.6,0.8]$, norm $=1$. Direction preserved, length capped.

</details>

<details><summary>Why must you feed raw logits, not softmax probabilities, to <code>F.cross_entropy</code>?</summary>

`F.cross_entropy` computes log-softmax directly with the log-sum-exp trick — stable on large logits, with no `log(0)`. If you softmax first, large logits can overflow to `NaN`, and small probabilities can underflow to $0$ so the log is $-\infty$. Passing raw logits lets PyTorch do the whole thing in one numerically safe step.

</details>

## Next

Numerical stability keeps the arithmetic finite; the next lesson explains the machinery that computes the gradients in the first place — **automatic differentiation**, the exact algorithm behind `loss.backward()`, and why neural networks use reverse mode.

Continue to [36 · Automatic differentiation](lessons/module-12/lesson-02.md).
