# 27 · SGD, momentum, RMSProp, Adam, AdamW

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-01.md">26 · Gradient descent</a> — you must be comfortable with the update $\theta \leftarrow \theta - \eta\,\nabla L$, the learning rate $\eta$, and the PyTorch loop <code>zero_grad</code> / <code>backward</code> / <code>step</code>. Vectors and element-wise operations from <a href="../module-02/lesson-01.md">the vectors module</a>.</p>
<p><strong>You will learn:</strong> why plain gradient descent struggles, and five optimizers that fix it — SGD, momentum, Nesterov momentum, RMSProp, Adam, and AdamW. For each you get the problem it solves, the equations with every symbol named, the internal <strong>state</strong> it stores, a tiny hand-worked numeric example, and the PyTorch call. Special depth on Adam's first/second moments and bias correction, and on how AdamW's <strong>decoupled</strong> weight decay differs from Adam-with-L2.</p>
<p><strong>Why this matters for ML:</strong> GPT-2 and essentially every modern Transformer is trained with AdamW. Understanding the moving averages it keeps, and why the weight decay is decoupled, is the difference between copying <code>torch.optim.AdamW(...)</code> as a magic incantation and knowing exactly what happens to every weight on every step.</p>
</div>

## 0. Why plain gradient descent is not enough

Plain gradient descent, $\theta \leftarrow \theta - \eta\,\nabla L$, has three weaknesses that bite hard on real networks:

1. **It has no memory.** Each step uses only the current gradient. In a long, gently-sloping valley it inches forward; on a steep-but-narrow ravine it zig-zags across the walls instead of running down the floor.
2. **One learning rate for all parameters.** A single $\eta$ must serve every weight, even though some weights sit in steep parts of the landscape and others in flat parts. Too big for one is too small for another.
3. **Noisy mini-batch gradients.** With mini-batch gradients (from [lesson 26](lesson-01.md)) the direction jitters step to step; raw descent follows every jitter.

The optimizers below are increasingly clever answers. Each keeps a little extra **state** — running averages carried from step to step — to smooth the direction, adapt the step size per parameter, or both. Throughout, $g$ denotes the current gradient $\nabla L(\theta)$ (element-wise, one number per parameter), and $t = 1, 2, 3, \dots$ is the step counter.

## 1. SGD (plain) — the baseline, no state

This is exactly [lesson 26](lesson-01.md), restated so the family is complete:

$$
\theta \leftarrow \theta - \eta\,g.
$$

- $\theta$ — parameters. $\eta$ — learning rate. $g$ — current gradient.
- **State stored: none.** SGD remembers nothing between steps; give it the gradient and it steps.

In PyTorch: `torch.optim.SGD(params, lr=0.1)`. Everything that follows adds *state* to this bare update.

## 2. Momentum — remember where you were going

**Problem it solves.** Weakness (1): no memory, so zig-zagging in ravines and crawling on plateaus. Momentum adds a "velocity" that accumulates consistent gradients, like a heavy ball rolling downhill that keeps its speed through flat patches and averages out sideways rattling.

**The mathematics.**

$$
v \leftarrow \mu\,v + g, \qquad \theta \leftarrow \theta - \eta\,v.
$$

- $v$ — the **velocity**, a running (leaky) sum of past gradients, one number per parameter. Starts at $0$.
- $\mu$ (mu) — the **momentum coefficient**, typically $0.9$. It sets how much of the old velocity survives each step. Larger $\mu$ = longer memory.
- $g$, $\eta$, $\theta$ — as before. Note the step is now $-\eta v$, using the *accumulated* velocity, not the raw gradient.

**State stored: the velocity $v$** (same shape as $\theta$).

**Numeric example.** Suppose the gradient is a steady $g = 1$ every step (a long consistent slope), with $\mu = 0.9$, $\eta = 0.1$, $v_0 = 0$:

| step | $v \leftarrow 0.9\,v + g$ | step size $\eta\,v$ |
|---|---|---|
| 1 | $0.9(0) + 1 = 1.00$ | $0.100$ |
| 2 | $0.9(1.00) + 1 = 1.90$ | $0.190$ |
| 3 | $0.9(1.90) + 1 = 2.71$ | $0.271$ |

Plain SGD would step $0.1$ every time; momentum accelerates — $0.10, 0.19, 0.271, \dots$ — because the velocity keeps building while the gradient stays consistent. If it continued, $v$ would approach $1/(1-\mu) = 1/0.1 = 10$, a tenfold speed-up in the steady direction. In a rattling direction, where $g$ flips sign, those contributions cancel in $v$ instead of accumulating — so momentum damps the zig-zag and amplifies the consistent descent, exactly what we wanted.

```python
import torch
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# per step: buf = 0.9*buf + grad ; p -= 0.1 * buf
```

## 3. Nesterov momentum — look before you leap

**Problem it solves.** Plain momentum computes the gradient at the *current* point, then takes the big velocity jump — so it can overshoot before noticing the slope changed. Nesterov momentum peeks ahead: it evaluates the gradient at the point the velocity is *about to carry it to*, then corrects.

**The modified update.** Conceptually, compute the gradient at the look-ahead point $\theta - \eta\,\mu\,v$ instead of at $\theta$:

$$
g_{\text{ahead}} = \nabla L(\theta - \eta\,\mu\,v), \qquad v \leftarrow \mu\,v + g_{\text{ahead}}, \qquad \theta \leftarrow \theta - \eta\,v.
$$

- $g_{\text{ahead}}$ — the gradient measured *after* tentatively applying the momentum jump.
- All other symbols as in section 2. **State stored: still just the velocity $v$.**

**Intuition.** Ordinary momentum is a ball that commits to its momentum jump and only then feels the new slope. Nesterov is a ball that first rolls forward along its current momentum, feels the slope *there*, and uses that to decide the real step. Because it senses the upcoming terrain, it brakes earlier when the valley floor curves up, reducing overshoot. In practice it often converges a little faster and more stably than plain momentum. In PyTorch it is one flag: `torch.optim.SGD(..., momentum=0.9, nesterov=True)`.

## 4. RMSProp — a personal learning rate per parameter

**Problem it solves.** Weakness (2): a single $\eta$ for all parameters. RMSProp gives each parameter its *own* effective step size by dividing the update by the recent typical magnitude (the root-mean-square) of that parameter's gradients. Parameters with big gradients get scaled *down*; parameters with tiny gradients get scaled *up*, so all directions progress at comparable speed.

**The mathematics.**

$$
s \leftarrow \rho\,s + (1 - \rho)\,g^2, \qquad \theta \leftarrow \theta - \eta\,\frac{g}{\sqrt{s} + \varepsilon}.
$$

- $s$ — a running average of the **squared** gradient, one number per parameter. Starts at $0$. ($g^2$ is element-wise squaring.)
- $\rho$ (rho) — the decay rate for that average, typically $0.9$ or $0.99$. Larger $\rho$ = longer memory of past magnitudes.
- $\sqrt{s}$ — the root-mean-square of recent gradients: the "typical size" of this parameter's gradient.
- $\varepsilon$ (epsilon) — a tiny constant like $10^{-8}$ added to avoid dividing by zero when $s$ is near $0$.
- Dividing $g$ by $\sqrt{s}$ makes the update roughly *unit-sized* regardless of the raw gradient scale.

**State stored: the squared-gradient average $s$** (same shape as $\theta$).

**Numeric example.** Steady $g = 1$, $\rho = 0.9$, $\eta = 0.1$, $\varepsilon$ negligible, $s_0 = 0$:

| step | $s \leftarrow 0.9\,s + 0.1\,g^2$ | step $=\eta\,g/\sqrt{s}$ |
|---|---|---|
| 1 | $0.9(0) + 0.1(1) = 0.100$ | $0.1/\sqrt{0.100} = 0.3162$ |
| 2 | $0.9(0.1) + 0.1(1) = 0.190$ | $0.1/\sqrt{0.190} = 0.2294$ |
| 3 | $0.9(0.19) + 0.1(1) = 0.271$ | $0.1/\sqrt{0.271} = 0.1921$ |

As $s$ climbs toward its steady value ($g^2 = 1$ here), the step settles toward $\eta \cdot g/\sqrt{1} = 0.1$. The point is *normalization*: whether this parameter's gradients are habitually huge or tiny, dividing by their RMS brings the step into the same ballpark, so one $\eta$ works for all parameters at once. In PyTorch: `torch.optim.RMSprop(params, lr=0.01, alpha=0.9)` (PyTorch calls $\rho$ `alpha`).

## 5. Adam — momentum *and* adaptive rates together

Adam ("adaptive moment estimation") is the workhorse. It combines momentum's smoothed direction (section 2) with RMSProp's per-parameter scaling (section 4), plus a correction that fixes a start-up bias. We give it the full six-pass treatment.

### 5.1 Intuition

Keep two running averages for each parameter: one of the gradient itself (which way am I consistently going — momentum), and one of the gradient *squared* (how big are my gradients typically — the RMSProp scale). Step in the smoothed direction, but scaled down in directions where gradients are large and up where they are small. Then fix the fact that both averages start at $0$ and are therefore artificially small in the first few steps.

### 5.2 The mathematics

$$
m \leftarrow \beta_1\,m + (1 - \beta_1)\,g, \qquad v \leftarrow \beta_2\,v + (1 - \beta_2)\,g^2,
$$
$$
\hat{m} = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1 - \beta_2^{\,t}}, \qquad \theta \leftarrow \theta - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon}.
$$

- $m$ — the **first moment**: a running average of the gradient. This is the *mean* (average direction) of recent gradients, i.e. momentum. Starts at $0$.
- $v$ — the **second moment**: a running average of the squared gradient. This is the *uncentered variance* (average squared magnitude) of recent gradients, i.e. the RMSProp scale. Starts at $0$. (This $v$ is a squared-gradient average, unrelated to the velocity $v$ of section 2 despite the reused letter.)
- $\beta_1$ (beta-one) — decay rate for $m$, default $0.9$.
- $\beta_2$ (beta-two) — decay rate for $v$, default $0.999$ (longer memory, because variance estimates need more samples to be stable).
- $\beta_1^{\,t}$, $\beta_2^{\,t}$ — the decay rates raised to the power of the step number $t$.
- $\hat{m}, \hat{v}$ — the **bias-corrected** moments (explained in 5.3).
- $\eta$ — learning rate, default $0.001$. $\varepsilon$ — stabiliser, default $10^{-8}$.

### 5.3 Why bias correction is needed

Both averages start at $m = 0$ and $v = 0$. On step $1$ with a gradient $g$, we get $m = (1-\beta_1)g = 0.1g$ — only a *tenth* of the gradient, because the average is still mostly the initial zero. The estimate is **biased toward zero** in the early steps. Bias correction undoes this exactly: dividing by $1 - \beta_1^{\,t}$ rescales the average back up. On step $1$, $1 - \beta_1^{1} = 1 - 0.9 = 0.1$, so $\hat{m} = 0.1g / 0.1 = g$ — the correction recovers the true first step. As $t$ grows, $\beta_1^{\,t} \to 0$ and the correction factor $\to 1$: it matters early and fades away.

### 5.4 A full 1-parameter numeric example ($t = 1$)

Take a single parameter, gradient $g = 0.1$, with the defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.001$, $\varepsilon = 10^{-8}$, and $m_0 = v_0 = 0$.

Step $t = 1$:

- First moment: $m = 0.9(0) + 0.1(0.1) = 0.01$.
- Second moment: $v = 0.999(0) + 0.001(0.1^2) = 0.001 \times 0.01 = 1\times 10^{-5}$.
- Bias-correct: $\hat{m} = 0.01 / (1 - 0.9^1) = 0.01 / 0.1 = 0.1$. So $\hat m = g$, as promised.
- $\hat{v} = 10^{-5} / (1 - 0.999^1) = 10^{-5} / 0.001 = 0.01$. So $\hat v = g^2$.
- The step: $\dfrac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon} = \dfrac{0.1}{\sqrt{0.01} + 10^{-8}} = \dfrac{0.1}{0.1 + 10^{-8}} \approx 1.0$, so
$$
\Delta\theta = -\eta \times 1.0 = -0.001 = -\eta.
$$

The effective step is $\approx \eta$. This is a signature of Adam: because $\hat m \approx g$ and $\sqrt{\hat v} \approx |g|$ at $t=1$, their ratio is $\approx g/|g| = \pm 1$, so **the first step has magnitude about $\eta$ regardless of how large or small the gradient is.** Adam largely decouples the step *size* from the gradient *magnitude* — you set the scale with $\eta$.

For completeness, step $t = 2$ with the same $g = 0.1$: $m = 0.9(0.01) + 0.1(0.1) = 0.019$, corrected $\hat m = 0.019/(1-0.81) = 0.019/0.19 = 0.1$; $v = 0.999(10^{-5}) + 0.001(0.01) = 1.999\times 10^{-5}$, corrected $\hat v = 1.999\times10^{-5}/(1-0.998001) = 1.999\times10^{-5}/0.001999 = 0.01$; step again $\approx \eta$. For a constant gradient, Adam marches at a steady $\approx \eta$ per step.

### 5.5 Why ML needs it

Deep networks have parameters at wildly different scales (embeddings, attention weights, layer norms), gradients that are noisy from mini-batching, and landscapes full of plateaus and ravines. Adam's momentum smooths the noise, its per-parameter scaling handles the different magnitudes without hand-tuning $\eta$ for each, and bias correction keeps the very first steps sane. That robustness is why Adam and its descendant AdamW became the default for training Transformers.

### 5.6 In PyTorch

```python
import torch

w = torch.tensor([0.5], requires_grad=True)
opt = torch.optim.Adam([w], lr=0.001, betas=(0.9, 0.999), eps=1e-8)

loss = (w - 3.)**2
opt.zero_grad()
loss.backward()          # w.grad = 2(w-3) = 2(0.5-3) = -5.0
opt.step()               # updates w using m, v, bias correction

print(opt.state[w])      # {'step': ..., 'exp_avg': m, 'exp_avg_sq': v}
```

`betas` is $(\beta_1, \beta_2)$; `eps` is $\varepsilon$. The optimizer object owns the state.

### 5.7 What PyTorch is doing under the hood

Adam keeps **two extra tensors per parameter**: `exp_avg` (the first moment $m$) and `exp_avg_sq` (the second moment $v$), plus a scalar `step` count $t$ per parameter for the bias correction. These live in `opt.state[p]`, *not* in `p` itself — the parameter tensor still holds only its value and `.grad`. Memory cost: Adam roughly *triples* the memory of the parameters (the weights, plus $m$, plus $v$). For a 124M-parameter GPT-2 that is a real cost, and it is exactly what `opt.state` stores. On each `opt.step()`, for every parameter with a gradient, PyTorch updates $m$ and $v$ in place, computes $\hat m$ and $\hat v$ from the current `step`, and applies the update under `torch.no_grad()` — the same hand-update you wrote in [lesson 26](lesson-01.md), just with the moment bookkeeping wrapped around it.

<div class="callout key"><p>Adam stores, per parameter, the first moment $m$ (smoothed gradient = momentum) and the second moment $v$ (smoothed squared gradient = adaptive scale). Bias correction ($/(1-\beta^t)$) fixes the zero-start of those averages so early steps are not artificially tiny. At $t=1$ the effective step size is about $\eta$.</p></div>

## 6. AdamW — the same, but with *decoupled* weight decay

### 6.1 Intuition

**Weight decay** is a regulariser that gently pulls every weight toward zero each step, discouraging the model from relying on large weights (which tends to improve generalisation). The question is *how* to inject that pull into Adam. The old way (Adam with "L2 regularisation") folds it into the gradient; AdamW keeps it separate. That separation, small as it looks, is why AdamW is the default optimizer for Transformers.

### 6.2 The mathematics

AdamW runs Adam's moment machinery on the *plain* gradient $g$ (no decay mixed in), then subtracts a decay term directly from $\theta$:

$$
\theta \leftarrow \theta - \eta\left(\frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon} + \lambda\,\theta\right).
$$

- $\lambda$ (lambda) — the **weight-decay coefficient**, e.g. $0.01$ or $0.1$. It sets how strongly each weight is pulled toward $0$.
- $\lambda\,\theta$ — the decay term, proportional to the weight itself, applied **directly to $\theta$**, not routed through $m$ or $v$.
- $\hat m, \hat v, \eta, \varepsilon$ — exactly as in Adam; the moments are computed from $g$ alone.

**State stored: the same as Adam** — $m$ and $v$ per parameter (weight decay adds no state).

### 6.3 How this differs from Adam + L2, precisely

"Adam with L2 regularisation" instead adds the decay *into the gradient*:

$$
g' = g + \lambda\,\theta, \qquad \text{then run ordinary Adam on } g'.
$$

Because $g'$ then flows into both moving averages, the decay gets smoothed into $m$ and — crucially — **divided by $\sqrt{\hat v}$** in the final step. The effective decay on each weight becomes $\lambda\theta / (\sqrt{\hat v} + \varepsilon)$: weights whose gradients are large (big $\hat v$) get *less* decay, and weights with small gradients get *more*. The regularisation strength ends up tangled with each parameter's gradient history — almost never what you intended.

AdamW's decoupled form applies $\lambda\theta$ *outside* the adaptive denominator, so **every weight is decayed by the same relative amount $\eta\lambda$ per step**, independent of its gradient magnitude. Clean, predictable regularisation.

### 6.4 Numeric example: the decay term

Take the Adam $t=1$ result from 5.4 (effective adaptive step $\approx 1.0$, so $-\eta \times 1.0 = -0.001$) for a weight currently at $\theta = 0.5$, with $\lambda = 0.01$:

- Adaptive part: $-\eta \cdot \dfrac{\hat m}{\sqrt{\hat v}+\varepsilon} \approx -0.001 \times 1.0 = -0.001000$.
- Decay part: $-\eta\,\lambda\,\theta = -0.001 \times 0.01 \times 0.5 = -0.000005$.
- New weight: $\theta \leftarrow 0.5 - 0.001000 - 0.000005 = 0.498995$.

The decay contributes a small, steady $-\eta\lambda\theta$ pull toward zero, computed straight from $\theta$ — it never touched $m$ or $v$. In Adam+L2, that same $\lambda\theta = 0.005$ would have been added to $g$ *before* the moments, then scaled by $1/\sqrt{\hat v} \approx 1/0.1 = 10$, distorting it enormously here.

### 6.5 Why AdamW is the default for Transformers

Transformers train best with meaningful weight decay, and their parameters have very different gradient scales across layers. With Adam+L2, decay strength would silently vary per parameter (scaled by $1/\sqrt{\hat v}$), making the regulariser unreliable. AdamW decouples the two so learning rate and weight decay can be tuned independently and behave consistently across all weights. This is why the GPT-2 / GPT-3 training recipes, and virtually every Transformer since, specify AdamW.

### 6.6 In PyTorch, and param_groups

```python
import torch

# One optimizer, TWO groups: decay the matrices, do NOT decay biases / LayerNorm.
decay_params   = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() <  2]

opt = torch.optim.AdamW(
    [
        {"params": decay_params,   "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
)

# opt.param_groups is a list of dicts, each with its own lr / weight_decay / betas.
# opt.state[p] still holds {'step', 'exp_avg' (m), 'exp_avg_sq' (v)} per parameter.
```

`param_groups` lets different parameters use different hyperparameters — here weight matrices are decayed while biases and LayerNorm gains (1-D parameters) are not, which is the standard GPT-2 recipe. Each group carries its own `lr`, `weight_decay`, and `betas`; the per-parameter moment state in `opt.state` is identical in structure to Adam.

### 6.7 What PyTorch is doing under the hood

AdamW's `step()` is Adam's `step()` with one extra line: *before* (PyTorch applies it as) the moment-based update, it multiplies each weight by $(1 - \eta\lambda)$ — i.e. subtracts $\eta\lambda\theta$ — directly, entirely separately from anything involving `exp_avg` or `exp_avg_sq`. The moments `m` and `v` are still computed from the raw gradient `p.grad` only. So the stored state is exactly Adam's (`step`, `exp_avg`, `exp_avg_sq`), and the *only* behavioural change from Adam-with-L2 is *where* the $\lambda\theta$ term enters: on $\theta$ directly, not inside the gradient. That one-line difference is the entire content of the Adam-to-AdamW change.

## Check yourself

<details><summary>What state does each optimizer store per parameter: SGD, momentum, RMSProp, Adam?</summary>

SGD: nothing. Momentum: the velocity $v$. RMSProp: the squared-gradient average $s$. Adam: two tensors, the first moment $m$ and second moment $v$ (plus a step count $t$). AdamW stores the same as Adam.

</details>

<details><summary>In Adam, why divide the moments by $1 - \beta^{\,t}$?</summary>

Both $m$ and $v$ start at $0$, so early on they are biased toward zero (too small). Dividing by $1-\beta^{\,t}$ rescales them back to unbiased estimates. At $t=1$ this exactly recovers $\hat m = g$; as $t$ grows the correction factor approaches $1$ and stops mattering.

</details>

<details><summary>Concretely, how does AdamW differ from running Adam with L2 regularisation?</summary>

Adam+L2 adds $\lambda\theta$ into the gradient $g$, so the decay flows through $m$ and $v$ and gets divided by $\sqrt{\hat v}$ (uneven per parameter). AdamW applies $\lambda\theta$ directly to $\theta$, outside the adaptive denominator, so every weight decays by the same relative amount $\eta\lambda$. The moments are computed from the plain gradient only.

</details>

<details><summary>At $t=1$ with default settings, roughly how big is Adam's step, and why?</summary>

About $\eta$ in magnitude. Because $\hat m \approx g$ and $\sqrt{\hat v} \approx |g|$, their ratio is $\approx \pm 1$, so the step is $\approx \eta$ regardless of the gradient's size — Adam sets the step scale through $\eta$, not through the raw gradient magnitude.

</details>

## Next

You can now train with the optimizer that actually trains GPT-2. Every optimizer so far treats each weight as an independent number in a flat bag — even Adam scales each entry on its own. The next lesson introduces Muon, which does something genuinely different: it looks at a weight *matrix* as a geometric object and reshapes the whole update using its singular structure.

Continue to [28 · Muon](lesson-03.md).
