# 26 · Gradient descent

<div class="prereq">
<p><strong>Prerequisites:</strong> the gradient from <a href="#/lessons/module-06/lesson-04">18 · Partial derivatives and the gradient</a> (you must know that $\nabla L$ points uphill and has the same shape as the parameters), and <a href="#/lessons/module-08/lesson-01">backpropagation</a> — the machine that actually computes $\nabla L$ for a network.</p>
<p><strong>You will learn:</strong> the gradient-descent update $\theta \leftarrow \theta - \eta\,\nabla L(\theta)$ with every symbol named, worked numerically for one and two parameters, how the learning rate $\eta$ controls crawling versus overshooting versus divergence, the shape of loss landscapes (minima, saddle points, plateaus), the difference between batch, stochastic, and mini-batch gradient descent, and how to perform the update by hand in PyTorch before letting <code>optimizer.step()</code> do it for you.</p>
<p><strong>Why this matters for ML:</strong> every neural network you will ever train, from a two-parameter toy to GPT-2, learns by repeating this one update millions of times. Backpropagation supplies the gradient; gradient descent is what turns that gradient into an actual change in the weights.</p>
</div>

## 1. The one equation that trains every network

Here is the entire idea of training, in one line:

$$
\theta \leftarrow \theta - \eta\,\nabla L(\theta).
$$

Read it as an assignment (the arrow $\leftarrow$ means "replace the left side with the right side," exactly like `theta = theta - ...` in code). Let us name every symbol, because this equation is the spine of the whole course from here on:

- $\theta$ (theta) — the **parameters**: every weight and bias in the model, gathered into one big collection. For our first example $\theta$ is a single number $w$; for GPT-2 it is about 124 million numbers.
- $L(\theta)$ — the **loss**: one number saying how wrong the model currently is. It depends on $\theta$, so as you change the parameters the loss changes.
- $\nabla L(\theta)$ — the **gradient** of the loss: the vector of all partial derivatives $\partial L/\partial\theta_i$, one per parameter. From [lesson 18](lessons/module-06/lesson-04.md) you know it has the same shape as $\theta$ and points in the direction of steepest *ascent* — the fastest way to make the loss *bigger*.
- $\eta$ (eta) — the **learning rate**: a small positive number (like $0.1$ or $0.001$) that controls how big a step you take. It is a knob *you* choose, not something the math hands you.

The minus sign is the whole trick. The gradient points uphill; we want to go *downhill* (smaller loss); so we step in the *opposite* direction, $-\nabla L$. The learning rate scales how far.

<div class="callout key"><p>Gradient descent: stand on the loss landscape, look at which way is steepest uphill ($\nabla L$), turn around, and take a small step ($\eta$ controls the size). Repeat. That is all training is.</p></div>

### The downhill-walk analogy

Imagine you are standing on a foggy hillside and want to reach the valley floor. You cannot see far, but you *can* feel the slope under your feet. A reasonable strategy: feel which direction is steepest downhill, step a little that way, and repeat. The gradient is the "feel the slope" step; $-\eta\,\nabla L$ is the "step a little downhill" part. You never need a map of the whole hill — only the local slope where you are standing.

## 2. A worked example in one dimension

Let the loss be

$$
L(w) = (w - 3)^2.
$$

This is a parabola with its lowest point at $w = 3$, where $L = 0$. That $w = 3$ is the answer gradient descent should discover on its own.

**The gradient.** With one parameter the gradient is just the ordinary derivative. Using the chain rule on $(w-3)^2$:

$$
\nabla L(w) = \frac{dL}{dw} = 2(w - 3).
$$

- At $w < 3$ this is negative (slope tilts down to the right, so uphill is to the left, so we should move right — and $-\eta\nabla L$ is positive, moving $w$ up toward $3$). Good.
- At $w > 3$ it is positive (we move left, back toward $3$).
- At $w = 3$ it is exactly $0$ — no slope, no step, we have arrived.

**The update.** Start at $w_0 = 0$ with learning rate $\eta = 0.1$. The rule is

$$
w_{k+1} = w_k - \eta\,\nabla L(w_k) = w_k - 0.1\cdot 2(w_k - 3) = w_k - 0.2\,(w_k - 3).
$$

Let us grind through the arithmetic. At each step, compute the gradient, multiply by $-\eta$ to get the step, and add it to $w$:

| step $k$ | $w_k$ | $\nabla L = 2(w_k-3)$ | step $=-\eta\nabla L$ | $w_{k+1}$ | $L(w_k) = (w_k-3)^2$ |
|---|---|---|---|---|---|
| 0 | 0.0000 | $2(0-3)=-6.0000$ | $+0.6000$ | 0.6000 | 9.0000 |
| 1 | 0.6000 | $2(0.6-3)=-4.8000$ | $+0.4800$ | 1.0800 | 5.7600 |
| 2 | 1.0800 | $2(1.08-3)=-3.8400$ | $+0.3840$ | 1.4640 | 3.6864 |
| 3 | 1.4640 | $2(1.464-3)=-3.0720$ | $+0.3072$ | 1.7712 | 2.3593 |
| 4 | 1.7712 | $2(1.7712-3)=-2.4576$ | $+0.2458$ | 2.0170 | 1.5099 |

Watch the parameter climb toward $3$: $0 \to 0.6 \to 1.08 \to 1.464 \to 1.7712 \to 2.017$, and the loss fall $9 \to 5.76 \to 3.69 \to 2.36 \to 1.51$. Each step covers $80\%$ of the remaining gap: if you define the error $e_k = w_k - 3$, then $e_{k+1} = 0.8\,e_k$ (because $w_{k+1}-3 = (w_k-3) - 0.2(w_k-3) = 0.8(w_k-3)$). The errors are $-3, -2.4, -1.92, -1.536, -1.2288, \dots$, shrinking geometrically toward $0$. Gradient descent never jumps straight to the answer; it *converges* to it.

## 3. Two parameters: each moves by its own partial

Real models have many parameters, and the gradient handles them all at once. Take

$$
L(w_1, w_2) = (w_1 - 2)^2 + (w_2 - 5)^2,
$$

whose minimum sits at $(w_1, w_2) = (2, 5)$. The gradient is the vector of both partials:

$$
\nabla L = \begin{bmatrix} \partial L/\partial w_1 \\ \partial L/\partial w_2 \end{bmatrix} = \begin{bmatrix} 2(w_1 - 2) \\ 2(w_2 - 5) \end{bmatrix}.
$$

Start at $(w_1, w_2) = (0, 0)$ with $\eta = 0.1$ and take **one** step. Evaluate the gradient:

$$
\nabla L(0,0) = \begin{bmatrix} 2(0-2) \\ 2(0-5) \end{bmatrix} = \begin{bmatrix} -4 \\ -10 \end{bmatrix}.
$$

Now apply the update componentwise:

$$
w_1 \leftarrow 0 - 0.1(-4) = 0.4, \qquad w_2 \leftarrow 0 - 0.1(-10) = 1.0.
$$

Notice $w_2$ moved $1.0$ while $w_1$ moved only $0.4$ — **each parameter moved by an amount proportional to its own partial derivative**. The steeper the loss is in a given parameter's direction, the bigger that parameter's step. The vector equation $\theta \leftarrow \theta - \eta\nabla L$ is really $n$ separate scalar updates happening simultaneously, one per parameter, coupled only through the single shared learning rate.

## 4. The learning rate: crawl, converge, or explode

The learning rate $\eta$ is the single most important knob in training. For the 1-D loss $L(w) = (w-3)^2$ the update multiplies the error by a factor $(1 - 2\eta)$ each step (from $e_{k+1} = (1 - 2\eta)e_k$). The behaviour depends entirely on that factor:

**Too small ($\eta = 0.01$).** The factor is $1 - 0.02 = 0.98$. The error shrinks by only $2\%$ per step; reaching the minimum takes hundreds of iterations. Correct, but painfully slow — you are taking baby steps down the hill.

**Just right ($\eta = 0.1$, our example).** The factor is $0.8$; steady, healthy progress. In fact for this exact parabola $\eta = 0.5$ is magic: the factor is $1 - 1.0 = 0$, and

$$
w_1 = 0 - 0.5\cdot 2(0 - 3) = 0 - 0.5(-6) = 3,
$$

hitting the minimum in a single step. (Real losses are not perfect parabolas, so no such magic value exists in practice — but it shows why a bigger $\eta$ can be faster, up to a point.)

**Too large ($\eta = 1.1$).** Now the factor is $1 - 2.2 = -1.2$, whose magnitude exceeds $1$. The error *grows* and flips sign every step — the iterate overshoots the valley, lands higher up the far wall, overshoots back even farther, and diverges. Watch it happen on the same $L(w)=(w-3)^2$, starting $w_0 = 0$:

$$
w_{k+1} = w_k - 1.1\cdot 2(w_k - 3) = -1.2\,w_k + 6.6.
$$

| step | $w_k$ | $L(w_k) = (w_k-3)^2$ |
|---|---|---|
| 0 | 0.000 | 9.00 |
| 1 | 6.600 | 12.96 |
| 2 | $-1.320$ | 18.66 |
| 3 | 8.184 | 26.87 |
| 4 | $-3.221$ | 38.70 |

The parameter bounces $0 \to 6.6 \to -1.32 \to 8.18 \to -3.22$, and the loss *rises* $9 \to 12.96 \to 18.66 \to 26.87 \to 38.70$ instead of falling. This is the classic "loss went to NaN" failure: the learning rate was too big, the steps overshot, and the whole thing blew up.

<div class="callout warn"><p>If training loss increases, oscillates wildly, or becomes <code>NaN</code>, the first suspect is a learning rate that is too large. If loss falls but agonisingly slowly, the learning rate may be too small. Almost all of "tuning" a model is finding an $\eta$ between these failure modes.</p></div>

## 5. The shape of the landscape

Our examples were simple bowls, so descent glided to the bottom. Real loss landscapes are bumpier, and the geometry has names worth knowing:

```text
      loss
       |
   \   |        ___
    \  |       /   \        /   <- local minimum (a dip that
     \ |      /     \      /       is not the deepest one)
      \|     /       \    /
   ----+----/         \--/
       |   ^           ^
   global  plateau    local
   minimum  (flat)    minimum
```

- **Global minimum** — the deepest valley, the parameters that make the loss smallest. What we ultimately want.
- **Local minimum** — a dip that is lowest *in its neighbourhood* but not overall. The gradient is $0$ here, so plain descent can get stuck. In the millions of dimensions of a real network these are rarely a serious problem, but conceptually they exist.
- **Saddle point** — a spot that curves *up* in some directions and *down* in others, like the middle of a horse's saddle or a mountain pass. The gradient is $0$, so descent slows to a near-halt, yet it is not a minimum: there is still a way down, just not an obvious one. High-dimensional landscapes are full of these.
- **Plateau** — a nearly flat region where the gradient is tiny but not zero. Steps become minuscule and progress stalls, even though you are not at any minimum. This is one reason momentum and adaptive optimizers (next lesson) exist.

You do not need to master this geometry now. The takeaway: gradient descent only ever sees the *local* slope, so where it ends up depends on where it starts and how it steps.

## 6. Batch, stochastic, and mini-batch gradient descent

There is one more choice hiding in $\nabla L$: **the loss over *what data*?** A model's loss is an average over training examples, and you can compute the gradient over different amounts of data:

- **Batch (full-batch) gradient descent.** $\nabla L$ is averaged over the *entire* training set every step. Most accurate direction, but for a dataset of millions of examples, one step is enormously expensive — you touch all the data just to move once.
- **Stochastic gradient descent (SGD).** $\nabla L$ is computed from *a single* training example each step. Each gradient is a noisy, cheap estimate of the true full-batch gradient. You take many quick, jittery steps instead of few careful ones. The noise even helps escape shallow local minima and saddle points.
- **Mini-batch gradient descent.** The practical middle ground, and what everyone actually uses: $\nabla L$ is averaged over a small **batch** of examples (say 32, or for GPT-2 a batch of sequences). Enough examples to make the gradient reasonably accurate and to use the GPU efficiently, few enough to step often.

The update equation $\theta \leftarrow \theta - \eta\nabla L$ is *identical* in all three; only the set of examples that $\nabla L$ averages over changes. When people say "SGD" in modern deep learning they almost always mean mini-batch SGD.

<div class="callout key"><p>Batch vs. stochastic vs. mini-batch is purely a statement about how many training examples the gradient $\nabla L$ is averaged over each step: all of them, one of them, or a small batch. The step rule does not change.</p></div>

## 7. In PyTorch: doing the update by hand

Let us reproduce the section 2 numbers exactly, performing the update ourselves so nothing is hidden. Loss $L(w) = (w-3)^2$, start $w = 0$, $\eta = 0.1$:

```python
import torch

w = torch.tensor(0., requires_grad=True)   # parameter, starts at 0
lr = 0.1                                    # learning rate eta

for step in range(5):
    loss = (w - 3.)**2                      # forward: compute L(w)
    loss.backward()                         # backward: fill w.grad with dL/dw = 2(w-3)

    with torch.no_grad():                   # do NOT record this update in the graph
        w -= lr * w.grad                    # the update: w <- w - eta * grad, in place

    w.grad.zero_()                          # reset the gradient slot for next step
    print(f"step {step}: w = {w.item():.4f}, grad was {2*(w.item()-3) + lr*0:.4f}")

# Prints w = 0.6000, 1.0800, 1.4640, 1.7712, 2.0170
# exactly the by-hand table in section 2.
```

The printed parameters `0.6000, 1.0800, 1.4640, 1.7712, 2.0170` match our hand table line for line. Three lines in that loop deserve a close look, because they are the whole mechanism.

## 8. What PyTorch is doing under the hood

**`with torch.no_grad():`** Normally every operation on a tensor with `requires_grad=True` is *recorded* into the computational graph so it can be differentiated later. But the update $w \leftarrow w - \eta\,\nabla L$ is *not* part of the model's math — it is us hand-editing the parameter. We do not want autograd to track it (that would build an ever-growing graph and try to differentiate the optimizer itself). `torch.no_grad()` temporarily switches recording off, so the subtraction just mutates the number without touching the graph.

**`w -= lr * w.grad`** This is an **in-place** update: it modifies the existing `w` tensor rather than creating a new one. That matters because `w` is a *leaf* parameter with a `.grad` attached; we want to keep the same tensor object (and its gradient slot) around across steps, just change its value. Writing `w = w - lr * w.grad` (without the in-place `-=`) inside `no_grad` would rebind `w` to a fresh tensor that is no longer a leaf with `requires_grad`, and the next `backward()` would fail — a classic beginner trap.

**`w.grad.zero_()`** Recall from [lesson 18](lessons/module-06/lesson-04.md) that `backward()` *accumulates* into `.grad` — it adds, it does not overwrite. If we skipped this line, step 2's gradient would be added on top of step 1's leftover, and every subsequent step would use a corrupted, ever-larger gradient. Zeroing the slot before (or after) each step keeps each update honest.

<div class="callout pt"><p>An optimizer is just this loop, packaged. <code>optimizer.step()</code> runs exactly <code>with torch.no_grad(): p -= lr * p.grad</code> for every parameter <code>p</code> it manages, and <code>optimizer.zero_grad()</code> runs <code>p.grad.zero_()</code> for each. The equivalent PyTorch idiom is:</p></div>

```python
import torch

w = torch.tensor(0., requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)    # plain gradient descent

for step in range(5):
    loss = (w - 3.)**2
    optimizer.zero_grad()                   # w.grad.zero_()  (or set to None)
    loss.backward()                         # fill w.grad = 2(w-3)
    optimizer.step()                        # w -= 0.1 * w.grad, under no_grad

# Same numbers: 0.6000, 1.0800, 1.4640, 1.7712, 2.0170
```

`torch.optim.SGD` with no extra options *is* the update from section 1 — nothing more. Everything in the next lesson (momentum, RMSProp, Adam, AdamW) is a smarter rule for turning `w.grad` into a step, but the surrounding loop — `zero_grad`, `backward`, `step` — never changes.

## Check yourself

<details><summary>For $L(w) = (w-3)^2$ with $w_0 = 0$ and $\eta = 0.1$, what is $w_1$, and why is it not $3$?</summary>

$\nabla L(0) = 2(0-3) = -6$, so $w_1 = 0 - 0.1(-6) = 0.6$. It is not $3$ because gradient descent takes a step *proportional* to the slope, not a jump to the minimum; it converges over many steps (here closing $80\%$ of the gap each step).

</details>

<details><summary>Why does the minus sign appear in $\theta \leftarrow \theta - \eta\nabla L$?</summary>

The gradient $\nabla L$ points in the direction of steepest *increase* of the loss. We want to *decrease* the loss, so we move in the opposite direction, $-\nabla L$.

</details>

<details><summary>Training loss keeps rising and then becomes <code>NaN</code>. What is the most likely single cause?</summary>

The learning rate is too large. The steps overshoot the minimum and grow each iteration (like the $\eta = 1.1$ divergence example), until the numbers overflow to infinity/NaN.

</details>

<details><summary>What exactly is the difference between batch, stochastic, and mini-batch gradient descent?</summary>

Only what $\nabla L$ is averaged over each step: the whole training set (batch), a single example (stochastic), or a small batch of examples (mini-batch). The update rule $\theta \leftarrow \theta - \eta\nabla L$ is the same in all three.

</details>

## Next

You now have the core loop: compute the gradient, step against it, repeat. Plain gradient descent works, but it crawls through plateaus, rattles across steep-and-narrow valleys, and forces you to hand-pick one learning rate for millions of differently-scaled parameters. The next lesson fixes all three with momentum and adaptive learning rates, building up to Adam and AdamW — the optimizers that actually train GPT-2.

Continue to [27 · SGD, momentum, RMSProp, Adam, AdamW](lessons/module-09/lesson-02.md).
