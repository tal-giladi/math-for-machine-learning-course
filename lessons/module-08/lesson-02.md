# 24 · Loss functions

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-01.md">23 · The linear layer, forward and backward by hand</a> (you saw squared error used as a loss), the softmax and its derivative from <a href="../module-06/lesson-03.md">17 · Derivatives of exp, log, sigmoid, softmax</a>, and logarithms from <a href="../module-01/lesson-03.md">03 · Powers, roots, exponentials, logarithms</a>.</p>
<p><strong>You will learn:</strong> what a <strong>loss function</strong> is, the two regression losses (<strong>MSE</strong>, <strong>MAE</strong>), and the classification stack — <strong>softmax</strong>, <strong>log-softmax</strong>, <strong>cross-entropy</strong>, <strong>negative log likelihood</strong> (NLL), and <strong>binary cross-entropy</strong> (BCE) — with a worked language-model example from logits all the way to a single loss number and its gradient.</p>
<p><strong>Why this matters for ML:</strong> the loss is the one number training minimizes, and its gradient is where every backward pass begins (it was $\partial L/\partial y$ in the previous lesson). GPT-2 is trained with cross-entropy on next-token predictions, so understanding that loss — and why PyTorch fuses it with log-softmax — is understanding the objective the whole model is optimizing.</p>
</div>

## 1. What a loss function is

A **loss function** takes the model's prediction and the true target and returns a single non-negative number measuring how wrong the prediction is. Smaller is better; zero means perfect. Training is nothing but making this number small.

$$
L = \ell(\text{prediction},\ \text{target}).
$$

- $\ell$ — the chosen loss function (which one depends on the task).
- prediction — whatever the model output: a number, a vector, a probability distribution.
- target — the ground truth we compare against.
- $L$ — a scalar, so we can take $\partial L / \partial(\text{parameters})$ and step downhill.

Two families cover almost everything. **Regression** losses (MSE, MAE) are for predicting a continuous number. **Classification** losses (cross-entropy and friends) are for predicting *which category* — and that is what a language model does at every position: which token comes next.

## 2. Regression losses

### 2.1 Mean squared error (MSE)

MSE averages the squared differences between predictions and targets:

$$
L_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2.
$$

- $n$ — number of examples (or elements).
- $\hat{y}_i$ — the model's prediction for element $i$ (the hat means "estimate").
- $y_i$ — the true target for element $i$.
- $(\hat{y}_i - y_i)$ — the **residual**, squared so that positive and negative errors both count as positive and large errors are penalized disproportionately.

**Tiny example.** Predictions $\hat{y} = [2.5,\ 0.0,\ 2.0]$, targets $y = [3.0,\ -0.5,\ 2.0]$. Residuals are $[-0.5,\ 0.5,\ 0.0]$; squared, $[0.25,\ 0.25,\ 0]$. Average:

$$
L_{\text{MSE}} = \frac{0.25 + 0.25 + 0}{3} = \frac{0.5}{3} = 0.1667.
$$

The single-element MSE was exactly the loss $L = (y-t)^2$ in [lesson 23](lesson-01.md), and its derivative $2(\hat{y}-y)/n$ is the $\partial L/\partial y$ that launched that backward pass. Squaring makes MSE smooth and heavily punish outliers — one badly-wrong prediction dominates the sum.

### 2.2 Mean absolute error (MAE)

MAE averages the *absolute* differences instead of the squared ones:

$$
L_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^{n}\lvert \hat{y}_i - y_i \rvert.
$$

Same symbols; $\lvert\cdot\rvert$ is absolute value. On the same numbers, absolute residuals are $[0.5,\ 0.5,\ 0]$:

$$
L_{\text{MAE}} = \frac{0.5 + 0.5 + 0}{3} = \frac{1.0}{3} = 0.3333.
$$

The difference that matters: MAE grows *linearly* with the error, so a single outlier pulls it less hard than MSE. MAE is more robust to outliers; MSE is smoother and easier to differentiate at zero. Neither is used for GPT-2 — language modeling is classification — but you meet them everywhere in regression.

## 3. From scores to probabilities: softmax and log-softmax

Classification needs the model to output a **probability distribution** over categories, not a single number. The model first produces raw scores called **logits** — one per category, any real value — and softmax turns them into probabilities.

### 3.1 Softmax (recap)

$$
\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}.
$$

- $\mathbf{z}$ — the logit vector, one entry $z_i$ per category.
- $e^{z_i}$ — exponentiate to make every value positive.
- $\sum_j e^{z_j}$ — the normalizer, so the outputs sum to $1$.
- The result is a vector of probabilities: each in $(0,1)$, summing to $1$.

We derived this and its Jacobian in [lesson 17](../module-06/lesson-03.md). Larger logits get exponentially larger shares of the probability mass.

### 3.2 Log-softmax

Cross-entropy (next) needs the *logarithm* of a softmax probability. Computing softmax then taking `log` works mathematically but is numerically fragile (section 7). The **log-softmax** computes the log directly:

$$
\log\text{softmax}(\mathbf{z})_i = z_i - \log\sum_j e^{z_j}.
$$

This is just $\log$ of the softmax formula, using $\log\frac{a}{b} = \log a - \log b$ and $\log e^{z_i} = z_i$. The term $\log\sum_j e^{z_j}$ is the **log-sum-exp** of the logits — remember it, it returns in section 7. Log-softmax outputs are all $\le 0$ (the log of a number below 1).

## 4. Cross-entropy and NLL: the language-model loss

### 4.1 The idea

For classification, the target is one true category. **Cross-entropy** loss says: look at the probability the model assigned to the *correct* category, and penalize it by the negative logarithm of that probability. If the model gave the right category probability near $1$, $-\log(1) = 0$ (no loss). If it gave it near $0$, $-\log(\text{tiny}) \to \infty$ (huge loss). It is a confidence-aware "how surprised was the model by the truth."

### 4.2 The mathematics

With a **one-hot** target $\mathbf{y}$ (a vector that is $1$ at the true category and $0$ elsewhere) and predicted probabilities $\mathbf{p} = \text{softmax}(\mathbf{z})$:

$$
L_{\text{CE}} = -\sum_{i} y_i \log p_i.
$$

Because $\mathbf{y}$ is one-hot, every term is zero except the true class $c$, so the sum collapses to a single term:

$$
L_{\text{CE}} = -\log p_c.
$$

That single-term form, $-\log p_c$, is exactly the **negative log likelihood** (NLL). So cross-entropy with a one-hot target *is* NLL of the correct class. The only wrinkle in PyTorch: `NLLLoss` expects you to have already taken the log (it consumes log-probabilities), whereas cross-entropy takes raw logits and does the log-softmax itself. Same math, split at a different seam.

- $y_i$ — target, $1$ for the true class $c$, else $0$.
- $p_i$ — predicted probability of class $i$.
- $p_c$ — predicted probability of the *true* class.
- $-\log p_c$ — the loss: zero when $p_c = 1$, rising without bound as $p_c \to 0$.

### 4.3 The worked LM example

This is the pipeline every GPT-2 training step runs at every position. A three-word vocabulary and one prediction:

```text
vocabulary = ["cat", "dog", "fish"]
logits     = [2.0, 1.0, 0.0]
target     = "dog"          (index 1)
```

**Step 1 — softmax.** Exponentiate the logits:

$$
e^{2.0} = 7.389, \quad e^{1.0} = 2.718, \quad e^{0.0} = 1.000.
$$

Sum: $7.389 + 2.718 + 1.000 = 11.107$. Divide each by the sum:

$$
\mathbf{p} = \left[\frac{7.389}{11.107},\ \frac{2.718}{11.107},\ \frac{1.000}{11.107}\right] = [0.6652,\ 0.2447,\ 0.0900].
$$

They sum to $1.0000$. ✓ The model thinks "cat" is most likely ($0.665$).

**Step 2 — probability of the target.** The target is "dog", index $1$, so $p_c = p_{\text{dog}} = 0.2447$.

**Step 3 — negative log probability.**

$$
-\log p_{\text{dog}} = -\ln(0.2447) = 1.4076.
$$

**Step 4 — the loss.** That *is* the cross-entropy loss for this example: $L_{\text{CE}} = 1.4076$. (Equivalently, log-softmax of the logits is $[-0.4076, -1.4076, -2.4076]$, and NLL picks out index 1 and negates it: $-(-1.4076) = 1.4076$.) The model was moderately surprised: it gave the truth only a $24\%$ chance.

### 4.4 The gradient: the clean $\mathbf{p} - \mathbf{y}$ result

Now the payoff. The gradient of cross-entropy with respect to the **logits** (not the probabilities) is astonishingly simple:

$$
\frac{\partial L_{\text{CE}}}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y},
$$

the predicted probabilities minus the one-hot target. This is proved in [lesson 17](../module-06/lesson-03.md): the messy softmax Jacobian and the $-\log$ derivative combine and cancel almost everything, leaving just $\mathbf{p} - \mathbf{y}$. For our example, the one-hot target for "dog" is $\mathbf{y} = [0, 1, 0]$, so

$$
\frac{\partial L_{\text{CE}}}{\partial \mathbf{z}} = [0.6652,\ 0.2447,\ 0.0900] - [0,\ 1,\ 0] = [0.6652,\ -0.7553,\ 0.0900].
$$

Read the signs: the correct class "dog" has a **negative** gradient ($-0.7553$), so training will *raise* its logit; the two wrong classes have positive gradients, so their logits get *pushed down*. The gradient magnitude on "dog" is largest because that is where the model is most wrong. This single vector is what flows back into the network as the starting upstream gradient of the entire backward pass — the classification analogue of $\partial L/\partial y$ from [lesson 23](lesson-01.md).

<div class="callout key"><p>Cross-entropy on a one-hot target reduces to $-\log p_c$, the negative log probability of the true class. Its gradient with respect to the logits is simply $\mathbf{p} - \mathbf{y}$: predicted minus target. That clean form is why classification networks are trained on logits with cross-entropy — the backward pass starts from a trivial subtraction.</p></div>

## 5. Binary cross-entropy (BCE)

When there are only **two** outcomes (yes/no, spam/not-spam), you do not need a full softmax. The model outputs one probability $p$ for the positive class (via a sigmoid), and the negative class gets $1 - p$. BCE is cross-entropy specialized to this two-outcome case:

$$
L_{\text{BCE}} = -\big[y\log p + (1 - y)\log(1 - p)\big].
$$

- $y$ — the true label, $1$ (positive) or $0$ (negative).
- $p$ — predicted probability of the positive class, in $(0,1)$.
- When $y = 1$ only the first term survives: $-\log p$. When $y = 0$ only the second: $-\log(1-p)$.

**Tiny examples.** Model says $p = 0.8$ and the truth is $y = 1$:

$$
L = -\log(0.8) = 0.2231.
$$

Model says $p = 0.3$ and the truth is $y = 0$:

$$
L = -\log(1 - 0.3) = -\log(0.7) = 0.3567.
$$

Both losses are small-ish because the model leaned the right way; a confident wrong answer ($p=0.05$ when $y=1$) would give $-\log(0.05) = 3.0$.

## 6. In PyTorch

Every loss above is one call in `torch.nn.functional` (imported as `F`). The snippet reproduces the numbers from sections 2–5.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.0])
target = torch.tensor(1)               # class index for "dog"

# softmax and log-softmax
print(F.softmax(logits, dim=0))        # tensor([0.6652, 0.2447, 0.0900])
print(F.log_softmax(logits, dim=0))    # tensor([-0.4076, -1.4076, -2.4076])

# cross-entropy takes RAW LOGITS (it does log_softmax + NLL internally)
loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
print(loss)                            # tensor(1.4076)

# same thing done in two explicit pieces: log_softmax then NLL
logp = F.log_softmax(logits, dim=0)
print(F.nll_loss(logp.unsqueeze(0), target.unsqueeze(0)))  # tensor(1.4076)

# regression losses
pred = torch.tensor([2.5, 0.0, 2.0])
tgt  = torch.tensor([3.0, -0.5, 2.0])
print(F.mse_loss(pred, tgt))           # tensor(0.1667)
print(F.l1_loss(pred, tgt))            # tensor(0.3333)   (L1 = MAE)

# binary cross-entropy from a raw logit (numerically stable form)
z = torch.tensor([1.0])                # one logit
y = torch.tensor([1.0])                # positive label
print(F.binary_cross_entropy_with_logits(z, y))  # tensor(0.3133)
# sigmoid(1.0) = 0.7311, and -log(0.7311) = 0.3133
```

Now verify the cross-entropy gradient is $\mathbf{p} - \mathbf{y}$ via autograd:

```python
logits = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
target = torch.tensor(1)
loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
loss.backward()
print(logits.grad)     # tensor([ 0.6652, -0.7553,  0.0900])
```

That is exactly $\mathbf{p} - \mathbf{y} = [0.6652, 0.2447, 0.0900] - [0, 1, 0]$, matching section 4.4 to four decimals.

<div class="callout pt"><p><code>F.cross_entropy</code> takes <strong>raw logits</strong>, not probabilities — it applies log-softmax and NLL for you. Passing it softmax outputs (or hand-softmaxed probabilities) is a very common bug: you would be taking the log-softmax of already-normalized numbers and get a wrong, too-small loss. Rule: give <code>cross_entropy</code> the logits straight from the final linear layer.</p></div>

## 7. What PyTorch is doing under the hood

Why does `F.cross_entropy` fuse log-softmax and NLL into one op instead of letting you call `softmax`, then `log`, then `nll_loss`? **Numerical stability.**

Softmax exponentiates the logits. Real logits can be large — a logit of $100$ gives $e^{100} \approx 2.7\times 10^{43}$, and $e^{1000}$ simply **overflows** to `inf` in floating point. Then $\text{inf}/\text{inf} = $ `NaN`, and the loss is ruined. Taking `log` of a softmax output that underflowed to exactly $0.0$ gives $\log(0) = -\text{inf}$, equally fatal.

Log-softmax sidesteps both. Using the log-sum-exp term from section 3.2, PyTorch computes $z_i - \log\sum_j e^{z_j}$, and it evaluates that $\log\sum e$ with the **max-subtraction trick**: factor out the largest logit $m$ so the exponentials are all $e^{z_j - m} \le 1$ — no overflow — while the answer is mathematically unchanged. Combining log-softmax with NLL in one kernel means the fragile intermediate probability is never materialized at all. This is the same log-sum-exp stability idea we treat in depth in [module 12](../module-12/lesson-01.md); for now the practical rule stands: feed logits to `cross_entropy`, never pre-softmaxed probabilities. The same reasoning is why `binary_cross_entropy_with_logits` is preferred over applying `sigmoid` then `binary_cross_entropy`.

## Check yourself

<details><summary>You have logits <code>[0.0, 0.0, 0.0]</code> for a 3-class problem. What is the cross-entropy loss regardless of the target, and why?</summary>

Softmax of equal logits is uniform: $[1/3, 1/3, 1/3]$. The true class has probability $1/3$, so $L = -\log(1/3) = \log 3 \approx 1.0986$. With no information, the loss equals the log of the number of classes — a useful sanity-check baseline for an untrained classifier.

</details>

<details><summary>Why does <code>F.cross_entropy</code> want logits while <code>F.nll_loss</code> wants log-probabilities?</summary>

`cross_entropy` = `log_softmax` + `nll_loss` fused. `nll_loss` does *only* the "negate the log-probability of the true class" step, so it expects the log-softmax to have already been applied. `cross_entropy` bundles the log-softmax in, so you hand it the raw logits.

</details>

<details><summary>For the example, the gradient on "dog" was $-0.7553$ and on "cat" was $+0.6652$. In one training step, what happens to the two logits?</summary>

Gradient descent steps against the gradient. "Dog" has a negative gradient, so its logit goes *up* (the model becomes more likely to predict the correct token). "Cat" has a positive gradient, so its logit goes *down*. The loss on this example drops.

</details>

<details><summary>Why prefer MAE over MSE when your data has a few wild outliers?</summary>

MSE squares the residuals, so a single large error contributes its square and dominates the loss and gradient, dragging the fit toward the outlier. MAE grows only linearly with the error, so outliers have proportionally less pull. MAE is the more robust choice.

</details>

## Next

You can now turn any prediction into a single loss number and — crucially — you have its gradient, the seed of the backward pass. The next lesson ties the by-hand backward pass of lesson 23 and the $\mathbf{p} - \mathbf{y}$ gradient here to the machinery inside PyTorch: `requires_grad`, `grad_fn`, the graph, gradient accumulation, and `zero_grad()`. That is backpropagation and autograd, demystified end to end.

Continue to [25 · Backpropagation and autograd](lesson-03.md).
