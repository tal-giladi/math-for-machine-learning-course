# 17 · Derivatives of exp, log, sigmoid, softmax

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-06/lesson-02">16 · Derivative rules and the chain rule</a> (power, quotient, and chain rules) and <a href="#/lessons/module-01/lesson-03">03 · Powers, roots, exponentials, logarithms</a> for what $e^x$ and $\ln x$ mean.</p>
<p><strong>You will learn:</strong> the derivatives of the four functions that appear in every neural network's forward pass — the exponential $e^x$, the natural logarithm $\ln x$, the <strong>sigmoid</strong> $\sigma(x)$ (derived in full), and the <strong>softmax</strong> (with its Jacobian derived and worked numerically).</p>
<p><strong>Why this matters for ML:</strong> sigmoid and softmax are the nonlinearities that turn raw network scores into probabilities, and $e^x$/$\ln x$ are their building blocks and the core of the cross-entropy loss. Their derivatives appear in <em>every</em> backward pass, so knowing them by hand demystifies what autograd computes.</p>
</div>

## 1. The exponential: its own derivative

The natural exponential $e^x$ (where $e \approx 2.71828$ is Euler's number, introduced in [module 1](lessons/module-01/lesson-03.md)) has the remarkable property that differentiating it changes nothing:

$$
\frac{d}{dx}\,e^x = e^x
$$

- $e^x$ — the exponential function; $e$ is a fixed constant, $x$ the variable.

That is the whole rule: the slope of $e^x$ at any point equals the *height* of $e^x$ at that point. This is in fact the defining property of $e$ — it is the one base for which the exponential is its own rate of change.

**Numerical feel.** At $x = 0$, $e^0 = 1$, so the slope there is $1$. Check with a small step $h = 0.001$: $\frac{e^{0.001} - e^{0}}{0.001} = \frac{1.0010005 - 1}{0.001} \approx 1.0005$, closing in on $1$. At $x = 1$, $e^1 \approx 2.71828$, and the slope there is also $\approx 2.71828$.

Combined with the chain rule, a scaled exponent gives $\frac{d}{dx}e^{kx} = k\,e^{kx}$, and in particular $\frac{d}{dx}e^{-x} = -e^{-x}$ — we need that one in a moment for the sigmoid.

**ML use:** $e^x$ is the engine inside softmax (it turns any real score into a positive number) and appears in Gaussian densities; its self-derivative keeps those gradients clean.

## 2. The natural logarithm

The natural logarithm $\ln x$ is the inverse of $e^x$ (it answers "$e$ to what power gives $x$?"). Its derivative is one of the tidiest in all of calculus:

$$
\frac{d}{dx}\,\ln x = \frac{1}{x} \qquad (x > 0)
$$

- $\ln x$ — natural log, defined for positive $x$.

**Worked at $x = 2$:** $\frac{1}{2} = 0.5$. Numerical check with $h = 0.001$: $\frac{\ln(2.001) - \ln(2)}{0.001} = \frac{0.6936471 - 0.6931472}{0.001} \approx 0.4999$, essentially $0.5$. 

**ML use:** the cross-entropy loss for a language model is $-\ln(\text{probability of the correct token})$. Its derivative therefore carries a $\frac{1}{x}$ factor, which — combined with the softmax derivative below — produces the famously simple training signal we preview in section 5.

## 3. The sigmoid, derived in full

### 3.1 Intuition

The **sigmoid** squashes any real number into the open interval $(0, 1)$, making it readable as a probability. Big positive inputs go near $1$, big negative inputs near $0$, and $0$ maps to exactly $\tfrac12$. Its S-shape is smooth, so it has a derivative everywhere.

### 3.2 The mathematics

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

- $\sigma(x)$ (read "sigma of x") — the sigmoid output, always in $(0,1)$.
- $e^{-x}$ — the exponential of $-x$; large when $x$ is very negative, tiny when $x$ is very positive.

We claim its derivative is

$$
\sigma'(x) = \sigma(x)\big(1 - \sigma(x)\big),
$$

a beautiful formula expressing the slope purely in terms of the output. Let us derive it.

### 3.3 The derivation, step by step

Write $\sigma(x) = (1 + e^{-x})^{-1}$ and use the chain rule with inner function $u = 1 + e^{-x}$, so $\sigma = u^{-1}$.

**Step 1 — outer derivative.** By the power rule, $\dfrac{d\sigma}{du} = -u^{-2} = -\dfrac{1}{u^2}$.

**Step 2 — inner derivative.** $\dfrac{du}{dx} = \dfrac{d}{dx}(1 + e^{-x}) = 0 + (-e^{-x}) = -e^{-x}$ (constant rule on the $1$; the $e^{-x}$ rule from section 1).

**Step 3 — multiply (chain rule).**

$$
\sigma'(x) = \frac{d\sigma}{du}\cdot\frac{du}{dx} = \left(-\frac{1}{u^2}\right)\!\big(-e^{-x}\big) = \frac{e^{-x}}{(1+e^{-x})^2}.
$$

The two minus signs cancel, leaving a positive slope — correct, since the sigmoid is always increasing.

**Step 4 — rewrite in terms of $\sigma$.** Split the fraction:

$$
\frac{e^{-x}}{(1+e^{-x})^2} = \underbrace{\frac{1}{1+e^{-x}}}_{\sigma(x)} \cdot \underbrace{\frac{e^{-x}}{1+e^{-x}}}_{1-\sigma(x)}.
$$

The first factor is $\sigma(x)$ by definition. For the second, note $1 - \sigma(x) = 1 - \frac{1}{1+e^{-x}} = \frac{(1+e^{-x}) - 1}{1+e^{-x}} = \frac{e^{-x}}{1+e^{-x}}$ — exactly the second factor. Therefore

$$
\sigma'(x) = \sigma(x)\big(1 - \sigma(x)\big). \qquad\checkmark
$$

### 3.4 Evaluate at $x = 0$

$\sigma(0) = \frac{1}{1 + e^{0}} = \frac{1}{1+1} = 0.5$. So

$$
\sigma'(0) = 0.5\,(1 - 0.5) = 0.5 \cdot 0.5 = 0.25.
$$

The sigmoid is steepest at the origin, with slope $0.25$, and flattens toward $0$ at the extremes (when $\sigma$ is near $0$ or $1$, the product $\sigma(1-\sigma)$ is near $0$). That flattening is the origin of the **vanishing-gradient** problem: deep stacks of sigmoids multiply many small slopes together, shrinking the backward signal toward zero.

<div class="callout key"><p>$\sigma'(x) = \sigma(x)(1-\sigma(x))$ — the sigmoid's slope is expressed entirely through its own output. Maximum $0.25$ at $x=0$; it vanishes as the output saturates near $0$ or $1$.</p></div>

## 4. Softmax and its Jacobian

### 4.1 Intuition

Sigmoid turns one score into one probability. **Softmax** turns a whole *vector* of scores into a probability distribution: all outputs positive and summing to $1$. In a language model the scores are the **logits** — one per vocabulary token — and softmax converts them into the probability of each token being next.

### 4.2 The mathematics

For an input vector $\mathbf{z} = (z_1, z_2, \dots, z_K)$ of $K$ scores, the $i$-th softmax output is

$$
\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}.
$$

- $z_i$ — the $i$-th input score (logit); any real number.
- $e^{z_i}$ — its exponential, always positive.
- $\sum_{k=1}^{K} e^{z_k}$ — the normalizer: the sum of all exponentials, which forces the outputs to add to $1$.
- $K$ — the number of scores (the vocabulary size $V$ in a language model).

Write $s_i = \operatorname{softmax}(\mathbf{z})_i$ for short. Because output $s_i$ depends on *every* input $z_j$ (they all sit in the shared denominator), there is not one derivative but a whole grid of them — the **Jacobian**, a matrix of partial derivatives $\partial s_i / \partial z_j$ (partial derivatives are the subject of [lesson 4](lessons/module-06/lesson-04.md); here we just need the pattern). The result is

$$
\frac{\partial s_i}{\partial z_j} = s_i\big(\delta_{ij} - s_j\big),
$$

where $\delta_{ij}$ is the **Kronecker delta**: $\delta_{ij} = 1$ when $i = j$ and $\delta_{ij} = 0$ when $i \neq j$. So there are two cases:

- **Diagonal** ($i = j$): $\dfrac{\partial s_i}{\partial z_i} = s_i(1 - s_i)$ — note this echoes the sigmoid's $\sigma(1-\sigma)$, no coincidence.
- **Off-diagonal** ($i \neq j$): $\dfrac{\partial s_i}{\partial z_j} = -\,s_i s_j$ — always negative: raising one logit steals probability from the others.

### 4.3 A tiny numerical example

Take two logits $\mathbf{z} = (1, 0)$ (a 2-token vocabulary). Exponentiate: $e^{1} = 2.71828$, $e^{0} = 1$. The normalizer is $2.71828 + 1 = 3.71828$. So

$$
s_1 = \frac{2.71828}{3.71828} = 0.73106, \qquad s_2 = \frac{1}{3.71828} = 0.26894,
$$

and indeed $s_1 + s_2 = 1$. Now build the $2\times2$ Jacobian:

$$
\frac{\partial s_1}{\partial z_1} = s_1(1 - s_1) = 0.73106 \times 0.26894 = 0.19661,
$$
$$
\frac{\partial s_1}{\partial z_2} = -\,s_1 s_2 = -0.73106 \times 0.26894 = -0.19661,
$$
$$
\frac{\partial s_2}{\partial z_1} = -\,s_2 s_1 = -0.19661, \qquad
\frac{\partial s_2}{\partial z_2} = s_2(1 - s_2) = 0.26894 \times 0.73106 = 0.19661.
$$

$$
J = \begin{bmatrix} 0.19661 & -0.19661 \\ -0.19661 & 0.19661 \end{bmatrix}.
$$

Two sanity checks. First, each *column* sums to $0$ ($0.19661 - 0.19661 = 0$): that must happen, because the outputs always sum to $1$, so nudging any single $z_j$ cannot change the total — the probability it adds to one token it removes from the others. Second, for two classes softmax reduces to a sigmoid: $s_1 = \sigma(z_1 - z_2) = \sigma(1) = 0.73106$, and its diagonal slope $s_1(1-s_1) = 0.19661$ is exactly $\sigma'(1)$. The pieces fit together.

### 4.4 The result that comes later

In practice softmax is almost never used alone — it is paired with the **cross-entropy loss**. When you differentiate that combined "softmax then cross-entropy" pipeline with respect to the logits, the messy Jacobian above collapses into something startlingly clean:

$$
\frac{\partial L}{\partial z_i} = s_i - y_i = p_i - y_i,
$$

the predicted probability minus the target (a one-hot label). We will prove this cancellation in **module 8**, on loss functions; flag it now as the reason softmax + cross-entropy is the standard output layer for classification and language modeling — the gradient it hands back is simply "prediction minus truth."

## 5. Why ML needs these specific derivatives

Sigmoid and softmax are the two functions that convert a network's internal scores into probabilities: sigmoid for a single yes/no output, softmax for a distribution over a vocabulary. Because they sit at (or near) the end of the forward pass, their derivatives are the *first* thing the backward pass touches on the way back toward the parameters. Every gradient that flows into GPT-2's weights has been multiplied by a softmax derivative on its way out of the output layer. Meanwhile $e^x$ and $\ln x$ are the atoms these are built from — $e^x$ inside softmax, $\ln x$ inside cross-entropy — so their simple derivatives ($e^x$ and $1/x$) are what make the end-to-end gradient tractable.

## 6. In PyTorch

### Sigmoid — verify $\sigma'(0) = 0.25$

```python
import torch

x = torch.tensor(0., requires_grad=True)
y = torch.sigmoid(x)       # forward: sigma(0) = 0.5
y.backward()
print(y)                   # tensor(0.5000, grad_fn=<SigmoidBackward0>)
print(x.grad)              # tensor(0.2500)   ->  sigma(0)*(1-sigma(0)) = 0.5*0.5
```

`x.grad` is `tensor(0.2500)`, exactly the $\sigma(0)(1-\sigma(0)) = 0.25$ we derived by hand.

### Exp and log

```python
a = torch.tensor(0., requires_grad=True)
torch.exp(a).backward()
print(a.grad)              # tensor(1.)      ->  d/dx e^x at 0 is e^0 = 1

b = torch.tensor(2., requires_grad=True)
torch.log(b).backward()    # torch.log is the NATURAL log (ln)
print(b.grad)              # tensor(0.5000)  ->  1/x at x=2
```

Both match sections 1 and 2: $e^0 = 1$ and $1/2 = 0.5$.

### Softmax — recover a Jacobian row

Differentiating a vector output needs a little care: `.backward()` wants a scalar, so to read the derivatives of $s_1$ with respect to every logit we call backward *on $s_1$ alone*. The resulting `z.grad` is the first row of the Jacobian.

```python
import torch.nn.functional as F

z = torch.tensor([1., 0.], requires_grad=True)
s = F.softmax(z, dim=0)    # tensor([0.7311, 0.2689])
s[0].backward()            # differentiate s_1 w.r.t. all of z
print(z.grad)              # tensor([ 0.1966, -0.1966])
```

`z.grad` is `[0.1966, -0.1966]` — precisely the first row $[\,s_1(1-s_1),\ -s_1 s_2\,]$ of the Jacobian $J$ we built in section 4.3.

<div class="callout pt"><p><code>torch.sigmoid</code>, <code>torch.exp</code>, <code>torch.log</code> (natural log), and <code>F.softmax</code> each carry their exact derivative rule. In real models you rarely call <code>softmax</code> then cross-entropy separately; you use <code>F.cross_entropy</code>, which fuses them and uses the tidy $p - y$ gradient from section 4.4 for stability.</p></div>

## 7. What PyTorch is doing under the hood

For the sigmoid, the forward pass stores the *output* $0.5$ on the node `y` (with `grad_fn=<SigmoidBackward0>`), not just the input. That is deliberate: the derivative rule $\sigma' = \sigma(1-\sigma)$ needs only the output, so `SigmoidBackward0` reads the saved output $s$ and returns $s(1-s) = 0.5 \cdot 0.5 = 0.25$ — no re-exponentiation required. This is a recurring autograd theme: an operation caches whichever forward values make its backward cheapest.

For softmax the backward node holds the full output vector $\mathbf{s}$ and applies the Jacobian rule $s_i(\delta_{ij} - s_j)$. When you called `s[0].backward()`, PyTorch effectively pushed the vector $(1, 0)$ (select the first output) back through that Jacobian, which extracts its first row — giving `z.grad = [0.1966, -0.1966]`. This "push a vector back through a Jacobian" operation is the **vector–Jacobian product** at the heart of reverse-mode autograd; you will meet it formally when we generalize the chain rule to vectors in [module 7](lessons/module-07/lesson-01.md). The key point for now: autograd never builds the giant $V \times V$ softmax Jacobian for a real vocabulary — it only ever multiplies a vector *through* it, which is far cheaper.

## Check yourself

<details><summary>What is $\frac{d}{dx}e^{-x}$, and why does the sign appear?</summary>

$-e^{-x}$. By the chain rule the inner function is $-x$ with derivative $-1$, and $\frac{d}{du}e^u = e^u$, so the result is $e^{-x}\cdot(-1) = -e^{-x}$. The exponential is decreasing in $x$, hence the negative slope.

</details>

<details><summary>Compute $\sigma'(x)$ when $\sigma(x) = 0.8$.</summary>

$\sigma'(x) = \sigma(1-\sigma) = 0.8 \times 0.2 = 0.16$. (You do not even need $x$ — the slope depends only on the output.)

</details>

<details><summary>For softmax outputs $s = (0.7, 0.2, 0.1)$, what is $\partial s_1/\partial z_1$ and $\partial s_1/\partial z_2$?</summary>

Diagonal: $\partial s_1/\partial z_1 = s_1(1-s_1) = 0.7 \times 0.3 = 0.21$. Off-diagonal: $\partial s_1/\partial z_2 = -s_1 s_2 = -0.7 \times 0.2 = -0.14$.

</details>

<details><summary>Why do the columns of the softmax Jacobian sum to zero?</summary>

Because the softmax outputs always sum to $1$. Changing a single logit $z_j$ cannot change that total, so the increases and decreases it causes across the outputs must cancel — i.e. $\sum_i \partial s_i/\partial z_j = 0$.

</details>

## Next

You can now differentiate the nonlinearities that end a network's forward pass. But real losses depend on *many* variables at once — every weight in the model — so we need derivatives with respect to one variable while the others are held fixed. That is the partial derivative, and collecting all of them gives the gradient: the vector that training walks downhill.

Continue to [18 · Partial derivatives and the gradient](lessons/module-06/lesson-04.md).
