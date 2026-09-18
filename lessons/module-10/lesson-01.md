# 29 · Probability and statistics for ML

<div class="prereq">
<p><strong>Prerequisites:</strong> logarithms and exponentials from <a href="../module-01/lesson-03.md">03 · Powers, roots, exponentials, logarithms</a>, summation notation from <a href="../module-01/lesson-04.md">04 · Summation, products, inequalities, rates</a>, and the cross-entropy / softmax stack from <a href="../module-08/lesson-02.md">24 · Loss functions</a>. Gradient descent from <a href="../module-09/lesson-01.md">26 · Gradient descent</a> is helpful for the "why minimize NLL" part.</p>
<p><strong>You will learn:</strong> probability as numbers in $[0,1]$ that sum to $1$; random variables and distributions (Bernoulli, categorical, a word on Gaussian); expectation, variance, standard deviation, and covariance; conditional probability and Bayes' theorem; and the information-theory quartet that <em>is</em> the language-model objective — entropy, cross-entropy, KL divergence, and maximum likelihood / negative log likelihood.</p>
<p><strong>Why this matters for ML:</strong> a GPT-2 model does exactly one thing at each position — it outputs a probability distribution over the vocabulary for the next token. Training makes the real next tokens as probable as possible. That single sentence is maximum likelihood, and it is identical to minimizing cross-entropy, which is identical to minimizing negative log likelihood. This lesson is the probability behind the loss you already met.</p>
</div>

## 1. Probability: numbers in $[0,1]$ that sum to $1$

A **probability** is a number measuring how likely an outcome is. It obeys two rules and nothing more:

- Every probability is between $0$ and $1$: $0 \le P(\text{outcome}) \le 1$. Zero means impossible, one means certain.
- Over a complete set of mutually exclusive outcomes, the probabilities **sum to $1$**: $\sum_i P(\text{outcome}_i) = 1$. Something must happen.

That is the whole foundation. A fair coin has $P(\text{heads}) = P(\text{tails}) = 0.5$, and $0.5 + 0.5 = 1$. A fair six-sided die has $P(\text{each face}) = 1/6$, and six of those sum to $1$.

For a programmer: think of a probability distribution as a dictionary mapping each possible outcome to a non-negative weight, with the constraint that the weights add up to exactly $1$. The "sum to one" rule is the one you will see enforced in code over and over — it is precisely what softmax guarantees.

<div class="callout key"><p>A probability distribution is a list of non-negative numbers, one per outcome, summing to $1$. Softmax exists to manufacture exactly such a list out of arbitrary real-valued scores.</p></div>

## 2. Random variables

A **random variable** is a quantity whose value is decided by a random outcome. We write it with a capital letter, usually $X$. It is not a fixed number; it is a placeholder for "whatever value comes out."

- A coin flip: let $X = 1$ for heads, $X = 0$ for tails. $X$ is a random variable taking values in $\{0, 1\}$.
- A die: let $X$ be the face shown, taking values in $\{1, 2, 3, 4, 5, 6\}$.
- Next-token prediction: let $X$ be the index of the next token, taking values in $\{0, 1, \dots, V-1\}$ where $V$ is the vocabulary size.

A random variable plus the probability of each of its values is a **distribution**. Writing $P(X = x)$ means "the probability that the random variable $X$ takes the specific value $x$." The lowercase $x$ is a particular value; the uppercase $X$ is the variable itself.

## 3. Distributions you need

### 3.1 Bernoulli — one yes/no trial

A **Bernoulli** distribution describes a single event with two outcomes, "success" (1) with probability $p$ and "failure" (0) with probability $1-p$:

$$
P(X = 1) = p, \qquad P(X = 0) = 1 - p.
$$

- $p$ — the success probability, a single number in $[0,1]$.
- The two probabilities sum to $1$ automatically. A biased coin with $p = 0.7$ is Bernoulli.

This is the distribution behind **binary** classification (spam / not-spam), and it pairs with the binary cross-entropy loss from [lesson 24](../module-08/lesson-02.md).

### 3.2 Categorical — one draw from $V$ outcomes

A **categorical** distribution generalizes Bernoulli from two outcomes to $V$ outcomes, each with its own probability:

$$
P(X = i) = p_i, \qquad p_i \ge 0, \qquad \sum_{i=0}^{V-1} p_i = 1.
$$

- $i$ — an outcome index, $0$ to $V-1$.
- $p_i$ — the probability of outcome $i$; the whole vector $\mathbf{p} = [p_0, \dots, p_{V-1}]$ is the distribution.

This is *the* distribution of language modeling. Roll a die: $\mathbf{p} = [1/6, \dots, 1/6]$. Predict the next token over a vocabulary of $V$ words: $\mathbf{p}$ is a length-$V$ vector of token probabilities. Softmax produces a categorical distribution.

### 3.3 Gaussian — a word

Not everything is discrete. The **Gaussian** (normal) distribution describes a continuous quantity clustered around a mean $\mu$ with spread $\sigma$; its bell-curve density is $\frac{1}{\sigma\sqrt{2\pi}}\exp\!\big(-\tfrac{(x-\mu)^2}{2\sigma^2}\big)$. You do not need it for the next-token loss, but it is how weights are often **initialized** (small random numbers drawn from a Gaussian) and how noise is modeled. We will only use it in passing; the discrete categorical distribution is what carries this module.

## 4. Expectation: the probability-weighted average

### 4.1 Intuition

The **expectation** (or expected value, or mean) of a random variable is the long-run average value you would see if you sampled it many times. It is not necessarily a value the variable can actually take — the expected value of a die is $3.5$.

### 4.2 The mathematics

$$
\mathbb{E}[X] = \sum_i p_i \, x_i.
$$

- $x_i$ — a value the variable can take.
- $p_i$ — the probability of that value.
- $\mathbb{E}[X]$ — read "expectation of $X$," often written $\mu$ (the Greek letter mu). It is a single number.

Each possible value is weighted by how likely it is, and the weighted values are summed. If all outcomes are equally likely this reduces to the ordinary average.

### 4.3 A tiny worked example

Let $X$ take values $x = [0, 1, 2]$ with probabilities $p = [0.2, 0.5, 0.3]$ (they sum to $1$ ✓):

$$
\mathbb{E}[X] = 0 \cdot 0.2 + 1 \cdot 0.5 + 2 \cdot 0.3 = 0 + 0.5 + 0.6 = 1.1.
$$

So $\mu = 1.1$. The most likely single value is $1$, but the mass leaning toward $2$ pulls the average up to $1.1$.

### 4.4 Why ML needs it

Every loss you minimize is an **expectation**: the training loss is the expected (average) per-example loss over your data. "Minimize the loss" formally means "minimize the expected loss." When we write $\mathbb{E}$ in a paper, in code it is almost always `.mean()` over a batch.

## 5. Variance and standard deviation: the spread

### 5.1 The mathematics

**Variance** measures how far, on average, the variable lands from its mean — using squared distances:

$$
\text{Var}(X) = \mathbb{E}\big[(X - \mu)^2\big] = \sum_i p_i (x_i - \mu)^2.
$$

- $\mu = \mathbb{E}[X]$ — the mean from section 4.
- $(x_i - \mu)^2$ — squared distance of value $i$ from the mean (squared so positive and negative deviations both count).
- The result is in squared units. The **standard deviation** $\sigma = \sqrt{\text{Var}(X)}$ brings it back to the original units.

### 5.2 A tiny worked example

Same distribution: $x = [0,1,2]$, $p = [0.2, 0.5, 0.3]$, $\mu = 1.1$. Compute each squared deviation times its probability:

$$
(0 - 1.1)^2 = 1.21, \quad 1.21 \cdot 0.2 = 0.242,
$$
$$
(1 - 1.1)^2 = 0.01, \quad 0.01 \cdot 0.5 = 0.005,
$$
$$
(2 - 1.1)^2 = 0.81, \quad 0.81 \cdot 0.3 = 0.243.
$$

Sum them:

$$
\text{Var}(X) = 0.242 + 0.005 + 0.243 = 0.49, \qquad \sigma = \sqrt{0.49} = 0.7.
$$

The typical distance from the mean is about $0.7$. (Sanity check via the identity $\text{Var}(X) = \mathbb{E}[X^2] - \mu^2$: $\mathbb{E}[X^2] = 0\cdot0.2 + 1\cdot0.5 + 4\cdot0.3 = 1.7$, and $1.7 - 1.1^2 = 1.7 - 1.21 = 0.49$ ✓.)

### 5.3 Why ML needs it

Variance is the heart of **normalization** (LayerNorm, coming in module 13): a layer rescales its activations to zero mean and unit variance so the numbers flowing through the network stay well-behaved. Variance also describes how noisy a gradient estimate is — the whole point of averaging over a batch is to reduce the variance of the gradient.

## 6. Covariance (brief)

Where variance describes one variable, **covariance** describes how two vary *together*:

$$
\text{Cov}(X, Y) = \mathbb{E}\big[(X - \mu_X)(Y - \mu_Y)\big].
$$

- Positive: when $X$ is above its mean, $Y$ tends to be above its mean too (they move together).
- Negative: they move oppositely.
- Near zero: no linear relationship.

You will not compute covariance by hand in this course, but the idea underlies correlation, whitening, and the covariance matrix — and it explains why a weight matrix's columns being correlated is something optimizers like Muon (module 9) care about.

## 7. Conditional probability

### 7.1 The idea

A **conditional probability** is the probability of one event *given that* another has happened. Notation: $P(A \mid B)$, read "probability of $A$ given $B$." Knowing $B$ narrows the world to only the cases where $B$ is true.

### 7.2 The mathematics

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}.
$$

- $P(A \cap B)$ — probability that $A$ **and** $B$ both happen (the intersection).
- $P(B)$ — probability of the condition; must be nonzero.
- Dividing by $P(B)$ rescales so that, within the world where $B$ holds, probabilities again sum to $1$.

### 7.3 A tiny worked example

Ten emails. Four are spam. Of the ten, five contain the word "free," and three emails are *both* spam and contain "free." Let $A$ = "spam," $B$ = "contains free." Then $P(B) = 5/10 = 0.5$ and $P(A \cap B) = 3/10 = 0.3$. So

$$
P(\text{spam} \mid \text{free}) = \frac{P(A \cap B)}{P(B)} = \frac{0.3}{0.5} = 0.6.
$$

An unconditioned email is $40\%$ likely to be spam, but *given* it says "free," it is $60\%$ likely. The condition changed the probability — that is the entire value of conditioning.

### 7.4 Why ML needs it

A language model is a giant conditional-probability machine. It computes $P(\text{next token} \mid \text{all previous tokens})$. Everything GPT-2 outputs is a conditional distribution given the context.

## 8. Bayes' theorem

### 8.1 The idea

Bayes' theorem flips a conditional around. It lets you compute $P(A \mid B)$ from $P(B \mid A)$ — updating a prior belief in light of evidence.

### 8.2 The mathematics

$$
P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}.
$$

- $P(A)$ — the **prior**: belief in $A$ before evidence.
- $P(B \mid A)$ — the **likelihood**: how probable the evidence is if $A$ holds.
- $P(B)$ — the **evidence**: total probability of $B$, expanded as $P(B \mid A)P(A) + P(B \mid \neg A)P(\neg A)$.
- $P(A \mid B)$ — the **posterior**: updated belief after seeing $B$.

### 8.3 A tiny worked example

A disease affects $1\%$ of people: $P(D) = 0.01$. A test catches it $90\%$ of the time when present: $P(+ \mid D) = 0.9$. It falsely fires $5\%$ of the time when absent: $P(+ \mid \neg D) = 0.05$. You test positive — what is the chance you are actually sick?

First the evidence, the total probability of a positive test:

$$
P(+) = P(+ \mid D)P(D) + P(+ \mid \neg D)P(\neg D) = 0.9 \cdot 0.01 + 0.05 \cdot 0.99 = 0.009 + 0.0495 = 0.0585.
$$

Then Bayes:

$$
P(D \mid +) = \frac{P(+ \mid D)P(D)}{P(+)} = \frac{0.009}{0.0585} = 0.1538.
$$

Only about $15\%$ — surprisingly low, because the disease is rare and false positives outnumber true positives. This "base-rate" reasoning is why Bayes' theorem is famous.

### 8.4 Why ML needs it

Bayes is the conceptual root of probabilistic modeling: priors on weights (weight decay has a Bayesian reading as a Gaussian prior), Bayesian inference, and the derivation of many loss functions. For this course the main takeaway is the mindset — treat model outputs as beliefs to be updated by data.

## 9. Entropy — the six-pass concept

### 9.1 Intuition

**Entropy** measures the uncertainty in a distribution — the average "surprise" of its outcomes. A fair coin is maximally uncertain; a two-headed coin has zero uncertainty (you always know the result). Entropy is highest when the distribution is flat and lowest when it is concentrated on one outcome.

### 9.2 The mathematics

$$
H(p) = -\sum_i p_i \log p_i.
$$

- $p_i$ — probability of outcome $i$.
- $-\log p_i$ — the **surprise** (or information content) of outcome $i$: rare events ($p_i$ small) are very surprising ($-\log p_i$ large); certain events ($p_i = 1$) carry zero surprise ($-\log 1 = 0$).
- $H(p)$ — the probability-weighted average surprise, i.e. $\mathbb{E}[-\log p]$. It is never negative.

The base of the log sets the unit. Natural log ($\ln$) gives **nats**; log base 2 gives **bits**. PyTorch and this course use $\ln$, so entropy is in nats.

### 9.3 A tiny worked example

**Fair coin** $p = [0.5, 0.5]$:

$$
H = -\big(0.5 \ln 0.5 + 0.5 \ln 0.5\big) = -\ln 0.5 = \ln 2 = 0.6931 \text{ nats.}
$$

**Skewed coin** $p = [0.9, 0.1]$:

$$
H = -\big(0.9 \ln 0.9 + 0.1 \ln 0.1\big) = -\big(0.9 \cdot (-0.10536) + 0.1 \cdot (-2.30259)\big),
$$
$$
H = -(-0.09482 - 0.23026) = 0.3251 \text{ nats.}
$$

The skewed coin ($0.3251$) has less than half the entropy of the fair coin ($0.6931$): it is more predictable, so less uncertain. A distribution pinned entirely on one outcome, $[1, 0]$, has $H = -1\ln 1 - 0\ln 0 = 0$ (using the convention $0 \ln 0 = 0$): no uncertainty at all.

### 9.4 Why ML needs it

Entropy is the floor of the language-model loss. The cross-entropy loss (next) can never drop below the entropy of the true next-token distribution — that irreducible uncertainty is the inherent unpredictability of language. Entropy also appears directly as a regularizer: policies and samplers are sometimes pushed toward higher entropy to keep them from collapsing to a single choice.

### 9.5 In PyTorch

```python
import torch
from torch.distributions import Categorical

fair   = Categorical(probs=torch.tensor([0.5, 0.5]))
skewed = Categorical(probs=torch.tensor([0.9, 0.1]))
print(fair.entropy())    # tensor(0.6931)
print(skewed.entropy())  # tensor(0.3251)

d = Categorical(probs=torch.tensor([0.2, 0.5, 0.3]))
print(d.entropy())       # tensor(1.0297)
print(d.log_prob(torch.tensor(1)))  # log(0.5) = tensor(-0.6931)
print(d.sample())        # a random index 0,1,2 drawn with those probs
```

### 9.6 Under the hood

`Categorical` stores the probabilities (or logits) and computes `entropy()` as exactly $-\sum p_i \log p_i$; `log_prob(i)` returns $\log p_i$ (which is what a loss needs — see NLL below); and `sample()` draws an index according to the distribution (this is the sampling that generates text). When you build it from logits, it applies log-softmax internally for numerical stability — the same fusion trick used by `F.cross_entropy`.

## 10. Cross-entropy and KL divergence — the six-pass concept

### 10.1 Intuition

Now there are **two** distributions: the true one $p$ and the model's guess $q$. **Cross-entropy** $H(p, q)$ is the average surprise you actually incur when reality follows $p$ but you *encoded your expectations using* $q$. If $q$ matches $p$, cross-entropy equals entropy — you paid the minimum. If $q$ is wrong, you pay extra. **KL divergence** is exactly that extra cost — the penalty for using the wrong distribution.

### 10.2 The mathematics

$$
H(p, q) = -\sum_i p_i \log q_i, \qquad D_{\text{KL}}(p \,\|\, q) = \sum_i p_i \log \frac{p_i}{q_i}.
$$

- $p_i$ — true probability of outcome $i$.
- $q_i$ — model's predicted probability of outcome $i$.
- Cross-entropy weights the model's surprise $-\log q_i$ by the *true* probabilities $p_i$.
- KL weights the log-ratio $\log(p_i/q_i)$ by $p_i$.

The three quantities are bound by one identity:

$$
H(p, q) = H(p) + D_{\text{KL}}(p \,\|\, q).
$$

Because $H(p)$ does not depend on the model, minimizing cross-entropy over the model is **identical** to minimizing KL divergence. And KL has a crucial property: $D_{\text{KL}}(p \,\|\, q) \ge 0$, with equality only when $q = p$ (this is Gibbs' inequality). So cross-entropy $\ge$ entropy, always — the entropy floor from section 9.4.

### 10.3 A tiny worked example

True $p = [0.5, 0.5]$, model $q = [0.8, 0.2]$.

Cross-entropy:

$$
H(p, q) = -\big(0.5 \ln 0.8 + 0.5 \ln 0.2\big) = -\big(0.5 \cdot (-0.22314) + 0.5 \cdot (-1.60944)\big) = 0.9163.
$$

Entropy of $p$ (from section 9.3): $H(p) = \ln 2 = 0.6931$.

KL divergence:

$$
D_{\text{KL}}(p \,\|\, q) = 0.5 \ln\frac{0.5}{0.8} + 0.5 \ln\frac{0.5}{0.2} = 0.5\ln(0.625) + 0.5\ln(2.5),
$$
$$
= 0.5 \cdot (-0.47000) + 0.5 \cdot (0.91629) = -0.23500 + 0.45815 = 0.2231.
$$

Verify the identity: $H(p) + D_{\text{KL}} = 0.6931 + 0.2231 = 0.9163 = H(p, q)$ ✓. And $D_{\text{KL}} = 0.2231 > 0$, as guaranteed. The model paid $0.2231$ nats extra for being wrong.

### 10.4 Why ML needs it — the language-model connection

Here is the whole point of this lesson. In language modeling the "true" distribution $p$ for a training position is a **one-hot** vector: probability $1$ on the token that actually came next, $0$ on all others. The model's $q$ is its softmax output over the vocabulary. Cross-entropy against a one-hot $p$ collapses to a single term:

$$
H(p, q) = -\sum_i p_i \log q_i = -\log q_c,
$$

where $c$ is the true next token. That is **exactly** the cross-entropy loss from [lesson 24](../module-08/lesson-02.md) — $-\log$ of the probability the model assigned to the correct token. Because $p$ is one-hot, its entropy $H(p) = 0$, so here cross-entropy *equals* KL divergence: minimizing the loss is directly driving the model's distribution toward the truth.

### 10.5 In PyTorch

```python
import torch
import torch.nn.functional as F

p = torch.tensor([0.5, 0.5])
q = torch.tensor([0.8, 0.2])

H_pq = -(p * q.log()).sum()          # cross-entropy
kl   = (p * (p / q).log()).sum()     # KL divergence
print(H_pq)   # tensor(0.9163)
print(kl)     # tensor(0.2231)

# built-in KL (expects log-probabilities of q, and probs of p as target)
print(F.kl_div(q.log(), p, reduction='sum'))   # tensor(0.2231)
```

### 10.6 Under the hood

`F.cross_entropy` you already know fuses log-softmax with the one-hot selection $-\log q_c$. It never builds the one-hot $p$ vector — it just indexes $\log q$ at the target class, because every other term of $-\sum_i p_i \log q_i$ is zero. `F.kl_div` similarly works in log-space to stay stable. The reason all of this lives in log-space is the same numerical-stability story as [lesson 24](../module-08/lesson-02.md): logs turn the fragile product of probabilities into a safe sum.

## 11. Maximum likelihood and negative log likelihood

### 11.1 The idea

**Maximum likelihood estimation** (MLE) chooses the parameters that make the observed data most probable. If your model assigns probability $P(\text{data} \mid \theta)$ to what you actually saw, MLE picks the $\theta$ maximizing it.

Because examples are assumed independent, the likelihood of a whole dataset is the **product** of per-example probabilities:

$$
\mathcal{L}(\theta) = \prod_{n=1}^{N} P(\text{example}_n \mid \theta).
$$

- $\theta$ — the model parameters (all the weights).
- $N$ — number of examples.
- The product runs over every example; each factor is the probability the model gave to that example.

### 11.2 From product to sum: negative log likelihood

Products of many small probabilities underflow to zero and are awkward to differentiate. Taking the log turns the product into a sum (since $\log(ab) = \log a + \log b$), and negating turns "maximize" into "minimize":

$$
\text{NLL}(\theta) = -\log \mathcal{L}(\theta) = -\sum_{n=1}^{N} \log P(\text{example}_n \mid \theta).
$$

Maximizing likelihood $\Leftrightarrow$ minimizing negative log likelihood. And $-\log P$ per example is precisely the cross-entropy loss. So the chain is complete:

$$
\text{maximize likelihood} \;=\; \text{minimize NLL} \;=\; \text{minimize cross-entropy}.
$$

### 11.3 A tiny worked example

You flip a coin $5$ times and see $3$ heads, $2$ tails. Model it as Bernoulli with unknown $\theta = P(\text{heads})$. The likelihood is

$$
\mathcal{L}(\theta) = \theta^3 (1 - \theta)^2.
$$

The log-likelihood is $\ell(\theta) = 3 \ln \theta + 2 \ln(1 - \theta)$. Set the derivative to zero:

$$
\frac{d\ell}{d\theta} = \frac{3}{\theta} - \frac{2}{1 - \theta} = 0 \;\Rightarrow\; 3(1 - \theta) = 2\theta \;\Rightarrow\; 3 = 5\theta \;\Rightarrow\; \theta = 0.6.
$$

The maximum-likelihood estimate is $\theta = 3/5 = 0.6$ — exactly the observed fraction of heads. MLE recovers the intuitive answer, and it did so by minimizing the negative log likelihood.

### 11.4 Why ML needs it — and why we minimize NLL

Training GPT-2 *is* maximum likelihood over text. The "examples" are (context, next-token) pairs; the model's probability for each is $q_c$, its softmax output on the true token. Maximizing the product of those probabilities across the whole corpus is minimizing $-\sum \log q_c$, summed over every position — which is the total cross-entropy loss. That is why the training loop minimizes NLL: it is the mathematically exact way to say "make the real text as likely as possible under the model." When you watch the loss curve fall from $\sim\!11$ (an untrained model is roughly uniform over $V \approx 50000$ tokens: $\ln 50000 \approx 10.8$ nats) toward $\sim\!3$, you are watching likelihood climb.

<div class="callout key"><p>One idea wearing four hats: <strong>maximum likelihood</strong> = minimize <strong>negative log likelihood</strong> = minimize <strong>cross-entropy</strong> = minimize <strong>KL divergence</strong> to the true distribution. Every GPT-2 training step is doing all four at once.</p></div>

## Check yourself

<details><summary>Why must a language model output a probability distribution over the vocabulary, rather than just picking one word?</summary>

Because the true next-token target is expressed as a distribution (a one-hot vector), and training minimizes cross-entropy between that target and the model's output. Cross-entropy needs a full distribution $q$ so it can read off $q_c$, the probability assigned to the correct token, and penalize $-\log q_c$. A single hard pick has no gradient to push on and no notion of "how confident"; a distribution gives a smooth, differentiable objective and lets you sample varied text at generation time.

</details>

<details><summary>The entropy of a fair coin is $\ln 2 \approx 0.6931$ nats. What is the entropy of a fair 3-sided die, and is it larger or smaller?</summary>

$H = -\sum \frac13 \ln \frac13 = \ln 3 \approx 1.0986$ nats. Larger — more equally-likely outcomes means more uncertainty. For a uniform distribution over $k$ outcomes, entropy is $\ln k$.

</details>

<details><summary>A model predicts $q = [0.7, 0.3]$ and the true next token is index 0. What is the loss in nats?</summary>

Cross-entropy against the one-hot target $[1, 0]$ is $-\log q_0 = -\ln 0.7 = 0.3567$ nats. Only the true-class probability enters.

</details>

<details><summary>Why does minimizing cross-entropy equal minimizing KL divergence during training?</summary>

Because $H(p, q) = H(p) + D_{\text{KL}}(p \,\|\, q)$, and $H(p)$ (the entropy of the fixed true distribution) does not depend on the model's parameters. So changing the model only changes the KL term; driving cross-entropy down drives KL down by the same amount. With a one-hot target, $H(p) = 0$ and the two are literally equal.

</details>

## Next

You now know that a language model emits a categorical distribution over the vocabulary and that training minimizes negative log likelihood, i.e. cross-entropy, i.e. KL to the truth. The next lesson opens up the mechanism that produces that distribution: how a hidden vector becomes logits through the LM head's weight matrix, how softmax normalizes them, and how temperature reshapes the result for sampling.

Continue to [30 · Softmax and language-modeling math](lesson-02.md).
