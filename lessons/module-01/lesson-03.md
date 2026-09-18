# 03 · Powers, roots, exponentials, logarithms

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-01.md">Lesson 01</a> (numbers, expressions) and <a href="lesson-02.md">Lesson 02</a> (functions and $f(x)$).</p>
<p><strong>You will learn:</strong> Powers and their rules, negative and fractional exponents (which are roots), the square root; the exponential function $e^x$ and why the number $e$ is special; the natural logarithm $\ln$ as the inverse of $e^x$, and the logarithm rules; and why logarithms are everywhere in ML.</p>
<p><strong>Why this matters for ML:</strong> Probabilities come out of an exponential (the softmax uses $e^x$). The loss that trains every language model is built from $\ln$ (cross-entropy is a negative log-probability). And logarithms are the standard trick for keeping tiny probabilities from underflowing to zero in floating point. You cannot read a loss function without these four tools.</p>
</div>

This lesson assembles a small toolkit. Each tool is simple on its own; the reason we gather them here is that ML's probabilities and losses are built by snapping these four together.

## 1. Powers (exponents)

A **power** is repeated multiplication. The expression $x^a$ means "multiply $x$ by itself $a$ times". Here $x$ is called the **base** and $a$ the **exponent**. Read $x^a$ as "$x$ to the power $a$" or "$x$ to the $a$".

$$
2^3 = 2 \cdot 2 \cdot 2 = 8
$$

The base is $2$, the exponent is $3$, and because there are three factors of $2$, the value is $8$. Likewise $5^2 = 5 \cdot 5 = 25$ ("five squared") and $2^4 = 16$.

### The exponent rules

These rules are not arbitrary; each one falls straight out of "repeated multiplication". They are worth knowing cold because they are used constantly to simplify ML expressions.

**Rule 1 — multiplying same-base powers adds exponents:**

$$
x^a \cdot x^b = x^{a+b}
$$

Why: $x^a$ is $a$ copies of $x$, and $x^b$ is $b$ copies; multiply them and you have $a+b$ copies. Check: $2^2 \cdot 2^3 = 4 \cdot 8 = 32$, and $2^{2+3} = 2^5 = 32$. Agrees.

**Rule 2 — a power of a power multiplies exponents:**

$$
(x^a)^b = x^{a \cdot b}
$$

Why: you have $b$ copies of "$a$ copies of $x$", so $a \cdot b$ copies in total. Check: $(2^2)^3 = 4^3 = 64$, and $2^{2 \cdot 3} = 2^6 = 64$. Agrees.

**Rule 3 — anything to the zero power is one:**

$$
x^0 = 1 \qquad (\text{for } x \neq 0)
$$

Why: Rule 1 forces it. We need $x^0 \cdot x^1 = x^{0+1} = x^1$, so $x^0$ must be the number that leaves $x^1$ unchanged when multiplied — and that number is $1$. So $2^0 = 1$, $7^0 = 1$.

**Rule 4 — negative exponents mean reciprocals (one over):**

$$
x^{-a} = \frac{1}{x^a}
$$

Why: again from Rule 1, $x^{-a} \cdot x^{a} = x^{0} = 1$, so $x^{-a}$ must be whatever you multiply $x^a$ by to get $1$ — that is $\frac{1}{x^a}$. Worked: $2^{-1} = \frac{1}{2^1} = \frac{1}{2} = 0.5$, and $2^{-3} = \frac{1}{8} = 0.125$.

<div class="callout key"><p><strong>Key idea.</strong> Positive exponent = repeated multiplication. Zero exponent = 1. Negative exponent = one-over. Multiplying same-base powers adds exponents; nesting powers multiplies them.</p></div>

## 2. Roots and fractional exponents

A **square root** of a number is the value that, squared, gives it back. Since $3 \cdot 3 = 9$, the square root of $9$ is $3$, written $\sqrt{9} = 3$. In general $\sqrt{x}$ is the (non-negative) number whose square is $x$.

The beautiful part: **roots are just fractional exponents.** Specifically,

$$
x^{1/2} = \sqrt{x}
$$

Why must this be true? Use Rule 1. If we square $x^{1/2}$ we get $x^{1/2} \cdot x^{1/2} = x^{1/2 + 1/2} = x^1 = x$. So $x^{1/2}$ is exactly the thing whose square is $x$ — the definition of the square root. Worked: $9^{1/2} = \sqrt{9} = 3$, since $3^2 = 9$.

The same logic gives cube roots and beyond: $x^{1/3}$ is the number whose *cube* is $x$, because $\big(x^{1/3}\big)^3 = x^{3/3} = x^1$. Worked: $8^{1/3} = 2$, since $2^3 = 8$. And a general fractional exponent combines a power and a root: $x^{a/b} = \big(x^{1/b}\big)^a$.

<div class="callout key"><p><strong>Key idea.</strong> A fractional exponent is a root. $x^{1/2}$ is the square root, $x^{1/3}$ the cube root, $x^{a/b}$ the $b$-th root raised to the $a$-th power. Every exponent rule from Section 1 still applies.</p></div>

## 3. The exponential function $e^x$

So far the exponent has been a fixed number. Now we let the exponent be the **variable**. Fix a base $b$ and consider the function $f(x) = b^x$ — this is an **exponential function** (the variable is up in the exponent). As $x$ grows, $b^x$ grows explosively; this is what "exponential growth" means.

Among all possible bases, one is so convenient that it is *the* default in mathematics and ML: the number

$$
e \approx 2.71828
$$

called **Euler's number**. Like $\pi$, it is an irrational real number with a never-ending decimal. The function $f(x) = e^x$ is called *the* exponential function and is also written $\exp(x)$.

Why $e$ and not, say, $2$ or $10$? The short honest answer, which we can only fully justify after calculus (Module 6): $e^x$ is the one exponential function whose *rate of change equals itself*. Its slope at every point is exactly its own height. That single property makes its derivative maximally clean ($e^x$ differentiates to $e^x$), which cascades into simpler formulas everywhere downstream — including the derivative of the softmax that produces token probabilities. For now, take $e$ as "the base nature prefers", and remember these anchor values:

$$
e^0 = 1, \qquad e^1 = e \approx 2.71828, \qquad e^2 \approx 7.389
$$

The first is just Rule 3 ($e^0 = 1$). The others are $e$ itself and $e$ squared. Note $e^x$ is **always positive** — you can never get zero or a negative out of it, no matter how negative $x$ is (a very negative $x$ gives a tiny positive number close to $0$). That "always positive" fact is exactly why softmax uses $e^x$ to turn arbitrary scores into valid probabilities.

## 4. The natural logarithm $\ln$

A logarithm answers the reverse question. Exponential asks "given the exponent, what is the value?" The logarithm asks **"given the value, what was the exponent?"** The natural logarithm, written $\ln$, is the inverse of $e^x$:

$$
\ln(y) = x \quad\text{means exactly}\quad e^x = y
$$

In words, $\ln(y)$ is "the power you must raise $e$ to in order to get $y$". Inverse functions undo each other, so $\ln(e^x) = x$ and $e^{\ln(y)} = y$. Anchor values, each read straight off the exponential anchors above:

$$
\ln(1) = 0, \qquad \ln(e) = 1, \qquad \ln(7.389) \approx 2
$$

Check them against Section 3: $\ln(1) = 0$ because $e^0 = 1$; $\ln(e) = 1$ because $e^1 = e$; $\ln(7.389) \approx 2$ because $e^2 \approx 7.389$. The logarithm just reads the exponent back out.

<div class="callout warn"><p><strong>Gotcha.</strong> $\ln$ is only defined for <em>positive</em> inputs, because $e^x$ only ever outputs positive numbers — there is no power of $e$ that yields $0$ or a negative. So $\ln(0)$ is undefined (it heads to $-\infty$), and $\ln$ of a negative number is undefined. This matters in ML: a probability that has collapsed exactly to $0$ makes $\ln(0)$ blow up, which is one source of <code>NaN</code> during training. We handle it in Module 12.</p></div>

Two notation notes. "$\log$" without a subscript sometimes means base 10 and sometimes base $e$; in ML, an unadorned $\log$ almost always means the *natural* log, i.e. base $e$, the same thing as $\ln$. We will use $\ln$ when we want to be unambiguous. Changing the base only rescales the log by a constant factor, so it never changes the shape of anything.

### The logarithm rules

Logs convert the harder operations into easier ones — multiplication becomes addition, powers become multiplication. These three rules are the reason logs appear all over ML math.

**Product rule — log of a product is the sum of logs:**

$$
\ln(ab) = \ln(a) + \ln(b)
$$

**Quotient rule — log of a quotient is the difference of logs:**

$$
\ln\!\left(\frac{a}{b}\right) = \ln(a) - \ln(b)
$$

**Power rule — log of a power brings the exponent down front:**

$$
\ln(a^b) = b \cdot \ln(a)
$$

These all inherit from the exponent rules of Section 1, because $\ln$ is just "read the exponent". For instance, the product rule mirrors Rule 1 ($x^a x^b = x^{a+b}$): multiplying values adds their exponents, so the log of the product is the sum of the logs. Quick check of the power rule: $\ln(e^2) = 2 \cdot \ln(e) = 2 \cdot 1 = 2$, matching $\ln(7.389)\approx 2$ from above.

<div class="callout key"><p><strong>Key idea.</strong> $\ln$ turns multiplication into addition and powers into multiplication. That is precisely why ML takes the log of products of probabilities: a product of many tiny numbers becomes a sum of manageable numbers.</p></div>

## 5. Why ML needs logs and exponentials

Three concrete reasons, each of which you will meet in full later:

**Probabilities are made with $e^x$.** A model outputs raw scores (called *logits*) that can be any real number, positive or negative. To turn them into probabilities — positive numbers that sum to $1$ — we exponentiate each with $e^x$ (guaranteeing positivity) and divide by the total. That is the *softmax*, the last step of every GPT forward pass (Module 10).

**The loss is a negative log-probability.** Training a language model means making the correct next token as probable as possible. The loss we minimize is $-\ln(\hat{p})$, where $\hat{p}$ is the probability the model assigned to the correct token. If the model is confident and right ($\hat{p}$ near $1$), then $\ln(\hat p)$ is near $0$ and the loss is near $0$. If it is confident and wrong ($\hat{p}$ near $0$), then $\ln(\hat p)$ heads to $-\infty$ and the loss is huge. This is *cross-entropy*, and $\ln$ is the heart of it (Module 8 and Module 10).

**Logs prevent underflow.** The probability of a whole sentence is a product of many per-token probabilities, each less than $1$. Multiply a thousand numbers like $0.02$ together and the result is so tiny that a float rounds it to exactly $0$ — *underflow*, and all information is lost. Take logs first, and by the product rule the giant product becomes a sum of logs: $\ln(p_1 p_2 \cdots) = \ln p_1 + \ln p_2 + \cdots$. Sums of moderate negative numbers stay perfectly representable. This is why ML code works in "log space" almost everywhere (Module 12).

## 6. In PyTorch

Every operation in this lesson is a one-liner in PyTorch. The expected output is shown as a comment, and each matches a value we computed by hand.

```python
import torch

# Powers: the ** operator, same as math x^a.
print(torch.tensor(2.) ** 3)     # tensor(8.)      <- 2^3 = 8
print(torch.tensor(2.) ** -1)    # tensor(0.5000)  <- 2^(-1) = 1/2

# Square root two equivalent ways: torch.sqrt, or the exponent 1/2.
print(torch.sqrt(torch.tensor(9.)))   # tensor(3.)  <- sqrt(9) = 3
print(torch.tensor(9.) ** 0.5)        # tensor(3.)  <- 9^(1/2) = 3

# Exponential e^x.
print(torch.exp(torch.tensor(0.)))    # tensor(1.)       <- e^0 = 1
print(torch.exp(torch.tensor(1.)))    # tensor(2.7183)   <- e^1 = e

# Natural logarithm ln. torch.log IS the natural log (base e), not base 10.
print(torch.log(torch.tensor(1.)))       # tensor(0.)      <- ln(1) = 0
print(torch.log(torch.exp(torch.tensor(1.))))  # tensor(1.) <- ln(e) = 1
print(torch.log(torch.tensor(7.389)))    # tensor(2.0000)  <- ln(7.389) ~ 2
```

<div class="callout pt"><p><strong>PyTorch note.</strong> In PyTorch (and NumPy, and most ML code) <code>log</code> means the <em>natural</em> logarithm, base $e$ — not base 10. If you ever need base 10 or base 2 there are separate functions <code>torch.log10</code> and <code>torch.log2</code>, but you will reach for plain <code>torch.log</code> almost every time, because it is the inverse of <code>torch.exp</code>.</p></div>

## 7. What PyTorch is doing under the hood

Two things worth knowing early.

First, these functions are **elementwise**. Hand `torch.exp` a tensor of a hundred numbers and it applies $e^x$ to each independently, returning a hundred results, in one call. That is how softmax exponentiates an entire vocabulary of logits at once.

Second, each of these operations is **differentiable**, and PyTorch already knows every derivative from this lesson's rules — that $e^x$ differentiates to $e^x$, that $\ln(x)$ differentiates to $\frac{1}{x}$, that $x^a$ differentiates to $a\,x^{a-1}$. So when a loss is built out of `torch.exp` and `torch.log`, PyTorch can automatically compute how the loss responds to each parameter. In other words, the clean derivatives that made $e$ "the base nature prefers" are exactly what let autograd train the network efficiently. We prove those derivatives in Module 6 and watch autograd use them in Module 8.

## Check yourself

1. Evaluate $2^{-3}$ and $16^{1/2}$.

<details><summary>Answer</summary>

$2^{-3} = \frac{1}{2^3} = \frac{1}{8} = 0.125$. And $16^{1/2} = \sqrt{16} = 4$, since $4^2 = 16$.

</details>

2. Without a calculator, what is $\ln(1)$ and why? What is $e^0$?

<details><summary>Answer</summary>

$\ln(1) = 0$, because $\ln(y)$ asks "what power of $e$ gives $y$?", and $e^0 = 1$. So the power that yields $1$ is $0$. And $e^0 = 1$ by the zero-exponent rule.

</details>

3. Use a log rule to simplify $\ln(a^3)$.

<details><summary>Answer</summary>

By the power rule, $\ln(a^3) = 3\ln(a)$ — the exponent comes down to the front as a multiplier.

</details>

4. A model assigns probability $\hat{p} = 0.9$ to the correct token, versus $\hat{p} = 0.1$ for a different example. The loss is $-\ln(\hat p)$. Which loss is larger, and does that match "we want high probability on the right answer"?

<details><summary>Answer</summary>

$-\ln(0.9) \approx 0.105$ (small), while $-\ln(0.1) \approx 2.303$ (much larger). The low-probability case incurs the larger loss, so minimizing $-\ln(\hat p)$ pushes the model toward assigning high probability to the correct token. Exactly what we want.

</details>

## Next

You now have exponentials and logs — the machinery behind probabilities and loss. Next we pick up the compact notation for *adding up* and *multiplying together* many terms (the $\sum$ and $\prod$ symbols), plus inequalities and the first whisper of calculus: average rate of change.

[04 · Summation, products, inequalities, rates →](lesson-04.md)
