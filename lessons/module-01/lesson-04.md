# 04 · Summation, products, inequalities, rates

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-01/lesson-01">Lesson 01</a> (variables, expressions), <a href="#/lessons/module-01/lesson-02">Lesson 02</a> (functions), <a href="#/lessons/module-01/lesson-03">Lesson 03</a> (powers).</p>
<p><strong>You will learn:</strong> Summation notation $\sum$ read index by index, product notation $\prod$, absolute value $|x|$, the inequality symbols and intervals, fractions and ratios, and the average rate of change $\frac{\Delta y}{\Delta x}$ — the direct runway to the derivative.</p>
<p><strong>Why this matters for ML:</strong> A loss is an <em>average over a batch</em>, which is a $\sum$ divided by a count. A likelihood is a <em>product</em> of probabilities, a $\prod$. Distances and errors use absolute value. And the average rate of change is the idea the derivative is built from — the single most important concept in the entire training story.</p>
</div>

This lesson is about compact notation for "do this to many things and combine them", plus a first bridge toward calculus. None of it is hard; the goal is that $\sum$ and $\prod$ stop looking like hieroglyphics and start reading like `for` loops.

## 1. Summation notation $\sum$

When you want to add up a list of terms, writing $a_1 + a_2 + a_3 + \cdots + a_{100}$ is unbearable. Mathematics compresses it with the capital Greek letter sigma, $\sum$, which simply means "add up". The full notation looks intimidating but is really a `for` loop:

$$
\sum_{i=1}^{n} a_i
$$

Read it piece by piece — this is the whole skill:

- $\sum$ — "sum up / add together". The instruction to accumulate.
- $i$ — the **index**, a counter variable. It plays the exact role of the loop variable `i` in `for (int i = ...)`. It takes integer values one at a time.
- $i = 1$ underneath — the **starting value** of the index. Begin the counter at $1$.
- $n$ on top — the **ending value**. Stop when the counter reaches $n$ (inclusive).
- $a_i$ — the **body**: the term to compute for each value of $i$ and add into the running total. The subscript $i$ means "the $i$-th item" (subscripts get a full treatment in Lesson 05).

Altogether: "for $i$ running from $1$ to $n$, compute $a_i$ and add them all up." In C# it is exactly:

```
double total = 0;
for (int i = 1; i <= n; i++) { total += a[i]; }
```

### A worked sum

Let the list be $a_1 = 2$, $a_2 = 5$, $a_3 = 1$ (so $n = 3$). Then

$$
\sum_{i=1}^{3} a_i = a_1 + a_2 + a_3 = 2 + 5 + 1 = 8
$$

Watch the index march: at $i=1$ we grab $a_1 = 2$; at $i=2$ we grab $a_2 = 5$; at $i=3$ we grab $a_3 = 1$; then we stop (because $i$ reached the top value $3$) and report the total $8$.

The body can be any expression in $i$, not just "look up the $i$-th item". For example, summing the first four whole numbers:

$$
\sum_{i=1}^{4} i = 1 + 2 + 3 + 4 = 10
$$

Here the body *is* $i$ itself, so we add the index values directly. And summing their squares:

$$
\sum_{i=1}^{3} i^2 = 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
$$

<div class="callout key"><p><strong>Key idea.</strong> $\sum_{i=1}^{n} (\text{body})$ is a <code>for</code> loop: start the index at the bottom number, stop at the top number, evaluate the body each time, add everything up. The index name ($i$, $j$, $k$) is arbitrary.</p></div>

### The average, written with $\sum$

An **average** (mean) of $n$ numbers is their sum divided by how many there are:

$$
\text{mean} = \frac{1}{n} \sum_{i=1}^{n} a_i
$$

For our list $2, 5, 1$: the mean is $\frac{1}{3}(2 + 5 + 1) = \frac{8}{3} \approx 2.67$. Hold this formula — it is *literally* the shape of a training loss.

## 2. Product notation $\prod$

Everything about $\sum$ carries over to multiplication, using the capital Greek letter pi, $\prod$, which means "multiply together":

$$
\prod_{i=1}^{n} a_i = a_1 \cdot a_2 \cdots a_n
$$

Same index, same bounds, same body — the only change is that we *multiply* the terms instead of adding them. For $a_1 = 2$, $a_2 = 5$, $a_3 = 1$:

$$
\prod_{i=1}^{3} a_i = 2 \cdot 5 \cdot 1 = 10
$$

Where does a product of many terms show up? In a **likelihood**. If a model assigns probability $p_i$ to each observed token independently, the probability of the whole sequence is the product $\prod_i p_i$. That product of many numbers below $1$ gets tiny fast — which is exactly why, per Lesson 03, we take its logarithm and turn the $\prod$ into a $\sum$:

$$
\ln\!\left(\prod_{i=1}^{n} p_i\right) = \sum_{i=1}^{n} \ln(p_i)
$$

<div class="callout key"><p><strong>Key idea.</strong> $\sum$ adds a list, $\prod$ multiplies a list. The log rule from Lesson 03 converts a $\prod$ of probabilities into a $\sum$ of log-probabilities — turning an underflow-prone product into a stable sum. This connection is the mathematical backbone of the training loss.</p></div>

## 3. Absolute value $|x|$

The **absolute value** of a number, written $|x|$ with vertical bars, is its distance from zero — always non-negative. Formally, it strips off any minus sign:

$$
|3| = 3, \qquad |-3| = 3, \qquad |0| = 0
$$

Think of it as "how far from zero, ignoring direction". It shows up whenever we care about the *size* of an error regardless of sign: if a prediction is off by $-2$ or by $+2$, both are errors of size $2$, and $|{-2}| = |2| = 2$ captures that. The *mean absolute error* loss (Module 8) is built from exactly this.

<div class="callout warn"><p><strong>Gotcha.</strong> The same vertical-bar symbol, when wrapped around a <em>vector</em> and doubled — $\|\mathbf{x}\|$ — means the vector's length, not "absolute value of each entry". Single bars around a number = absolute value; double bars around a vector = norm. We preview the norm in Lesson 05 and study it in Module 2.</p></div>

## 4. Inequalities and intervals

An **equation** claims two things are equal. An **inequality** claims one is bigger or smaller. Four symbols cover it:

- $a < b$ — "$a$ is strictly less than $b$" (e.g. $2 < 5$).
- $a \le b$ — "$a$ is less than or equal to $b$" (true if $a$ equals $b$ too).
- $a > b$ — "$a$ is strictly greater than $b$".
- $a \ge b$ — "$a$ is greater than or equal to $b$".

A useful pattern is the **chained** inequality: $0 \le p \le 1$ means "$p$ is between $0$ and $1$, inclusive on both ends" — precisely the range every probability lives in.

A set of numbers described by inequalities is an **interval**. Two notations:

- **Bracket notation:** $[0, 1]$ means all reals $x$ with $0 \le x \le 1$ — the square brackets mean "endpoint included" ("closed"). $(0, 1)$ with round brackets means $0 < x < 1$ — endpoints excluded ("open"). You can mix: $[0, 1)$ includes $0$ but not $1$.
- Probabilities are usually described as lying in $[0, 1]$; a strictly positive number lives in $(0, \infty)$, where $\infty$ (infinity) is always paired with a round bracket because you can never "reach" it.

<div class="callout key"><p><strong>Key idea.</strong> $\le$ and $\ge$ include the boundary; $<$ and $>$ do not. Square bracket = included endpoint, round bracket = excluded. A probability lives in the closed interval $[0,1]$.</p></div>

## 5. Fractions and ratios

A **fraction** $\frac{a}{b}$ means $a$ divided by $b$; the top is the **numerator**, the bottom the **denominator**. A **ratio** is the same object viewed as a comparison — "$a$ per $b$", how much of one quantity there is for each unit of another. Speed is a ratio (miles per hour); a probability is a ratio (favorable outcomes per total outcomes).

Two facts we lean on constantly:

- **Dividing by a fraction flips it:** $\dfrac{a}{\;b/c\;} = a \cdot \dfrac{c}{b}$. This is why "divide by $b/c$" and "multiply by $c/b$" are interchangeable.
- **A fraction is undefined when the denominator is $0$.** Division by zero has no value, which is why the softmax carefully divides by a sum that is guaranteed positive, and why "stability" terms called *epsilon* get added to denominators (Module 12) to keep them away from zero.

## 6. Average rate of change — the runway to derivatives

Here is the most important idea in the lesson, because the entire second half of the course grows out of it.

Suppose a quantity $y$ depends on a quantity $x$ through some function. As $x$ changes, $y$ changes. The **average rate of change** measures *how much $y$ moves per unit that $x$ moves*, between two points. We use the Greek capital delta, $\Delta$ (read "delta"), to mean "the change in":

$$
\text{average rate of change} = \frac{\Delta y}{\Delta x} = \frac{y_2 - y_1}{x_2 - x_1}
$$

- $\Delta x = x_2 - x_1$ is how far the input moved.
- $\Delta y = y_2 - y_1$ is how far the output moved as a result.
- Their ratio is "output change per unit of input change" — the *slope* of the straight line connecting the two points $(x_1, y_1)$ and $(x_2, y_2)$.

### A worked rate

Take the function $f(x) = x^2$ from Lesson 02, and measure its average rate of change between $x_1 = 1$ and $x_2 = 3$. First the outputs:

$$
y_1 = f(1) = 1^2 = 1, \qquad y_2 = f(3) = 3^2 = 9
$$

Now the changes and their ratio:

$$
\Delta x = 3 - 1 = 2, \qquad \Delta y = 9 - 1 = 8, \qquad \frac{\Delta y}{\Delta x} = \frac{8}{2} = 4
$$

So between $x = 1$ and $x = 3$, the function rises on average $4$ units of output per $1$ unit of input. Geometrically, $4$ is the slope of the straight line joining the points $(1, 1)$ and $(3, 9)$ on the curve.

### Why this is the runway to the derivative

The average rate of change uses two points that are some distance apart. The **derivative**, which we build in Module 6, asks: what happens to $\frac{\Delta y}{\Delta x}$ as the two points slide *infinitely close together*? The answer is the *instantaneous* rate of change — the slope of the curve at a single point, i.e. its sensitivity to a tiny nudge in the input. That "sensitivity of output to a tiny nudge in a parameter" is precisely what training needs to know in order to improve a model. So this humble ratio $\frac{\Delta y}{\Delta x}$ is the seed of gradients, backpropagation, and gradient descent.

<div class="callout key"><p><strong>Key idea.</strong> $\frac{\Delta y}{\Delta x}$ is the slope between two points — output change per unit of input change. Shrink the gap between the points to zero and you get the derivative: sensitivity at a single point. Everything about training is downstream of this.</p></div>

## 7. In PyTorch

The combining operations are direct calls; the batch loss ties them together.

```python
import torch

a = torch.tensor([2., 5., 1.])

print(torch.sum(a))    # tensor(8.)   <- 2 + 5 + 1, the worked sum
print(torch.prod(a))   # tensor(10.)  <- 2 * 5 * 1, the worked product
print(a.mean())        # tensor(2.6667) <- (2+5+1)/3, the average

print(torch.abs(torch.tensor([-3., 2., -0.5])))
# tensor([3.0000, 2.0000, 0.5000])   <- |-3|, |2|, |-0.5|
```

Now the pattern you will see in every training script: a **mean squared error** loss. It is exactly "average of the squared differences" — a $\sum$ of squares, divided by the count, i.e. a `.mean()`:

```python
import torch

preds   = torch.tensor([2., 3.])   # what the model predicted
targets = torch.tensor([1., 5.])   # the correct answers

loss = (preds - targets).pow(2).mean()
print(loss)   # tensor(2.5000)
```

By hand, to confirm the number: the differences are $2 - 1 = 1$ and $3 - 5 = -2$; squaring gives $1^2 = 1$ and $(-2)^2 = 4$; the mean of $1$ and $4$ is $\frac{1 + 4}{2} = 2.5$. That matches `tensor(2.5000)`.

<div class="callout pt"><p><strong>PyTorch note.</strong> Read <code>(preds - targets).pow(2).mean()</code> as one mathematical expression: $\frac{1}{n}\sum_{i=1}^{n}(\text{pred}_i - \text{target}_i)^2$. The subtraction and squaring happen elementwise (per item), and <code>.mean()</code> is the $\frac{1}{n}\sum$ that collapses the whole batch into one number. Squaring (rather than absolute value) makes big errors count disproportionately and keeps the function smooth for the derivatives we will need.</p></div>

## 8. What PyTorch is doing under the hood

When you write `(preds - targets).pow(2).mean()`, PyTorch does not just compute the number $2.5$. If `preds` came from a model with trainable parameters, PyTorch also records the chain of operations — subtract, square, average — as a graph, so it can later compute how the loss changes when each parameter changes. That "how the loss changes when a parameter changes" is a rate of change, $\frac{\Delta L}{\Delta \theta}$, taken in the limit — the derivative from Section 6, applied to the loss. The `.mean()` at the end is what makes the loss a single number, which matters because you can only ask "which way is downhill" about one number, not a whole list. We will see all of this made mechanical in Modules 6 through 8.

## Check yourself

1. Compute $\sum_{i=1}^{4} 2i$ (the body is $2i$).

<details><summary>Answer</summary>

Run the index $1$ to $4$: $2\cdot1 + 2\cdot2 + 2\cdot3 + 2\cdot4 = 2 + 4 + 6 + 8 = 20$.

</details>

2. What is $|-7|$, and why might we prefer $|x|$ over $x$ when measuring an error?

<details><summary>Answer</summary>

$|-7| = 7$. We often want the <em>size</em> of an error regardless of whether the prediction was too high or too low; absolute value discards the sign so that $-7$ and $+7$ both count as an error of size $7$.

</details>

3. Describe the interval $[0, 1)$ in words. Is $1$ included?

<details><summary>Answer</summary>

All real numbers $x$ with $0 \le x < 1$: from $0$ included up to $1$ excluded. $1$ is **not** included (round bracket), but $0$ is (square bracket).

</details>

4. For $f(x) = x^2$, find the average rate of change between $x = 2$ and $x = 4$.

<details><summary>Answer</summary>

$y_1 = 2^2 = 4$, $y_2 = 4^2 = 16$. So $\Delta y = 16 - 4 = 12$, $\Delta x = 4 - 2 = 2$, and $\frac{\Delta y}{\Delta x} = \frac{12}{2} = 6$.

</details>

## Next

You have now met most of the operator symbols. Before we leave Part 1, we do a guided tour of the more exotic symbols the course will use later — $\partial$, $\nabla$, $\|x\|$, transpose, inverse, and friends — so that when they appear in Module 6 and beyond, they are familiar faces rather than strangers.

[05 · The notation zoo: ∂ ∇ ‖x‖ indices transpose →](lessons/module-01/lesson-05.md)
