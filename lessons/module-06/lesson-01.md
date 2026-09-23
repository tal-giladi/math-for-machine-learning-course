# 15 · Limits, slope, and the derivative

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-01/lesson-02">02 · Functions and the meaning of f(x)</a>, <a href="#/lessons/module-01/lesson-04">04 · Summation, products, inequalities, rates</a> (average rate of change), and <a href="#/lessons/module-05/lesson-01">functions and composition</a>. You need to be comfortable reading $f(x)$ as "feed in $x$, get out a number."</p>
<p><strong>You will learn:</strong> what a limit is (informally), what continuity means, the slope of a line as rise over run, the average rate of change of a function over an interval, and then the star of the module — the <strong>derivative</strong>, defined as the limit of that average rate as the interval shrinks to zero. You will derive $f'(x)=2x$ for $f(x)=x^2$ by hand and verify it numerically.</p>
<p><strong>Why this matters for ML:</strong> training a neural network <em>is</em> computing derivatives of a loss with respect to millions of parameters and nudging each one downhill. Every <code>.backward()</code> call in PyTorch is evaluating derivatives. This lesson is where that whole machine starts.</p>
</div>

## 1. A limit, without the heavy machinery

Here is the one idea you need: a **limit** is the value a function *approaches* as its input approaches some target, whether or not the function ever actually reaches it.

Think of it the way a programmer thinks about a loop that keeps halving a distance. Consider the function

$$
g(x) = \frac{x^2 - 1}{x - 1}.
$$

At exactly $x = 1$ this is $\frac{0}{0}$ — undefined, a division-by-zero your code would throw on. But ask a different question: *as $x$ gets closer and closer to $1$, what number does $g(x)$ head toward?* Factor the top: $x^2 - 1 = (x-1)(x+1)$, so for every $x \neq 1$,

$$
g(x) = \frac{(x-1)(x+1)}{x - 1} = x + 1.
$$

Feed in values creeping toward $1$: at $x = 0.9$, $g = 1.9$; at $x = 0.99$, $g = 1.99$; at $x = 1.001$, $g = 2.001$. The outputs pile up around $2$. We write

$$
\lim_{x \to 1} g(x) = 2,
$$

read "the limit of $g(x)$ as $x$ approaches $1$ is $2$." The symbol $\lim$ means "the value being approached," and $x \to 1$ (read "$x$ tends to $1$") means we let $x$ slide toward $1$ without demanding it arrive. The function is undefined *at* $1$, yet the limit exists — that gap between "value at a point" and "value approached near a point" is exactly the subtlety we will exploit to define the derivative.

- $\lim$ — the limiting (approached) value.
- $x \to a$ — "$x$ approaches $a$"; $a$ is the target input.
- The whole expression is a single number (here, $2$), not a function.

That is all the limit machinery this module needs. We will not do epsilon-delta proofs; we only ever ask "what number is this expression heading toward as the step size shrinks?"

## 2. Continuity, informally

A function is **continuous** if you can draw its graph without lifting your pen — no sudden jumps, holes, or teleports. Formally it means the limit as you approach any point equals the function's actual value there, but the pen picture is enough for us.

Why care? Because the derivative we are about to define relies on zooming in on a smooth curve. Polynomials ($x^2$, $3x+1$), the exponential $e^x$, the sigmoid, and the smooth activation functions inside a Transformer are all continuous, so the machinery applies cleanly. A function with a jump (a step) has no well-defined slope at the jump — a fact that will matter when we discuss why ML prefers smooth activations.

## 3. Slope of a line: rise over run

Start with the simplest possible curve, a straight line. Its **slope** measures steepness: how much the output rises for a given run along the input axis.

$$
\text{slope} = \frac{\text{rise}}{\text{run}} = \frac{\Delta y}{\Delta x} = \frac{y_2 - y_1}{x_2 - x_1}
$$

- $\Delta y$ (read "delta y") — the change in output between two points; $\Delta$ means "change in."
- $\Delta x$ — the change in input.
- $(x_1, y_1)$ and $(x_2, y_2)$ — two points on the line.

Take the line $y = 3x + 1$ and the points at $x_1 = 2$ (so $y_1 = 7$) and $x_2 = 5$ (so $y_2 = 16$):

$$
\text{slope} = \frac{16 - 7}{5 - 2} = \frac{9}{3} = 3.
$$

The slope is $3$, which is exactly the coefficient of $x$ — for a straight line the slope is the same everywhere, and it means "each time the input grows by 1, the output grows by 3." That constancy is special to lines. A curve like $x^2$ is steeper in some places than others, and that is the problem the derivative solves.

## 4. Average rate of change over an interval

For a curved function $f$, we cannot read off a single slope. But we *can* pick two input points, $x$ and $x + h$, and measure the slope of the straight line connecting them — the **secant line**. The horizontal gap is $h$ (a small step), and the vertical gap is $f(x + h) - f(x)$. Their ratio is the **average rate of change** over the interval $[x, x+h]$:

$$
\frac{f(x + h) - f(x)}{h}
$$

- $x$ — where we start.
- $h$ — the step size (the width of the interval); it can be tiny.
- $f(x + h) - f(x)$ — how much the output changed over that step (the rise).
- The whole ratio — the rise divided by the run $h$: the average steepness across the interval.

This expression has a name you will see everywhere: the **difference quotient**. It answers "on average, how fast did the output change per unit of input, between $x$ and $x + h$?" It is exactly the rise-over-run of section 3, applied to two points on a curve instead of a line.

## 5. The derivative: shrink the interval to zero

The average rate over $[x, x+h]$ still depends on how wide we made the interval. Make $h$ smaller and the secant line pivots until it kisses the curve at a single point — becoming the **tangent line**, whose slope is the *instantaneous* rate of change right at $x$. "Instantaneous" is the limit of "average" as the interval collapses. That limit is the **derivative**:

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

- $f'(x)$ (read "f prime of x") — the derivative of $f$ at $x$; itself a function of $x$.
- $\lim_{h \to 0}$ — take $h$ toward $0$ (but never plug in $0$, which would divide by zero, the section-1 trick).
- The result is the slope of the curve exactly at $x$.

Three names for the same number, all worth carrying in your head:

- **Instantaneous rate of change** — how fast the output is changing right now.
- **Local slope** — the steepness of the curve at that single point.
- **Local sensitivity** — how much the output changes per tiny change in the input. This is the phrasing to keep for ML: the derivative tells you *how sensitive the output is to a small wiggle in the input.*

<div class="callout key"><p>The derivative $f'(x)$ is the limit of the difference quotient as the step $h \to 0$. Read it as <strong>local sensitivity</strong>: how much the output changes per tiny change in the input. That single quantity is what training uses to decide which way to nudge each parameter.</p></div>

## 6. Worked example: the derivative of $f(x) = x^2$

Let us actually compute one, by hand, from the definition. Take $f(x) = x^2$.

**Step 1 — write the difference quotient.** Substitute $f(x+h) = (x+h)^2$ and $f(x) = x^2$:

$$
\frac{f(x+h) - f(x)}{h} = \frac{(x+h)^2 - x^2}{h}
$$

**Step 2 — expand the square.** Recall $(x+h)^2 = x^2 + 2xh + h^2$:

$$
\frac{x^2 + 2xh + h^2 - x^2}{h} = \frac{2xh + h^2}{h}
$$

The $x^2$ terms cancel — that cancellation is what makes the limit finite.

**Step 3 — divide through by $h$.** Every remaining term on top has a factor of $h$, so:

$$
\frac{2xh + h^2}{h} = 2x + h
$$

This is the average rate of change over $[x, x+h]$: a clean $2x + h$.

**Step 4 — take the limit $h \to 0$.** As $h$ shrinks to nothing, the $h$ term vanishes and we are left with:

$$
f'(x) = \lim_{h \to 0} (2x + h) = 2x
$$

So the derivative of $x^2$ is $2x$. At $x = 3$:

$$
f'(3) = 2 \cdot 3 = 6.
$$

The curve $y = x^2$ has slope $6$ at the point $x = 3$. In sensitivity language: near $x = 3$, nudging the input up by a tiny $\varepsilon$ pushes the output up by about $6\varepsilon$.

### Verify it numerically

The definition says $2x + h$ should be very close to $6$ when $x = 3$ and $h$ is small. Take $h = 0.001$:

$$
\frac{f(3.001) - f(3)}{0.001} = \frac{3.001^2 - 3^2}{0.001} = \frac{9.006001 - 9}{0.001} = \frac{0.006001}{0.001} = 6.001.
$$

We got $6.001$ — which is exactly $2x + h = 6 + 0.001$, as the algebra promised. Shrink $h$ further and it homes in on $6$. The hand-derivative and the numerical estimate agree.

## 7. Notation: $f'(x)$ and $\frac{df}{dx}$

Two notations for the derivative appear throughout ML code and papers, and they mean the same thing:

- **Lagrange notation** $f'(x)$ — compact, good when there is one obvious input. Read "f prime."
- **Leibniz notation** $\dfrac{df}{dx}$ — reads "the derivative of $f$ with respect to $x$." The $d$ is an infinitesimal (vanishingly small) change; $\frac{df}{dx}$ literally suggests "tiny change in $f$ divided by tiny change in $x$," the difference quotient in the limit. This form is indispensable once several variables are in play, because it names *which* variable you are differentiating with respect to — the seed of partial derivatives in [lesson 4](lessons/module-06/lesson-04.md).

For $f(x) = x^2$ we write either $f'(x) = 2x$ or $\dfrac{df}{dx} = 2x$; in Leibniz style people also write $\dfrac{d}{dx}(x^2) = 2x$, treating $\frac{d}{dx}$ as the "take the derivative" operator.

## 8. Why ML needs this

Training a neural network means minimizing a **loss** $L$ — a single number measuring how wrong the model's predictions are. Every weight and bias in the model is a knob $\theta$ (theta) that $L$ depends on. The question training must answer, billions of times, is: *if I nudge this one knob a hair, does the loss go up or down, and how fast?* That is precisely the derivative $\dfrac{\partial L}{\partial \theta}$ — the local sensitivity of the loss to that parameter.

Once you know the sensitivity of $L$ to every parameter, you nudge each parameter in the direction that lowers $L$. That is gradient descent, and it is the entire training loop. So the humble limit-of-a-difference-quotient you just computed is not a warm-up — it is the mathematical event that happens every training step, for every one of GPT-2's 124 million parameters. The rest of this module and the next build the tools to compute these derivatives for the functions real networks use.

## 9. In PyTorch

PyTorch computes derivatives for you. Watch it reproduce our hand result $f'(3) = 6$:

```python
import torch

x = torch.tensor(3., requires_grad=True)  # a scalar input we want the derivative w.r.t.
y = x ** 2                                 # forward pass: y = x^2 = 9.0
y.backward()                               # compute dy/dx and store it
print(x.grad)                              # tensor(6.)
```

The printed `tensor(6.)` is exactly $f'(3) = 2 \cdot 3 = 6$ — the number we derived by hand and confirmed numerically. No difference quotient, no small $h$: PyTorch knows the derivative of squaring is "multiply by $2x$" and evaluates it at $x = 3$.

<div class="callout pt"><p><code>requires_grad=True</code> marks a tensor as something we want derivatives with respect to. <code>y.backward()</code> fills in the derivative. You then read it from <code>x.grad</code>. For a scalar output, <code>x.grad</code> holds $\frac{dy}{dx}$ evaluated at the current value of <code>x</code>.</p></div>

## 10. What PyTorch is doing under the hood

`y.backward()` is not estimating a slope with a tiny $h$. It is applying the exact derivative rules — the same $\frac{d}{dx}(x^2) = 2x$ you derived — automatically. Here is the mental model:

1. **Forward pass records structure.** When you wrote `y = x ** 2`, PyTorch did two things: it computed the value $9.0$, and it recorded that `y` was produced by *squaring* `x`. That recorded operation is stored on `y` as its `grad_fn` (you can print `y.grad_fn` and see `<PowBackward0>`). PyTorch is quietly building a small graph: `x → (square) → y`.

2. **Each operation knows its local derivative.** The squaring node knows that the derivative of "square the input" is "multiply by $2 \times \text{input}$." At the recorded input value $x = 3$, that local derivative is $2 \cdot 3 = 6$.

3. **Backward applies it.** Calling `y.backward()` walks the recorded graph from `y` back to `x`, evaluating each node's local derivative and depositing the result in the input's `.grad` field. With a single node, the result is just the local derivative $6$, so `x.grad` becomes `tensor(6.)`.

Only tensors you marked with `requires_grad=True` (called **leaf** tensors — the graph's inputs) receive a `.grad`. This is the whole essence of PyTorch's autograd, and it is nothing more than "record the operations on the way forward, then apply their known derivatives on the way back." In [lesson 2](lessons/module-06/lesson-02.md) we collect the derivative rules autograd uses, and the **chain rule** that lets it stitch many operations together — which, in the next module, becomes backpropagation itself.

## Check yourself

<details><summary>Using the definition, what is the average rate of change of $f(x)=x^2$ over $[4, 4+h]$, and what does it become as $h \to 0$?</summary>

$\frac{(4+h)^2 - 4^2}{h} = \frac{16 + 8h + h^2 - 16}{h} = \frac{8h + h^2}{h} = 8 + h$. As $h \to 0$ this tends to $8$, so $f'(4) = 8$ — matching $2x = 2\cdot 4$.

</details>

<details><summary>Why can we cancel the $h$ in the denominator even though we are heading toward $h = 0$?</summary>

Because we never set $h = 0$; we only let it approach $0$. For every nonzero $h$ the cancellation is ordinary algebra, giving $2x + h$. The limit then reads off the value that clean expression approaches — $2x$ — without ever dividing by zero.

</details>

<details><summary>In one sentence, what does $f'(3) = 6$ tell you about the output near $x = 3$?</summary>

Near $x = 3$, the output changes about $6$ times as fast as the input — a tiny input bump of $\varepsilon$ raises the output by roughly $6\varepsilon$.

</details>

<details><summary>Which tensor gets a <code>.grad</code> after <code>y.backward()</code>, and why?</summary>

The leaf tensor `x`, because it was created with `requires_grad=True`. `.grad` is filled only for the leaf inputs we asked to track; `y` is an intermediate result of the computation, not a leaf we are optimizing.

</details>

## Next

You can now compute a derivative from its definition and you have seen PyTorch produce the same number. Deriving every derivative from scratch would be exhausting, so next we collect the **rules** — power, product, quotient, and above all the **chain rule** — that let you differentiate any composition quickly. The chain rule is the single most important idea for backpropagation.

Continue to [16 · Derivative rules and the chain rule](lessons/module-06/lesson-02.md).
