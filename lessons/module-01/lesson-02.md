# 02 · Functions and the meaning of f(x)

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-01.md">Lesson 01</a> — numbers, variables, expressions, substitution, and evaluation.</p>
<p><strong>You will learn:</strong> Exactly what $f$, $x$, and $f(x)$ each mean; what a function is as a mapping from inputs to outputs; domain and range; functions of several inputs like $f(x,y)$; and the precise sense in which "machine learning is learning a function".</p>
<p><strong>Why this matters for ML:</strong> The one-sentence definition of machine learning is "learning a function from data". A neural network <em>is</em> a function; its parameters <em>are</em> the dials that shape which function it is. Every forward pass is a function evaluation. If you own the idea of a function, the rest of the course is filling in what kind of function and how we learn it.</p>
</div>

Functions are the center of gravity for this entire course, so we give this lesson the full six-pass treatment: intuition, the mathematics, a tiny worked example, why ML needs it, the PyTorch version, and what PyTorch is doing underneath. Take it slowly.

## 1. Intuition

A **function** is a machine that takes something in and gives something out, with one rule: for the same input, you always get the same output. Put in $2$, get $5$; put in $2$ again, get $5$ again, every time.

As a programmer you already have a perfect mental model: a function is like a *pure method*. Consider this C#:

```
int F(int x) { return 2 * x; }
```

`F` is the machine's name. `x` is the input parameter. `2 * x` is the rule. Call `F(3)` and you get `6`. A mathematical function is exactly this idea, minus the types and the memory — just "name, input, rule, output". The word "pure" matters: a mathematical function has no side effects and no hidden state. Its output depends only on its input.

<div class="callout key"><p><strong>Key idea.</strong> A function is a reliable input → output mapping. Same input, same output, always. Think "pure method with no side effects".</p></div>

## 2. The mathematics: decoding $f(x)$

Here is the notation everyone uses, taken apart piece by piece. Written in full, a function definition looks like:

$$
f(x) = 2x
$$

Read it slowly, symbol by symbol:

- $f$ is the **name** of the function. It is a label, like a method name. We could call it $g$, $h$, or $\sigma$; $f$ is just the traditional first choice ("f" for function).
- $x$ is the **input**, also called the *argument*. It is a variable in the sense of Lesson 01 — a placeholder for whatever value we feed in.
- $f(x)$ — read aloud as "$f$ of $x$" — is the **output**: the value the function produces for that input. The parentheses do **not** mean multiplication here; $f(x)$ is not "$f$ times $x$". It means "$f$ applied to $x$". This trips up nearly everyone once, so say it in your head as "f *of* x".
- The $= 2x$ part is the **rule**: it tells you how to turn the input into the output. Whatever comes in as $x$, double it.

So $f(x) = 2x$ altogether reads: "the function named $f$ takes an input $x$ and returns $2x$."

To actually *use* it, we substitute (Lesson 01) a specific number for $x$:

$$
f(3) = 2 \cdot 3 = 6
$$

Here $f(3)$ means "apply $f$ to the input $3$". We replaced $x$ with $3$ everywhere it appeared in the rule, then evaluated.

A cleaner way to picture the whole thing is the **mapping arrow** notation:

$$
f : x \mapsto 2x
$$

Read: "$f$ sends $x$ to $2x$." A function is fundamentally a mapping — an assignment of exactly one output to each allowed input.

### Domain and range

Two more names, both simple:

- The **domain** is the set of inputs the function is allowed to take. For $f(x) = 2x$, any real number works, so the domain is all of $\mathbb{R}$.
- The **range** is the set of outputs the function actually produces as $x$ ranges over the domain. For $f(x) = 2x$, doubling any real number can give you any real number, so the range is also all of $\mathbb{R}$.

Domains are not always everything. For $g(x) = \frac{1}{x}$, the input $x = 0$ is forbidden (division by zero), so the domain is "all reals except $0$". Knowing a function's domain is knowing which inputs are legal — the mathematical cousin of "what argument values won't throw".

<div class="callout key"><p><strong>Key idea.</strong> Domain = allowed inputs. Range = achievable outputs. A function maps each domain element to exactly one output.</p></div>

## 3. A tiny numerical example

Let us fully work two functions by hand so the machine feels concrete.

**Function A: $f(x) = 2x$.** Feed it a few inputs and tabulate:

| input $x$ | rule $2x$ | output $f(x)$ |
|:--:|:--:|:--:|
| $0$ | $2 \cdot 0$ | $0$ |
| $1$ | $2 \cdot 1$ | $2$ |
| $2$ | $2 \cdot 2$ | $4$ |
| $3$ | $2 \cdot 3$ | $6$ |
| $-1$ | $2 \cdot (-1)$ | $-2$ |

Each input has exactly one output. Doubling is a *straight-line* (linear) relationship: every step of $+1$ in $x$ produces a step of $+2$ in $f(x)$.

**Function B: $f(x) = x^2$.** The notation $x^2$ means "$x$ times itself", $x \cdot x$ (Lesson 03 goes deep on powers). Tabulate:

| input $x$ | rule $x^2$ | output $f(x)$ |
|:--:|:--:|:--:|
| $0$ | $0 \cdot 0$ | $0$ |
| $1$ | $1 \cdot 1$ | $1$ |
| $2$ | $2 \cdot 2$ | $4$ |
| $3$ | $3 \cdot 3$ | $9$ |
| $-2$ | $(-2)\cdot(-2)$ | $4$ |

Two things to notice. First, squaring is *not* a straight line: the outputs $0, 1, 4, 9$ grow faster and faster. Second, both $x = 2$ and $x = -2$ give output $4$. That is completely allowed — a function may send *different inputs to the same output*. What it may **never** do is send *one input to two different outputs*. One input, one output; that is the whole contract.

## 4. Functions of several inputs

Nothing says a function takes only one input. A function of two inputs is written $f(x, y)$ — "$f$ of $x$ and $y$" — and its rule uses both. For example:

$$
f(x, y) = 3x + 2y
$$

To evaluate, substitute both inputs. At $x = 2$ and $y = 5$:

$$
f(2, 5) = 3 \cdot 2 + 2 \cdot 5 = 6 + 10 = 16
$$

The domain is now pairs of numbers, written $(x, y)$; the output is still a single number. You can have three inputs, ten, or ten thousand. A neural network is a function of *all its inputs and all its parameters at once* — often millions of arguments — but the principle is identical to $f(x, y) = 3x + 2y$: substitute values for every input, evaluate the rule, get an output.

<div class="callout key"><p><strong>Key idea.</strong> Multi-input functions map a <em>tuple</em> of inputs to an output. More inputs change the bookkeeping, not the concept: substitute all of them, evaluate the rule.</p></div>

## 5. Why ML needs it: "learning a function"

Here is the sentence you will hear constantly, now made precise.

Suppose you want a system that reads a house's size and predicts its price, or reads the last 1000 words of text and predicts the next word. In each case there is some true input → output mapping out in the world that you wish you knew. Call it the *target function*. You do not have a formula for it. What you have is **examples**: pairs of (input, correct output).

Machine learning is: **pick a flexible function with adjustable dials, then tune the dials until the function matches the examples.** The flexible function is the model. The dials are the **parameters** (weights). "Learning" = "searching for the parameter values that make the function fit the data".

To make the dials explicit, we write the model as

$$
f(x; \theta)
$$

Read: "$f$ of $x$, parameterized by $\theta$." The Greek letter $\theta$ (theta) is the standard name for "all the parameters bundled together". The semicolon separates the two roles: **before** it is the input $x$ (the data, which you feed in), **after** it is $\theta$ (the dials, which training chooses). Same function-machine idea, with a second set of arguments that we get to set.

A concrete miniature: the model $f(x; \theta) = \theta x$ has a single dial $\theta$. If your examples suggest that outputs should be triple the inputs, training will discover $\theta = 3$, giving the specific function $f(x) = 3x$. A real network has the same structure with millions of $\theta$'s and a far richer rule, but the story is unchanged: **parameters shape which function you have; data drives the search for good parameters.**

<div class="callout key"><p><strong>Key idea.</strong> A model is a function $f(x; \theta)$. The input $x$ is data; the parameters $\theta$ are learnable dials. Training searches for the $\theta$ that makes $f$ agree with the examples. That search is the entire back half of this course.</p></div>

For GPT-style models specifically: the input $x$ is a sequence of tokens, the output is a probability for every possible next token, and $\theta$ is the hundreds of millions of weights inside the Transformer. When people say "GPT-2 has 124 million parameters", they mean $\theta$ has 124 million numbers in it. The forward pass evaluates $f(x; \theta)$; training adjusts those 124 million dials.

## 6. In PyTorch

Two levels. First, an ordinary Python function — the direct analogue of the mathematics:

```python
def f(x):
    return 2 * x

print(f(3))     # 6      <- matches f(3) = 2*3 = 6 from the table
print(f(-1))    # -2     <- matches the table row x = -1
```

Now the same idea on **tensors**, which is how ML actually computes. A tensor lets the "input" be many numbers processed together, and it carries the machinery for training:

```python
import torch

# A function on tensors: same rule f(x) = 2x, no nn module needed.
def f(x):
    return 2 * x

x = torch.tensor([0., 1., 2., 3.])   # four inputs at once
print(f(x))                          # tensor([0., 2., 4., 6.])
```

Notice the output `tensor([0., 2., 4., 6.])` is exactly the $f(x)$ column of Function A's table — PyTorch applied the rule to every input in the tensor at once.

And a function with an explicit parameter $\theta$, matching $f(x; \theta) = \theta x$:

```python
import torch

theta = torch.tensor(3.)             # the dial; here we set it by hand to 3

def model(x, theta):
    return theta * x                 # f(x; theta) = theta * x

x = torch.tensor(4.)
print(model(x, theta))               # tensor(12.)  <- 3 * 4 = 12
```

<div class="callout pt"><p><strong>PyTorch note.</strong> A real model bundles its parameters $\theta$ inside an object (a <code>torch.nn.Module</code>) so you do not pass them by hand, and it marks them <code>requires_grad=True</code> so they can be tuned. But conceptually a model is nothing more mysterious than this <code>model(x, theta)</code> — a function of data and dials. We build up to <code>nn.Module</code> in Module 8.</p></div>

## 7. What PyTorch is doing under the hood

When you called `model(x, theta)`, PyTorch evaluated the rule `theta * x` and produced `tensor(12.)`. That is the *forward pass*: a function evaluation, nothing more.

But there is a second, invisible half. If `theta` is marked as a parameter to be learned, PyTorch simultaneously records *how the output was built from the inputs* — here, "multiply `theta` by `x`". It keeps this as a little graph so that, later, it can answer the question training actually cares about: **"if I nudge $\theta$ a hair, which way and how much does the output move?"** Answering that question for every parameter is how the dials get turned. It is the subject of Module 6 (derivatives) and Module 8 (backpropagation).

For now, hold two facts together and you understand functions in ML completely:

1. **Forward** = evaluate the function $f(x; \theta)$ — substitute inputs and parameters, turn the crank, read the output.
2. **Learning** = adjust $\theta$ using the recorded graph so that $f$'s outputs match the target examples.

Everything else is detail layered onto this skeleton.

## Check yourself

1. In $f(x) = 2x$, what do $f$, $x$, and $f(x)$ each refer to? Is $f(x)$ a multiplication?

<details><summary>Answer</summary>

$f$ is the function's name; $x$ is the input (argument); $f(x)$ is the output for that input — read "f of x". It is **not** a multiplication: $f(x)$ means "$f$ applied to $x$", not "$f$ times $x$".

</details>

2. Can a function send two different inputs to the same output? Can it send one input to two different outputs?

<details><summary>Answer</summary>

Yes to the first — e.g. $f(x) = x^2$ sends both $2$ and $-2$ to $4$. No to the second — that is forbidden; each input must map to exactly one output, or it is not a function.

</details>

3. For $f(x, y) = 3x + 2y$, compute $f(1, 4)$.

<details><summary>Answer</summary>

$f(1, 4) = 3 \cdot 1 + 2 \cdot 4 = 3 + 8 = 11$.

</details>

4. In the notation $f(x; \theta)$, which part is the data and which part is learned during training?

<details><summary>Answer</summary>

$x$ (before the semicolon) is the input data, fed in each time. $\theta$ (after the semicolon) is the parameters — the dials that training searches for and sets.

</details>

## Next

You can now read $f(x)$ and think of a model as a function of data and dials. Those rules keep using operations like squaring, roots, exponentials, and logarithms — the toolkit that makes loss functions and probabilities work. We build that toolkit next.

[03 · Powers, roots, exponentials, logarithms →](lesson-03.md)
