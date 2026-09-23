# 19 · The chain rule and computational graphs

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-06/lesson-02">16 · Derivative rules and the chain rule</a> (the single-variable chain rule, power rule, sum rule) and <a href="#/lessons/module-06/lesson-04">18 · Partial derivatives and the gradient</a> (partial derivatives; that <code>.grad</code> holds a derivative). Functions and composition from <a href="#/lessons/module-05/lesson-01">the functions module</a> help.</p>
<p><strong>You will learn:</strong> how to read a calculation as a <strong>computational graph</strong>, how every node carries a <strong>local derivative</strong>, how the chain rule multiplies those local derivatives along a path, and how — when a variable feeds several paths — the contributions <strong>add</strong>. This is backpropagation stated as arithmetic on a graph.</p>
<p><strong>Why this matters for ML:</strong> training a network means computing $\partial L/\partial\theta$ for every parameter. Backpropagation does this by sweeping a computational graph from the loss backward, multiplying local derivatives. Everything in this lesson is that mechanism in miniature — once you see it on a three-node chain, you have seen what PyTorch does on a graph of millions of nodes.</p>
</div>

## 1. Intuition: a calculation is a pipeline

Think of a numerical calculation the way you think about a data pipeline in code: values flow through a series of small, simple stages, each transforming its input into an output that becomes the next stage's input. Consider this tiny pipeline, the example we will ride for the whole lesson:

$$
a = 2x, \qquad b = a^2, \qquad y = b + 3.
$$

Read top to bottom: take $x$, double it to get $a$; square $a$ to get $b$; add $3$ to get $y$. Each line is one elementary operation. Nobody computes $y$ from $x$ "all at once" — you go stage by stage. That stage-by-stage structure is exactly a **computational graph**, and it is the key to differentiating complicated functions: if you know how to differentiate each *small* stage, the chain rule tells you how to combine those pieces into the derivative of the whole.

A programmer's framing: the derivative $dy/dx$ answers "if I nudge the input $x$ by a tiny amount, how much does the final output $y$ move, and in which direction?" That sensitivity is what training needs — it tells us how to change inputs (parameters) to move an output (the loss).

## 2. The computational graph

Draw the pipeline as boxes (operations) connected by arrows (values):

```text
   x  ──▶ [ ×2 ] ──▶  a  ──▶ [ square ] ──▶  b  ──▶ [ +3 ] ──▶  y
```

- **Nodes / edges.** Each box is an *operation* (multiply-by-2, square, add-3). Each arrow carries a *value* ($x$, then $a$, then $b$, then $y$).
- **Forward pass.** Pushing numbers left-to-right through the boxes is the **forward pass** — it computes the output.
- **Local derivative.** Every box has a **local derivative**: the derivative of *its output with respect to its own input*, ignoring everything else in the graph. The multiply-by-2 box has local derivative $da/dx = 2$. The square box has local derivative $db/da = 2a$. The add-3 box has local derivative $dy/db = 1$ (adding a constant does not change the slope).

The whole trick of backprop is that these local derivatives are trivial — each box does one tiny operation whose derivative you already know from [module 6](lessons/module-06/lesson-02.md) — and the chain rule stitches them into the global derivative.

## 3. Computing $dy/dx$ two ways

### 3.1 Way one — substitute, then differentiate

Collapse the pipeline into a single formula by substitution. Starting from the last line and working back:

$$
y = b + 3 = a^2 + 3 = (2x)^2 + 3 = 4x^2 + 3.
$$

Now differentiate this one-variable function with the power rule ($\frac{d}{dx}x^2 = 2x$) and the constant rule ($\frac{d}{dx}3 = 0$):

$$
\frac{dy}{dx} = \frac{d}{dx}\big(4x^2 + 3\big) = 8x.
$$

At $x = 3$ that gives $\frac{dy}{dx} = 8 \cdot 3 = 24$. Clean, but it required us to flatten the whole pipeline into one expression by hand. For a real network with millions of operations, that flattening is impossible. We need a method that stays *local*.

### 3.2 Way two — the chain rule along the path

The **chain rule** says: to get the derivative of the output with respect to the input, **multiply the local derivatives along the path** connecting them. There is a single path $x \to a \to b \to y$, so

$$
\frac{dy}{dx} = \frac{dy}{db}\cdot\frac{db}{da}\cdot\frac{da}{dx}.
$$

Read it as a telescoping product: how $y$ responds to $b$, times how $b$ responds to $a$, times how $a$ responds to $x$. Each factor is one box's local derivative:

- $\dfrac{dy}{db} = 1$ (the $+3$ box).
- $\dfrac{db}{da} = 2a$ (the square box).
- $\dfrac{da}{dx} = 2$ (the $\times 2$ box).

Multiply:

$$
\frac{dy}{dx} = 1 \cdot (2a) \cdot 2 = 4a.
$$

This is in terms of the *intermediate* value $a$. Since $a = 2x$, we have $4a = 4(2x) = 8x$ — identical to way one. The two methods agree, as they must. The difference is that the chain-rule method never needed the collapsed formula $4x^2 + 3$; it only needed each box's own derivative and the intermediate value $a$.

<div class="callout key"><p>The chain rule multiplies <strong>local derivatives</strong> along the path from input to output: $\frac{dy}{dx} = \frac{dy}{db}\cdot\frac{db}{da}\cdot\frac{da}{dx}$. Each factor depends only on one operation and the values flowing through it. This locality is what makes it mechanical — and automatable.</p></div>

### 3.3 Evaluate everything at $x = 3$

Let us pin down every value on both passes at $x = 3$.

**Forward pass** (left to right):

| Quantity | Formula | Value at $x=3$ |
| --- | --- | --- |
| $x$ | input | $3$ |
| $a$ | $2x$ | $6$ |
| $b$ | $a^2$ | $36$ |
| $y$ | $b + 3$ | $39$ |

**Backward pass** (right to left), multiplying local derivatives as we go. We track a running quantity — call it the **upstream derivative** $\frac{dy}{d(\cdot)}$ — starting at $\frac{dy}{dy} = 1$ and picking up one local factor per box:

| At edge | Local derivative | Running $\dfrac{dy}{d(\cdot)}$ |
| --- | --- | --- |
| output $y$ | — | $\dfrac{dy}{dy} = 1$ |
| $b$ | $\dfrac{dy}{db} = 1$ | $1 \cdot 1 = 1$ |
| $a$ | $\dfrac{db}{da} = 2a = 12$ | $1 \cdot 12 = 12$ |
| $x$ | $\dfrac{da}{dx} = 2$ | $12 \cdot 2 = 24$ |

The final running value at $x$ is $\frac{dy}{dx} = 24$, matching $8x = 8\cdot 3 = 24$. Notice how the backward pass *reuses* the forward value $a = 6$ (it appears inside the square box's local derivative $2a = 12$). This is why frameworks store forward values: the backward pass needs them.

### 3.4 Verify numerically with a small nudge

A derivative is a limit of $\frac{\Delta y}{\Delta x}$ for a tiny step. Let us confirm $\frac{dy}{dx}\big|_{x=3} = 24$ by actually nudging. Take $x = 3.001$ and run the forward pass:

$$
y(3.001) = 4(3.001)^2 + 3 = 4(9.006001) + 3 = 36.024004 + 3 = 39.024004.
$$

We already have $y(3) = 39$. The finite-difference slope is

$$
\frac{y(3.001) - y(3)}{0.001} = \frac{39.024004 - 39}{0.001} = \frac{0.024004}{0.001} = 24.004,
$$

which is essentially $24$ (the tiny excess $0.004$ is the curvature; it shrinks as the step shrinks). The chain rule and the direct nudge agree — the derivative really is the sensitivity of $y$ to a wiggle in $x$.

## 4. Branching: when a value feeds two paths, gradients ADD

The single-path chain rule handles a straight pipeline. But real graphs branch: one value is used in more than one place. This is the case that matters most for ML, because a shared parameter is a value that fans out to many downstream uses.

The simplest branching example is $y = x \cdot x$ (which of course equals $x^2$, but we deliberately write it as a *product of two copies of $x$* to expose the branch). Draw it:

```text
        ┌──────────────▶  u ( = x )  ─┐
   x  ──┤                              ├──▶ [ × ] ──▶  y = u·v
        └──────────────▶  v ( = x )  ─┘
```

Here $x$ fans out along **two edges**: it becomes the left factor $u$ and the right factor $v$ of the multiply node, and then $y = u \cdot v$. The value $x$ influences $y$ through *two* paths.

### 4.1 The multivariable chain rule

When a variable reaches the output through several paths, you compute the derivative along *each* path and **sum them**. This is the **multivariable chain rule** (also called the total derivative):

$$
\frac{dy}{dx} = \frac{\partial y}{\partial u}\cdot\frac{du}{dx} \;+\; \frac{\partial y}{\partial v}\cdot\frac{dv}{dx}.
$$

- The first term is the path $x \to u \to y$.
- The second term is the path $x \to v \to y$.
- The $+$ between them is the whole point: **contributions from separate paths add**.

For the multiply node $y = u\cdot v$, the local (partial) derivatives are $\frac{\partial y}{\partial u} = v$ and $\frac{\partial y}{\partial v} = u$ (differentiate with the other factor held fixed — this is exactly a partial derivative from [lesson 18](lessons/module-06/lesson-04.md)). And $\frac{du}{dx} = \frac{dv}{dx} = 1$, since $u = x$ and $v = x$. So

$$
\frac{dy}{dx} = v\cdot 1 + u\cdot 1 = u + v = x + x = 2x.
$$

At $x = 3$: $\frac{dy}{dx} = 6$. Cross-check against the collapsed form $y = x^2$, whose derivative $2x = 6$ agrees. Had we (wrongly) followed only one path we would have gotten $x = 3$ — off by exactly the factor of two that the *second* incoming edge contributes. The two incoming gradients, $v = 3$ from the left edge and $u = 3$ from the right edge, sum to $6$.

<div class="callout key"><p>When a value fans out to several downstream uses, its gradient is the <strong>sum</strong> of the gradients coming back along each path. This "gradients add at a fan-out" rule is why a shared parameter — used in many positions — accumulates one gradient contribution per use.</p></div>

### 4.2 Why this is the rule for shared parameters

In a Transformer, the same weight matrix multiplies the input at every one of the $T$ sequence positions; the same token-embedding row is reused wherever that token appears; weight tying reuses one matrix as both the input embedding and the output projection. Each of these is a value that fans out to many places in the graph. The gradient PyTorch reports for such a parameter is the *sum* of the gradient contributions from all its uses — precisely the multivariable chain rule at a fan-out. This is also exactly why gradients **accumulate** into `.grad` (the behavior we flagged in [lesson 18](lessons/module-06/lesson-04.md)): accumulation is how the graph sums the branches.

## 5. Why ML needs this: backprop is a backward graph sweep

A neural network is one enormous computational graph. The forward pass runs left to right — tokens in, loss $L$ out — recording every operation and the values flowing along its edges. Training needs $\frac{\partial L}{\partial\theta}$ for every parameter $\theta$. Doing that by the "substitute everything into one formula" method (way one) is hopeless: there is no closed form for a 124-million-parameter loss. But the chain-rule method (way two) is perfectly mechanical:

1. Start at the output with $\frac{\partial L}{\partial L} = 1$.
2. Walk the graph **backward**. At each node, multiply the incoming upstream derivative by that node's *local* derivative to get the derivative with respect to the node's inputs.
3. Where a value fanned out, **add** the contributions coming back along each edge.
4. When you reach a parameter leaf, the accumulated number is $\frac{\partial L}{\partial\theta}$ — store it in that leaf's `.grad`.

That is **backpropagation**: the chain rule, organized as a right-to-left sweep over the graph, computing every parameter's gradient in a single pass. Our three-node chain did steps 1–4 in the table of section 3.3; a network does the same thing with matrix operations at each node. Nothing new is added later — only more nodes and bigger local derivatives.

## 6. In PyTorch

Let us build the exact $a, b, y$ chain, differentiate it, and confirm `x.grad == 24`.

```python
import torch

x = torch.tensor(3., requires_grad=True)   # the input leaf

a = 2 * x        # forward: a = 6.
b = a**2         # forward: b = 36.
y = b + 3        # forward: y = 39.

y.backward()     # sweep the graph backward from y
print(y)         # tensor(39., grad_fn=<AddBackward0>)
print(x.grad)    # tensor(24.)   ->  dy/dx = 8x = 24 at x=3
```

`x.grad` is `tensor(24.)`, exactly the hand result. Now the branching case $y = x\cdot x$, confirming the gradients add to $2x = 6$:

```python
x = torch.tensor(3., requires_grad=True)
y = x * x        # x fans out to BOTH factors of the multiply
y.backward()
print(x.grad)    # tensor(6.)   ->  v + u = 3 + 3, the two edges summed
```

`x.grad` is `6.`, not `3.`: PyTorch summed the contribution from each edge, just as the multivariable chain rule prescribes.

### Inspecting the graph with `grad_fn`

Every non-leaf tensor produced by an operation carries a `grad_fn` — a handle to the backward function for the operation that made it. Following the `grad_fn` chain reveals the recorded graph:

```python
x = torch.tensor(3., requires_grad=True)
a = 2 * x
b = a**2
y = b + 3

print(y.grad_fn)                          # <AddBackward0>    (the +3 box)
print(y.grad_fn.next_functions[0][0])     # <PowBackward0>    (the square box)
print(b.grad_fn)                          # <PowBackward0>
print(a.grad_fn)                          # <MulBackward0>    (the *2 box)
print(x.grad_fn)                          # None  -> x is a leaf, nothing made it
```

Reading `y.grad_fn` and its `next_functions` walks the graph backward: `AddBackward0` (the $+3$) points to `PowBackward0` (the square), which points to `MulBackward0` (the $\times 2$), which points to the leaf $x$. That linked structure — each node knowing its inputs' backward functions — *is* the graph `backward()` traverses.

<div class="callout pt"><p>A tensor with <code>requires_grad=True</code> that you created directly is a <strong>leaf</strong> (its <code>grad_fn</code> is <code>None</code>); gradients land in its <code>.grad</code>. Every tensor produced by an op is a non-leaf carrying a <code>grad_fn</code>. <code>y.backward()</code> follows the <code>grad_fn</code> links from <code>y</code> down to the leaves, multiplying local derivatives on the way.</p></div>

## 7. What PyTorch is doing under the hood

During the **forward pass**, each operation does two things: it computes its output, and — because a tensor in the computation has `requires_grad=True` — it *records* itself onto the graph, saving whatever inputs its backward rule will need. The square node $b = a^2$ saves the value $a = 6$, because its local derivative is $2a$ and needs $a$. The multiply node saves nothing extra beyond the constant $2$. This recording is the "tape" of reverse-mode automatic differentiation: a list of operations in the order they ran, each annotated with the values needed to differentiate it.

When you call `y.backward()`, autograd walks that tape in **reverse**. It seeds the output derivative with $1$ (the derivative of $y$ with respect to itself). Then, node by node from $y$ back toward $x$, it applies each node's backward rule: take the upstream derivative arriving from the output side, multiply by this node's local derivative (using the saved forward values), and pass the result further back. At the add node it multiplies by $1$; at the square node it multiplies by $2a = 12$ (using the saved $a=6$); at the multiply node it multiplies by $2$ — yielding $1 \cdot 1 \cdot 12 \cdot 2 = 24$, deposited into `x.grad`. Where a value fanned out to several nodes (as in $x \cdot x$), the backward contributions arriving from each consumer are **summed** into that value's gradient — the accumulation we saw give $6$.

The general operation applied at each node — "take an upstream vector and push it back through this operation's local derivative" — is called a **vector–Jacobian product** (VJP). For a scalar chain it is just multiplication by a number, as here. In the next lessons we generalize local derivatives to vectors, matrices, and tensors, and the VJP becomes the workhorse that lets autograd differentiate a whole network without ever building a giant derivative matrix.

## Check yourself

<details><summary>For the chain $a = 2x,\ b = a^2,\ y = b + 3$, what are the three local derivatives, and what is $dy/dx$ at $x = 5$?</summary>

Local derivatives: $\frac{dy}{db} = 1$, $\frac{db}{da} = 2a$, $\frac{da}{dx} = 2$. Product: $1\cdot 2a\cdot 2 = 4a = 8x$. At $x=5$: $8\cdot 5 = 40$. (Check: $y = 4x^2+3$, $y' = 8x = 40$.)

</details>

<details><summary>In the backward table, why does the value $a = 6$ from the forward pass reappear during the backward pass?</summary>

Because the square node's local derivative is $\frac{db}{da} = 2a$, which needs the forward value of $a$. The backward pass reuses saved forward values — this is exactly why PyTorch stores intermediates during forward.

</details>

<details><summary>For $y = x\cdot x$, one student computes $dy/dx = v = x$ (following only the left edge) and gets $3$ at $x=3$. What did they miss, and what is the correct answer?</summary>

They followed only one of the two paths from $x$ into the multiply node. The multivariable chain rule sums both: $v\cdot 1 + u\cdot 1 = x + x = 2x = 6$. Gradients from a fan-out add.

</details>

<details><summary>A weight matrix is used at all $T = 4$ sequence positions of a Transformer. How many gradient contributions get summed into its <code>.grad</code>, and why?</summary>

Four — one per position. The matrix is a value that fans out to four downstream uses, and by the fan-out rule the gradient is the sum over all uses. This summation is exactly gradient accumulation.

</details>

## Next

You now have backpropagation in miniature: a graph, local derivatives, a right-to-left sweep that multiplies them, and a sum wherever a value fans out. The next step is to be precise about the *shapes* of the things being differentiated — a single partial $\partial L/\partial x_i$ versus the whole gradient vector $\nabla_x L$, and what it means to take a gradient with respect to a matrix or an arbitrary tensor of parameters.

Continue to [20 · Gradients: ∂L/∂x vs ∇, with respect to matrices and tensors](lessons/module-07/lesson-02.md).
