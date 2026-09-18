# 18 · Partial derivatives and the gradient

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-01.md">15 · Limits, slope, and the derivative</a> and <a href="lesson-02.md">16 · Derivative rules and the chain rule</a> (you need the power, product, and sum rules), plus vectors from <a href="../module-02/lesson-01.md">the vectors module</a>.</p>
<p><strong>You will learn:</strong> what a function of several variables is, the <strong>partial derivative</strong> (differentiate with respect to one variable while holding the others fixed), and the <strong>gradient</strong> — the vector of all partial derivatives — including its shape rule and its meaning as the direction of steepest ascent.</p>
<p><strong>Why this matters for ML:</strong> a loss depends not on one number but on millions of parameters at once. The gradient is the vector of every $\partial L/\partial\theta$, and training moves the parameters along $-\nabla L$ — downhill. This lesson is the direct bridge from single-variable calculus to gradient descent.</p>
</div>

## 1. Functions of several variables

So far every function took one input: $f(x)$. But most quantities in ML depend on many inputs at once. A **function of several variables** takes a handful of numbers and returns one number. Write it $f(x, y)$ for two inputs:

$$
f(x, y) = x^2 + 3xy + y^2.
$$

- $x, y$ — two independent inputs.
- $f(x, y)$ — a single output number computed from both.

Feed in $x = 1, y = 2$: $f(1,2) = 1^2 + 3\cdot 1\cdot 2 + 2^2 = 1 + 6 + 4 = 11$. You can picture $f(x,y)$ as a landscape: the two inputs are east–west and north–south position, and the output is the *height* of the terrain there. A neural network loss is exactly this kind of function, only with millions of input axes instead of two — a landscape in a million-dimensional space, whose height is how wrong the model currently is.

## 2. The partial derivative

### 2.1 Intuition

On a two-input landscape, "the slope" is ambiguous — slope *in which direction*? The simplest choice: slope if you walk purely east (change $x$, keep $y$ fixed), or purely north (change $y$, keep $x$ fixed). Those two slopes are the **partial derivatives**. A partial derivative asks: *how sensitive is the output to one input, holding all the others constant?*

That "hold the others constant" is the entire trick, and it is one a programmer will find natural: treat the other variables as if they were fixed constants for the moment, then differentiate normally.

### 2.2 The mathematics

The partial derivative of $f$ with respect to $x$ is written with a curly $\partial$ (read "partial" or "del") instead of the straight $d$:

$$
\frac{\partial f}{\partial x} = \lim_{h \to 0}\frac{f(x + h,\, y) - f(x,\, y)}{h}.
$$

- $\partial$ — signals a partial derivative: we differentiate with respect to *one* named variable.
- $\frac{\partial f}{\partial x}$ — the rate of change of $f$ as $x$ varies and **$y$ is held fixed** (notice only the first slot changes in $f(x+h, y)$).
- Likewise $\frac{\partial f}{\partial y}$ holds $x$ fixed and varies $y$.

The straight-$d$ notation $\frac{df}{dx}$ from [lesson 1](lesson-01.md) is for single-variable functions; the curly $\frac{\partial f}{\partial x}$ says "there are other variables, and I am freezing them." Otherwise it is the same limit-of-a-difference-quotient idea, and — crucially — the *same rules from [lesson 2](lesson-02.md) apply*. You just regard every non-target variable as a constant.

### 2.3 Worked example

Take $f(x, y) = x^2 + 3xy + y^2$ and compute both partials.

**Partial with respect to $x$** (treat $y$ as a constant):

- $\frac{\partial}{\partial x}(x^2) = 2x$ (power rule).
- $\frac{\partial}{\partial x}(3xy) = 3y$ — here $3y$ is a *constant multiple* of $x$, so its derivative is that constant, $3y$ (constant-multiple rule; $\frac{\partial}{\partial x}x = 1$).
- $\frac{\partial}{\partial x}(y^2) = 0$ — with $y$ held fixed, $y^2$ is a pure constant, and the derivative of a constant is $0$.

Adding up (sum rule):

$$
\frac{\partial f}{\partial x} = 2x + 3y.
$$

**Partial with respect to $y$** (now treat $x$ as a constant):

- $\frac{\partial}{\partial y}(x^2) = 0$ (constant).
- $\frac{\partial}{\partial y}(3xy) = 3x$ ($3x$ is the constant multiple of $y$).
- $\frac{\partial}{\partial y}(y^2) = 2y$.

So

$$
\frac{\partial f}{\partial y} = 3x + 2y.
$$

### 2.4 Evaluate at a point

Plug in $(x, y) = (1, 2)$:

$$
\frac{\partial f}{\partial x}\bigg|_{(1,2)} = 2(1) + 3(2) = 2 + 6 = 8,
$$
$$
\frac{\partial f}{\partial y}\bigg|_{(1,2)} = 3(1) + 2(2) = 3 + 4 = 7.
$$

Reading them as sensitivities: at the point $(1,2)$, a tiny nudge to $x$ raises the output about $8$ times as fast as the nudge, while the same nudge to $y$ raises it about $7$ times as fast. The output is slightly more sensitive to $x$ than to $y$ right here.

<div class="callout key"><p>A partial derivative $\frac{\partial f}{\partial x}$ is an ordinary derivative taken with every <em>other</em> variable frozen as a constant. All the rules from lesson 2 still apply; you just treat the frozen variables as numbers.</p></div>

## 3. The gradient

### 3.1 Intuition

We now have two separate slopes at $(1,2)$: $8$ in the $x$-direction and $7$ in the $y$-direction. Bundle them into a single vector and you get the **gradient** — an arrow that packages all the partial derivatives at once. Its components tell you how the output responds to each input, and, taken together, the arrow points in a very special direction: straight uphill.

### 3.2 The mathematics

The gradient of $f$, written $\nabla f$ (the symbol $\nabla$ is "nabla" or "del"), is the vector of all first partial derivatives:

$$
\nabla f = \begin{bmatrix} \dfrac{\partial f}{\partial x} \\[2mm] \dfrac{\partial f}{\partial y} \end{bmatrix}.
$$

- $\nabla f$ — the gradient, a vector with one entry per input variable.
- Each entry is the partial derivative with respect to that variable.

For our $f$:

$$
\nabla f = \begin{bmatrix} 2x + 3y \\ 3x + 2y \end{bmatrix}, \qquad
\nabla f\big|_{(1,2)} = \begin{bmatrix} 8 \\ 7 \end{bmatrix}.
$$

### 3.3 Shape rule

The gradient has exactly the **same shape as the input**. Here the input is a 2-vector $(x, y)$, so $\nabla f$ is a 2-vector. If a function took a length-$1000$ vector and returned one number, its gradient would be a length-$1000$ vector; if it took a $4\times 4$ matrix of parameters, its gradient would be a $4\times 4$ matrix. This rule is worth burning in, because it is precisely how PyTorch lays out gradients: `param.grad` always matches `param.shape`.

<div class="callout key"><p>$\nabla f$ is the vector of all partial derivatives, and it has the <strong>same shape as the input</strong>. One number in, scalar output: the gradient collects "how the output moves per unit move in each input direction," one entry at a time.</p></div>

### 3.4 Direction of steepest ascent

Here is the fact that makes the gradient the center of all of training: **$\nabla f$ points in the direction of steepest ascent** — the direction you would step to increase $f$ fastest — and its length is how steep that fastest climb is. Any other direction climbs more slowly (or descends).

The consequence is immediate: the **negative** gradient $-\nabla f$ points in the direction of steepest *descent* — the fastest way *down*. At $(1,2)$ our gradient is $[8, 7]$, so the steepest way to *increase* $f$ is to move in the direction $[8, 7]$, and the steepest way to *decrease* it is to move in $[-8, -7]$. If $f$ were a loss you wanted to minimize, you would take a small step in the direction $[-8, -7]$. That single sentence is the seed of **gradient descent**:

$$
\theta \leftarrow \theta - \eta\,\nabla L(\theta),
$$

where $\theta$ is the parameter vector, $\eta$ (eta) is a small positive step size (the learning rate), and $\nabla L$ is the gradient of the loss. Each step nudges the parameters a little way downhill. We build this update in full in a later module; for now, notice it is nothing but "evaluate the gradient, then step against it."

## 4. Why ML needs this

A GPT-2 model has about 124 million parameters — call the whole collection $\theta$. The loss $L(\theta)$ is one number (how badly the model predicts the next token), computed from all 124 million inputs at once: a function of 124 million variables. Training must know, for each parameter, whether nudging it up raises or lowers the loss and by how much. That is $\frac{\partial L}{\partial\theta_i}$ for every parameter $i$, and stacking them all gives

$$
\nabla_\theta L = \begin{bmatrix} \partial L/\partial\theta_1 \\ \partial L/\partial\theta_2 \\ \vdots \\ \partial L/\partial\theta_{124\text{M}} \end{bmatrix},
$$

a vector with 124 million entries — the same shape as $\theta$ itself, by the shape rule. Every training step evaluates this gradient (by backpropagation, coming in the next module) and moves $\theta$ a small distance along $-\nabla_\theta L$. So the two-variable example you just worked by hand is the exact same operation the training loop performs, scaled up by a factor of sixty million. Everything that follows — backprop, SGD, AdamW, Muon — is machinery for computing this gradient and choosing how to step along it.

## 5. In PyTorch

PyTorch fills in the partial derivatives for us. First, two separate scalar leaves, reproducing the hand result $\nabla f|_{(1,2)} = [8, 7]$:

```python
import torch

x = torch.tensor(1., requires_grad=True)
y = torch.tensor(2., requires_grad=True)

f = x**2 + 3*x*y + y**2     # forward: 1 + 6 + 4 = 11.0
f.backward()                # compute df/dx and df/dy

print(x.grad)               # tensor(8.)   ->  2x + 3y = 2 + 6
print(y.grad)               # tensor(7.)   ->  3x + 2y = 3 + 4
```

`x.grad` and `y.grad` are `8.` and `7.`, exactly the partials we computed in section 2.4. A single `f.backward()` filled *both* — one backward pass produces the whole gradient.

Now the same thing with the inputs packed into one vector, which is how parameters actually live in a model:

```python
p = torch.tensor([1., 2.], requires_grad=True)          # p[0]=x, p[1]=y

f = p[0]**2 + 3*p[0]*p[1] + p[1]**2                      # forward: 11.0
f.backward()

print(p.grad)               # tensor([8., 7.])
```

`p.grad` is the vector `[8., 7.]` — the gradient $\nabla f$, and it has the same shape $(2,)$ as `p`, exactly as the shape rule promised.

<div class="callout pt"><p>One <code>.backward()</code> call computes <em>every</em> partial derivative at once and deposits each into the corresponding leaf's <code>.grad</code>. When parameters are packed in a tensor, <code>.grad</code> is a tensor of the same shape holding the full gradient — this is what an optimizer reads to take its step.</p></div>

## 6. What PyTorch is doing under the hood

`.grad` is a slot attached to every leaf tensor (one created with `requires_grad=True`). It starts as `None`. When you call `f.backward()`, autograd walks the recorded computational graph backward from `f`, and for each leaf it computes $\partial f/\partial(\text{that leaf})$ and **writes it into that leaf's `.grad`**. So one backward sweep populates the gradient for every parameter simultaneously — you do not loop over parameters yourself.

Two details that trip people up, both flowing from "`.grad` is a slot that gets written":

- **Same shape as the leaf.** Because $\partial f/\partial\theta_i$ exists for each component $\theta_i$, the collected gradient has exactly the leaf's shape. `p.grad` matches `p.shape`; this is the section 3.3 rule made concrete.
- **Gradients accumulate.** `backward()` *adds* into `.grad` rather than overwriting it. Call `backward()` twice without clearing and you get the sum of two gradients. That is deliberate (it lets you accumulate gradients over several mini-batches), but it means the training loop must reset the slots with `optimizer.zero_grad()` before each new backward pass — otherwise this step's gradient is contaminated by last step's. We return to `zero_grad`, `grad_fn`, and the full graph traversal in [module 7](../module-07/lesson-01.md), where the gradient computation itself — backpropagation — is finally assembled from the chain rule.

## Check yourself

<details><summary>For $f(x,y) = x^2 + 3xy + y^2$, compute $\nabla f$ at $(x,y) = (2, 1)$.</summary>

$\frac{\partial f}{\partial x} = 2x + 3y = 4 + 3 = 7$; $\frac{\partial f}{\partial y} = 3x + 2y = 6 + 2 = 8$. So $\nabla f|_{(2,1)} = [7, 8]$.

</details>

<details><summary>Find both partials of $g(x, y) = x^2 y$.</summary>

Holding $y$ fixed: $\frac{\partial g}{\partial x} = 2xy$. Holding $x$ fixed ($x^2$ is a constant multiple of $y$): $\frac{\partial g}{\partial y} = x^2$.

</details>

<details><summary>If the gradient of a loss at the current parameters is $[3, -4]$, which way do you step to reduce the loss fastest, and what is the slope of steepest descent?</summary>

Step along the negative gradient, $[-3, 4]$. The magnitude $\|[3,-4]\| = \sqrt{9 + 16} = 5$ is how steeply the loss rises in the uphill direction, so descent drops at rate $5$ per unit step.

</details>

<details><summary>Why does <code>param.grad</code> always have the same shape as <code>param</code>?</summary>

Because the gradient has one partial derivative per input component (the shape rule). A parameter tensor of shape $S$ has one entry $\partial L/\partial\theta_i$ for each of its $S$ components, so the collected gradient is itself a tensor of shape $S$.

</details>

## Next

You can now take partial derivatives, assemble them into a gradient, and you understand that training walks the parameters along $-\nabla L$. What is still missing is the *mechanism* that computes $\nabla L$ efficiently for a deep network: the chain rule organized over a computational graph, swept backward. That is backpropagation — the heart of the whole course — and it is where we go next.

Continue to [19 · The chain rule as a graph](../module-07/lesson-01.md).
