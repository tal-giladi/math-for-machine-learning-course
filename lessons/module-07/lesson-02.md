# 20 · Gradients: ∂L/∂x vs ∇, with respect to matrices and tensors

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-06/lesson-04">18 · Partial derivatives and the gradient</a> (the partial derivative and the gradient vector, and the shape rule) and <a href="#/lessons/module-07/lesson-01">19 · The chain rule and computational graphs</a>. Vectors and matrices from <a href="#/lessons/module-02/lesson-01">module 2</a> and <a href="#/lessons/module-03/lesson-01">module 3</a>.</p>
<p><strong>You will learn:</strong> the precise difference between a single partial derivative $\partial L/\partial x_i$ and the whole gradient $\nabla_x L$; that the gradient with respect to a <strong>matrix</strong> is itself a matrix, and with respect to any <strong>tensor</strong> is a tensor of the same shape; and exactly what "the gradient of the loss with respect to a parameter" means, entry by entry.</p>
<p><strong>Why this matters for ML:</strong> every parameter in GPT-2 — vectors, matrices, and higher tensors — needs a gradient, and PyTorch stores it in <code>p.grad</code> right beside the parameter <code>p</code>, always the same shape. Understanding the shape invariant is understanding how an optimizer knows how to step every parameter at once.</p>
</div>

## 1. One partial derivative vs. the whole gradient

Two pieces of notation get used constantly and are easy to confuse. Let us pin them down.

A **single partial derivative** $\dfrac{\partial L}{\partial x_i}$ is *one number*: how the scalar loss $L$ changes as you nudge *one* component $x_i$ of the input, holding every other component fixed. The subscript $i$ picks out a specific slot.

The **gradient** $\nabla_x L$ (read "nabla-x of L", or "grad of L with respect to $x$") is the *whole collection* of those partial derivatives, packed into an object of the same shape as $x$:

$$
\nabla_x L = \begin{bmatrix} \dfrac{\partial L}{\partial x_1} \\[2mm] \dfrac{\partial L}{\partial x_2} \\[1mm] \vdots \\[1mm] \dfrac{\partial L}{\partial x_n} \end{bmatrix}.
$$

- $L$ — a scalar (a single number, such as a loss).
- $x$ — the input, here a vector with components $x_1, \dots, x_n$.
- $\partial L/\partial x_i$ — one entry: the sensitivity of $L$ to the $i$-th component alone.
- $\nabla_x L$ — the vector holding *all* $n$ of those entries.

So $\partial L/\partial x_i$ is a single ingredient; $\nabla_x L$ is the full recipe. When someone writes $\partial L/\partial x$ without a subscript (no specific $i$), they almost always mean the entire gradient object $\nabla_x L$ — the two notations $\partial L/\partial x$ and $\nabla_x L$ are used interchangeably for "the gradient." The subscripted $\partial L/\partial x_i$ is the only one that denotes a lone number.

<div class="callout key"><p>$\dfrac{\partial L}{\partial x_i}$ is <strong>one</strong> partial derivative (a scalar). $\nabla_x L$ (equivalently the unsubscripted $\partial L/\partial x$) is the <strong>whole gradient</strong> — every partial derivative packed into an object <strong>the same shape as $x$</strong>.</p></div>

## 2. The gradient keeps the shape of what you differentiate against

In [lesson 18](lessons/module-06/lesson-04.md) you met the shape rule for a vector input: $\nabla_x L$ has the same shape as $x$. That rule does not stop at vectors. It holds for *any* container of numbers you differentiate the scalar $L$ against.

### 2.1 Gradient with respect to a matrix

Suppose $L$ depends on a matrix $\mathbf{W}$ of shape $m\times n$ (that is, $W_{ij}$ for row $i = 1..m$ and column $j = 1..n$). Then the gradient is a matrix of the *same* $m\times n$ shape, whose $(i,j)$ entry is the partial derivative of $L$ with respect to that one weight:

$$
\left(\frac{\partial L}{\partial \mathbf{W}}\right)_{ij} = \frac{\partial L}{\partial W_{ij}}, \qquad \frac{\partial L}{\partial \mathbf{W}} \text{ has shape } m\times n.
$$

Concretely, for a $2\times 2$ weight matrix,

$$
\mathbf{W} = \begin{bmatrix} W_{11} & W_{12} \\ W_{21} & W_{22} \end{bmatrix}
\quad\Longrightarrow\quad
\frac{\partial L}{\partial \mathbf{W}} = \begin{bmatrix} \dfrac{\partial L}{\partial W_{11}} & \dfrac{\partial L}{\partial W_{12}} \\[3mm] \dfrac{\partial L}{\partial W_{21}} & \dfrac{\partial L}{\partial W_{22}} \end{bmatrix}.
$$

There is one partial derivative per weight, laid out in the same grid as the weights. Nothing deep is happening — "gradient with respect to a matrix" is just "collect the partial derivative for each entry and keep them in the same arrangement."

### 2.2 Gradient with respect to an arbitrary tensor

The same idea extends to any shape. If $L$ depends on a tensor $T$ of shape, say, $(B, T, C)$ — a common Transformer activation shape — then $\partial L/\partial T$ is a tensor of shape $(B, T, C)$, with one partial derivative per element. Whatever the shape of the thing you differentiate against, the gradient wears the same shape.

<div class="callout key"><p><strong>Shape invariant:</strong> the gradient of a scalar $L$ with respect to any object (vector, matrix, or higher tensor) is an object of the <strong>same shape</strong>, holding one partial derivative per element. Scalar in the numerator, arbitrary shape in the denominator — the gradient copies the denominator's shape.</p></div>

## 3. What "gradient of the loss with respect to a parameter" means

Strip away the notation and here is the plain-English definition, which is worth stating exactly:

> For each individual entry of the parameter, the gradient records **how much the loss $L$ changes per unit increase in that entry, with every other entry held fixed** — evaluated at the current values.

So if $\big(\partial L/\partial \mathbf{W}\big)_{12} = -0.4$, it means: at the model's current weights, increasing $W_{12}$ by a tiny amount $\varepsilon$ decreases the loss by about $0.4\varepsilon$ (negative sign: increasing this weight *reduces* loss), while all other weights stay put. Do this bookkeeping for every entry and you have the full gradient. That is the entire content of "the gradient of the loss with respect to a parameter": a per-entry sensitivity of the one output number $L$ to each knob, holding the other knobs fixed.

Because each entry says "which way, and how fast, does $L$ move if I push this knob," the negative gradient tells the optimizer which way to push *every* knob to reduce the loss — which is why the update rule $\theta \leftarrow \theta - \eta\,\nabla_\theta L$ works on the whole parameter object at once.

## 4. A tiny neural-network example

Let us make it concrete with the smallest possible "network": one weight, one input, one target, and a squared-error loss. This is the atom every linear layer is built from.

### 4.1 Scalar version

Model a single prediction as $\hat y = w\,x$ (weight times input), with squared-error loss against a target $t$:

$$
L = (w\,x - t)^2.
$$

- $w$ — the (scalar) parameter we differentiate against.
- $x$ — the input (a fixed number for this example).
- $t$ — the target (also fixed).
- $L$ — the scalar loss.

To differentiate with respect to $w$, treat $x$ and $t$ as constants and use the chain rule from [lesson 19](lessons/module-07/lesson-01.md). The outer function is $(\cdot)^2$ with derivative $2(\cdot)$; the inner function is $wx - t$ with derivative (with respect to $w$) equal to $x$. So

$$
\frac{\partial L}{\partial w} = 2\,(w\,x - t)\cdot x.
$$

**Numbers.** Take $w = 2$, $x = 3$, $t = 5$. Forward: $\hat y = wx = 6$, residual $wx - t = 6 - 5 = 1$, loss $L = 1^2 = 1$. Gradient:

$$
\frac{\partial L}{\partial w} = 2\,(1)\,(3) = 6.
$$

Interpretation: nudging $w$ up by $\varepsilon$ raises the loss by about $6\varepsilon$ here — so to *reduce* the loss we should move $w$ *down*. The gradient's sign points uphill; we step against it.

### 4.2 Vector version

Now let the weight be a 2-vector $\mathbf{w} = (w_1, w_2)$ and the input a 2-vector $\mathbf{x} = (x_1, x_2)$, with the prediction being the dot product $\hat y = \mathbf{w}\cdot\mathbf{x} = w_1 x_1 + w_2 x_2$ and the same squared loss:

$$
L = (\mathbf{w}\cdot\mathbf{x} - t)^2.
$$

Now there are two knobs, so the gradient $\nabla_{\mathbf w} L$ is a 2-vector. Differentiate with respect to each component (the residual $r = \mathbf{w}\cdot\mathbf{x} - t$ has $\partial r/\partial w_i = x_i$):

$$
\frac{\partial L}{\partial w_1} = 2\,r\,x_1, \qquad \frac{\partial L}{\partial w_2} = 2\,r\,x_2
\quad\Longrightarrow\quad
\nabla_{\mathbf w} L = 2\,r\,\mathbf{x} = 2(\mathbf{w}\cdot\mathbf{x} - t)\,\mathbf{x}.
$$

The whole gradient is just the scalar $2r$ times the input vector $\mathbf{x}$ — and it is a 2-vector, the same shape as $\mathbf{w}$.

**Numbers.** Take $\mathbf{w} = (1, 2)$, $\mathbf{x} = (3, 4)$, $t = 10$. Forward: $\mathbf{w}\cdot\mathbf{x} = 1\cdot 3 + 2\cdot 4 = 3 + 8 = 11$, residual $r = 11 - 10 = 1$, loss $L = 1$. Gradient:

$$
\nabla_{\mathbf w} L = 2(1)\begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 6 \\ 8 \end{bmatrix}.
$$

Two numbers, one per weight, arranged like $\mathbf{w}$. Component $\partial L/\partial w_1 = 6$, component $\partial L/\partial w_2 = 8$: the loss is more sensitive to $w_2$ here, because its input $x_2 = 4$ is larger.

## 5. Why ML needs this: the shape invariant runs the optimizer

GPT-2's parameters are a zoo of shapes: 1-D bias vectors, 2-D weight matrices for every linear layer, a large 2-D embedding matrix of shape $(V, C)$, LayerNorm gain/bias vectors, and so on. The update rule is the same for all of them:

$$
\theta \leftarrow \theta - \eta\,\nabla_\theta L.
$$

For this subtraction to even type-check, $\nabla_\theta L$ must have exactly the same shape as $\theta$ — you can only subtract a matrix from a matrix, a vector from a vector, element by element. The shape invariant from section 2 is precisely what guarantees this. That is *why* PyTorch stores each parameter's gradient in a field `p.grad` living right next to `p` and shaped identically: the optimizer can then step every parameter with one elementwise formula, never needing to know whether a given parameter is a bias vector or a weight matrix. The invariant `p.grad.shape == p.shape` is the contract that makes generic optimizers possible.

## 6. In PyTorch

### 6.1 The gradient matches the parameter's shape — vector and matrix

First the vector case from section 4.2, then a $2\times 2$ matrix parameter, showing `.grad.shape == p.shape` both times.

```python
import torch

# --- vector parameter (section 4.2) ---
w = torch.tensor([1., 2.], requires_grad=True)
x = torch.tensor([3., 4.])          # input (no grad needed)
t = torch.tensor(10.)

pred = w @ x                        # dot product: 1*3 + 2*4 = 11.
L = (pred - t)**2                   # (11 - 10)^2 = 1.
L.backward()

print(w.grad)                       # tensor([6., 8.])  -> 2*r*x
print(w.grad.shape, w.shape)        # torch.Size([2]) torch.Size([2])  (equal)
```

`w.grad` is `[6., 8.]`, matching the hand computation, and its shape equals `w`'s. Now a matrix parameter. We reduce to a scalar with `.sum()` because `backward()` needs a scalar to start from (more on that in section 7):

```python
# --- 2x2 matrix parameter ---
W = torch.tensor([[1., 2.],
                  [3., 4.]], requires_grad=True)

L = (W**2).sum()                    # scalar: sum of squares = 1+4+9+16 = 30.
L.backward()

print(W.grad)                       # tensor([[2., 4.],
                                    #         [6., 8.]])   -> dL/dW_ij = 2*W_ij
print(W.grad.shape, W.shape)        # torch.Size([2, 2]) torch.Size([2, 2])
```

Here $L = \sum_{ij} W_{ij}^2$, so $\partial L/\partial W_{ij} = 2W_{ij}$, giving the gradient matrix $\begin{bmatrix} 2 & 4 \\ 6 & 8\end{bmatrix}$ — same $2\times 2$ shape as `W`, one partial per entry, exactly as section 2.1 said.

<div class="callout pt"><p><code>p.grad</code> is always allocated to match <code>p.shape</code> — vector, matrix, or higher tensor. This is the shape invariant made physical: the optimizer reads <code>p.grad</code> and updates <code>p</code> with one elementwise rule, regardless of the parameter's rank.</p></div>

## 7. What PyTorch is doing under the hood

Two under-the-hood facts follow directly from this lesson.

**`backward()` requires a scalar (or a supplied upstream gradient).** In every snippet above we called `backward()` on a single number — the loss, or a `.sum()`. That is not incidental. Backpropagation, as built in [lesson 19](lessons/module-07/lesson-01.md), seeds the sweep with $\frac{\partial L}{\partial L} = 1$, and "$= 1$" only makes sense when $L$ is *one* number. If you call `.backward()` on a non-scalar tensor, PyTorch raises `RuntimeError: grad can be implicitly created only for scalar outputs`, because there is no single "$1$" to seed with — it would need one seed per output element. You either reduce to a scalar first (as with `.sum()`), or hand `backward()` an explicit seed vector: `y.backward(v)`. That seed $v$ is the *upstream gradient*, and the operation "push $v$ back through the graph" is the **vector–Jacobian product** — the topic that the next two lessons build up formally. This is why a loss is always a scalar: it gives the backward pass its single starting number.

**`.grad` accumulates, matching the fan-out rule.** Each leaf's `.grad` starts at `None`; the first `backward()` writes the gradient into it; subsequent `backward()` calls *add* to it rather than overwrite. This is the same summation as the fan-out rule from [lesson 19](lessons/module-07/lesson-01.md) — a parameter used in several places receives one contribution per use, and they sum. It is deliberate (it enables gradient accumulation across mini-batches), but it means the training loop must call `optimizer.zero_grad()` to reset each `.grad` before the next backward pass, or this step's gradient would be polluted by the last. We assemble the full loop in [module 8](lessons/module-08/lesson-01.md).

## Check yourself

<details><summary>What is the difference between $\partial L/\partial x_3$ and $\nabla_x L$?</summary>

$\partial L/\partial x_3$ is a single number: the sensitivity of $L$ to the third component of $x$ alone. $\nabla_x L$ is the whole gradient — every partial $\partial L/\partial x_i$ packed into an object the same shape as $x$. The first is one entry of the second.

</details>

<details><summary>If a parameter is a matrix of shape $4\times 768$, what shape is its gradient, and how many individual partial derivatives does it contain?</summary>

Shape $4\times 768$ (same as the parameter), containing $4 \times 768 = 3072$ partial derivatives, one per weight.

</details>

<details><summary>For $L = (wx - t)^2$ with $w = 3$, $x = 2$, $t = 5$, compute $\partial L/\partial w$.</summary>

Residual $wx - t = 6 - 5 = 1$. $\partial L/\partial w = 2(wx - t)x = 2(1)(2) = 4$.

</details>

<details><summary>Why does <code>loss.backward()</code> work but <code>activations.backward()</code> (on a non-scalar) raise an error?</summary>

Backprop seeds the sweep with $\partial L/\partial L = 1$, which requires a single output number. A non-scalar has no single "$1$" to start from; you must reduce it to a scalar or pass an explicit upstream-gradient vector to <code>backward()</code>. That vector is what makes it a vector–Jacobian product.

</details>

## Next

You can now say precisely what a gradient is for a vector, a matrix, or a tensor, and why its shape always matches the parameter. But so far the *output* has always been a single scalar. What if a function outputs a whole vector — like a linear layer, or softmax? Then one derivative is not enough; we need a grid of partials, one per (output, input) pair. That grid is the Jacobian.

Continue to [21 · Jacobians](lessons/module-07/lesson-03.md).
