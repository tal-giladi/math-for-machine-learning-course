# 23 · The linear layer, forward and backward by hand

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-07/lesson-01">19 · The chain rule as a graph</a>, <a href="#/lessons/module-07/lesson-03">21 · Vector-Jacobian products</a>, and <a href="#/lessons/module-07/lesson-04">22 · Backprop identities for a linear layer</a> (the four gradient rules we will reuse). You also need the sigmoid and its derivative from <a href="#/lessons/module-06/lesson-03">17 · Derivatives of exp, log, sigmoid, softmax</a>, and matrix-vector products from <a href="#/lessons/module-03/lesson-01">the matrices module</a>.</p>
<p><strong>You will learn:</strong> how to run an entire neural network — a 2→2→1 network with one hidden layer — <strong>forward and backward, by hand</strong>, computing every activation and every parameter gradient with real numbers, and then how to reproduce the exact same numbers in PyTorch with <code>loss.backward()</code>.</p>
<p><strong>Why this matters for ML:</strong> this is the milestone the last three modules were building toward. Everything GPT-2 does in training is this same forward-then-backward loop, just with more layers and bigger matrices. Once you have done it once by hand and watched PyTorch agree to four decimal places, <code>.backward()</code> stops being magic forever.</p>
</div>

## 1. The whole network on one page

We are going to build the smallest network that still has every moving part of a real one: an input, a hidden layer with a nonlinearity, an output layer, and a loss. Two inputs, two hidden neurons, one output. That is why it is called a **2→2→1** network.

Here is the complete pipeline, top to bottom:

```text
x            (2 inputs)
  |   z1 = W1 @ x + b1        linear layer 1
  v
z1           (2 pre-activations)
  |   a1 = sigmoid(z1)        nonlinearity
  v
a1           (2 hidden activations)
  |   z2 = W2 @ a1 + b2       linear layer 2
  v
y = z2       (1 output, no activation)
  |   L = (y - t)^2           squared-error loss
  v
L            (1 number)
```

Every arrow is a function. The whole network is one big composition of functions — exactly the picture from [module 5](lessons/module-05/lesson-01.md) — and the loss $L$ is a single number at the end that says how wrong we are. Training will eventually nudge the parameters to make $L$ smaller, but first we must be able to (a) compute $L$ from the inputs (the **forward pass**) and (b) compute the gradient of $L$ with respect to every parameter (the **backward pass**). This lesson does both, by hand.

### 1.1 The exact numbers

So that you can check every digit — and so PyTorch produces the identical output — we fix concrete values. These never change through the lesson:

| Symbol | Meaning | Shape | Value |
|---|---|---|---|
| $\mathbf{x}$ | input | $(2,)$ | $[1.0,\ 2.0]$ |
| $\mathbf{W}_1$ | layer-1 weights | $(2,2)$ | $\begin{bmatrix}0.10 & 0.20\\ 0.30 & 0.40\end{bmatrix}$ |
| $\mathbf{b}_1$ | layer-1 bias | $(2,)$ | $[0.10,\ 0.20]$ |
| $\mathbf{W}_2$ | layer-2 weights | $(1,2)$ | $[0.50,\ 0.60]$ |
| $b_2$ | layer-2 bias | scalar | $0.10$ |
| $t$ | target | scalar | $1.0$ |

The activation on the hidden layer is the **sigmoid**, $\sigma(u) = \dfrac{1}{1 + e^{-u}}$, from [lesson 17](lessons/module-06/lesson-03.md). The output has no activation (a "linear" output). The loss is the squared error $L = (y - t)^2$; we will see in section 6 that this matches `torch.nn.MSELoss` on a single element, and that its derivative is $\dfrac{\partial L}{\partial y} = 2(y - t)$.

## 2. Forward pass, by hand

### 2.1 Layer 1: the linear part

A **linear layer** computes $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$: multiply the input vector by a weight matrix and add a bias vector. Here

$$
\mathbf{z}_1 = \mathbf{W}_1\mathbf{x} + \mathbf{b}_1.
$$

- $\mathbf{W}_1$ is $(2,2)$ and $\mathbf{x}$ is $(2,)$, so $\mathbf{W}_1\mathbf{x}$ is $(2,)$; adding $\mathbf{b}_1$ (also $(2,)$) keeps it $(2,)$.
- Each output entry is a dot product of one row of $\mathbf{W}_1$ with $\mathbf{x}$, plus the matching bias.

Row by row:

$$
z_{1,0} = (0.10)(1.0) + (0.20)(2.0) + 0.10 = 0.10 + 0.40 + 0.10 = 0.60,
$$
$$
z_{1,1} = (0.30)(1.0) + (0.40)(2.0) + 0.20 = 0.30 + 0.80 + 0.20 = 1.30.
$$

So $\mathbf{z}_1 = [0.60,\ 1.30]$. These are the **pre-activations**: the raw scores before the nonlinearity.

### 2.2 Layer 1: the sigmoid

Now squash each pre-activation through the sigmoid $\sigma(u) = 1/(1 + e^{-u})$, element by element:

$$
a_{1,0} = \sigma(0.60) = \frac{1}{1 + e^{-0.60}} = \frac{1}{1 + 0.548812} = \frac{1}{1.548812} = 0.645656,
$$
$$
a_{1,1} = \sigma(1.30) = \frac{1}{1 + e^{-1.30}} = \frac{1}{1 + 0.272532} = \frac{1}{1.272532} = 0.785835.
$$

So $\mathbf{a}_1 = [0.6457,\ 0.7858]$ (keeping four decimals). These are the **hidden activations** — the network's internal representation of the input.

<div class="callout warn"><p>Keep several decimals through the whole computation and only round at the end. If you round $\mathbf{a}_1$ to two decimals here, the error compounds and your final gradients will disagree with PyTorch in the third decimal place. Carry six figures; report four.</p></div>

### 2.3 Layer 2 and the output

The second linear layer collapses the two hidden activations into one number:

$$
z_2 = \mathbf{W}_2\mathbf{a}_1 + b_2 = (0.50)(0.645656) + (0.60)(0.785835) + 0.10.
$$

Compute the two products:

- $(0.50)(0.645656) = 0.322828$,
- $(0.60)(0.785835) = 0.471501$.

Add them with the bias:

$$
z_2 = 0.322828 + 0.471501 + 0.10 = 0.894329.
$$

There is no activation on the output, so the prediction is $y = z_2 = 0.894329$.

### 2.4 The loss

$$
L = (y - t)^2 = (0.894329 - 1.0)^2 = (-0.105671)^2 = 0.011166.
$$

The forward pass is done. In summary:

| Quantity | Value |
|---|---|
| $\mathbf{z}_1$ | $[0.6000,\ 1.3000]$ |
| $\mathbf{a}_1$ | $[0.6457,\ 0.7858]$ |
| $y = z_2$ | $0.8943$ |
| $L$ | $0.0112$ |

## 3. The backward pass: the plan

The backward pass computes $\dfrac{\partial L}{\partial(\text{each parameter})}$. We do it by walking the pipeline **in reverse**, carrying a running derivative of $L$ with respect to whatever quantity we are currently standing on. That running quantity — "$\partial L$ divided by the thing here" — is called the **upstream gradient**, and each layer's job is to turn the upstream gradient on its output into the upstream gradient on its input, picking off the parameter gradients along the way. This is exactly the chain-rule-over-a-graph procedure from [lesson 19](lessons/module-07/lesson-01.md).

We reuse four identities, each derived in [lesson 22](lessons/module-07/lesson-04.md). For a linear layer $\mathbf{z} = \mathbf{W}\mathbf{a} + \mathbf{b}$ with upstream gradient $\dfrac{\partial L}{\partial \mathbf{z}}$:

$$
\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{z}}\,\mathbf{a}^\top, \qquad
\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}}, \qquad
\frac{\partial L}{\partial \mathbf{a}} = \mathbf{W}^\top\frac{\partial L}{\partial \mathbf{z}}.
$$

And for the element-wise sigmoid $\mathbf{a} = \sigma(\mathbf{z})$, the upstream gradient passes through by multiplying element-wise with the sigmoid's local derivative $\sigma'(z) = \sigma(z)\,(1 - \sigma(z))$ = $a\,(1-a)$:

$$
\frac{\partial L}{\partial \mathbf{z}} = \frac{\partial L}{\partial \mathbf{a}} \odot \mathbf{a}\odot(1 - \mathbf{a}),
$$

where $\odot$ is the element-wise (Hadamard) product. The **shape rule** from [lesson 18](lessons/module-06/lesson-04.md) will be our safety check: every gradient has the same shape as the thing it is the gradient of. $\dfrac{\partial L}{\partial \mathbf{W}_1}$ is $(2,2)$, $\dfrac{\partial L}{\partial b_2}$ is a scalar, and so on.

## 4. Backward pass, by hand

### 4.1 Start at the loss

The very first upstream gradient is the derivative of the loss with respect to the prediction. With $L = (y-t)^2$, the power and chain rules give

$$
\frac{\partial L}{\partial y} = 2(y - t) = 2(0.894329 - 1.0) = 2(-0.105671) = -0.211342.
$$

Because the output has no activation, $y = z_2$, so the gradient flows straight through:

$$
\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial y} = -0.211342.
$$

The sign is meaningful: it is negative because $y$ is *below* the target, so increasing $y$ would decrease the loss.

### 4.2 Gradients of the output layer ($\mathbf{W}_2, b_2$)

Apply the three linear-layer identities with $\mathbf{a} = \mathbf{a}_1$, $\dfrac{\partial L}{\partial z_2} = -0.211342$.

**Weights** $\dfrac{\partial L}{\partial \mathbf{W}_2} = \dfrac{\partial L}{\partial z_2}\,\mathbf{a}_1^\top$. Since $z_2$ is a scalar this is just the scalar times the activation vector:

$$
\frac{\partial L}{\partial \mathbf{W}_2} = -0.211342 \times [0.645656,\ 0.785835] = [-0.136454,\ -0.166080].
$$

Shape $(1,2)$ — matches $\mathbf{W}_2$. ✓

**Bias** $\dfrac{\partial L}{\partial b_2} = \dfrac{\partial L}{\partial z_2} = -0.211342$. A scalar — matches $b_2$. ✓

**Down to the hidden activations** $\dfrac{\partial L}{\partial \mathbf{a}_1} = \mathbf{W}_2^\top\dfrac{\partial L}{\partial z_2}$. With $\mathbf{W}_2 = [0.50, 0.60]$:

$$
\frac{\partial L}{\partial \mathbf{a}_1} = [0.50,\ 0.60] \times (-0.211342) = [-0.105671,\ -0.126805].
$$

Shape $(2,)$ — matches $\mathbf{a}_1$. ✓ This is the upstream gradient we hand to the sigmoid.

### 4.3 Back through the sigmoid

The local derivative of the sigmoid at each hidden unit uses $\sigma'(z) = a(1-a)$, and we already have $\mathbf{a}_1$:

$$
a_{1,0}(1 - a_{1,0}) = 0.645656 \times (1 - 0.645656) = 0.645656 \times 0.354344 = 0.228784,
$$
$$
a_{1,1}(1 - a_{1,1}) = 0.785835 \times (1 - 0.785835) = 0.785835 \times 0.214165 = 0.168298.
$$

Multiply the upstream gradient element-wise by these local derivatives:

$$
\frac{\partial L}{\partial z_{1,0}} = (-0.105671)(0.228784) = -0.024176,
$$
$$
\frac{\partial L}{\partial z_{1,1}} = (-0.126805)(0.168298) = -0.021341.
$$

So $\dfrac{\partial L}{\partial \mathbf{z}_1} = [-0.024176,\ -0.021341]$. Shape $(2,)$ — matches $\mathbf{z}_1$. ✓

### 4.4 Gradients of the first layer ($\mathbf{W}_1, \mathbf{b}_1$)

Now apply the linear-layer identities again, this time with input $\mathbf{x}$ and upstream $\dfrac{\partial L}{\partial \mathbf{z}_1}$.

**Weights** $\dfrac{\partial L}{\partial \mathbf{W}_1} = \dfrac{\partial L}{\partial \mathbf{z}_1}\,\mathbf{x}^\top$. This is an **outer product**: a $(2,)$ column times a $(2,)$ row gives a $(2,2)$ matrix. Entry $(i, j)$ is $\dfrac{\partial L}{\partial z_{1,i}}\, x_j$. With $\mathbf{x} = [1.0, 2.0]$:

$$
\frac{\partial L}{\partial \mathbf{W}_1} =
\begin{bmatrix}
(-0.024176)(1.0) & (-0.024176)(2.0)\\
(-0.021341)(1.0) & (-0.021341)(2.0)
\end{bmatrix}
=
\begin{bmatrix}
-0.024176 & -0.048352\\
-0.021341 & -0.042682
\end{bmatrix}.
$$

Shape $(2,2)$ — matches $\mathbf{W}_1$. ✓

**Bias** $\dfrac{\partial L}{\partial \mathbf{b}_1} = \dfrac{\partial L}{\partial \mathbf{z}_1} = [-0.024176,\ -0.021341]$. Shape $(2,)$ — matches $\mathbf{b}_1$. ✓

### 4.5 Gradient with respect to the input

We do not train the input, but the same rule gives it, and it is what would flow to an earlier layer in a deeper network: $\dfrac{\partial L}{\partial \mathbf{x}} = \mathbf{W}_1^\top\dfrac{\partial L}{\partial \mathbf{z}_1}$. With $\mathbf{W}_1^\top = \begin{bmatrix}0.10 & 0.30\\ 0.20 & 0.40\end{bmatrix}$:

$$
\frac{\partial L}{\partial x_0} = (0.10)(-0.024176) + (0.30)(-0.021341) = -0.002418 - 0.006402 = -0.008820,
$$
$$
\frac{\partial L}{\partial x_1} = (0.20)(-0.024176) + (0.40)(-0.021341) = -0.004835 - 0.008536 = -0.013372.
$$

So $\dfrac{\partial L}{\partial \mathbf{x}} = [-0.008820,\ -0.013372]$. Shape $(2,)$ — matches $\mathbf{x}$. ✓

### 4.6 Every gradient in one table

| Gradient | Shape | Value |
|---|---|---|
| $\partial L/\partial y$ | scalar | $-0.2113$ |
| $\partial L/\partial \mathbf{W}_2$ | $(1,2)$ | $[-0.1365,\ -0.1661]$ |
| $\partial L/\partial b_2$ | scalar | $-0.2113$ |
| $\partial L/\partial \mathbf{a}_1$ | $(2,)$ | $[-0.1057,\ -0.1268]$ |
| $\partial L/\partial \mathbf{z}_1$ | $(2,)$ | $[-0.02418,\ -0.02134]$ |
| $\partial L/\partial \mathbf{W}_1$ | $(2,2)$ | $\begin{bmatrix}-0.02418 & -0.04835\\ -0.02134 & -0.04268\end{bmatrix}$ |
| $\partial L/\partial \mathbf{b}_1$ | $(2,)$ | $[-0.02418,\ -0.02134]$ |
| $\partial L/\partial \mathbf{x}$ | $(2,)$ | $[-0.00882,\ -0.01337]$ |

<div class="callout key"><p>The backward pass is just the chain rule applied layer by layer, right to left. Each layer receives the gradient of the loss on its output, and returns the gradient on its input — producing its own parameter gradients on the way. Every gradient has the same shape as the thing it differentiates. Nothing here is more than multiplication and addition.</p></div>

## 5. Why ML needs exactly this

A GPT-2 forward pass is this same picture with 12 Transformer blocks instead of one hidden layer, matrices of size $768 \times 768$ instead of $2 \times 2$, and a cross-entropy loss instead of squared error. But the skeleton is identical: **linear layers separated by nonlinearities, ending in a scalar loss**, run forward to get $L$, then run backward to get $\partial L / \partial \theta$ for all 124 million parameters $\theta$. The four identities you used above are the *only* per-layer rules backprop needs; a deep network just chains more of them. When you later read a training loop and see `loss.backward()`, that one call is doing section 4 for the entire model — the same arithmetic, at scale.

## 6. In PyTorch

Now we set up the identical network in PyTorch, run it forward, call `.backward()`, and confirm every gradient matches the hand computation. We build it from raw parameter tensors (rather than `nn.Linear`) so the mapping to the math is one-to-one; `nn.Linear` would store exactly these same `weight` and `bias` tensors internally.

```python
import torch

# Inputs and target (not trained -> no requires_grad needed here)
x  = torch.tensor([1.0, 2.0])
t  = torch.tensor([1.0])

# Parameters, set to the SAME values as the by-hand example
W1 = torch.tensor([[0.10, 0.20],
                   [0.30, 0.40]], requires_grad=True)   # (2, 2)
b1 = torch.tensor([0.10, 0.20],  requires_grad=True)    # (2,)
W2 = torch.tensor([[0.50, 0.60]], requires_grad=True)   # (1, 2)
b2 = torch.tensor([0.10],         requires_grad=True)   # (1,)

# ---- forward pass ----
z1 = W1 @ x + b1          # (2,)   -> [0.6000, 1.3000]
a1 = torch.sigmoid(z1)    # (2,)   -> [0.6457, 0.7858]
z2 = W2 @ a1 + b2         # (1,)   -> [0.8943]
y  = z2
L  = ((y - t) ** 2).sum() # scalar -> 0.0112

# ---- backward pass ----
L.backward()

print("L      :", L.item())      # 0.011166
print("W2.grad:", W2.grad)       # [[-0.1365, -0.1661]]
print("b2.grad:", b2.grad)       # [-0.2113]
print("W1.grad:", W1.grad)       # [[-0.0242, -0.0484], [-0.0213, -0.0427]]
print("b1.grad:", b1.grad)       # [-0.0242, -0.0213]
```

`((y - t) ** 2).sum()` on a one-element tensor is exactly $(y-t)^2$, which is what `torch.nn.MSELoss()` computes for a single element (its default reduction is the mean, and the mean of one number is that number). So `torch.nn.MSELoss()(y, t)` would give the same `L` and the same gradients.

To also get $\partial L / \partial \mathbf{x}$, ask PyTorch to track the input too:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
# ... rerun the forward pass and L.backward() ...
print("x.grad :", x.grad)        # [-0.0088, -0.0134]
```

### 6.1 Hand vs PyTorch

Every printed number matches section 4 to four decimals:

| Gradient | By hand | PyTorch |
|---|---|---|
| $\partial L/\partial \mathbf{W}_2$ | $[-0.1365,\ -0.1661]$ | `[[-0.1365, -0.1661]]` |
| $\partial L/\partial b_2$ | $-0.2113$ | `[-0.2113]` |
| $\partial L/\partial \mathbf{W}_1$ | $\begin{bmatrix}-0.0242 & -0.0484\\ -0.0213 & -0.0427\end{bmatrix}$ | `[[-0.0242, -0.0484], [-0.0213, -0.0427]]` |
| $\partial L/\partial \mathbf{b}_1$ | $[-0.0242,\ -0.0213]$ | `[-0.0242, -0.0213]` |
| $\partial L/\partial \mathbf{x}$ | $[-0.0088,\ -0.0134]$ | `[-0.0088, -0.0134]` |

They agree because PyTorch is running *the same chain rule you just ran by hand* — not an approximation, the exact analytic derivatives.

## 7. What PyTorch is doing under the hood

Three things happened, and each has a name you will meet constantly.

**Leaf tensors.** The four tensors you created with `requires_grad=True` (`W1, b1, W2, b2`) are **leaf tensors**: parameters at the edge of the graph, not computed from anything else. They are the only tensors that get a `.grad` populated by default. Intermediate results like `z1` and `a1` are non-leaf; PyTorch computes gradients *through* them but does not keep those gradients unless you ask (that is `retain_grad`, covered in [lesson 25](lessons/module-08/lesson-03.md)).

**The graph.** As the forward pass runs, each operation records a node — `W1 @ x` records a matmul node, `torch.sigmoid(z1)` a sigmoid node, and so on. Each node remembers its inputs and how to compute its local derivative. By the time `L` exists, PyTorch holds a full **computational graph** from the leaves up to `L`. You can see the last node via `L.grad_fn` (something like `<SumBackward0>`).

**`.backward()` fills `.grad`.** Calling `L.backward()` walks that graph from `L` down to the leaves, applying exactly the identities from section 3 at each node and multiplying the results together (the chain rule). When it reaches each leaf, it stores the accumulated derivative in that leaf's `.grad` slot. So `W1.grad` is literally $\partial L / \partial \mathbf{W}_1$, sitting in a tensor of the same shape as `W1` — the shape rule made physical. This is precisely the by-hand backward pass of section 4, automated. We open up the graph, `grad_fn`, and gradient accumulation in full in [lesson 25](lessons/module-08/lesson-03.md).

## Check yourself

<details><summary>Why does <code>a1.grad</code> print <code>None</code> after <code>L.backward()</code>, even though we clearly computed $\partial L/\partial \mathbf{a}_1 = [-0.1057, -0.1268]$ along the way?</summary>

Because `a1` is a **non-leaf** tensor (it was computed from `z1`). PyTorch computes the gradient through it internally but only *stores* gradients on leaf tensors by default, to save memory. If you want it kept, call `a1.retain_grad()` before `backward()`.

</details>

<details><summary>The hand computation gave $\partial L/\partial b_2 = \partial L/\partial z_2$. Why are the bias gradient and the pre-activation gradient identical?</summary>

Because $z_2 = \mathbf{W}_2\mathbf{a}_1 + b_2$, and $\partial z_2 / \partial b_2 = 1$. Adding the bias shifts $z_2$ one-for-one, so the loss's sensitivity to $b_2$ equals its sensitivity to $z_2$. This is the general rule $\partial L/\partial \mathbf{b} = \partial L/\partial \mathbf{z}$.

</details>

<details><summary>All four parameter gradients came out negative. What does that tell you about how to change the parameters to reduce the loss?</summary>

A negative $\partial L/\partial \theta$ means increasing that parameter *decreases* the loss. Gradient descent steps against the gradient ($\theta \leftarrow \theta - \eta\,\partial L/\partial\theta$), so with negative gradients every parameter would be nudged *up* slightly — which makes sense, since $y = 0.894$ is below the target $1.0$ and we want to raise it.

</details>

<details><summary>If you changed the output activation from "none" to sigmoid, which single step of the backward pass would change first?</summary>

Section 4.1: instead of $\partial L/\partial z_2 = \partial L/\partial y$, you would multiply by the sigmoid's local derivative, $\partial L/\partial z_2 = \partial L/\partial y \cdot y(1-y)$. Everything downstream (the $\mathbf{W}_2, b_2, \mathbf{a}_1$ gradients) would then use this new $\partial L/\partial z_2$.

</details>

## Next

You have now run a complete network forward and backward by hand and watched PyTorch reproduce it exactly. The one piece we treated as given is the loss and its derivative $\partial L/\partial y$. Real models — especially language models — use richer losses, and the next lesson builds them from scratch: MSE, MAE, and then the classification losses (softmax, cross-entropy, NLL, BCE) that turn logits into a single training signal.

Continue to [24 · Loss functions](lessons/module-08/lesson-02.md).
