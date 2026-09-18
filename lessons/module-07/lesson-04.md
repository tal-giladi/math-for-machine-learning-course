# 22 · Matrix calculus and VJPs

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-03.md">21 · Jacobians</a> (the Jacobian, the linear map's Jacobian $W$, the chain rule as a Jacobian product), <a href="lesson-02.md">20 · Gradients with respect to matrices and tensors</a> (the shape invariant), and matrix multiplication and transpose from <a href="../module-03/lesson-03.md">module 3</a>.</p>
<p><strong>You will learn:</strong> the derivative of a scalar with respect to a vector (a gradient vector) and with respect to a matrix (a gradient matrix); the <strong>vector–Jacobian product (VJP)</strong> and <strong>Jacobian–vector product (JVP)</strong>, and why reverse-mode uses the VJP; and the three workhorse identities for $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ — with every shape checked and a full numeric example.</p>
<p><strong>Why this matters for ML:</strong> the linear layer $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ is the most common operation in GPT-2, and its three backward formulas ($\partial L/\partial\mathbf{x} = \mathbf{W}^\top\mathbf{g}$, $\partial L/\partial\mathbf{W} = \mathbf{g}\mathbf{x}^\top$, $\partial L/\partial\mathbf{b} = \mathbf{g}$) are literally what PyTorch's <code>Linear</code> backward computes. This lesson is the direct on-ramp to module 8.</p>
</div>

## 1. Motivation: don't panic about "matrix calculus"

"Matrix calculus" sounds forbidding, but you already have every piece. We are only combining two ideas you have met:

1. **The shape invariant** ([lesson 20](lesson-02.md)): the derivative of a *scalar* with respect to some object is another object of the *same shape*.
2. **The chain rule as a Jacobian product** ([lesson 21](lesson-03.md)): compositions multiply Jacobians, and because the loss is scalar we only ever push a *vector* back through each Jacobian.

Put together, matrix calculus for neural networks reduces to a short list of "if the loss is scalar and this operation happened in the forward pass, here is how the gradient passes through it." We will derive that list for the linear layer and check every shape. No new theory — just careful bookkeeping.

## 2. Derivative of a scalar with respect to a vector and a matrix

Recall the two objects from [lesson 20](lesson-02.md), now named as the building blocks of matrix calculus.

**Scalar with respect to a vector = gradient vector.** If $L$ is a scalar and $\mathbf{x}$ is an $n$-vector, then $\dfrac{\partial L}{\partial\mathbf{x}}$ is the $n$-vector with entries $\big(\partial L/\partial\mathbf{x}\big)_i = \partial L/\partial x_i$. Same shape as $\mathbf{x}$.

**Scalar with respect to a matrix = gradient matrix.** If $L$ is a scalar and $\mathbf{W}$ is $m\times n$, then $\dfrac{\partial L}{\partial\mathbf{W}}$ is the $m\times n$ matrix with entries $\big(\partial L/\partial\mathbf{W}\big)_{ij} = \partial L/\partial W_{ij}$. Same shape as $\mathbf{W}$.

Both are just "one partial derivative per element, arranged like the element." We will keep the convention — used throughout ML and by PyTorch — that **the gradient of a scalar has the same shape as the thing it is taken against**. This is what makes $\theta \leftarrow \theta - \eta\,\nabla_\theta L$ type-check for every parameter.

## 3. VJP and JVP: two ways to touch a Jacobian

A Jacobian $J$ (shape $m\times n$) can be multiplied by a vector on either side, and the two products have very different roles.

**Jacobian–vector product (JVP):** $J\,\mathbf{u}$, where $\mathbf{u}$ is an $n$-vector (input-sized). Result: an $m$-vector (output-sized). It answers "if I perturb the input by direction $\mathbf{u}$, how does the output move?" — pushing a perturbation *forward*. This is **forward-mode** differentiation.

**Vector–Jacobian product (VJP):** $\mathbf{v}^\top J$, where $\mathbf{v}$ is an $m$-vector (output-sized). Result: an $n$-vector (input-sized). It answers "given a sensitivity $\mathbf{v}$ of the loss to the output, what is the sensitivity to the input?" — pulling a gradient *backward*. This is **reverse-mode** differentiation, i.e. backprop. (We write $\mathbf{v}^\top J$; as a column vector the same quantity is $J^\top\mathbf{v}$, which is how code usually spells it.)

<div class="callout key"><p><strong>JVP</strong> $J\mathbf{u}$ pushes an input perturbation forward (input-sized in, output-sized out). <strong>VJP</strong> $\mathbf{v}^\top J$ pulls an output gradient backward (output-sized in, input-sized out). Backprop is a chain of VJPs.</p></div>

### 3.1 Why reverse-mode uses the VJP

The loss $L$ is a single scalar. So at the very top of the backward pass the "sensitivity of the loss to the network's output" is a single vector $\mathbf{g} = \partial L/\partial(\text{output})$ — one number per output component. A VJP takes exactly such an output-sized vector and returns an input-sized one, so we can feed $\mathbf{g}$ into the last layer's VJP, get the gradient with respect to that layer's input, feed *that* into the previous layer's VJP, and so on back to the parameters. Every step stays vector-sized; no Jacobian is ever built.

Forward-mode (JVPs) would instead propagate one input perturbation at a time forward, which is efficient when there are few inputs and many outputs — the opposite of a neural network, where there are millions of inputs (parameters) and one output (the loss). That asymmetry is the whole reason training uses reverse-mode: one scalar output, so one VJP sweep computes all input gradients at once. (We revisit forward vs reverse mode in the automatic-differentiation module.)

## 4. The linear layer: the three workhorse identities

Now the centerpiece. A linear layer computes

$$
\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b},
$$

with these shapes:

- $\mathbf{x}$ — input vector, shape $(n,)$.
- $\mathbf{W}$ — weight matrix, shape $(m, n)$.
- $\mathbf{b}$ — bias vector, shape $(m,)$.
- $\mathbf{y}$ — output (pre-activation) vector, shape $(m,)$.

During backprop, this layer receives from downstream the **upstream gradient**

$$
\mathbf{g} = \frac{\partial L}{\partial\mathbf{y}}, \qquad \text{shape } (m,) \text{ — same as } \mathbf{y}.
$$

$\mathbf{g}$ is "how much the loss cares about each output component of this layer." The layer must produce three gradients: with respect to its input $\mathbf{x}$ (to pass further back), and with respect to its parameters $\mathbf{W}$ and $\mathbf{b}$ (to update them). Here are the identities; derivations follow.

$$
\boxed{\;\frac{\partial L}{\partial\mathbf{x}} = \mathbf{W}^\top\mathbf{g}\;}\qquad
\boxed{\;\frac{\partial L}{\partial\mathbf{W}} = \mathbf{g}\,\mathbf{x}^\top\;}\qquad
\boxed{\;\frac{\partial L}{\partial\mathbf{b}} = \mathbf{g}\;}
$$

### 4.1 Gradient with respect to the input: $\partial L/\partial\mathbf{x} = \mathbf{W}^\top\mathbf{g}$

The Jacobian of $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ with respect to $\mathbf{x}$ is $\mathbf{W}$ (the linear-map result from [lesson 21](lesson-03.md); the $+\mathbf{b}$ adds a constant and does not affect the derivative). The chain rule (VJP) says $\partial L/\partial\mathbf{x} = \mathbf{v}^\top J$ with $\mathbf{v} = \mathbf{g}$ and $J = \mathbf{W}$, i.e. as a column vector $\mathbf{W}^\top\mathbf{g}$. Component-wise, since $y_i = \sum_k W_{ik}x_k + b_i$, input $x_j$ influences every output $y_i$ through $W_{ij}$, so their contributions **sum** (the fan-out rule from [lesson 19](lesson-01.md)):

$$
\frac{\partial L}{\partial x_j} = \sum_{i=1}^{m}\frac{\partial L}{\partial y_i}\frac{\partial y_i}{\partial x_j} = \sum_{i=1}^{m} g_i\,W_{ij} = (\mathbf{W}^\top\mathbf{g})_j.
$$

**Shape check:** $\mathbf{W}^\top$ is $(n, m)$, $\mathbf{g}$ is $(m,)$, product is $(n,)$ — same shape as $\mathbf{x}$. ✓

### 4.2 Gradient with respect to the weights: $\partial L/\partial\mathbf{W} = \mathbf{g}\,\mathbf{x}^\top$

Weight $W_{ij}$ enters only output $y_i$, and there it multiplies $x_j$: $\partial y_i/\partial W_{ij} = x_j$ (and $\partial y_k/\partial W_{ij} = 0$ for $k\ne i$). So

$$
\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial y_i}\frac{\partial y_i}{\partial W_{ij}} = g_i\,x_j.
$$

The matrix whose $(i,j)$ entry is $g_i x_j$ is the **outer product** $\mathbf{g}\,\mathbf{x}^\top$ (column vector times row vector).

**Shape check:** $\mathbf{g}$ is $(m,1)$, $\mathbf{x}^\top$ is $(1,n)$, outer product is $(m,n)$ — same shape as $\mathbf{W}$. ✓ (This is the shape invariant doing its job: the gradient of the scalar $L$ with respect to the $m\times n$ matrix $\mathbf{W}$ is itself $m\times n$.)

### 4.3 Gradient with respect to the bias: $\partial L/\partial\mathbf{b} = \mathbf{g}$

Bias $b_i$ enters only output $y_i$, added directly: $\partial y_i/\partial b_i = 1$. So $\partial L/\partial b_i = g_i\cdot 1 = g_i$, i.e.

$$
\frac{\partial L}{\partial\mathbf{b}} = \mathbf{g}.
$$

**Shape check:** $\mathbf{g}$ is $(m,)$ — same shape as $\mathbf{b}$. ✓ The bias gradient is simply the upstream gradient, unchanged.

<div class="callout key"><p>For $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ with upstream $\mathbf{g} = \partial L/\partial\mathbf{y}$: $\;\partial L/\partial\mathbf{x} = \mathbf{W}^\top\mathbf{g}$ (pull the gradient back through the weights), $\;\partial L/\partial\mathbf{W} = \mathbf{g}\mathbf{x}^\top$ (outer product of upstream with input), $\;\partial L/\partial\mathbf{b} = \mathbf{g}$. Memorize these three — they are 90% of backprop.</p></div>

## 5. A tiny numeric example, all by hand

Fix concrete values:

$$
\mathbf{W} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
\mathbf{x} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}, \quad
\mathbf{b} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}.
$$

**Forward pass.** $\mathbf{W}\mathbf{x} = \begin{bmatrix} 1\cdot 2 + 2\cdot 3 \\ 3\cdot 2 + 4\cdot 3 \end{bmatrix} = \begin{bmatrix} 8 \\ 18 \end{bmatrix}$, then add $\mathbf{b}$:

$$
\mathbf{y} = \begin{bmatrix} 8 \\ 18 \end{bmatrix} + \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 9 \\ 19 \end{bmatrix}.
$$

Suppose the downstream loss is $L = y_1 + 2y_2$ (so $L = 9 + 38 = 47$). Then the upstream gradient is

$$
\mathbf{g} = \frac{\partial L}{\partial\mathbf{y}} = \begin{bmatrix} \partial L/\partial y_1 \\ \partial L/\partial y_2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}.
$$

**Backward pass — apply the three identities.**

Input gradient, $\mathbf{W}^\top\mathbf{g}$ with $\mathbf{W}^\top = \begin{bmatrix} 1 & 3 \\ 2 & 4\end{bmatrix}$:

$$
\frac{\partial L}{\partial\mathbf{x}} = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}\begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1\cdot 1 + 3\cdot 2 \\ 2\cdot 1 + 4\cdot 2 \end{bmatrix} = \begin{bmatrix} 7 \\ 10 \end{bmatrix}. \quad\text{shape }(2,)=\text{shape }\mathbf{x}\ \checkmark
$$

Weight gradient, outer product $\mathbf{g}\,\mathbf{x}^\top$ with $\mathbf{x}^\top = \begin{bmatrix} 2 & 3\end{bmatrix}$:

$$
\frac{\partial L}{\partial\mathbf{W}} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}\begin{bmatrix} 2 & 3 \end{bmatrix} = \begin{bmatrix} 1\cdot 2 & 1\cdot 3 \\ 2\cdot 2 & 2\cdot 3 \end{bmatrix} = \begin{bmatrix} 2 & 3 \\ 4 & 6 \end{bmatrix}. \quad\text{shape }(2,2)=\text{shape }\mathbf{W}\ \checkmark
$$

Bias gradient:

$$
\frac{\partial L}{\partial\mathbf{b}} = \mathbf{g} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}. \quad\text{shape }(2,)=\text{shape }\mathbf{b}\ \checkmark
$$

Spot-check one weight against first principles: $L = y_1 + 2y_2$ and $y_2 = 3x_1 + 4x_2 + b_2$, so $\partial L/\partial W_{21} = 2\cdot\partial y_2/\partial W_{21} = 2\cdot x_1 = 2\cdot 2 = 4$, matching the $(2,1)$ entry of $\partial L/\partial\mathbf{W}$. The identities are correct.

## 6. Why ML needs this

Every linear layer in GPT-2 — the query/key/value projections, the attention output projection, both matrices of every MLP block, and the final projection to logits — is exactly $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ (batched over positions). During training, each such layer, on the backward pass, does precisely three things using the identities above: compute $\mathbf{W}^\top\mathbf{g}$ to hand the gradient to the layer beneath it, compute $\mathbf{g}\mathbf{x}^\top$ to know how to update its own weights, and pass $\mathbf{g}$ through as the bias gradient. Chain these across all layers and you have the entire backward pass of a Transformer. Module 8 assembles a full network from this single brick; everything there is these three formulas applied layer after layer.

## 7. In PyTorch

Reproduce the section-5 example and confirm all three gradients. We build $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ manually and use the loss $L = y_1 + 2y_2$ so that $\mathbf{g} = [1, 2]$:

```python
import torch

W = torch.tensor([[1., 2.],
                  [3., 4.]], requires_grad=True)
x = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([1., 1.], requires_grad=True)

y = W @ x + b                      # forward: [9., 19.]
L = y[0] + 2*y[1]                  # scalar loss = 47.; dL/dy = [1., 2.]
L.backward()

print(x.grad)                      # tensor([ 7., 10.])   -> W^T g
print(W.grad)                      # tensor([[2., 3.],
                                   #         [4., 6.]])    -> g x^T (outer product)
print(b.grad)                      # tensor([1., 2.])      -> g
```

All three match the hand computation exactly: `x.grad = [7, 10]`, `W.grad = [[2,3],[4,6]]`, `b.grad = [1, 2]`. Now the same thing with `nn.Linear`, whose `.weight.grad` follows the identical rule:

```python
import torch, torch.nn as nn

layer = nn.Linear(2, 2)            # holds layer.weight (2x2) and layer.bias (2,)
with torch.no_grad():              # set the params to our numbers
    layer.weight.copy_(torch.tensor([[1., 2.], [3., 4.]]))
    layer.bias.copy_(torch.tensor([1., 1.]))

x = torch.tensor([2., 3.])
y = layer(x)                       # [9., 19.]
L = y[0] + 2*y[1]
L.backward()

print(layer.weight.grad)           # tensor([[2., 3.],
                                   #         [4., 6.]])    -> g x^T
print(layer.bias.grad)             # tensor([1., 2.])      -> g
```

`layer.weight.grad` is the same outer product $\mathbf{g}\mathbf{x}^\top$, and `layer.bias.grad` is $\mathbf{g}$ — `nn.Linear`'s backward is exactly our three identities.

<div class="callout pt"><p>The gradients PyTorch deposits in <code>weight.grad</code>, <code>bias.grad</code>, and the input's <code>.grad</code> are computed by the formulas $\mathbf{g}\mathbf{x}^\top$, $\mathbf{g}$, and $\mathbf{W}^\top\mathbf{g}$ — not by any numerical approximation. Each is a single matrix/vector operation, and each matches its parameter's shape.</p></div>

## 8. What PyTorch is doing under the hood

`nn.Linear`'s registered backward function *is* these three formulas. When autograd reaches the linear node during the backward sweep, it holds the upstream gradient $\mathbf{g} = \partial L/\partial\mathbf{y}$ (computed by whatever came after this layer). Using the input $\mathbf{x}$ and weight $\mathbf{W}$ that the forward pass saved on the node, it computes:

- $\mathbf{W}^\top\mathbf{g}$ — the VJP through the weights — and passes it further back as the input's gradient (which becomes the *next* node's upstream $\mathbf{g}$).
- $\mathbf{g}\mathbf{x}^\top$ — accumulated into `weight.grad`.
- $\mathbf{g}$ — accumulated into `bias.grad`.

Notice that only the input-gradient $\mathbf{W}^\top\mathbf{g}$ continues down the chain; the parameter gradients $\mathbf{g}\mathbf{x}^\top$ and $\mathbf{g}$ are *deposited* at this node (accumulated, per [lesson 20](lesson-02.md)) and read later by the optimizer. This is the VJP of section 3 made completely concrete: an output-sized vector $\mathbf{g}$ goes in, an input-sized vector $\mathbf{W}^\top\mathbf{g}$ comes out, and the Jacobian $\mathbf{W}$ is used but never explicitly built. Every layer in the network applies its own VJP this way, and the composition of all of them — from the loss back to the first parameter — is the complete backpropagation algorithm. You are now ready to build a whole network from these bricks.

## Check yourself

<details><summary>For $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ with $\mathbf{W}$ shape $(4, 3)$, what shapes are $\mathbf{g} = \partial L/\partial\mathbf{y}$, $\partial L/\partial\mathbf{x}$, and $\partial L/\partial\mathbf{W}$?</summary>

$\mathbf{g}$ is $(4,)$ (same as $\mathbf{y}$); $\partial L/\partial\mathbf{x} = \mathbf{W}^\top\mathbf{g}$ is $(3,)$ (same as $\mathbf{x}$); $\partial L/\partial\mathbf{W} = \mathbf{g}\mathbf{x}^\top$ is $(4,3)$ (same as $\mathbf{W}$).

</details>

<details><summary>With $\mathbf{W} = \begin{bmatrix} 1 & 2 \\ 3 & 4\end{bmatrix}$, $\mathbf{x} = \begin{bmatrix} 2 \\ 3\end{bmatrix}$, and upstream $\mathbf{g} = \begin{bmatrix} 1 \\ 2\end{bmatrix}$, compute $\partial L/\partial\mathbf{x}$.</summary>

$\mathbf{W}^\top\mathbf{g} = \begin{bmatrix} 1 & 3 \\ 2 & 4\end{bmatrix}\begin{bmatrix} 1 \\ 2\end{bmatrix} = \begin{bmatrix} 7 \\ 10\end{bmatrix}$.

</details>

<details><summary>Why does reverse-mode (VJP) suit neural-network training better than forward-mode (JVP)?</summary>

Training has one scalar output (the loss) and millions of inputs (parameters). A single VJP sweep, seeded by the scalar loss's gradient, produces every input gradient at once. Forward-mode would need one JVP pass per input, which is prohibitive when inputs vastly outnumber outputs.

</details>

<details><summary>Which of the three gradients gets passed further back down the network, and which get deposited at the layer?</summary>

$\partial L/\partial\mathbf{x} = \mathbf{W}^\top\mathbf{g}$ is passed back (it becomes the previous layer's upstream gradient). $\partial L/\partial\mathbf{W} = \mathbf{g}\mathbf{x}^\top$ and $\partial L/\partial\mathbf{b} = \mathbf{g}$ are deposited (accumulated) at this layer for the optimizer to read.

</details>

## Next

You now hold the three identities that every linear layer uses on the backward pass, with shapes that always line up. That is the last mathematical brick. In module 8 we stack these bricks into a real two-layer network, run a complete forward and backward pass by hand with small numbers, and then reproduce the exact same gradients in PyTorch — the milestone where all of this calculus becomes a working neural network.

Continue to [module 8 · Linear layers and neural-network mathematics](../module-08/lesson-01.md).
