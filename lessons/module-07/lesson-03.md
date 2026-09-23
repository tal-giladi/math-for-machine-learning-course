# 21 · Jacobians

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-07/lesson-02">20 · Gradients: ∂L/∂x vs ∇, with respect to matrices and tensors</a>, the chain rule and computational graphs from <a href="#/lessons/module-07/lesson-01">lesson 19</a>, and the softmax Jacobian from <a href="#/lessons/module-06/lesson-03">17 · Derivatives of exp, log, sigmoid, softmax</a>. Matrix multiplication from <a href="#/lessons/module-03/lesson-03">module 3</a>.</p>
<p><strong>You will learn:</strong> what the <strong>Jacobian</strong> of a vector-valued function is, its exact shape rule ($m\times n$), a fully worked $2\times 2$ example, why a linear map's Jacobian is the matrix itself, why an elementwise activation has a <strong>diagonal</strong> Jacobian, and the chain rule as a <strong>product of Jacobians</strong> — plus why PyTorch never builds big Jacobians explicitly.</p>
<p><strong>Why this matters for ML:</strong> every layer is a vector-valued function, so every layer has a Jacobian. Backpropagation is, formally, the chain rule multiplying these Jacobians backward. Seeing that linear layers give $W$, activations give a diagonal, and the whole chain is a matrix product is what turns "backprop" from a black box into linear algebra.</p>
</div>

## 1. From scalar output to vector output

Until now our function returned a single number $L$, and its derivative object (the gradient) had the shape of the *input*. Now let the output be a whole **vector**. A **vector-valued function** takes an $n$-vector and returns an $m$-vector:

$$
f : \mathbb{R}^n \to \mathbb{R}^m, \qquad f(\mathbf{x}) = \big(f_1(\mathbf{x}),\, f_2(\mathbf{x}),\, \dots,\, f_m(\mathbf{x})\big).
$$

- $n$ — number of inputs (the length of $\mathbf{x}$).
- $m$ — number of outputs.
- $f_i(\mathbf{x})$ — the $i$-th output component; each is itself a scalar function of all $n$ inputs.

A linear layer $\mathbf{z} = \mathbf{W}\mathbf{x}$ that maps a $C$-vector to a $C'$-vector is exactly this kind of function, with $n = C$ and $m = C'$. So is softmax, which maps a $V$-vector of logits to a $V$-vector of probabilities.

Now "the derivative" is not a single number, and not even a single gradient vector. Each of the $m$ outputs has its own sensitivity to each of the $n$ inputs — that is $m\times n$ numbers. Collect them in a grid and you get the **Jacobian**.

## 2. The Jacobian matrix and its shape

The **Jacobian** $J$ of $f : \mathbb{R}^n \to \mathbb{R}^m$ is the $m\times n$ matrix whose $(i,j)$ entry is the partial derivative of output $i$ with respect to input $j$:

$$
J_{ij} = \frac{\partial f_i}{\partial x_j}, \qquad
J = \begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\[2mm]
\vdots & \ddots & \vdots \\[1mm]
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix}.
$$

- **Row $i$** collects how output $f_i$ depends on *every* input — it is exactly the gradient $\nabla f_i$ laid out as a row.
- **Column $j$** collects how *every* output depends on input $x_j$.

<div class="callout key"><p><strong>Shape rule:</strong> for $f:\mathbb{R}^n\to\mathbb{R}^m$ the Jacobian is $m\times n$ — <em>outputs index the rows, inputs index the columns</em>, and $J_{ij}=\partial f_i/\partial x_j$. A scalar output ($m=1$) makes $J$ a single $1\times n$ row: the gradient transposed.</p></div>

Notice the special case $m = 1$: the Jacobian is a single row of $n$ partials — that is the gradient from [lesson 20](lessons/module-07/lesson-02.md), written horizontally. The Jacobian is the natural generalization of the gradient to many outputs.

## 3. A worked $2\times 2$ example

Take the function $f : \mathbb{R}^2 \to \mathbb{R}^2$ (so $m = n = 2$):

$$
f(x_1, x_2) = \begin{bmatrix} x_1^2\, x_2 \\ x_1 + 3x_2 \end{bmatrix}.
$$

The two output components are $f_1 = x_1^2 x_2$ and $f_2 = x_1 + 3x_2$. Compute all four partials, treating the other variable as a constant each time (partial derivatives, [lesson 18](lessons/module-06/lesson-04.md)):

- $\dfrac{\partial f_1}{\partial x_1} = 2x_1 x_2$ (power rule on $x_1^2$, with $x_2$ a constant multiple).
- $\dfrac{\partial f_1}{\partial x_2} = x_1^2$ ($x_1^2$ is the constant multiple of $x_2$).
- $\dfrac{\partial f_2}{\partial x_1} = 1$ (derivative of $x_1$; the $3x_2$ term is constant).
- $\dfrac{\partial f_2}{\partial x_2} = 3$ (derivative of $3x_2$; the $x_1$ term is constant).

Assemble, outputs as rows:

$$
J = \begin{bmatrix} 2x_1 x_2 & x_1^2 \\ 1 & 3 \end{bmatrix}.
$$

**Evaluate at $(x_1, x_2) = (2, 3)$:** $2x_1x_2 = 2\cdot 2\cdot 3 = 12$ and $x_1^2 = 4$, so

$$
J\big|_{(2,3)} = \begin{bmatrix} 12 & 4 \\ 1 & 3 \end{bmatrix}.
$$

Read entry $(1,1) = 12$: at this point, output $f_1$ rises about $12$ units per unit rise in $x_1$. Entry $(2,2) = 3$: output $f_2$ rises $3$ per unit rise in $x_2$, everywhere (that partial is constant). Each entry is a plain partial derivative; the Jacobian just organizes them.

## 4. Two Jacobians that appear in every network

Two shapes of Jacobian dominate neural networks, and both are simple.

### 4.1 A linear map's Jacobian is the matrix itself

Let $f(\mathbf{x}) = \mathbf{W}\mathbf{x}$ for a fixed $m\times n$ matrix $\mathbf{W}$. Output component $i$ is the dot product of row $i$ of $\mathbf{W}$ with $\mathbf{x}$:

$$
f_i = \sum_{k=1}^{n} W_{ik}\,x_k.
$$

Differentiate with respect to $x_j$. In that sum, only the $k = j$ term contains $x_j$, and its derivative is $W_{ij}$; every other term is constant. So

$$
\frac{\partial f_i}{\partial x_j} = W_{ij} \quad\Longrightarrow\quad J = \mathbf{W}.
$$

The Jacobian of a linear layer (with respect to its input) is exactly its weight matrix — no computation needed. This is the single most-used Jacobian in all of deep learning, and it is why matrix multiplication is the backbone of the backward pass.

### 4.2 An elementwise activation has a diagonal Jacobian

An activation like sigmoid or ReLU is applied **elementwise**: output $i$ depends only on input $i$, $f_i(\mathbf{x}) = \phi(x_i)$ for a scalar function $\phi$. Then for $i \ne j$ the output $f_i$ does not involve $x_j$ at all, so $\partial f_i/\partial x_j = 0$; and on the diagonal $\partial f_i/\partial x_i = \phi'(x_i)$. The Jacobian is therefore **diagonal**:

$$
J = \begin{bmatrix} \phi'(x_1) & 0 & \cdots & 0 \\ 0 & \phi'(x_2) & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & \phi'(x_n) \end{bmatrix} = \operatorname{diag}\big(\phi'(x_1), \dots, \phi'(x_n)\big).
$$

For sigmoid, $\phi'(x_i) = \sigma(x_i)\,(1 - \sigma(x_i))$ — the formula derived in [lesson 17](lessons/module-06/lesson-03.md). Example: for inputs whose sigmoid outputs are $\mathbf{s} = (0.5, 0.8)$, the Jacobian is $\operatorname{diag}(0.5\cdot 0.5,\ 0.8\cdot 0.2) = \operatorname{diag}(0.25,\ 0.16)$. A diagonal Jacobian is cheap: multiplying a vector by it is just elementwise scaling, no full matrix needed.

### 4.3 Softmax: a dense Jacobian (recap)

Softmax is the contrast. Because every output shares the same denominator, output $s_i$ depends on *every* input $z_j$, so its Jacobian is **dense** (no zeros to exploit). We derived it in full in [lesson 17](lessons/module-06/lesson-03.md):

$$
\frac{\partial s_i}{\partial z_j} = s_i\big(\delta_{ij} - s_j\big),
$$

with $\delta_{ij}$ the Kronecker delta ($1$ if $i=j$, else $0$). For $\mathbf{z} = (1, 0)$ we computed there

$$
J_{\text{softmax}} = \begin{bmatrix} 0.19661 & -0.19661 \\ -0.19661 & 0.19661 \end{bmatrix},
$$

diagonal entries $s_i(1-s_i)$, off-diagonal $-s_i s_j$. See [module 6, lesson 3](lessons/module-06/lesson-03.md) for the derivation and the column-sums-to-zero sanity check.

## 5. The chain rule is a product of Jacobians

Here is the payoff. For a composition $\mathbf{y} = f(g(\mathbf{x}))$ — first apply $g$, then $f$ — the chain rule for vector-valued functions says the overall Jacobian is the **matrix product** of the individual Jacobians:

$$
J_{\mathbf{y}} = J_f \cdot J_g.
$$

- $J_g$ is the Jacobian of the inner function $g$ (evaluated at $\mathbf{x}$).
- $J_f$ is the Jacobian of the outer function $f$ (evaluated at $\mathbf{u} = g(\mathbf{x})$).
- The product is ordinary matrix multiplication.

This is the exact generalization of the scalar chain rule "multiply the local derivatives" from [lesson 19](lessons/module-07/lesson-01.md): for scalars the factors were numbers; for vector functions the factors are Jacobian matrices, and "multiply" becomes matrix multiply. **The shapes have to line up**, and they do: if $g:\mathbb{R}^n\to\mathbb{R}^p$ then $J_g$ is $p\times n$; if $f:\mathbb{R}^p\to\mathbb{R}^m$ then $J_f$ is $m\times p$; the product $J_f\,J_g$ is $(m\times p)(p\times n) = m\times n$ — exactly the Jacobian shape for the composite $\mathbb{R}^n\to\mathbb{R}^m$. The inner dimension $p$ (the width of the intermediate layer) cancels, just as it does in any matrix multiply.

### 5.1 A numeric check

Let $g(\mathbf{x}) = \big(x_1^2 x_2,\ x_1 + 3x_2\big)$ — the function from section 3, so $J_g\big|_{(2,3)} = \begin{bmatrix} 12 & 4 \\ 1 & 3\end{bmatrix}$ — and let $f(\mathbf{u}) = u_1 + u_2$ (a scalar output, $m = 1$), whose Jacobian is the row $J_f = \begin{bmatrix} 1 & 1 \end{bmatrix}$. The composite is $h(\mathbf{x}) = x_1^2 x_2 + x_1 + 3x_2$. By the Jacobian product,

$$
J_h = J_f\,J_g = \begin{bmatrix} 1 & 1 \end{bmatrix}\begin{bmatrix} 12 & 4 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 12 + 1 & 4 + 3 \end{bmatrix} = \begin{bmatrix} 13 & 7 \end{bmatrix}.
$$

Check directly: $\partial h/\partial x_1 = 2x_1x_2 + 1 = 12 + 1 = 13$ and $\partial h/\partial x_2 = x_1^2 + 3 = 4 + 3 = 7$. The Jacobian product $[13,\ 7]$ matches the direct computation exactly. A whole network's forward pass is a long composition $f_L \circ \cdots \circ f_2 \circ f_1$, so its Jacobian is a long product $J_{f_L}\cdots J_{f_2} J_{f_1}$ — and *that product, evaluated backward, is backpropagation.*

## 6. Why we don't build big Jacobians in PyTorch

The Jacobian product $J_{f_L}\cdots J_{f_1}$ is the right idea but the wrong thing to compute directly. Consider GPT-2's output layer mapping a hidden vector to logits over a $V \approx 50000$ vocabulary. A single layer's Jacobian there would be $50000 \times 768$ — about 38 million entries for *one* layer, one token. Materializing such matrices for every layer and multiplying them would be absurdly expensive in both memory and time.

The escape is that in training we never need the full Jacobian — we only need to push the scalar loss's sensitivity backward. The loss is one number, so what enters the top of the backward pass is a single **vector** (the upstream gradient), and every backward step only needs the product of that vector with a Jacobian, not the Jacobian itself. This product $\mathbf{v}^\top J$ is the **vector–Jacobian product (VJP)**, and it is cheap: for a linear layer it is just another matrix-vector multiply by $\mathbf{W}^\top$; for a diagonal activation Jacobian it is an elementwise multiply; for softmax it can be done in $O(V)$ without ever forming the $V\times V$ matrix. Computing VJPs backward through the graph — never building the Jacobians — is exactly **reverse-mode automatic differentiation**, which is what `backward()` does. The next lesson makes the VJP precise and derives the linear-layer formulas from it.

<div class="callout key"><p>Backprop = the chain rule as a Jacobian product, but evaluated as a sequence of <strong>vector–Jacobian products</strong> so the big Jacobians are never materialized. "Multiply Jacobians backward" is the math; "push a vector back through each op" is the implementation.</p></div>

## 7. In PyTorch

PyTorch *can* build a small Jacobian on request, which is handy for verifying by hand. Use `torch.autograd.functional.jacobian` on the function from section 3:

```python
import torch
from torch.autograd.functional import jacobian

def f(x):                                   # R^2 -> R^2
    return torch.stack([x[0]**2 * x[1],     # f1 = x1^2 * x2
                        x[0] + 3*x[1]])      # f2 = x1 + 3*x2

x = torch.tensor([2., 3.])
J = jacobian(f, x)
print(J)
# tensor([[12.,  4.],
#         [ 1.,  3.]])
```

The result is exactly the $\begin{bmatrix} 12 & 4 \\ 1 & 3\end{bmatrix}$ we computed by hand. But note what this call does under the hood: to fill in $m = 2$ rows it effectively runs the backward pass *twice* (once per output). That is fine for a $2\times 2$ toy, ruinous for a real layer.

Contrast it with the normal training path, which reduces to a scalar and calls `.backward()` once — a single VJP, not the whole Jacobian:

```python
x = torch.tensor([2., 3.], requires_grad=True)
h = x[0]**2 * x[1] + x[0] + 3*x[1]          # scalar composite from section 5.1
h.backward()
print(x.grad)                               # tensor([13.,  7.])
```

`x.grad` is `[13., 7.]` — the same $J_h$ row we got from the Jacobian product $J_f J_g$, obtained in *one* backward sweep because the output is scalar. This is the everyday case: one scalar loss, one cheap backward pass, no explicit Jacobian.

<div class="callout pt"><p><code>torch.autograd.functional.jacobian</code> materializes the full $m\times n$ Jacobian (running backward once per output row) — use it only for small functions or checking. Real training calls <code>loss.backward()</code>, which computes a single vector–Jacobian product per node and never forms a full Jacobian.</p></div>

## 8. What PyTorch is doing under the hood

Each backward node in the graph implements one thing: given the upstream gradient vector $\mathbf{v} = \partial L/\partial(\text{node output})$, return $\mathbf{v}^\top J_{\text{node}} = \partial L/\partial(\text{node input})$ — the VJP for that operation. It does this *without* forming $J_{\text{node}}$:

- A **linear** node $\mathbf{z} = \mathbf{W}\mathbf{x}$ has $J = \mathbf{W}$ (section 4.1), so its VJP returns $\mathbf{W}^\top\mathbf{v}$ — a matrix-vector product.
- An **elementwise activation** node has diagonal $J = \operatorname{diag}(\phi'(x_i))$ (section 4.2), so its VJP returns the elementwise product $\mathbf{v}\odot\phi'(\mathbf{x})$.
- A **softmax** node uses the rule $s_i(\delta_{ij}-s_j)$ (section 4.3) to compute $\mathbf{v}^\top J$ directly in $O(V)$.

Chaining these VJPs from the loss backward multiplies all the Jacobians in exactly the order $J_{f_L}\cdots J_{f_1}$ — the composite Jacobian of the whole network — but only ever as vector-sized quantities. That is the entire mechanism of reverse-mode autodiff, and the next lesson derives the specific VJP formulas for the linear layer $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ that you will use throughout module 8.

## Check yourself

<details><summary>A function maps $\mathbb{R}^5 \to \mathbb{R}^3$. What is the shape of its Jacobian, and what does entry $J_{2,4}$ mean?</summary>

Shape $3\times 5$ (outputs $\times$ inputs). $J_{2,4} = \partial f_2/\partial x_4$: how the 2nd output changes as the 4th input changes, others fixed.

</details>

<details><summary>For $f(x_1, x_2) = (x_1^2 x_2,\ x_1 + 3x_2)$, give the Jacobian at $(1, 1)$.</summary>

$J = \begin{bmatrix} 2x_1x_2 & x_1^2 \\ 1 & 3\end{bmatrix}$, so at $(1,1)$: $\begin{bmatrix} 2 & 1 \\ 1 & 3\end{bmatrix}$.

</details>

<details><summary>Why is the Jacobian of an elementwise activation diagonal, and the Jacobian of softmax dense?</summary>

An elementwise activation has each output depending only on the matching input, so all off-diagonal partials are zero (diagonal). Softmax's outputs share one denominator, so every output depends on every input — no zeros, hence dense.

</details>

<details><summary>For a linear layer with $W$ of shape $50000 \times 768$, why does PyTorch compute $W^\top v$ instead of the Jacobian?</summary>

The Jacobian is $W$ itself, a $50000\times768$ matrix (38M entries) that is wasteful to store and multiply. Since the loss is scalar, only the vector–Jacobian product $W^\top v$ is needed — a single matrix-vector multiply — which gives $\partial L/\partial x$ without materializing anything new.

</details>

## Next

You now see backprop as a product of Jacobians, executed as vector–Jacobian products so the big matrices never appear. The final piece is to make the VJP concrete for the operation that dominates every network — the linear layer $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ — and derive, with checked shapes, the three gradient formulas ($\partial L/\partial\mathbf{x}$, $\partial L/\partial\mathbf{W}$, $\partial L/\partial\mathbf{b}$) that module 8 is built on.

Continue to [22 · Matrix calculus and vector–Jacobian products](lessons/module-07/lesson-04.md).
