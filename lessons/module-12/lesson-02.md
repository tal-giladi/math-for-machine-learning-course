# 36 · Automatic differentiation

<div class="prereq">
<p><strong>Prerequisites:</strong> the chain rule and computational graphs from <a href="../module-07/lesson-01.md">19 · The chain rule and computational graphs</a>; Jacobians from <a href="../module-07/lesson-03.md">21 · Jacobians</a> and vector-Jacobian products from <a href="../module-07/lesson-04.md">22 · Matrix calculus and VJPs</a>; and <code>backward()</code>, <code>grad_fn</code>, <code>requires_grad</code> from <a href="../module-08/lesson-03.md">25 · Backpropagation and autograd</a>.</p>
<p><strong>You will learn:</strong> the three ways a computer can compute a derivative — symbolic, numerical (finite differences), and automatic — with their exact tradeoffs; how automatic differentiation decomposes code into elementary operations with known local derivatives; the difference between <strong>forward-mode</strong> and <strong>reverse-mode</strong> AD; and precisely why neural networks use reverse mode.</p>
<p><strong>Why this matters for ML:</strong> <code>loss.backward()</code> is reverse-mode automatic differentiation. Understanding it — and understanding what forward mode does instead — tells you why one backward pass yields the gradient with respect to all 124 million GPT-2 parameters at once, why that costs about as much as a single forward pass, and what <code>.backward(v)</code>'s mysterious argument really supplies.</p>
</div>

## 1. Three ways to get a derivative

You want $\dfrac{df}{dx}$ for a function $f$ expressed as code. There are exactly three strategies, and they are genuinely different — not three implementations of one idea.

1. **Symbolic differentiation** — manipulate the *formula* with the calculus rules to produce a new formula for the derivative. This is what you did by hand and what Mathematica/SymPy do.
2. **Numerical differentiation** — never touch the formula; just evaluate $f$ at nearby points and take a difference quotient. This is the finite-difference method.
3. **Automatic differentiation (AD)** — work on the *code*, decomposing it into elementary operations whose derivatives are known, and combining them with the chain rule. This is what PyTorch does.

The rest of the lesson takes each in turn, then splits AD into its two modes.

## 2. Symbolic: exact, but the expression explodes

Symbolic differentiation applies the rules from [module 6](../module-06/lesson-02.md) to an expression tree and returns a new expression tree. For $f(x) = (2x+1)^2$ it produces $f'(x) = 4(2x+1)$ — exact, closed form, evaluatable anywhere.

The problem is **expression swell**. The product rule and chain rule each *duplicate* sub-expressions: differentiating a product $uv$ gives $u'v + uv'$, which contains both $u$ and $v$ again. Apply this through the dozens of nested operations in a neural network and the symbolic derivative becomes an astronomically large expression full of repeated sub-terms — often far larger than the original function, and wasteful to evaluate because the same sub-expression is recomputed many times. Symbolic differentiation is perfect for a textbook formula and hopeless for a million-operation program.

## 3. Numerical: approximate, and it costs you one evaluation per input

Numerical (finite-difference) differentiation goes straight back to the definition of the derivative as a limit (from [lesson 15](../module-06/lesson-01.md)), but stops short of the limit:

$$
f'(x) \approx \frac{f(x+h) - f(x)}{h}\qquad\text{for a small } h.
$$

It needs only the ability to *evaluate* $f$ — no formula, no code inspection. But it has two fatal weaknesses for ML.

**Weakness 1: it is inexact, and you cannot simply shrink $h$ to fix it.** There are two competing errors. The *truncation* error (from stopping short of the limit) is proportional to $h$, so it shrinks as $h\to 0$. But the *roundoff* error grows as $h\to 0$: when $h$ is tiny, $f(x+h)$ and $f(x)$ are almost equal, and subtracting two nearly-equal floating-point numbers destroys precision (catastrophic cancellation, from [lesson 35](lesson-01.md)). The total error is smallest at some middling $h$ and gets *worse* if you go smaller. Watch it, for $f(x)=x^2$ at $x=1$ (true derivative $2$):

| $h$ | approximation | error |
|---|---|---|
| $0.1$ | $2.1$ | $0.1$ |
| $0.01$ | $2.01$ | $0.01$ |
| $10^{-4}$ | $2.0001$ | $10^{-4}$ |
| $10^{-8}$ | $1.99999999$ | $\approx 10^{-8}$ |
| $10^{-12}$ | $2.00018$ | $\approx 2\times10^{-4}$ |
| $10^{-14}$ | $1.9984$ | $\approx 2\times10^{-3}$ |

The error falls with $h$ until about $h=10^{-8}$, then roundoff takes over and it climbs again. You can never get a truly exact derivative this way.

**Weakness 2: the cost scales with the number of inputs.** To get the partial derivative with respect to *each* input you must perturb that input separately and re-evaluate $f$. A function with $P$ inputs needs about $P$ extra evaluations of $f$ to get the full gradient. For GPT-2, $P \approx 124{,}000{,}000$: computing the gradient by finite differences would mean 124 million forward passes per step. Utterly impossible.

Finite differences remain useful for one thing: **checking** an AD implementation on a tiny function, where a slightly-wrong-but-independent number confirms the exact one.

## 4. Automatic: exact and efficient, by working on the code

Automatic differentiation is the resolution. Its founding observation: any function expressed as code, however complicated, is ultimately a composition of a small set of **elementary operations** — add, multiply, `exp`, `log`, `sin`, matmul, and so on — and for each of those we know the local derivative exactly. AD evaluates the function *and*, alongside it, propagates derivatives through each elementary operation using the chain rule.

So AD is **exact** (it uses true local derivatives, not difference quotients — no truncation error, no roundoff-from-cancellation) *and* **efficient** (it reuses intermediate values instead of duplicating sub-expressions the way symbolic does). It is neither of the other two methods; it is a third thing that keeps the good properties of both.

Concretely, take $f(x) = (2x+1)^2$ and decompose it into a chain of elementary steps (a computational graph, from [lesson 19](../module-07/lesson-01.md)):

$$
a = 2x + 1, \qquad y = a^2.
$$

Each step has a known local derivative: $\dfrac{da}{dx} = 2$ and $\dfrac{dy}{da} = 2a$. AD's job is to combine these two local derivatives via the chain rule, $\dfrac{dy}{dx} = \dfrac{dy}{da}\cdot\dfrac{da}{dx}$. The two *modes* of AD differ only in the **order** they do that combining.

## 5. Forward mode: carry derivatives forward alongside values

**Forward-mode** AD walks the graph in the same direction as the function itself (inputs to outputs). At each step it carries two numbers: the value, and the derivative of that value with respect to the chosen input. Pick $x=1.5$ and track $\dfrac{d(\cdot)}{dx}$:

- Start: $x = 1.5$, and $\dfrac{dx}{dx} = 1$.
- Step $a = 2x+1$: value $a = 4$; derivative $\dfrac{da}{dx} = 2\cdot\dfrac{dx}{dx} = 2$.
- Step $y = a^2$: value $y = 16$; derivative $\dfrac{dy}{dx} = 2a\cdot\dfrac{da}{dx} = 2(4)(2) = 16$.

Out pops $\dfrac{dy}{dx} = 16$ in a single forward sweep. The clean way to formalize "carry a value and its derivative together" is **dual numbers**: replace each number $v$ by a pair $v + v'\epsilon$, where $\epsilon$ is a formal symbol with $\epsilon^2 = 0$, and $v'$ is the derivative. Arithmetic on dual numbers automatically applies the derivative rules — e.g. $(a + a'\epsilon)^2 = a^2 + 2aa'\epsilon$, whose $\epsilon$-part $2aa'$ is exactly the chain rule for squaring.

**The cost of forward mode.** Each forward sweep tracks the derivative with respect to *one* input. To get the gradient with respect to all $n$ inputs you need $n$ sweeps (or one sweep carrying $n$ derivatives). So **forward-mode cost scales with the number of inputs.**

## 6. Reverse mode: record forward, then propagate backward

**Reverse-mode** AD does it in two phases. First a **forward pass** that computes the values and *records the graph* (which operation fed which, and the local derivative at each node). Then a **backward pass** that starts at the output and propagates the derivative back to every input, multiplying by local derivatives as it goes — exactly the procedure from [lesson 25](../module-08/lesson-03.md).

Same example, $y = (2x+1)^2$ at $x=1.5$:

- **Forward** (record): $a = 4$, $y = 16$; remember locals $\dfrac{dy}{da} = 2a = 8$ and $\dfrac{da}{dx} = 2$.
- **Backward** (propagate): seed $\dfrac{dy}{dy} = 1$. Then $\dfrac{dy}{da} = 1\cdot 8 = 8$. Then $\dfrac{dy}{dx} = 8\cdot 2 = 16$.

Same answer, $16$, but arrived at from the output backward. The point of doing it backward is what happens when there are **many inputs and one output**. A single backward pass, seeded once at the output, delivers the derivative of that output with respect to *every* input simultaneously — because the propagation visits every node on its way back to the inputs. So **reverse-mode cost scales with the number of outputs**, not inputs.

## 7. Why neural networks use reverse mode

Now the decisive comparison. A neural network's loss function has the shape

$$
L = f(\theta_1, \theta_2, \ldots, \theta_P), \qquad P \approx \text{millions}, \quad L \text{ a single scalar.}
$$

Millions of inputs (the parameters), **one** output (the scalar loss). Match that to the two modes:

- **Forward mode** costs one sweep per input $\Rightarrow$ millions of sweeps to get the full gradient. Hopeless.
- **Reverse mode** costs one sweep per output $\Rightarrow$ **one** backward pass gives $\dfrac{\partial L}{\partial\theta_i}$ for *all* $P$ parameters at once.

That is the entire reason `loss.backward()` is a reverse-mode engine. Training needs the gradient of one scalar with respect to millions of parameters, and reverse mode delivers exactly that in a single sweep costing roughly as much as one forward pass. (The mirror-image rule: if you had one input and millions of outputs, forward mode would win. Networks are the opposite shape, so reverse wins.)

<div class="callout key"><p>Forward-mode cost scales with the number of <em>inputs</em>; reverse-mode cost scales with the number of <em>outputs</em>. A neural network maps millions of parameters to one scalar loss, so reverse mode computes the whole gradient in a single backward pass — this is why training uses reverse-mode AD.</p></div>

## 8. Reverse mode is repeated vector-Jacobian products

Here is where this lesson meets [lessons 21–22](../module-07/lesson-04.md). Each elementary operation, viewed as a vector-to-vector map, has a **Jacobian** — the matrix of every output's partial derivative with respect to every input. For a layer that maps a $768$-vector to a $768$-vector that Jacobian is $768\times768$; forming one per operation would be as ruinous as symbolic swell.

Reverse mode never forms them. At each node it computes only a **vector-Jacobian product** (VJP): given the upstream gradient row-vector $\mathbf{v} = \dfrac{\partial L}{\partial(\text{node output})}$, it returns $\mathbf{v}^\top J$ — the downstream gradient — directly, in one cheap operation, without ever materializing $J$. The backward pass is nothing but a chain of VJPs, one per node, from the loss back to the parameters. For a linear layer the VJP is the $\mathbf{W}^\top(\partial L/\partial \mathbf{z})$ identity you derived by hand; for `exp` it is elementwise multiply by $e^x$; and so on. That is precisely why the gradient of a scalar loss with respect to millions of parameters is cheap: no Jacobian is ever built.

(Forward mode is the dual: it computes **Jacobian-vector products**, $J\mathbf{u}$, pushing a perturbation of the inputs forward — which is why `torch.func.jvp` is the forward-mode tool and `grad`/`vjp` are the reverse-mode ones.)

## 9. Three-way agreement in PyTorch

Let us verify that the exact methods agree, and that finite differences lands close, on one concrete number. Take $f(x) = (2x+1)^2$ at $x = 1.5$, where we computed the derivative to be $16$ three different ways above.

```python
import torch

x = torch.tensor(1.5, requires_grad=True)

# --- automatic differentiation (reverse mode) ---
y = (2 * x + 1) ** 2
y.backward()
print("autograd:       ", x.grad.item())      # 16.0

# --- symbolic derivative 4*(2x+1), evaluated by hand-coded formula ---
symbolic = 4 * (2 * 1.5 + 1)
print("symbolic:       ", symbolic)           # 16.0

# --- numerical finite difference ---
def f(v): return (2 * v + 1) ** 2
h = 1e-4
fd = (f(1.5 + h) - f(1.5)) / h
print("finite-diff:    ", fd)                  # 16.0004...  (approx, not exact)
```

Autograd prints `16.0` and the symbolic formula prints `16.0` — **exactly** equal, because both are exact methods. The finite difference prints about `16.0004`: close enough to confirm the others, but visibly inexact because of the truncation error from section 3. That gap is the whole difference between "approximate" and "exact" made concrete on a single number.

You can also see the two modes as separate APIs:

```python
import torch
from torch.func import grad, jvp

g = torch.tensor(1.5)
# reverse mode: derivative of a scalar output
print(grad(lambda v: (2*v+1)**2)(g))                         # tensor(16.)
# forward mode: push a unit perturbation through (Jacobian-vector product)
val, deriv = jvp(lambda v: (2*v+1)**2, (g,), (torch.tensor(1.0),))
print(val, deriv)                                             # tensor(16.) tensor(16.)
```

Both give $16$; they differ in *how* they compute it — `jvp` carries the perturbation forward (forward mode), `grad` propagates backward (reverse mode).

## 10. Under the hood: the dynamic graph and `.backward(v)`

Everything from [lesson 25](../module-08/lesson-03.md) is reverse-mode AD wearing PyTorch clothes:

- **`requires_grad` propagation** decides which tensors participate: if any input to an op needs grad, the output does, so tracking spreads forward through the computation.
- The **dynamic graph** is the recording phase of reverse mode, built op-by-op *as the forward code runs*. Each non-leaf tensor's **`grad_fn`** is that node's stored VJP function — it holds whatever it needs to turn an upstream gradient into a downstream one.
- **`.backward()`** is the backward propagation phase: seed $\dfrac{\partial L}{\partial L}=1$ at the loss, then walk the recorded graph in reverse, applying each `grad_fn`'s VJP, accumulating results into each leaf's `.grad`.

One detail this lesson finally explains: **`.backward(v)`'s optional argument**. When you call `loss.backward()` on a scalar, PyTorch seeds the backward pass with $1$ automatically. But if you call `.backward(v)` on a *non-scalar* tensor, `v` is the upstream vector for the first VJP — the $\mathbf{v}$ in $\mathbf{v}^\top J$. You are supplying the "gradient of some downstream scalar with respect to this tensor" yourself. That is why `y.backward()` on a non-scalar raises an error asking for a `gradient` argument: reverse mode needs a starting vector for its first vector-Jacobian product, and only a scalar output supplies one ($1$) for free.

<div class="callout pt"><p><code>loss.backward()</code> is reverse-mode AD: the forward pass records a graph of VJP functions (<code>grad_fn</code>), the backward pass seeds $\partial L/\partial L=1$ and chains those VJPs to fill every leaf's <code>.grad</code>. For a non-scalar output you must pass the seed yourself via <code>.backward(v)</code> — <code>v</code> is the upstream vector of the first vector-Jacobian product.</p></div>

## Check yourself

<details><summary>Why is symbolic differentiation impractical for a large neural network, even though it is exact?</summary>

Expression swell: the product and chain rules duplicate sub-expressions, so the symbolic derivative of a million-operation program becomes an astronomically large formula with vast redundancy — far bigger and slower than the original function. AD keeps exactness but reuses intermediate values instead of duplicating them.

</details>

<details><summary>Finite differences: why can't you just make $h$ as small as possible to get an exact derivative?</summary>

Two competing errors. Truncation error $\propto h$ shrinks as $h\to0$, but roundoff error grows because $f(x+h)$ and $f(x)$ become nearly equal and subtracting them loses precision (catastrophic cancellation). Total error is minimized at a middling $h$ (around $10^{-8}$ in float64) and gets worse if you go smaller.

</details>

<details><summary>A function has 1,000,000 inputs and 1 output. Which AD mode is cheaper, and by how much?</summary>

Reverse mode. Its cost scales with the number of outputs (here 1), so one backward pass yields all 1,000,000 partial derivatives. Forward mode scales with inputs, needing ~1,000,000 sweeps. Reverse mode is about a million times cheaper — this is the neural-network case.

</details>

<details><summary>What does the <code>v</code> in <code>tensor.backward(v)</code> represent?</summary>

The upstream gradient vector for the first vector-Jacobian product — the $\mathbf{v}$ in $\mathbf{v}^\top J$. For a scalar loss PyTorch seeds it as $1$ automatically; for a non-scalar output you must supply it, which is why non-scalar `.backward()` without an argument errors.

</details>

## Next

You now know what computes the gradients (reverse-mode AD) and how to keep the arithmetic finite ([lesson 35](lesson-01.md)). The next lesson assembles everything in the course into one object: a complete GPT-2-style training step, every tensor named, shaped, and classified as parameter, activation, gradient, or optimizer state.

Continue to [37 · Training a GPT-2-style model end to end](lesson-03.md).
