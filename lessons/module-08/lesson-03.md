# 25 · Backpropagation and autograd

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-08/lesson-01">23 · The linear layer, forward and backward by hand</a> (the by-hand backward pass we now automate), <a href="#/lessons/module-07/lesson-01">19 · The chain rule as a graph</a> (the branching rule especially), and <a href="#/lessons/module-07/lesson-03">21 · Vector-Jacobian products</a>. Loss gradients come from <a href="#/lessons/module-08/lesson-02">24 · Loss functions</a>.</p>
<p><strong>You will learn:</strong> exactly what <code>loss.backward()</code> does — how the forward pass builds a computational graph, how <code>backward()</code> traverses it in reverse applying the chain rule, and every piece of vocabulary around it: <code>requires_grad</code>, <code>grad_fn</code>, <code>.grad</code>, leaf tensors, graph construction and traversal order, gradient <strong>accumulation</strong>, and <code>zero_grad()</code>.</p>
<p><strong>Why this matters for ML:</strong> this is the engine of all training. Every GPT-2 step is forward → <code>loss.backward()</code> → read <code>.grad</code> → optimizer update → <code>zero_grad()</code>. Once you see that <code>backward()</code> is just the chain rule replayed over a recorded graph — not magic — you can debug any training loop, explain any <code>None</code> gradient, and know precisely where every number lives.</p>
</div>

## 1. The one-sentence summary

Backpropagation is: **the forward pass records a graph of every operation; `loss.backward()` walks that graph in reverse, applying the chain rule at each node, and deposits the result into each leaf's `.grad`.** That is the whole thing. This lesson unpacks each clause with a runnable snippet, so that by the end nothing in that sentence is mysterious.

You already did the hard part by hand in [lesson 23](lessons/module-08/lesson-01.md): starting from $\partial L/\partial y$, you pushed the gradient back through each layer, multiplying by local derivatives, until every parameter had its gradient. `backward()` does *exactly that*, automatically, for a graph of any size.

## 2. `requires_grad`: opting a tensor into the graph

A tensor tracks gradients only if you ask. Set `requires_grad=True` and PyTorch begins recording every operation that tensor takes part in.

```python
import torch

a = torch.tensor([2.0, 3.0], requires_grad=True)  # tracked
b = torch.tensor([4.0, 5.0])                       # NOT tracked (default False)

print(a.requires_grad)   # True
print(b.requires_grad)   # False
```

Model parameters are created with `requires_grad=True` (that is what `nn.Parameter` does). Inputs and targets usually are not — you do not train the data. If *any* input to an operation requires grad, the output does too, so the "tracked" property spreads forward through the computation automatically.

## 3. Leaf vs non-leaf tensors

A **leaf** tensor is one at the edge of the graph — created directly by you, not computed from other tracked tensors. Your parameters are leaves. A **non-leaf** tensor is the result of an operation on tracked tensors: `z1`, `a1`, `y` in the previous lessons.

The rule that trips everyone up: **`.grad` is populated only on leaf tensors by default.** PyTorch computes gradients *through* non-leaf tensors during the backward pass but discards them afterward to save memory.

```python
w = torch.tensor(3.0, requires_grad=True)   # leaf
x = torch.tensor(2.0)                        # leaf (but requires_grad=False)
y = w * x                                    # non-leaf: computed from w

print(w.is_leaf, y.is_leaf)   # True False

y.backward()
print(w.grad)   # tensor(2.0)   -> dy/dw = x = 2
print(y.grad)   # None + a warning: y is non-leaf, grad not retained
```

If you genuinely need a non-leaf's gradient (for debugging, or inspecting an intermediate activation), call `.retain_grad()` on it *before* `backward()`:

```python
y = w * x
y.retain_grad()
y.backward()
print(y.grad)   # tensor(1.0)   -> dy/dy = 1, now kept
```

This is exactly why `a1.grad` printed `None` in [lesson 23](lessons/module-08/lesson-01.md): `a1` was non-leaf.

## 4. `grad_fn`: the backward function on each node

When PyTorch computes a non-leaf tensor, it attaches a **`grad_fn`** — the function that knows how to compute the local derivative of that operation. This is the graph node.

```python
w = torch.tensor(3.0, requires_grad=True)
z = w * 2          # multiplication node
y = z + 1          # addition node
L = y ** 2         # power node

print(z.grad_fn)   # <MulBackward0>
print(y.grad_fn)   # <AddBackward0>
print(L.grad_fn)   # <PowBackward0>
```

Each `grad_fn` remembers what it needs to compute its local derivative (for `MulBackward0`, the other factor; for `PowBackward0`, the base and exponent) and holds pointers to the `grad_fn`s of *its* inputs. Those pointers are the edges of the graph. Leaf tensors have `grad_fn = None` — they are not computed from anything, so there is no backward function, just a `.grad` slot waiting to be filled.

## 5. Graph construction is dynamic

PyTorch builds the graph **during the forward pass, as the operations run** — not ahead of time. This is called a *dynamic* (define-by-run) graph. Every line of Python that touches a tracked tensor adds a node right then. Consequences worth internalizing:

- The graph reflects exactly the operations that ran this time, so ordinary Python control flow (`if`, `for`, variable-length loops) just works — the graph can differ from one forward pass to the next.
- The graph is built fresh every forward pass and, by default, **freed** as soon as you call `backward()`. Calling `backward()` a second time on the same graph raises an error unless you passed `retain_graph=True`.

```python
w = torch.tensor(3.0, requires_grad=True)
L = w ** 2
L.backward()
# L.backward()   # RuntimeError: graph freed; would need retain_graph=True
```

## 6. Graph traversal: reverse topological order

`backward()` starts at the loss and visits nodes in **reverse topological order** — a node is processed only after everything that consumed its output has been processed. That guarantees that by the time PyTorch reaches a node, it already has the full upstream gradient $\partial L/\partial(\text{node output})$ to multiply by that node's local derivative. It is the by-hand rule "work right to left, one layer at a time" from [lesson 23](lessons/module-08/lesson-01.md), made into an algorithm.

At each node the chain rule is one multiply: (upstream gradient) × (this node's local derivative) = (gradient to pass further back). Nothing more clever is happening.

## 7. Gradient accumulation and the branching rule

Here is the single most important operational detail: **`backward()` adds into `.grad`; it does not overwrite.** Each backward pass does `leaf.grad += (new gradient)`.

```python
w = torch.tensor(3.0, requires_grad=True)

L1 = w ** 2
L1.backward()
print(w.grad)      # tensor(6.0)      -> dL1/dw = 2w = 6

L2 = w ** 2
L2.backward()
print(w.grad)      # tensor(12.0)     -> 6 + 6, ACCUMULATED, not replaced
```

The second `backward()` did not compute $12$; it computed $6$ again and *added* it to the $6$ already sitting in `w.grad`. This is not a bug — it is the correct implementation of the **branching rule** from [lesson 19](lessons/module-07/lesson-01.md): when a quantity feeds into the loss through several paths, its total gradient is the *sum* of the contributions from each path. A parameter used in many places (like a weight reused across positions, or the embedding of a token that appears twice) receives one contribution per use, and accumulation sums them automatically. PyTorch simply treats "you called backward twice" as another form of "multiple contributions," and adds.

## 8. `zero_grad()`: why you must reset every step

Accumulation is exactly what you want *within* one backward pass and exactly what you must *undo* between training steps. If you do not clear `.grad`, step 2's gradient is added on top of step 1's leftover, and your update is corrupted with stale gradient. So every training step resets the slots first:

```python
# w.grad.zero_()                 # by hand, or:
# optimizer.zero_grad()          # what the training loop actually calls
```

Forget `zero_grad()` and your gradients pile up across iterations — a classic silent training bug where the model behaves as if the learning rate were growing every step.

<div class="callout warn"><p><code>backward()</code> <em>adds</em> to <code>.grad</code>. That is deliberate (it implements the branching rule — sum over paths), but it means you must call <code>zero_grad()</code> before each new training step, or gradients from previous steps contaminate the current one.</p></div>

## 9. A minimal end-to-end training loop

Everything above, assembled into the loop that trains every neural network. We minimize $L = (wx - t)^2$ with a single weight $w$, input $x = 1.0$, target $t = 2.0$, learning rate $\eta = 0.1$, starting from $w = 0.0$. Two full iterations, by hand and in code.

**By hand.** The gradient is $\dfrac{\partial L}{\partial w} = 2(wx - t)\,x$.

- **Iteration 1.** $w = 0.0$: prediction $y = 0.0$, loss $L = (0 - 2)^2 = 4.0$, gradient $= 2(0 - 2)(1) = -4.0$. Update: $w \leftarrow 0.0 - 0.1(-4.0) = 0.4$.
- **Iteration 2.** $w = 0.4$: prediction $y = 0.4$, loss $L = (0.4 - 2)^2 = 2.56$, gradient $= 2(0.4 - 2)(1) = -3.2$. Update: $w \leftarrow 0.4 - 0.1(-3.2) = 0.72$.

The loss fell from $4.0$ to $2.56$ (and the next prediction, $0.72$, gives $1.6384$). In code:

```python
import torch

x = torch.tensor(1.0)
t = torch.tensor(2.0)
w = torch.tensor(0.0, requires_grad=True)
lr = 0.1

for step in range(2):
    y = w * x                 # forward: builds the graph
    L = (y - t) ** 2          # scalar loss
    L.backward()              # backward: fills w.grad

    print(step, "loss", L.item(), "w", w.item(), "grad", w.grad.item())

    with torch.no_grad():     # update must NOT be recorded in the graph
        w -= lr * w.grad      # manual gradient-descent step
    w.grad.zero_()            # reset before the next iteration

# 0 loss 4.0  w 0.0  grad -4.0
# 1 loss 2.56 w 0.4  grad -3.2
```

Every printed number matches the hand computation. Three details to notice:

- The **forward** lines rebuild the graph each iteration (dynamic graph, section 5).
- The update is wrapped in `torch.no_grad()` so that changing `w` is *not* recorded as a graph operation — we are adjusting a parameter, not computing a new quantity to differentiate.
- `w.grad.zero_()` at the end is the section 8 reset. Remove it and iteration 1's `-4.0` would still be in `w.grad` when iteration 2 adds `-3.2`, giving a wrong `-7.2`.

A real optimizer (`optimizer.step()` then `optimizer.zero_grad()`) does precisely these last two lines for every parameter at once — the subject of the next module.

## 10. `backward()` is not magic

Put it all together and the demystification is complete. `loss.backward()`:

1. starts with $\partial L/\partial L = 1$ at the loss node;
2. visits nodes in reverse topological order (section 6);
3. at each node, multiplies the incoming upstream gradient by that node's local derivative (the `grad_fn`), summing contributions where a tensor branched (section 7);
4. writes the result into each leaf's `.grad` (section 3).

That is the chain rule of [module 7](lessons/module-07/lesson-01.md), executed over the graph recorded during the forward pass. It is the identical procedure you performed with a calculator in [lesson 23](lessons/module-08/lesson-01.md) — PyTorch just never gets bored or makes an arithmetic slip.

## 11. What PyTorch is doing under the hood: VJPs, not Jacobians

One deeper point about *how* step 3 stays efficient. A layer whose output and input are both large vectors has, in principle, a huge **Jacobian** — the full matrix of every output's derivative with respect to every input (from [module 7](lessons/module-07/lesson-03.md)). For a $768$-wide layer that Jacobian is $768 \times 768$ per position; forming them explicitly would be ruinous.

PyTorch never forms them. Reverse-mode autodiff only ever needs the product of the Jacobian with the upstream gradient *vector* — a **vector-Jacobian product** (VJP), covered in [lessons 21–22](lessons/module-07/lesson-03.md). Each `grad_fn` implements its VJP directly: given the upstream gradient, it returns the downstream gradient in one cheap operation, without materializing the matrix in between. For the linear layer, that VJP is exactly the $\mathbf{W}^\top(\partial L/\partial\mathbf{z})$ identity you used by hand — a matrix-vector product, not a Jacobian construction. This is why backprop over a 124-million-parameter model costs about the same as one or two forward passes, rather than exploding.

<div class="callout key"><p>Reverse-mode autograd computes <strong>vector-Jacobian products</strong>, never full Jacobians. Each operation's <code>grad_fn</code> maps the upstream gradient straight to the downstream gradient. That is what makes computing the gradient of a scalar loss with respect to millions of parameters cheap — one backward sweep, roughly the cost of the forward pass.</p></div>

## Check yourself

<details><summary>After <code>y = w * x; L = y**2; L.backward()</code>, why is <code>y.grad</code> <code>None</code> but <code>w.grad</code> a real number?</summary>

`w` is a leaf (`requires_grad=True`, created by you), so its `.grad` is populated. `y` is non-leaf (computed from `w`), and PyTorch does not retain non-leaf gradients by default. Call `y.retain_grad()` before `backward()` to keep it.

</details>

<details><summary>You train for 3 steps but forget <code>zero_grad()</code>. Qualitatively, what goes wrong?</summary>

Gradients accumulate across steps, so step 2 uses (grad1 + grad2), step 3 uses (grad1 + grad2 + grad3), and so on. Effective step sizes grow, updates overshoot, and the loss diverges or oscillates — as if the learning rate were increasing every iteration.

</details>

<details><summary>Why is the parameter update wrapped in <code>with torch.no_grad()</code>?</summary>

The update <code>w -= lr * w.grad</code> is a modification of a parameter, not part of the function we are differentiating. Without `no_grad()`, PyTorch would record it as a graph operation, corrupting the next backward pass (and wasting memory). `no_grad()` tells autograd to not track these operations.

</details>

<details><summary>A weight is reused at three positions in a sequence. How does its total gradient get computed?</summary>

Each use contributes a gradient during the single `backward()` call, and they are summed into the weight's `.grad` (the branching rule / accumulation of section 7). One backward pass, three contributions added — no manual bookkeeping needed.

</details>

## Next

You now understand training's engine end to end: forward builds the graph, `backward()` replays the chain rule over it as vector-Jacobian products, and gradients land in `.grad` — ready to be consumed. What consumes them is the **optimizer**: the rule that turns gradients into parameter updates. The next module builds it from the simplest case, plain gradient descent, up through momentum, Adam, and AdamW.

Continue to [26 · Gradient descent from scratch](lessons/module-09/lesson-01.md).
