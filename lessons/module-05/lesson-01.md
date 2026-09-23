# 14 · Composition and computational graphs

<div class="prereq">
<p><strong>Prerequisites:</strong> Functions and the notation $f(x)$ (<a href="#/lessons/module-01/lesson-04">04 · Functions and equations</a>), vectors as lists of numbers (<a href="#/lessons/module-02/lesson-01">06 · What a vector is</a>), matrix and matrix–vector multiplication (<a href="#/lessons/module-03/lesson-02">10 · Matrix multiplication</a>), and tensors and shapes (<a href="#/lessons/module-04/lesson-01">12 · Tensors and shapes</a>).</p>
<p><strong>You will learn:</strong> how functions take one input, many inputs, or produce whole vectors; what it means to <em>compose</em> functions as $f(g(x))$ and evaluate them inside-out; why composition order matters; and how to draw any composition as a <em>computational graph</em> whose nodes are operations and whose edges carry data.</p>
<p><strong>Why this matters for ML:</strong> a neural network is nothing but a big composition of functions. The <em>forward pass</em> that produces a prediction is just evaluating that composition left to right, and the stored intermediate values are exactly what backpropagation will reuse. This lesson is the bridge from algebra to training.</p>
</div>

## 1. Functions, from one input to many outputs

You already met a function as a machine: give it an input, it returns an output. This lesson stretches that idea in three directions, because a neural network needs all three.

### 1.1 One-input functions (recap)

A **one-input function** takes a single number and returns a single number. We write

$$
f(x) = x^2
$$

- $f$ — the name of the function (the machine).
- $x$ — the input, a single scalar (one plain number).
- $f(x)$ — the output, also a single scalar; here it is the input squared.

Feed it $x = 3$ and you get $f(3) = 9$. One number in, one number out. The whole of Part 1 lived here.

### 1.2 Multi-input functions $f(x, y)$

Most interesting functions depend on **several** inputs at once. A **multi-input function** takes two or more numbers and returns one number. For example

$$
f(x, y) = x^2 + 3xy + y^2
$$

- $x, y$ — two separate scalar inputs. The comma means "both of these, in this order."
- $f(x, y)$ — a single scalar output computed from both.

Evaluate at $x = 1, y = 2$: $f(1, 2) = 1 + 3\cdot 1 \cdot 2 + 4 = 1 + 6 + 4 = 11$. Two numbers in, one number out. When we later ask "how does the output change if I nudge only $x$?" we will be doing partial derivatives (Part 7) — but the function itself is just a rule with two input slots.

<div class="callout key"><p>The number of inputs is just the number of named slots in the parentheses. $f(x)$ has one slot, $f(x, y)$ has two. Nothing deeper is going on.</p></div>

### 1.3 Vector-valued functions $f : \mathbb{R}^n \to \mathbb{R}^m$

Now the payoff direction for ML. A function can take a whole vector as input and return a whole vector as output. We write its **type signature** as

$$
f : \mathbb{R}^n \to \mathbb{R}^m
$$

Read this literally: "$f$ maps a point in $n$-dimensional space to a point in $m$-dimensional space." The symbol $\mathbb{R}$ means the set of all real numbers, $\mathbb{R}^n$ means the set of all length-$n$ vectors of real numbers, and the arrow $\to$ separates the input space from the output space.

- $n$ — how many numbers go in (the input vector's length).
- $m$ — how many numbers come out (the output vector's length).
- $n$ and $m$ need **not** be equal: a function can turn a 3-vector into a 2-vector.

The concrete example you will meet everywhere is a **linear layer**. Take an input vector $\mathbf{x}$ of length $n$, a weight matrix $\mathbf{W}$ of shape $(m, n)$, and a bias vector $\mathbf{b}$ of length $m$, and define

$$
f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

- $\mathbf{x}$ — input vector, shape $(n,)$.
- $\mathbf{W}$ — weight matrix, shape $(m, n)$.
- $\mathbf{W}\mathbf{x}$ — matrix–vector product from Part 3, shape $(m,)$.
- $\mathbf{b}$ — bias vector, shape $(m,)$, added element-wise.
- $f(\mathbf{x})$ — output vector, shape $(m,)$.

This single function eats an $n$-vector and produces an $m$-vector, so its signature is $f : \mathbb{R}^n \to \mathbb{R}^m$. A tiny case: with $n = 2, m = 2$,

$$
\mathbf{W} = \begin{bmatrix} 1 & 0 \\ 2 & 3 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
$$

gives $\mathbf{W}\mathbf{x} = \begin{bmatrix} 1\cdot 2 + 0 \cdot 1 \\ 2\cdot 2 + 3\cdot 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 7 \end{bmatrix}$, and then $f(\mathbf{x}) = \begin{bmatrix} 2 \\ 7 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 7 \end{bmatrix}$. A whole vector went in; a whole vector came out. When a GPT layer "produces a hidden vector," this is exactly the kind of function that produced it.

## 2. Composition: doing one function after another

### 2.1 Intuition

Software is built from small functions that call each other: `parse` feeds `validate` feeds `save`. Mathematics has the same move, and calls it **composition**. If $g$ turns $x$ into something, and $f$ knows what to do with that something, then "do $g$, then do $f$" is itself a function. That combined function is written $f(g(x))$, or with the dedicated symbol $f \circ g$ (read "$f$ after $g$").

The word **after** is the whole point. Even though $f$ is written on the left, $g$ runs **first**. You evaluate from the inside out, exactly like nested parentheses in code: `f(g(x))` runs `g(x)` before `f` ever sees an argument.

### 2.2 The mathematics

Given two functions

$$
g(x) = 2x \qquad\text{and}\qquad f(u) = u^2,
$$

their composition is the function you get by feeding $g$'s output straight into $f$:

$$
y = f(g(x)) = \big(g(x)\big)^2 = (2x)^2 = 4x^2.
$$

- $x$ — the original input.
- $g(x) = 2x$ — the **inner** function; it runs first.
- $u$ — a name for $g$'s output, the value handed to $f$. Using a fresh letter $u$ keeps the two stages straight.
- $f(u) = u^2$ — the **outer** function; it runs second, on whatever $g$ produced.
- $y = f(g(x))$ — the final output of the whole pipeline.

Notice we substituted $u = g(x) = 2x$ into $f(u) = u^2$ and simplified to a single closed form $4x^2$. That simplification is optional; the composition is defined even when you cannot simplify it.

### 2.3 A tiny numerical example, step by step

Evaluate $y = f(g(x))$ at $x = 3$, inside-out:

1. **Inner first.** $u = g(3) = 2 \cdot 3 = 6$.
2. **Outer next.** $y = f(u) = f(6) = 6^2 = 36$.

Cross-check with the simplified form: $4x^2 = 4 \cdot 3^2 = 4 \cdot 9 = 36$. The two agree. The middle value $u = 6$ is not part of the final answer, yet we had to compute and hold onto it to get there. Remember that value — it is the seed of backpropagation.

### 2.4 A three-deep chain

Nothing stops us stacking a third function. Add $h(v) = v + 1$ on the outside and evaluate $h(f(g(x)))$ at $x = 3$:

1. $u = g(3) = 6$.
2. $v = f(6) = 36$.
3. $y = h(36) = 36 + 1 = 37$.

Three stages, three intermediate names $u, v, y$, each computed from the one before. A real network stacks dozens of such stages; the mechanics never change.

### 2.5 Order matters — composition is not commutative

With ordinary multiplication $a \cdot b = b \cdot a$. Composition has **no** such symmetry. Using the same $g(x) = 2x$ and $f(u) = u^2$:

$$
f(g(x)) = (2x)^2 = 4x^2 \qquad\text{but}\qquad g(f(x)) = 2(x^2) = 2x^2.
$$

At $x = 3$ that is $36$ versus $18$ — different functions entirely. Swapping the order changes the result, so in general

$$
f \circ g \ne g \circ f.
$$

This should feel familiar: in <a href="#/lessons/module-03/lesson-02">10 · Matrix multiplication</a> you saw that $\mathbf{A}\mathbf{B} \ne \mathbf{B}\mathbf{A}$. That is not a coincidence. Applying a matrix is a function, and stacking two linear layers *is* composing their functions — so the non-commutativity of matrix products is the non-commutativity of function composition wearing different clothes. The parentheses and the ordering are load-bearing; get them wrong and you have built a different network.

## 3. Computational graphs

### 3.1 The idea

Writing $y = 4x^2$ hides the steps. Writing $y = f(g(x))$ shows two steps but buries them in nested parentheses. A **computational graph** makes every step explicit by drawing each operation as a **node** and each value flowing between operations as an **edge**. Inputs enter on the left, data flows rightward along the edges, the final output leaves on the right.

Here is the graph for our example $g(x) = 2x$ then $f(u) = u^2$:

```text
        [ * 2 ]          [ square ]
  x ------------->  u  ------------->  y
 input   multiply     store    square   output
         by two     intermediate
```

Reading it: the value $x$ flows into the node `* 2`, which multiplies by two and emits $u$; $u$ flows into the node `square`, which emits $y$. Every arrow carries one concrete number during evaluation. At $x = 3$ the top arrow carries $3$, the middle carries $u = 6$, the last carries $y = 36$ — the exact inside-out trace from Section 2.3.

### 3.2 Every intermediate gets a name and is stored

The crucial detail is the labelled value $u$ sitting *between* the two nodes. It is not decoration. To evaluate the graph you must compute $u$ and **hold it in memory** before the `square` node can run. Each operation produces a named intermediate that later operations consume.

<div class="callout key"><p>A computational graph turns one tangled formula into a list of tiny operations, each with a named, stored output. Those stored intermediates are what the backward pass will reach back for when it computes gradients in Modules 6–7.</p></div>

Why store them at all, rather than recompute on demand? Because the backward pass needs them. To find how $y$ responds to a nudge in $x$, the chain rule (next module) multiplies the local sensitivity at each node, and the local sensitivity of the `square` node depends on the value $u$ that flowed into it. If you threw $u$ away after the forward pass, you would have to recompute it. Frameworks keep it instead — trading memory for speed. That trade is why training a large model needs so much more memory than merely running it.

## 4. The big payoff: a neural network is a composition

### 4.1 Machine-learning application — the forward pass

Everything above was rehearsal for one sentence: **a neural network is a composition of functions, and running it is evaluating that composition.** Each layer is a function; the network is the layers composed; a prediction is the inside-out evaluation of that stack.

Here is a small language-model-style pipeline drawn as a graph, with each edge labelled by the intermediate tensor it carries:

```text
 tokens        embedding       linear W1       activation      linear W2         loss
  ids  --->  [ lookup ] ---> [ x@W1+b1 ] ---> [  relu  ] ---> [ x@W2+b2 ] ---> [ cross-ent ]
              e (B,T,C)       z1 (B,T,H)       a1 (B,T,H)      z2 (B,T,V)       L (scalar)
```

Reading left to right, each box is a function and each label under an arrow is the stored output of the box to its left:

- `lookup` turns token ids into an embedding tensor $\mathbf{e}$ of shape $(B, T, C)$ — batch, time, channels, the dimensions from Part 4.
- the first linear layer computes $z_1 = \mathbf{e}\mathbf{W}_1 + \mathbf{b}_1$, a pre-activation of shape $(B, T, H)$.
- the activation applies $a_1 = \mathrm{relu}(z_1)$ element-wise, same shape.
- the second linear layer computes $z_2 = a_1 \mathbf{W}_2 + \mathbf{b}_2$, the logits of shape $(B, T, V)$ over the vocabulary.
- the loss node compares logits to targets and emits a single scalar $L$.

As one composition that reads

$$
L = \text{loss}\Big(\mathbf{W}_2\,\mathrm{relu}(\mathbf{W}_1\,\text{embed}(\text{ids}) + \mathbf{b}_1) + \mathbf{b}_2,\ \text{targets}\Big).
$$

<div class="callout key"><p>The <strong>forward pass</strong> is exactly this: evaluate the composition left to right, storing every intermediate ($\mathbf{e}, z_1, a_1, z_2, L$) as you go. That is all "running the model" means. The stored intermediates then feed the <strong>backward pass</strong> — the chain rule applied to this same graph — which you will build in Modules 6 and 7.</p></div>

### 4.2 In PyTorch

First, composition on a tensor exactly as in Section 2 — define $g$ and $f$ as plain Python functions and chain them:

```python
import torch

def g(x):
    return 2 * x        # inner function, runs first

def f(u):
    return u ** 2       # outer function, runs second

x = torch.tensor(3.0)
u = g(x)                # 6.0  -- the stored intermediate
y = f(u)                # 36.0
print(u, y)             # tensor(6.) tensor(36.)

# composing in one line reads outer(inner(x)):
print(f(g(x)))          # tensor(36.)
```

The one-liner `f(g(x))` runs `g` first and `f` second — the code order matches the math order. Now the two-layer network from the pipeline, written with plain tensor operations so every intermediate is visible:

```python
import torch

x  = torch.randn(4, 3)        # batch of 4 inputs, each length 3   -> (4, 3)
W1 = torch.randn(3, 5)        # first weight matrix                -> (3, 5)
W2 = torch.randn(5, 2)        # second weight matrix               -> (5, 2)

z1 = x @ W1                   # linear layer 1: pre-activation      (4, 5)
a1 = z1.relu()                # activation: element-wise ReLU       (4, 5)
z2 = a1 @ W2                  # linear layer 2: logits              (4, 2)

print(z1.shape, a1.shape, z2.shape)   # (4,5) (4,5) (4,2)
```

Read it as a composition: `z2 = relu(x @ W1) @ W2`. Each named variable `z1`, `a1`, `z2` is one stored intermediate — one edge in the graph. Writing the steps out (rather than cramming them into one expression) is not just for readability; it mirrors exactly how the framework will lay the graph out internally.

### 4.3 What PyTorch is doing under the hood

Here is the quiet part that this whole lesson has been setting up. When you write `z1 = x @ W1`, PyTorch does more than compute the product. If any input requires gradients, it also **records the operation as a node in a graph** and remembers the inputs it will need later — much like the labelled `u` we insisted on storing. Each result tensor carries a hidden reference (`grad_fn`) pointing back to the operation that produced it, so the chain of operations `x -> z1 -> a1 -> z2` becomes a literal linked graph in memory, built step by step **as you execute the forward pass**.

This is PyTorch's *dynamic* computational graph: it is not declared ahead of time, it is assembled on the fly by the ordinary act of running your Python. By the time you reach the loss, PyTorch holds the entire graph plus the stored intermediates, and a single call to `.backward()` will walk that graph in reverse to produce gradients. We are keeping this light on purpose — `grad_fn`, `requires_grad`, and `.backward()` get the full six-pass treatment once you have the chain rule. For now, hold one idea: the forward pass you just wrote *is* the graph, and every intermediate you stored is there because the backward pass is going to want it.

## Check yourself

<details><summary>With $g(x) = x + 1$ and $f(u) = 3u$, what is $f(g(2))$, and does it equal $g(f(2))$?</summary>

Inside-out: $g(2) = 3$, then $f(3) = 9$, so $f(g(2)) = 9$. The other order: $f(2) = 6$, then $g(6) = 7$, so $g(f(2)) = 7$. They differ ($9 \ne 7$) — composition is not commutative.

</details>

<details><summary>In the graph <code>x -> [*2] -> u -> [square] -> y</code>, why does PyTorch bother storing $u$ during the forward pass?</summary>

Because the backward pass needs it. The local sensitivity of the `square` node depends on the value $u$ that flowed into it, so $u$ must survive the forward pass to be reused when gradients are computed. Storing intermediates is the memory cost that makes backprop fast.

</details>

<details><summary>A linear layer has signature $f : \mathbb{R}^n \to \mathbb{R}^m$ with $f(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}$. What are the shapes of $\mathbf{W}$ and $\mathbf{b}$?</summary>

$\mathbf{W}$ has shape $(m, n)$ so that $\mathbf{W}\mathbf{x}$ turns a length-$n$ vector into a length-$m$ vector, and $\mathbf{b}$ has shape $(m,)$ so it can be added to that length-$m$ result.

</details>

## Next

You now have the one structural idea the rest of the course leans on: a network is a composition, its forward pass evaluates that composition and stores every intermediate, and that stored graph is what training reads backward. Next you meet the tool that reads it backward — the derivative, and then the chain rule that turns a graph into gradients.

Continue to <a href="#/lessons/module-06/lesson-01">15 · Limits, slope, and the derivative</a>.
