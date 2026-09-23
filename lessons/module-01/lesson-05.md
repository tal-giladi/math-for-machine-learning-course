# 05 · The notation zoo: ∂ ∇ ‖x‖ indices transpose

<div class="prereq">
<p><strong>Prerequisites:</strong> Lessons <a href="#/lessons/module-01/lesson-01">01</a>–<a href="#/lessons/module-01/lesson-04">04</a> — numbers, functions, powers/logs, and summation.</p>
<p><strong>You will learn:</strong> A first, friendly pass over the symbols the rest of the course leans on — subscripts and superscripts, indices, $\partial$, $\nabla$, $\|x\|$, transpose $A^\top$, inverse $A^{-1}$, elementwise $\odot$, and the relation symbols $\approx$, $\propto$, $\in$, plus $\arg\max$ / $\arg\min$.</p>
<p><strong>Why this matters for ML:</strong> ML papers and PyTorch code are dense with these symbols. Meeting them once, with a rough meaning and a pointer to where each gets its full lesson, means they will read as words instead of noise when you hit them for real.</p>
</div>

This lesson is deliberately different from the others: it is a **reference tour, not a deep dive**. Every symbol below gets a full, careful lesson later in the course — this page just removes the fear so the symbols stop being scary the first time you see them in the wild. Skim it now; come back to it whenever a symbol ambushes you.

<div class="callout key"><p><strong>Key idea.</strong> Everything on this page is a <em>preview</em>. You are not expected to compute with these yet. The goal is recognition: "ah, that triangle is the gradient, it collects rates of change, and Module 6 will teach it." Rough meaning now; rigor later.</p></div>

## 1. Subscripts and superscripts

A **subscript** is a small number or letter written below and to the right of a symbol, used to *index* — to say "which one". In the list $a_1, a_2, a_3$, the subscript picks out the first, second, third element. In ML, $x_i$ is the $i$-th input, $w_{ij}$ is the weight connecting unit $j$ to unit $i$ (two subscripts, a row and a column), and $\theta_k$ is the $k$-th parameter. A subscript is a *label*, not an operation: $x_2$ does not mean "$x$ times 2", it means "the second $x$".

A **superscript** is written above and to the right — and here is the one genuine trap in this lesson, because a superscript means **two completely different things** depending on context:

- As a **power** (Lesson 03): $x^2$ means $x \cdot x$. This is arithmetic.
- As an **index/label**: $x^{(1)}, x^{(2)}$ might mean "the first training example, the second training example". To signal "this is a label, not a power", the index is very often wrapped in **parentheses**: $x^{(t)}$ means "the value at step/layer $t$", not "$x$ raised to the $t$".

<div class="callout warn"><p><strong>Gotcha — the single most common notation confusion.</strong> A superscript in parentheses like $a^{(2)}$ is an <em>index</em> ("the second one"), while a bare superscript like $a^2$ is a <em>power</em> ("a squared"). When you see a superscript, ask: is it in parentheses (or clearly a label like a layer number)? Then it indexes. Otherwise it is an exponent. Subscripts, by contrast, are <em>always</em> indices, never powers.</p></div>

## 2. The reference table

Here is the zoo, one row per symbol. "Read as" is what to say in your head; "rough meaning" is the one-line intuition; "where in ML" is why you will care. Each is a preview — the "full lesson" column points to where it is taught properly.

| symbol | read as | rough meaning | where it appears in ML | full lesson |
|:--|:--|:--|:--|:--|
| $x_i$ | "x sub i" | the $i$-th element of a list | indexing tokens, features, parameters | Lesson 04, Module 2 |
| $x^{(t)}$ | "x super-paren t" | the item at index/step $t$ (a label, not a power) | training example $t$, layer $t$, time step $t$ | Module 4 |
| $x^2$ | "x squared" | a power: $x \cdot x$ | squared error, variance | Lesson 03 |
| $\sum_{i=1}^{n} a_i$ | "sum from i=1 to n of a-i" | add up the terms $a_1$ through $a_n$ | averaging a loss over a batch | Lesson 04 |
| $\prod_{i=1}^{n} a_i$ | "product from i=1 to n of a-i" | multiply the terms together | likelihood of a sequence | Lesson 04, Module 10 |
| $\|x\|$ | "absolute value of x" | distance of a number from $0$ | error size, L1 penalties | Lesson 04 |
| $\|\mathbf{x}\|$ | "norm of x" | length of a vector | measuring vector size, normalization | Module 2 |
| $\frac{d y}{d x}$ | "d y d x" | derivative: rate of change of $y$ as $x$ changes | how a loss responds to one input | Module 6 |
| $\frac{\partial L}{\partial \theta}$ | "partial L partial theta" | partial derivative: rate of change of $L$ wrt one variable, others held fixed | sensitivity of loss to one parameter | Module 6 |
| $\nabla_\theta L$ | "grad L" / "del L" | gradient: the vector of all partial derivatives | the direction training moves parameters | Module 6, Module 7 |
| $A^\top$ | "A transpose" | flip a matrix's rows and columns | $q \cdot k^\top$ in attention, weight reuse | Module 3, Module 11 |
| $A^{-1}$ | "A inverse" | the matrix that undoes $A$ | solving linear systems (rare in training) | Module 3 |
| $A \odot B$ | "A elementwise-times B" | multiply matching entries, one by one | gating, masks, dropout | Module 3, Module 4 |
| $\approx$ | "approximately equals" | close but not exactly equal | numerical values, float results | throughout |
| $\propto$ | "is proportional to" | equal up to a constant factor | softmax before normalizing | Module 10 |
| $\in$ | "is an element of" / "in" | belongs to a set | $x \in \mathbb{R}$, $p \in [0,1]$ | Lesson 01 |
| $\arg\max_i$ | "arg max over i" | the index that gives the largest value | picking the predicted token | Module 10 |
| $\arg\min_i$ | "arg min over i" | the index that gives the smallest value | choosing the best option | Module 9 |

The rest of the lesson expands the handful most worth a few extra sentences now.

## 3. $\partial$ — the partial derivative

The rounded-d symbol $\partial$ (say "partial" or "del") is used for a **partial derivative**. We define it carefully in Module 6, but the one-line preview is: *when a function has several inputs, the partial derivative measures its rate of change with respect to one of them while holding all the others fixed.*

It is written like a fraction:

$$
\frac{\partial L}{\partial \theta}
$$

Read "the partial derivative of $L$ with respect to $\theta$", and understood as: "if I nudge $\theta$ a tiny bit and freeze everything else, how fast does $L$ change?" That is the same "sensitivity" idea as the average rate of change from Lesson 04, sharpened to a single point and a single variable. It uses $\partial$ instead of the plain $d$ purely to signal "there are other variables around that we are holding still". For now: **$\partial$ = rate of change with respect to one variable, others held constant.**

## 4. $\nabla$ — the gradient

The upside-down triangle $\nabla$ (say "nabla" or "del") builds directly on $\partial$. The **gradient** of a function of many variables is simply the *list of all its partial derivatives, one per variable, packed into a vector*:

$$
\nabla_\theta L = \left[ \frac{\partial L}{\partial \theta_1}, \; \frac{\partial L}{\partial \theta_2}, \; \dots, \; \frac{\partial L}{\partial \theta_n} \right]
$$

Read "the gradient of $L$ with respect to $\theta$". Each entry says how sensitive $L$ is to one parameter; collected together, the gradient points in the direction that increases $L$ fastest — so its negative points *downhill*, toward smaller loss. That is the whole engine of training: "compute $\nabla_\theta L$, step the parameters a little in the $-\nabla$ direction, repeat." You will see this preview pay off in Module 6 (where we build it) and Module 9 (gradient descent). For now: **$\nabla$ = a vector of every partial derivative = the direction of steepest change.**

## 5. $\|\mathbf{x}\|$ — the norm

Double bars around a **bold** symbol (bold marks a vector, per the style of this course) mean the **norm** — the length of the vector, its distance from the origin. It generalizes the absolute value $|x|$ of a single number to a whole list of numbers. The most common norm is the Euclidean one, computed with the Pythagorean theorem: square each entry, add (there is our $\sum$ again), take the square root (there is Lesson 03's $x^{1/2}$):

$$
\|\mathbf{x}\| = \sqrt{\sum_{i} x_i^2}
$$

Read "the norm of $\mathbf{x}$" or "the length of $\mathbf{x}$". Norms measure how big a vector is, which matters for normalization (making a vector unit length) and for detecting exploding gradients. Full treatment in Module 2. For now: **$\|\mathbf{x}\|$ = the length of a vector.**

## 6. Transpose $A^\top$ and inverse $A^{-1}$

These act on **matrices** (rectangular grids of numbers, the subject of Module 3), so we only preview them.

**Transpose**, written $A^\top$ ("A transpose"), flips a matrix over its diagonal: rows become columns and columns become rows. A $2 \times 3$ matrix becomes $3 \times 2$. You will meet it constantly in attention, where the score computation is literally $q\,k^\top$ — the queries times the transpose of the keys. Rough meaning: **transpose = swap rows and columns.**

**Inverse**, written $A^{-1}$ ("A inverse"), is the matrix that undoes $A$: applying $A$ then $A^{-1}$ gets you back where you started, analogous to how $\frac{1}{5}$ undoes multiplication by $5$. Note the superscript $-1$ here is *not* the "one-over" power from Lesson 03 (a matrix is not a single number); it is a labelled operation meaning "the inverse matrix". Inverses are central to classical linear algebra but, usefully, appear only rarely in neural-network *training* (we almost never invert weight matrices). Rough meaning: **inverse = the matrix that reverses $A$.**

## 7. Elementwise $\odot$ and the relation symbols

**Elementwise product** $\odot$ (the circled dot, "elementwise times" or "Hadamard product") multiplies two equally shaped collections *entry by entry*: the first entry of one times the first entry of the other, and so on — as opposed to the more involved matrix multiplication of Module 3. In PyTorch this is just the ordinary `*` operator on two tensors. It shows up in gating, attention masks, and dropout. Rough meaning: **$\odot$ = multiply matching positions, one at a time.**

The relation symbols are quick:

- $\approx$ — "approximately equals". Used whenever we round: $e \approx 2.718$.
- $\propto$ — "is proportional to", i.e. equal after multiplying by some constant. Softmax computes something $\propto e^{\text{logit}}$ and then divides by the total to normalize (Module 10).
- $\in$ — "is an element of" / "belongs to". $x \in \mathbb{R}$ says "$x$ is a real number"; $p \in [0, 1]$ says "$p$ is a probability".

## 8. $\arg\max$ and $\arg\min$

A subtle but important pair. Given a list of values, $\max$ returns the **largest value**, while $\arg\max$ returns the **index (position) where that largest value occurs** — "the *argument* that maximizes". They answer different questions:

- $\max_i a_i$ → *how big* is the biggest? (a value)
- $\arg\max_i a_i$ → *where* is the biggest? (a position)

For the list $a = [2.0, 5.0, 1.0]$: $\max = 5.0$ (the value), but $\arg\max = 2$ (the position of that value, counting from 1; PyTorch counts from 0 and would say index 1). In language modeling, once the model produces a probability for each token, $\arg\max$ over those probabilities gives the *index of the most likely next token* — which you then map back to an actual word. $\arg\min$ is the mirror image: the position of the smallest value. Rough meaning: **max/min give the value; argmax/argmin give the location.**

## 9. In PyTorch (kept light)

A quick, real taste of three of these — the rest arrive with their full lessons.

```python
import torch

A = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])          # a 2x3 matrix

# Transpose: rows <-> columns, so 2x3 becomes 3x2.
print(torch.transpose(A, 0, 1).shape)     # torch.Size([3, 2])

# Norm (length) of a vector: sqrt(sum of squares).
x = torch.tensor([3., 4.])
print(torch.linalg.norm(x))               # tensor(5.)   <- sqrt(3^2 + 4^2) = sqrt(25) = 5

# Elementwise product: * multiplies matching entries, one by one.
a = torch.tensor([2., 5., 1.])
b = torch.tensor([10., 10., 10.])
print(a * b)                              # tensor([20., 50., 10.])

# argmax: the position of the largest entry (0-indexed in PyTorch).
print(torch.argmax(a))                    # tensor(1)   <- 5. is the largest, at index 1
```

<div class="callout pt"><p><strong>PyTorch note.</strong> Notice how the plain <code>*</code> operator is the elementwise $\odot$, while true matrix multiplication (Module 3) uses a different operator, <code>@</code>. Confusing the two is one of the most common bugs when translating math into PyTorch, so it is worth flagging now: <code>*</code> is entry-by-entry, <code>@</code> is matrix multiplication. Also note <code>torch.linalg.norm</code> gave exactly $5$ for the vector $[3, 4]$, the classic 3-4-5 right triangle — the norm really is just Pythagoras.</p></div>

## 10. What PyTorch is doing under the hood

Nothing new mechanically here — these are the same tensor operations from earlier lessons, and each is differentiable so autograd can flow gradients through them. The one conceptual point to carry forward is that PyTorch has a *separate function or operator for each symbol in the table*: `torch.sum` for $\sum$, `torch.prod` for $\prod$, `torch.abs` for $|x|$, `torch.exp` and `torch.log` for $e^x$ and $\ln$, `@` for matrix multiply, `*` for $\odot$, `.T` / `torch.transpose` for $A^\top$, `torch.linalg.norm` for $\|\mathbf{x}\|$, `torch.argmax` for $\arg\max$, and — once we reach training — `.backward()` to fill in the $\nabla$ and $\partial$ quantities automatically. Learning to read the math is, to a large degree, learning this dictionary between symbols and PyTorch calls. The rest of the course builds it entry by entry.

## Check yourself

1. In $x^{(3)}$ versus $x^3$, which is an index and which is a power? How can you tell?

<details><summary>Answer</summary>

$x^{(3)}$ (parentheses) is an <em>index</em> — "the third one". $x^3$ (bare superscript) is a <em>power</em> — $x \cdot x \cdot x$. The parentheses are the signal that it is a label rather than an exponent. Subscripts are always indices.

</details>

2. In one line each: what does $\partial$ measure, and what does $\nabla$ collect?

<details><summary>Answer</summary>

$\partial$ measures the rate of change of a function with respect to one variable while holding the others fixed (a partial derivative). $\nabla$ collects all of a function's partial derivatives into one vector — the gradient — which points in the direction of steepest increase.

</details>

3. For the list $[7, 2, 9, 4]$, what are $\max$ and $\arg\max$ (using 1-based positions)?

<details><summary>Answer</summary>

$\max = 9$ (the largest value). $\arg\max = 3$ (the position of that value, third in the list; PyTorch, counting from 0, would report index 2).

</details>

4. What is the difference between $|x|$ and $\|\mathbf{x}\|$?

<details><summary>Answer</summary>

$|x|$ is the absolute value of a single number — its distance from $0$. $\|\mathbf{x}\|$ is the norm (length) of a whole vector — the square root of the sum of the squares of its entries. Single bars around a scalar, double bars around a vector.

</details>

## Next

That closes Part 1: you can now read the language of mathematics — numbers, variables, functions, powers, logs, sums, products, and the symbol zoo. Part 2 puts that language to work on the first real ML object, the **vector**: a list of numbers that will represent a token, a hidden state, or a gradient.

[06 · What a vector is →](lessons/module-02/lesson-01.md)
