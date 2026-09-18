# 01 · Numbers, variables, expressions, equations

<div class="prereq">
<p><strong>Prerequisites:</strong> You can program (loops, variables, functions) in some language such as C#. No mathematics beyond arithmetic is assumed.</p>
<p><strong>You will learn:</strong> What the different kinds of numbers are and why ML lives in the real numbers (as floats); what a mathematical variable really is and how it differs from a programming variable; the difference between an expression and an equation; how to substitute and evaluate; and how to solve a simple linear equation.</p>
<p><strong>Why this matters for ML:</strong> A neural network is, underneath, one gigantic mathematical expression built from numbers and variables. "Training" is the process of choosing values for millions of those variables. If the vocabulary of numbers, variables, and expressions is fuzzy, every later page is fuzzy too. We fix that here.</p>
</div>

This is the ground floor. We are going to be slow and explicit, because everything else stacks on top of it. Even if some of this feels familiar, read it — the goal is that a symbol never surprises you later.

## 1. Numbers

Mathematics starts with numbers, but "number" is not one single thing. There are several families, and they nest inside each other like Russian dolls.

**Natural numbers** are the counting numbers: $0, 1, 2, 3, \dots$ (some authors start at 1). You use these for counting tokens, layers, or the size of a vocabulary.

**Integers** add the negatives: $\dots, -3, -2, -1, 0, 1, 2, 3, \dots$. The symbol for the whole set of integers is $\mathbb{Z}$. In ML, integers show up as *indices* — "the 5th token", "row 2 of a matrix" — and as *token IDs* (a word mapped to a whole number like 15496).

**Rational numbers** are anything you can write as a fraction $\frac{a}{b}$ where $a$ and $b$ are integers and $b \neq 0$. Examples: $\frac{1}{2} = 0.5$, $\frac{3}{4} = 0.75$, $-\frac{7}{2} = -3.5$. The set of rationals is written $\mathbb{Q}$. Every integer is also a rational (for example $5 = \frac{5}{1}$), which is what "nesting" means.

**Real numbers** include the rationals *and* the numbers that cannot be written as an exact fraction, such as $\sqrt{2} = 1.41421\ldots$ and $\pi = 3.14159\ldots$. These have decimal expansions that never terminate and never repeat. The set of reals is written $\mathbb{R}$. Think of the reals as every point on a continuous number line, with no gaps.

<div class="callout key"><p><strong>Key idea.</strong> The families nest: $\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R}$. The symbol $\subset$ means "is contained in". So every natural number is an integer, every integer is a rational, and every rational is a real.</p></div>

### Why ML uses real numbers

Machine learning is about *smooth adjustment*. During training we nudge a number a tiny bit — say from $0.7$ to $0.7001$ — and ask whether that made the model slightly better or slightly worse. That only makes sense if the numbers can vary continuously, taking any value on the line, not just whole steps. That is exactly what the real numbers $\mathbb{R}$ give us. A weight in a network is a real number; so is a probability like $0.63$; so is a loss value like $2.41$.

### Reals vs. floats: the honest version

There is a catch. A computer cannot store a number with infinitely many decimal digits, so it does not really store real numbers. It stores **floating-point numbers** ("floats"): real numbers rounded to a fixed budget of precision. A 32-bit float keeps roughly 7 decimal digits of accuracy.

So when the mathematics says $\frac{1}{3} = 0.3333\ldots$ forever, the computer stores something like $0.33333334$ and stops. Mathematically we reason in $\mathbb{R}$; in code we compute in floats. Ninety-nine percent of the time the gap does not matter, and we will treat tensors as if they held true real numbers. But that gap is the source of a whole later topic — overflow, underflow, and numerical stability — so it is worth knowing it exists from day one.

<div class="callout warn"><p><strong>Gotcha.</strong> Floats are approximations. In Python, <code>0.1 + 0.2</code> prints <code>0.30000000000000004</code>, not <code>0.3</code>. This is not a bug; it is the price of storing reals in finite memory. We return to it in Module 12.</p></div>

## 2. Variables and constants

A **constant** is a fixed number that does not change: $7$ is a constant, and so is $\pi$. It always means the same value.

A **variable** is a *name that stands in for a value*. This is the single most important idea to get right, because a mathematical variable is **not** the same thing as a programming variable, even though we reuse the word.

In C#, when you write `x = 2;` and later `x = x + 1;`, the name `x` is a **box in memory** whose contents you overwrite. It is *mutable*: `x` was 2, now it is 3. The assignment `x = x + 1` is a command that changes the box.

In mathematics, a variable like $x$ is a **name for one unknown-but-fixed value**, or a placeholder we will plug a value into. When a mathematician writes

$$
y = 3x + 1
$$

they are not issuing a command and they are not overwriting anything. They are stating a *relationship*: whatever $x$ turns out to be, $y$ is three times it plus one. Within one problem, $x$ does not "change"; we may not yet *know* its value, or we may be describing how $y$ depends on it, but $x = x + 1$ would be read as an *equation* (which happens to have no solution), not as an instruction.

<div class="callout key"><p><strong>Key idea.</strong> A programming variable is a mutable box you assign to. A mathematical variable is a name for a value — an unknown to solve for, or a placeholder to substitute into. Same word, different meaning. Keep the two pictures separate and a lot of confusion disappears.</p></div>

A quick vocabulary note: symbols playing the role of "fixed but unspecified numbers" are often called **parameters** or **coefficients**. In $y = 3x + 1$, the $3$ and the $1$ are constants (coefficients); $x$ and $y$ are variables. In a neural network, the numbers that get *learned* are called parameters — and, confusingly for a programmer, those are the ones that actually do change during training. We will keep the language precise as we go.

## 3. Expressions vs. equations

These two words look similar and are constantly confused. The difference is simple once you see it.

An **expression** is a recipe that produces a value. It has no equals sign. Examples:

$$
3x + 1, \qquad x^2 - 4, \qquad \frac{a + b}{2}
$$

An expression is like a function body with no `return` statement written yet — a computation waiting to be evaluated once you supply values for its variables.

An **equation** is a claim that two expressions are equal. It has an equals sign in the middle. Examples:

$$
3x + 1 = 7, \qquad x^2 - 4 = 0, \qquad y = 3x + 1
$$

An equation is either something you *solve* (find the $x$ that makes it true) or something that *defines a relationship* (like $y = 3x + 1$, which tells you how to get $y$ from $x$).

<div class="callout key"><p><strong>Key idea.</strong> No equals sign → expression (a value to compute). One equals sign → equation (a statement to solve or a relationship to use).</p></div>

## 4. Substitution and evaluation

**Substitution** means replacing a variable with a specific value. **Evaluation** means then doing the arithmetic to get a single number. This is the mathematical version of "calling a function with an argument".

Take the expression $3x + 1$ and let $x = 2$. Substitute:

$$
3x + 1 \;\longrightarrow\; 3 \cdot (2) + 1
$$

Now evaluate, respecting order of operations (multiply before you add):

$$
3 \cdot 2 + 1 = 6 + 1 = 7
$$

So when $x = 2$, the expression $3x + 1$ evaluates to $7$. Notice $3x$ means "3 times $x$"; in mathematics we usually drop the multiplication sign between a number and a variable.

Let us do one more, with the relationship $y = 3x + 1$, at $x = 4$:

$$
y = 3 \cdot 4 + 1 = 12 + 1 = 13
$$

That is the whole idea of a "forward pass" in miniature: you have an expression with variables, you plug in inputs, and you turn the crank to get an output.

## 5. Solving a simple linear equation

"Solving" an equation means finding the value(s) of the variable that make the equation true. The core technique: do the *same operation to both sides* so that the equality is preserved, step by step, until the variable is alone on one side.

Solve $3x + 1 = 7$.

Step 1 — subtract $1$ from both sides to peel off the $+1$:

$$
3x + 1 - 1 = 7 - 1 \quad\Longrightarrow\quad 3x = 6
$$

Step 2 — divide both sides by $3$ to peel off the multiplication:

$$
\frac{3x}{3} = \frac{6}{3} \quad\Longrightarrow\quad x = 2
$$

The solution is $x = 2$. You can **check** it by substituting back into the original: $3 \cdot 2 + 1 = 7$. True, so $x = 2$ is correct. Always check; it is free and it catches arithmetic slips.

Why "linear"? Because the variable appears only to the first power ($x$, never $x^2$ or $\sqrt{x}$). Graphed, $y = 3x + 1$ is a straight line. Linear equations are the workhorses of neural networks — every "linear layer" is a bundle of exactly this kind of expression, just with many variables at once.

## 6. Why ML needs all of this

Here is the payoff, stated plainly so it sits in the back of your mind for the rest of the course.

A neural network is one enormous **expression**. Its inputs are variables (the data you feed in). Buried inside it are millions of other variables called **parameters** (also "weights"). The network computes an output by substituting the input values and evaluating the whole expression — that is the *forward pass*, exactly like turning $3x+1$ into $7$, only with billions of arithmetic operations.

"Training" flips the problem around. We fix the inputs and outputs (we have example data), and we treat the *parameters* as the unknowns to solve for — like solving $3x + 1 = 7$ for $x$, except there is no neat algebra for millions of variables at once. Instead we solve it approximately, by nudging the parameters a little at a time. Everything in this course is building the machinery to do that nudging correctly.

<div class="callout key"><p><strong>Key idea.</strong> Forward pass = evaluate a big expression. Training = approximately solve a big equation for its parameters. Both are the humble ideas of this lesson, scaled up enormously.</p></div>

## 7. In PyTorch

PyTorch is the library we will use to make all of this concrete. Its basic object is the **tensor** — a container for numbers. The simplest tensor holds a single number and is called a *scalar*. Let us make one, substitute a value into an expression, and evaluate it.

```python
import torch

# A constant scalar. The trailing dot makes it a float (a real number),
# not an integer. torch.tensor(3) would be an int; torch.tensor(3.) is a float.
c = torch.tensor(3.)
print(c)          # tensor(3.)
print(c.dtype)    # torch.float32   <- 32-bit float, ~7 digits of precision

# A "variable" x that we substitute a value into.
x = torch.tensor(2.)

# The expression 3*x + 1, evaluated. This is substitution + evaluation.
y = 3 * x + 1
print(y)          # tensor(7.)  <- matches our hand calculation 3*2 + 1 = 7
```

Read this alongside the mathematics. `c = torch.tensor(3.)` is the constant $3$. `x = torch.tensor(2.)` substitutes the value $2$ for the variable $x$. The line `y = 3 * x + 1` builds and evaluates the expression $3x + 1$, and the result `tensor(7.)` is exactly the $7$ we got by hand.

<div class="callout pt"><p><strong>PyTorch note.</strong> Always write float literals like <code>2.</code> (with the dot) for numbers you will later train on. Integers cannot carry gradients, and training needs the smooth, continuous behavior of the real numbers — which floats approximate. If you write <code>torch.tensor(2)</code> and later try to train, PyTorch will complain. The dot is not decoration.</p></div>

## 8. What PyTorch is doing under the hood

When you wrote `y = 3 * x + 1`, PyTorch did two things that will matter enormously later:

1. It **evaluated** the expression and stored the number $7$ inside `y` — the part you can see.
2. If `x` had been marked as trainable (we will write `requires_grad=True` in later lessons), PyTorch would also have quietly **recorded that `y` was built from `x`** via "multiply by 3, then add 1". It remembers the recipe, not just the answer.

That recorded recipe is the *computational graph*, and it is what lets PyTorch later run the whole calculation backwards to figure out how to nudge `x`. Right now we are only doing the forward direction — substitute and evaluate — so there is nothing backward to see yet. But keep in mind that even in this trivial example, PyTorch is doing bookkeeping in preparation for training. We will unpack that bookkeeping thoroughly starting in Module 6.

For now, the mental model is complete: numbers are the values, variables are the names, an expression is a recipe, an equation is a claim of equality, and PyTorch tensors are how we hold numbers and evaluate expressions in code.

## Check yourself

1. Is $-\frac{5}{2}$ a natural number, an integer, a rational, and/or a real number?

<details><summary>Answer</summary>

It is a rational number (it is a ratio of integers, $\frac{-5}{2}$) and therefore also a real number. It is **not** an integer (it is not a whole number) and **not** a natural number.

</details>

2. In plain words, how does a mathematical variable differ from a C# variable?

<details><summary>Answer</summary>

A C# variable is a mutable box in memory that you assign and overwrite. A mathematical variable is a name standing for a value — an unknown to solve for, or a placeholder to substitute a value into. It does not get "overwritten"; within a problem it denotes one value.

</details>

3. Is $2x - 6$ an expression or an equation? What about $2x - 6 = 0$? Solve the second one.

<details><summary>Answer</summary>

$2x - 6$ is an expression (no equals sign — a value to compute once you know $x$). $2x - 6 = 0$ is an equation. Solving: add 6 to both sides to get $2x = 6$, then divide by 2 to get $x = 3$. Check: $2 \cdot 3 - 6 = 0$. Correct.

</details>

4. Evaluate the expression $5x - 2$ at $x = 3$.

<details><summary>Answer</summary>

$5 \cdot 3 - 2 = 15 - 2 = 13$.

</details>

## Next

You now have numbers, variables, expressions, and equations — the raw vocabulary. The single most important structure built from them is the **function**, the object that turns inputs into outputs and that machine learning is entirely about learning. That is next.

[02 · Functions and the meaning of f(x) →](lesson-02.md)
