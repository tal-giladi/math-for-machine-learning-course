# 16 · Derivative rules and the chain rule

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-01.md">15 · Limits, slope, and the derivative</a> (the definition $f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}$ and the result $\frac{d}{dx}x^2 = 2x$), plus function composition from <a href="../module-05/lesson-01.md">the functions and composition module</a>.</p>
<p><strong>You will learn:</strong> the shortcut rules that let you differentiate without going back to the limit every time — the constant, power, constant-multiple, sum, product, and quotient rules — and then the <strong>chain rule</strong>, the rule for differentiating a function inside a function. Each rule comes with a named-symbol formula, a tiny worked number, and a note on where it shows up in ML.</p>
<p><strong>Why this matters for ML:</strong> a neural network is a deep composition of functions, so its derivatives are computed by the chain rule applied over and over. The chain rule is <em>the</em> rule behind backpropagation; this lesson plants it, and <a href="../module-07/lesson-01.md">module 7</a> gives it the full computational-graph treatment.</p>
</div>

## 1. Why we want rules

In [lesson 1](lesson-01.md) we found $\frac{d}{dx}x^2 = 2x$ by expanding a difference quotient and taking a limit. Doing that for every function in a Transformer would be hopeless. Fortunately, a handful of rules cover almost everything, and they *compose*: once you know how to differentiate the pieces, you can differentiate any combination. Every rule below can be proven from the limit definition, but in practice you apply the rule and move on. We take them one at a time, each in the same shape: **name the formula, name every symbol, work a tiny number, note the ML use.**

Throughout, $f$ and $g$ are functions of $x$, $f'$ and $g'$ are their derivatives, $c$ and $n$ are constants (fixed numbers).

## 2. The constant rule

$$
\frac{d}{dx}\,c = 0
$$

- $c$ — any constant (a number that does not depend on $x$).

A constant does not change as $x$ changes, so its rate of change is zero — a flat horizontal line has slope $0$. Example: $\frac{d}{dx}\,7 = 0$.

**ML use:** hyperparameters like a fixed learning rate contribute nothing to gradients; only quantities that depend on the parameters do.

## 3. The power rule

$$
\frac{d}{dx}\,x^n = n\,x^{\,n-1}
$$

- $x$ — the variable.
- $n$ — a fixed exponent (works for any real $n$: positive, negative, fractional).

Bring the exponent down as a multiplier, then reduce the exponent by one. This is the workhorse. Check it against lesson 1: for $n = 2$, $\frac{d}{dx}x^2 = 2x^{2-1} = 2x$ — exactly what the limit gave us.

**Tiny example:** $\frac{d}{dx}x^3 = 3x^{2}$. At $x = 2$ that is $3 \cdot 4 = 12$.

It even handles negative exponents: $\frac{d}{dx}x^{-1} = -1\cdot x^{-2} = -\frac{1}{x^2}$ (we will re-derive this with the quotient rule in section 7).

**ML use:** squared terms appear in the mean-squared-error loss and in $L2$ weight decay; the power rule differentiates them.

## 4. The constant-multiple rule

$$
\frac{d}{dx}\big(c\,f(x)\big) = c\,f'(x)
$$

- $c$ — a constant scaling the whole function.

A constant factor rides along untouched; scaling a function scales its slope by the same amount. Example: $\frac{d}{dx}(5x^2) = 5 \cdot 2x = 10x$. At $x = 3$, that is $30$.

**ML use:** scaling a loss by a constant (say $\frac{1}{2}$ in MSE, or averaging over a batch of size $N$ via $\frac{1}{N}$) scales every gradient by that same constant.

## 5. The sum rule

$$
\frac{d}{dx}\big(f(x) + g(x)\big) = f'(x) + g'(x)
$$

Differentiate term by term — the derivative of a sum is the sum of the derivatives. Example:

$$
\frac{d}{dx}\big(x^2 + x^3\big) = 2x + 3x^2.
$$

At $x = 1$: $2 + 3 = 5$.

**ML use:** a total loss is often a sum of per-example losses (plus a regularization term); its gradient is the sum of the individual gradients. This is why gradients from a batch simply add up.

## 6. The product rule

Two functions **multiplied** together do *not* differentiate to the product of the derivatives. The correct rule is:

$$
(f g)' = f'g + f g'
$$

- $f, g$ — the two factors.
- $f', g'$ — their derivatives.

In words: derivative of the first times the second, plus the first times derivative of the second. There are two terms because either factor can change while the other holds.

**Tiny worked example.** Let $f(x) = x^2$ and $g(x) = x + 1$, so $y = x^2(x+1)$. Then $f' = 2x$ and $g' = 1$:

$$
(fg)' = f'g + fg' = 2x(x+1) + x^2(1) = 2x^2 + 2x + x^2 = 3x^2 + 2x.
$$

**Sanity check by expanding first:** $y = x^2(x+1) = x^3 + x^2$, whose derivative (power + sum rules) is $3x^2 + 2x$ — identical. At $x = 2$: $3\cdot 4 + 2\cdot 2 = 12 + 4 = 16$.

**ML use:** the product rule appears whenever two parameter-dependent quantities multiply — for instance inside the derivations of gated units and attention weights, and (in a two-factor form) in the LoRA update $BA$.

## 7. The quotient rule

For one function **divided** by another:

$$
\left(\frac{f}{g}\right)' = \frac{f' g - f g'}{g^2}
$$

- $f$ — the numerator, $g$ — the denominator (with $g \neq 0$).
- The result has $g^2$ in the denominator; mind the *minus* sign and the order in the numerator.

**Tiny worked example.** Take $y = \frac{1}{x}$, so $f = 1$ (constant, $f' = 0$) and $g = x$ ($g' = 1$):

$$
\left(\frac{1}{x}\right)' = \frac{0 \cdot x - 1 \cdot 1}{x^2} = \frac{-1}{x^2} = -\frac{1}{x^2}.
$$

At $x = 2$: $-\frac{1}{4} = -0.25$. This agrees with the power rule reading of $x^{-1}$: $-1\cdot x^{-2} = -x^{-2}$. Good — two routes, same answer.

**ML use:** the quotient rule is exactly what you use to differentiate the **sigmoid** $\sigma(x) = \frac{1}{1+e^{-x}}$ and the **softmax**, both of which are ratios; we do those derivations in full in [lesson 3](lesson-03.md).

<div class="callout key"><p>The five algebraic rules: constant $\to 0$; power $x^n \to n x^{n-1}$; constant-multiple pulls the constant out; sum differentiates term by term; product is $f'g+fg'$; quotient is $\frac{f'g-fg'}{g^2}$. Memorize product and quotient carefully — they are the ones people get wrong.</p></div>

## 8. The chain rule — the important one

### 8.1 Intuition

Suppose $y$ depends on $u$, and $u$ in turn depends on $x$ — a function *inside* a function, $y = f(g(x))$. If you wiggle $x$, that wiggles $u$ (at rate $\frac{du}{dx}$), and the wiggle in $u$ wiggles $y$ (at rate $\frac{dy}{du}$). The total sensitivity of $y$ to $x$ is those two sensitivities **multiplied**:

$$
\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}
$$

- $x$ — the outer input.
- $u = g(x)$ — the intermediate quantity.
- $y = f(u)$ — the final output.
- $\frac{dy}{du}$ — how fast $y$ changes per change in $u$ (differentiate the outer function, treating $u$ as its variable).
- $\frac{du}{dx}$ — how fast $u$ changes per change in $x$ (differentiate the inner function).

The programmer's picture: it is a gain chain. If turning the input knob moves a middle gear at rate $2$, and the middle gear moves the output at rate $3$, then the input moves the output at rate $2 \times 3 = 6$. Rates through a chain multiply. (The Leibniz notation makes it look like the $du$'s "cancel" — a useful mnemonic, though the real justification is the limit.)

### 8.2 Worked example: $y = (2x)^2$

We will differentiate $y = (2x)^2$ two ways and check they agree.

**By the chain rule.** Set the inner function $u = 2x$, so $y = u^2$.

- Outer derivative: $\dfrac{dy}{du} = 2u$ (power rule on $u^2$).
- Inner derivative: $\dfrac{du}{dx} = 2$ (constant-multiple rule on $2x$).
- Multiply: $\dfrac{dy}{dx} = 2u \cdot 2 = 4u$. Now substitute back $u = 2x$: $\dfrac{dy}{dx} = 4(2x) = 8x$.

**By expanding first.** $y = (2x)^2 = 4x^2$, whose derivative is $4 \cdot 2x = 8x$. 

Both give $8x$. At $x = 3$: $24$. The chain rule got the right answer without our ever multiplying out the square — which matters when the inside is too complicated to expand.

### 8.3 A multi-step chain

The rule extends to any depth: for each extra layer, multiply in one more local derivative. Take $y = (x^2 + 1)^3$. Let $u = x^2 + 1$, so $y = u^3$:

- $\dfrac{dy}{du} = 3u^2$ (power rule).
- $\dfrac{du}{dx} = 2x$ (power + sum rules).
- Chain: $\dfrac{dy}{dx} = 3u^2 \cdot 2x = 6x\,u^2 = 6x(x^2+1)^2$.

**Evaluate at $x = 1$:** $u = 1^2 + 1 = 2$, so $\dfrac{dy}{dx} = 6 \cdot 1 \cdot 2^2 = 6 \cdot 4 = 24$.

**Numerical check.** $y(1) = (1+1)^3 = 8$. Step to $x = 1.001$: $u = 1.002001$, $y = 1.002001^3 \approx 8.02404$. The difference quotient is $\frac{8.02404 - 8}{0.001} \approx 24.0$ — matching our chain-rule $24$.

If there were three nested functions, $x \to u \to v \to y$, you would multiply three local derivatives: $\frac{dy}{dx} = \frac{dy}{dv}\frac{dv}{du}\frac{du}{dx}$. That is the exact pattern module 7 turns into backpropagation.

<div class="callout key"><p>Chain rule: $\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}$. To differentiate a composition, differentiate the outer function (leaving the inside alone), then multiply by the derivative of the inside. Sensitivities multiply along a chain. Deep networks are deep chains, so this rule runs the entire backward pass.</p></div>

## 9. Why the chain rule is backpropagation

A neural network computes its output by composition: the input passes through a linear layer, then an activation, then another linear layer, then the loss — each stage a function fed the previous stage's output. Symbolically the loss is something like $L = \ell(\,f_3(\,f_2(\,f_1(x)\,)\,)\,)$, a deep nest.

To train, you need $\frac{\partial L}{\partial \theta}$ for every parameter $\theta$ buried anywhere in that nest. The chain rule says: multiply the local derivative at each stage from the loss back to the parameter. **Backpropagation is just the chain rule applied to the network's computational graph, evaluated from the output backward** so that shared sub-results are reused instead of recomputed. Every gradient PyTorch reports is a product of local derivatives assembled by the chain rule. We build the full machinery — computational graphs, local derivatives at each node, the backward sweep — in [module 7](../module-07/lesson-01.md). For now, the takeaway is that the single formula $\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}$ scales up to the whole network.

## 10. In PyTorch

Let PyTorch confirm the product-rule example from section 6. We had $y = x^2(x+1)$ with derivative $3x^2 + 2x$, equal to $16$ at $x = 2$:

```python
import torch

x = torch.tensor(2., requires_grad=True)
y = (x ** 2) * (x + 1)     # forward: 4 * 3 = 12.0
y.backward()
print(x.grad)              # tensor(16.)   ->  3*2^2 + 2*2 = 12 + 4 = 16
```

`x.grad` is `tensor(16.)`, matching the hand-computed $3x^2 + 2x$ at $x = 2$. Now the chain-rule example $y = (x^2+1)^3$, whose derivative we found to be $6x(x^2+1)^2 = 24$ at $x = 1$:

```python
x = torch.tensor(1., requires_grad=True)
y = (x ** 2 + 1) ** 3      # forward: (1+1)^3 = 8.0
y.backward()
print(x.grad)              # tensor(24.)   ->  6*1*(1^2+1)^2 = 6*4 = 24
```

Again the reported gradient equals our by-hand chain-rule result.

<div class="callout pt"><p>You never tell PyTorch which rule to use. It knows the derivative of each elementary operation (<code>**</code>, <code>*</code>, <code>+</code>, …) and composes them with the chain rule automatically. Your job is to write the forward expression; autograd differentiates it.</p></div>

## 11. What PyTorch is doing under the hood

When you wrote `y = (x ** 2 + 1) ** 3`, PyTorch built a small chain of recorded operations, each remembering how it was made:

```
x --(square)--> t1=x^2 --(add 1)--> t2=x^2+1 --(cube)--> y
```

Each node stores a `grad_fn` (`PowBackward0`, `AddBackward0`, `PowBackward0`) — a handle to that operation's local-derivative recipe. Calling `y.backward()` sweeps this chain from `y` back to `x`, and at each node it multiplies in that node's local derivative, evaluated at the value that flowed through on the forward pass:

- cube node: local derivative $3 t_2^2 = 3 \cdot 2^2 = 12$;
- add node: local derivative $1$ (adding a constant does not change the rate), running product $12$;
- square node: local derivative $2x = 2$, running product $12 \cdot 2 = 24$.

That accumulated product, $24$, is deposited in `x.grad`. This is the chain rule of section 8, executed mechanically: record local derivatives going forward, multiply them going backward. Notice the running product is built by **multiplication** exactly as $\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}$ prescribes. Nothing more mysterious than that is happening inside `.backward()`; module 7 shows how the same idea handles graphs that branch and merge.

## Check yourself

<details><summary>Differentiate $y = 5x^4 - 3x^2 + 7$.</summary>

Term by term (sum, constant-multiple, power, constant rules): $\frac{dy}{dx} = 5\cdot 4x^3 - 3\cdot 2x + 0 = 20x^3 - 6x$.

</details>

<details><summary>Use the product rule on $y = x^3(x+2)$, then check by expanding.</summary>

$f = x^3,\ g = x+2$; $f' = 3x^2,\ g' = 1$. $(fg)' = 3x^2(x+2) + x^3(1) = 3x^3 + 6x^2 + x^3 = 4x^3 + 6x^2$. Expanding first: $y = x^4 + 2x^3$, derivative $4x^3 + 6x^2$. Same.

</details>

<details><summary>Use the chain rule on $y = (3x + 1)^5$.</summary>

Let $u = 3x+1$, $y = u^5$. $\frac{dy}{du} = 5u^4$, $\frac{du}{dx} = 3$. So $\frac{dy}{dx} = 15u^4 = 15(3x+1)^4$.

</details>

<details><summary>In backprop terms, why does the chain rule mean gradients get <em>multiplied</em> as they pass back through layers?</summary>

Because a network output is a composition of layer functions, and the chain rule differentiates a composition by multiplying the layers' local derivatives. Passing a gradient back through one more layer multiplies it by that layer's local derivative — which is exactly what the backward pass does at each node.

</details>

## Next

You now have the rules to differentiate the algebraic functions that make up a network, plus the chain rule that stitches them together. Next we apply these rules to the specific nonlinear functions that appear in every model — the exponential, the logarithm, the sigmoid, and the softmax — and derive their derivatives in full.

Continue to [17 · Derivatives of exp, log, sigmoid, softmax](lesson-03.md).
