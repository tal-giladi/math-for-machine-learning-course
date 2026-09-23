# Math cheat sheet

<div class="prereq">
<p><strong>What this is:</strong> the most important formulas of the whole course on one scannable page, grouped by area, each with a one-line "what it means" and a link to the lesson that teaches it.</p>
<p><strong>How to read it:</strong> everything renders in KaTeX. The <code>[→ NN]</code> link jumps to the lesson where the formula is derived with a worked numerical example.</p>
<p><strong>Notation used course-wide:</strong> input $x$, weights $\mathbf{W}$, bias $\mathbf{b}$, pre-activation $z$, activation $a$, loss $L$, learning rate $\eta$, gradient $\nabla$. GPT dims: batch $B$, time/sequence $T$, channels $C=d_\text{model}$, vocab $V$, heads $n_h$, head dim $d_h$ with $C=n_h\,d_h$.</p>
</div>

## Notation & algebra

| Formula | What it means | Lesson |
|---|---|---|
| $f(x)$ | a function $f$ turns input $x$ into an output; $f$ is the rule, $x$ the argument | [→ 02](lessons/module-01/lesson-02.md) |
| $\sum_{i=1}^{n} a_i = a_1 + a_2 + \dots + a_n$ | add up a list; $i$ is the index, $n$ the count | [→ 04](lessons/module-01/lesson-04.md) |
| $\prod_{i=1}^{n} a_i = a_1 \cdot a_2 \cdots a_n$ | multiply a list together | [→ 04](lessons/module-01/lesson-04.md) |
| $x^a x^b = x^{a+b}$, $(x^a)^b = x^{ab}$, $x^{-1}=\tfrac1x$ | power rules: multiply → add exponents | [→ 03](lessons/module-01/lesson-03.md) |
| $\log(ab)=\log a+\log b$, $\log(a^b)=b\log a$ | log turns products into sums (this is why we log-add probabilities) | [→ 03](lessons/module-01/lesson-03.md) |
| $e^{\ln x}=x$, $\ln(e^x)=x$, $\ln 1 = 0$ | $\exp$ and $\ln$ are inverses; base $e\approx 2.71828$ | [→ 03](lessons/module-01/lesson-03.md) |
| $\lvert x\rvert$ | absolute value: distance from $0$, always $\ge 0$ | [→ 04](lessons/module-01/lesson-04.md) |
| $\operatorname*{argmax}_i s_i$ | the index $i$ of the largest entry (not the value) — the predicted token | [→ 05](lessons/module-01/lesson-05.md) |
| $x_i$, $x^{(k)}$, $W_{ij}$ | subscript = index into a vector; superscript in parens = which example/layer; two indices = row $i$, col $j$ | [→ 05](lessons/module-01/lesson-05.md) |

## Vectors

Let $\mathbf{x},\mathbf{y}\in\mathbb{R}^n$ (lists of $n$ numbers).

| Formula | What it means | Lesson |
|---|---|---|
| $\mathbf{x}\cdot\mathbf{y}=\sum_{i=1}^{n} x_i y_i$ | dot product: one number measuring alignment; core of every linear layer | [→ 07](lessons/module-02/lesson-02.md) |
| $\lVert\mathbf{x}\rVert = \sqrt{\sum_i x_i^2} = \sqrt{\mathbf{x}\cdot\mathbf{x}}$ | Euclidean norm: the length of the vector | [→ 08](lessons/module-02/lesson-03.md) |
| $\hat{\mathbf{x}} = \mathbf{x}/\lVert\mathbf{x}\rVert$ | unit vector: same direction, length $1$ | [→ 08](lessons/module-02/lesson-03.md) |
| $\cos\theta = \dfrac{\mathbf{x}\cdot\mathbf{y}}{\lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert}$ | cosine similarity: direction agreement in $[-1,1]$, ignores length | [→ 08](lessons/module-02/lesson-03.md) |
| $\operatorname{proj}_{\mathbf{y}}\mathbf{x} = \dfrac{\mathbf{x}\cdot\mathbf{y}}{\lVert\mathbf{y}\rVert^2}\,\mathbf{y}$ | projection: the shadow of $\mathbf{x}$ along $\mathbf{y}$ | [→ 08](lessons/module-02/lesson-03.md) |

## Matrices

$\mathbf{A}$ is $(m,n)$ = $m$ rows, $n$ columns.

| Formula | What it means | Lesson |
|---|---|---|
| $(m,n)\times(n,p)\to(m,p)$ | matmul shape rule: inner dims must match, they cancel | [→ 10](lessons/module-03/lesson-02.md) |
| $(\mathbf{A}\mathbf{B})_{ij}=\sum_{k} A_{ik}B_{kj}$ | entry $(i,j)$ = row $i$ of $\mathbf{A}$ dotted with column $j$ of $\mathbf{B}$ | [→ 10](lessons/module-03/lesson-02.md) |
| $(\mathbf{A}^\top)_{ij}=A_{ji}$ | transpose: swap rows and columns; $(m,n)\to(n,m)$ | [→ 11](lessons/module-03/lesson-03.md) |
| $\mathbf{I}$, $\mathbf{A}\mathbf{I}=\mathbf{I}\mathbf{A}=\mathbf{A}$ | identity: $1$ on the diagonal, $0$ elsewhere; the "do nothing" matrix | [→ 11](lessons/module-03/lesson-03.md) |
| $\mathbf{A}\mathbf{A}^{-1}=\mathbf{I}$; for $2\times2$: $\begin{bmatrix}a&b\\c&d\end{bmatrix}^{-1}=\dfrac{1}{ad-bc}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$ | inverse undoes the transform; exists only if $ad-bc\ne 0$ | [→ 11](lessons/module-03/lesson-03.md) |
| $\operatorname{rank}(\mathbf{A})$ | number of independent rows/columns = true dimensionality (key for LoRA) | [→ 11](lessons/module-03/lesson-03.md) |

<div class="callout key">
<p>The single most common operation in a Transformer is $\mathbf{y}=\mathbf{W}\mathbf{x}+\mathbf{b}$: a matrix multiply (a batch of dot products) plus a bias. Get comfortable with shapes and you can read most of GPT-2.</p>
</div>

## Calculus

| Formula | What it means | Lesson |
|---|---|---|
| $f'(x)=\displaystyle\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}$ | derivative: instantaneous slope = local sensitivity of output to input | [→ 15](lessons/module-06/lesson-01.md) |
| power: $\dfrac{d}{dx}x^n = n x^{n-1}$ | bring the exponent down, subtract one | [→ 16](lessons/module-06/lesson-02.md) |
| sum: $(f+g)'=f'+g'$ | derivative distributes over addition | [→ 16](lessons/module-06/lesson-02.md) |
| product: $(fg)'=f'g+fg'$ | | [→ 16](lessons/module-06/lesson-02.md) |
| quotient: $\left(\dfrac{f}{g}\right)'=\dfrac{f'g-fg'}{g^2}$ | | [→ 16](lessons/module-06/lesson-02.md) |
| chain: $\dfrac{d}{dx}f(g(x))=f'(g(x))\,g'(x)$ | multiply local slopes along the composition — the seed of backprop | [→ 16](lessons/module-06/lesson-02.md) |
| $\dfrac{d}{dx}e^x=e^x$ | exp is its own derivative | [→ 17](lessons/module-06/lesson-03.md) |
| $\dfrac{d}{dx}\ln x=\dfrac1x$ | | [→ 17](lessons/module-06/lesson-03.md) |
| $\sigma(x)=\dfrac{1}{1+e^{-x}}$, $\sigma'(x)=\sigma(x)\,(1-\sigma(x))$ | sigmoid squashes to $(0,1)$; its derivative reuses the output | [→ 17](lessons/module-06/lesson-03.md) |
| softmax Jacobian: $\dfrac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)$ | $\delta_{ij}=1$ if $i=j$ else $0$; couples every output to every logit | [→ 17](lessons/module-06/lesson-03.md) |

## Gradients & backprop

For a scalar loss $L$ and a vector/matrix parameter, the gradient has the **same shape** as the thing you differentiate with respect to.

| Formula | What it means | Lesson |
|---|---|---|
| $\dfrac{\partial f}{\partial x_i}$ | partial derivative: slope in one direction, others held fixed | [→ 18](lessons/module-06/lesson-04.md) |
| $\nabla_{\mathbf{x}} f = \left(\dfrac{\partial f}{\partial x_1},\dots,\dfrac{\partial f}{\partial x_n}\right)$ | gradient: vector of all partials; points uphill fastest | [→ 18](lessons/module-06/lesson-04.md) |
| $\dfrac{\partial L}{\partial x}$ vs $\nabla_x L$ | same object; $\nabla$ is the whole gradient, $\partial/\partial x_i$ one component | [→ 20](lessons/module-07/lesson-02.md) |
| chain rule as Jacobian product: $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial y}\dfrac{\partial y}{\partial x}$ | backprop multiplies Jacobians right-to-left | [→ 21](lessons/module-07/lesson-03.md) |
| for $\mathbf{y}=\mathbf{W}\mathbf{x}+\mathbf{b}$, given $\mathbf{g}=\partial L/\partial\mathbf{y}$: $\;\dfrac{\partial L}{\partial\mathbf{x}}=\mathbf{W}^\top\mathbf{g}$ | pull the gradient back through the weights (a transpose matmul) | [→ 23](lessons/module-08/lesson-01.md) |
| $\dfrac{\partial L}{\partial\mathbf{W}}=\mathbf{g}\,\mathbf{x}^\top$ | outer product: grad has $\mathbf{W}$'s shape | [→ 23](lessons/module-08/lesson-01.md) |
| $\dfrac{\partial L}{\partial\mathbf{b}}=\mathbf{g}$ | bias just passes the upstream gradient through (sum over the batch) | [→ 23](lessons/module-08/lesson-01.md) |
| residual $y=x+f(x)\Rightarrow \dfrac{dy}{dx}=1+f'(x)$ | the $+1$ is a gradient highway — why deep nets train | [→ 33](lessons/module-11/lesson-02.md) |

<div class="callout key">
<p>Backprop = the chain rule applied to the computational graph. Each node knows its local derivative; the backward pass multiplies these together from the loss back to every parameter. See [→ 25](lessons/module-08/lesson-03.md).</p>
</div>

## Loss functions

Prediction $\hat{y}$ / probabilities $\mathbf{p}$, target $y$ / one-hot $\mathbf{y}$, class count / vocab $V$.

| Formula | What it means | Lesson |
|---|---|---|
| MSE $=\dfrac1n\sum_i (\hat{y}_i-y_i)^2$ | mean squared error: regression loss, penalizes big misses | [→ 24](lessons/module-08/lesson-02.md) |
| MAE $=\dfrac1n\sum_i \lvert\hat{y}_i-y_i\rvert$ | mean absolute error: robust to outliers | [→ 24](lessons/module-08/lesson-02.md) |
| softmax $p_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | turn logits into a probability distribution | [→ 30](lessons/module-10/lesson-02.md) |
| log-softmax $\log p_i = z_i-\log\sum_j e^{z_j}$ | computed directly for numerical stability | [→ 30](lessons/module-10/lesson-02.md) |
| cross-entropy $H(\mathbf{y},\mathbf{p})=-\sum_i y_i\log p_i$ | penalty for the probability you gave the true class | [→ 24](lessons/module-08/lesson-02.md) |
| NLL $=-\log p_{\text{correct}}$ | negative log-likelihood; equals CE with a one-hot target | [→ 24](lessons/module-08/lesson-02.md) |
| BCE $=-\big[y\log\hat{y}+(1-y)\log(1-\hat{y})\big]$ | binary cross-entropy for a single yes/no output | [→ 24](lessons/module-08/lesson-02.md) |
| CE gradient wrt logits: $\dfrac{\partial L}{\partial z_i}=p_i-y_i$ | softmax + CE collapse to $\mathbf{p}-\mathbf{y}$ — beautifully simple | [→ 30](lessons/module-10/lesson-02.md) |

## Probability

| Formula | What it means | Lesson |
|---|---|---|
| $\mathbb{E}[X]=\sum_x x\,P(x)$ | expectation: the long-run average value | [→ 29](lessons/module-10/lesson-01.md) |
| $\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]=\mathbb{E}[X^2]-\mathbb{E}[X]^2$ | variance: spread; std dev $=\sqrt{\operatorname{Var}}$ | [→ 29](lessons/module-10/lesson-01.md) |
| $H(p)=-\sum_i p_i\log p_i$ | entropy: average surprise of a distribution | [→ 29](lessons/module-10/lesson-01.md) |
| $H(p,q)=-\sum_i p_i\log q_i$ | cross-entropy: surprise using $q$ when truth is $p$ | [→ 29](lessons/module-10/lesson-01.md) |
| $D_{\mathrm{KL}}(p\,\|\,q)=\sum_i p_i\log\dfrac{p_i}{q_i}$ | KL divergence: extra bits from using $q$ instead of $p$; $\ge 0$ | [→ 29](lessons/module-10/lesson-01.md) |
| $H(p,q)=H(p)+D_{\mathrm{KL}}(p\,\|\,q)$ | minimizing CE = minimizing KL (since $H(p)$ is fixed) | [→ 29](lessons/module-10/lesson-01.md) |
| $P(A\mid B)=\dfrac{P(B\mid A)\,P(A)}{P(B)}$ | Bayes' theorem: update a belief given evidence | [→ 29](lessons/module-10/lesson-01.md) |

## Optimization

Parameters $\theta$, gradient $g=\nabla_\theta L$, learning rate $\eta$, step $t$.

| Formula | What it means | Lesson |
|---|---|---|
| GD: $\theta \leftarrow \theta - \eta\,g$ | step downhill against the gradient | [→ 26](lessons/module-09/lesson-01.md) |
| SGD (mini-batch): $g$ from a batch, then GD step | noisy but cheap; the workhorse | [→ 27](lessons/module-09/lesson-02.md) |
| momentum: $v\leftarrow\mu v + g$; $\;\theta\leftarrow\theta-\eta v$ | accumulate a velocity to smooth and accelerate ($\mu\approx0.9$) | [→ 27](lessons/module-09/lesson-02.md) |
| RMSProp: $s\leftarrow\rho s+(1-\rho)g^2$; $\;\theta\leftarrow\theta-\dfrac{\eta}{\sqrt{s}+\epsilon}g$ | per-parameter learning rate from recent gradient size | [→ 27](lessons/module-09/lesson-02.md) |
| Adam: $m\leftarrow\beta_1 m+(1-\beta_1)g$, $\;v\leftarrow\beta_2 v+(1-\beta_2)g^2$ | first moment (mean) + second moment (variance) | [→ 27](lessons/module-09/lesson-02.md) |
| Adam bias-correct: $\hat{m}=\dfrac{m}{1-\beta_1^t}$, $\;\hat{v}=\dfrac{v}{1-\beta_2^t}$ | fix the cold-start bias toward zero | [→ 27](lessons/module-09/lesson-02.md) |
| Adam update: $\theta\leftarrow\theta-\eta\,\dfrac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}$ | momentum divided by RMS scale | [→ 27](lessons/module-09/lesson-02.md) |
| AdamW: $\theta\leftarrow\theta-\eta\Big(\dfrac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}+\lambda\theta\Big)$ | weight decay applied *decoupled*, straight to $\theta$ | [→ 27](lessons/module-09/lesson-02.md) |
| Muon: $O=\mathbf{U}\mathbf{V}^\top$ from $\mathbf{G}=\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$, via Newton–Schulz | orthogonalize the momentum matrix (set all singular values to $1$) before stepping | [→ 28](lessons/module-09/lesson-03.md) |

<div class="callout pt">
<p>Adam's $m$ and $v$ are the "optimizer state": two extra tensors per parameter, so Adam needs ~3× the memory of the weights alone. See [→ 27](lessons/module-09/lesson-02.md).</p>
</div>

## Normalization

For a feature vector $\mathbf{x}\in\mathbb{R}^C$, small $\epsilon$ for stability, learnable gain $\gamma$ and shift $\beta$.

| Formula | What it means | Lesson |
|---|---|---|
| $\mu=\dfrac1C\sum_i x_i$, $\;\sigma^2=\dfrac1C\sum_i (x_i-\mu)^2$ | mean and variance across the feature axis | [→ 34](lessons/module-11/lesson-03.md) |
| LayerNorm: $y_i=\dfrac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\,\gamma_i+\beta_i$ | recenter + rescale each token's features, then affine | [→ 34](lessons/module-11/lesson-03.md) |
| RMSNorm: $y_i=\dfrac{x_i}{\sqrt{\frac1C\sum_j x_j^2+\epsilon}}\,\gamma_i$ | drop the mean-subtraction; scale by RMS only (cheaper) | [→ 34](lessons/module-11/lesson-03.md) |

## Attention

Per head: $\mathbf{Q},\mathbf{K},\mathbf{V}$ are $(T,d_h)$.

| Formula | What it means | Lesson |
|---|---|---|
| $\operatorname{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V})=\operatorname{softmax}\!\Big(\dfrac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}+M\Big)\mathbf{V}$ | scores → scale → mask → softmax → weighted sum of values | [→ 32](lessons/module-11/lesson-01.md) |
| scores $\mathbf{Q}\mathbf{K}^\top$ shape $(T,T)$ | every query dotted with every key | [→ 32](lessons/module-11/lesson-01.md) |
| scale $1/\sqrt{d_h}$ | keep dot products from blowing up softmax | [→ 32](lessons/module-11/lesson-01.md) |
| causal mask $M_{ij}=-\infty$ for $j>i$ | a token can't see the future | [→ 32](lessons/module-11/lesson-01.md) |
| multi-head: split $C=n_h\,d_h$, run $n_h$ attentions, concat, project | reshape $(B,T,C)\leftrightarrow(B,n_h,T,d_h)$ | [→ 32](lessons/module-11/lesson-01.md) |

## LoRA (fine-tuning)

| Formula | What it means | Lesson |
|---|---|---|
| $\mathbf{W}'=\mathbf{W}+\mathbf{B}\mathbf{A}$ | freeze pretrained $\mathbf{W}$ $(d,k)$; learn a low-rank update | [→ 38](lessons/module-12/lesson-04.md) |
| $\mathbf{B}$ is $(d,r)$, $\mathbf{A}$ is $(r,k)$, rank $r\ll\min(d,k)$ | the product $\mathbf{B}\mathbf{A}$ is $(d,k)$ but rank $\le r$ | [→ 38](lessons/module-12/lesson-04.md) |
| trainable params $=r(d+k)$ | vs $dk$ for full fine-tuning — orders of magnitude fewer | [→ 38](lessons/module-12/lesson-04.md) |

## Next

If a formula here still looks like a wall of symbols, open its lesson: each one derives the formula from scratch with a small worked example. Pair this page with the [PyTorch cheat sheet](assets/pytorch-cheatsheet.md) to see what each formula becomes in code, and the [Glossary](assets/glossary.md) for any term.
