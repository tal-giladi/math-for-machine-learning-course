# PyTorch cheat sheet

<div class="prereq">
<p><strong>What this is:</strong> a two-column map from each <em>mathematical concept</em> to the <em>PyTorch that does it</em>. The framing the brief asks for: "I understand the math — now what does PyTorch actually do with it?"</p>
<p><strong>How to read it:</strong> every table row links to the lesson that explains it. Code lives in <code>python</code> fences; math notation is written in the prose around them (KaTeX does not render inside code fences).</p>
<p><strong>Setup for every snippet:</strong> assume <code>import torch</code> and <code>import torch.nn.functional as F</code>.</p>
</div>

## Tensors: creating, dtype, shape

| Math concept | PyTorch | Lesson |
|---|---|---|
| a scalar, vector, matrix, N-D tensor | `torch.tensor(2.)`, `torch.tensor([2.,3.])`, `torch.tensor([[1.,2.],[3.,4.]])` | [→ 12](lessons/module-04/lesson-01.md) |
| the shape $(B,T,C)$ | `x.shape` → `torch.Size([B, T, C])`; `x.ndim`, `x.numel()` | [→ 13](lessons/module-04/lesson-02.md) |
| real numbers vs integers | `dtype=torch.float32` (grads) vs `torch.long` (token IDs) | [→ 12](lessons/module-04/lesson-01.md) |
| reinterpret shape (contiguous) | `x.view(B, T, C)` | [→ 13](lessons/module-04/lesson-02.md) |
| reshape (copies if needed) | `x.reshape(B, T, C)` | [→ 13](lessons/module-04/lesson-02.md) |
| reorder axes (e.g. heads) | `x.permute(0, 2, 1, 3)`, `x.transpose(1, 2)` | [→ 13](lessons/module-04/lesson-02.md) |
| broadcasting a bias over a batch | shapes align right-to-left; size-1 dims stretch | [→ 13](lessons/module-04/lesson-02.md) |

```python
x = torch.arange(24.).reshape(2, 3, 4)   # (B=2, T=3, C=4)
x.view(2, 12)          # same data, shape (2, 12)
x.transpose(1, 2).shape  # torch.Size([2, 4, 3])
# broadcasting: (2,3,4) + (4,) -> bias added to every (B,T) position
x + torch.ones(4)
```

<div class="callout warn">
<p><code>view</code> needs contiguous memory and will error after some <code>permute</code>/<code>transpose</code> chains — call <code>.contiguous()</code> first, or use <code>reshape</code>, which handles it. See [→ 13](lessons/module-04/lesson-02.md).</p>
</div>

## Vector & matrix ops

| Math concept | PyTorch | Lesson |
|---|---|---|
| dot product $\mathbf{x}\cdot\mathbf{y}=\sum_i x_i y_i$ | `x @ y`, `torch.dot(x, y)` | [→ 07](lessons/module-02/lesson-02.md) |
| norm $\lVert\mathbf{x}\rVert$ | `x.norm()`, `torch.linalg.norm(x)` | [→ 08](lessons/module-02/lesson-03.md) |
| cosine similarity | `F.cosine_similarity(x, y, dim=0)` | [→ 08](lessons/module-02/lesson-03.md) |
| matrix multiply $(m,n)(n,p)\to(m,p)$ | `A @ B`, `torch.matmul(A, B)` | [→ 10](lessons/module-03/lesson-02.md) |
| batched matmul over leading dims | `torch.matmul` broadcasts batch dims | [→ 10](lessons/module-03/lesson-02.md) |
| transpose $\mathbf{A}^\top$ | `A.T`, `A.transpose(-2, -1)`, `A.mT` | [→ 11](lessons/module-03/lesson-03.md) |
| identity, inverse | `torch.eye(n)`, `torch.linalg.inv(A)` | [→ 11](lessons/module-03/lesson-03.md) |
| element-wise $\odot$, add, scale | `a * b`, `a + b`, `2 * a`, `torch.exp(a)`, `a ** 2` | [→ 12](lessons/module-04/lesson-01.md) |

```python
A = torch.tensor([[1., 2.], [3., 4.]])
x = torch.tensor([5., 6.])
A @ x            # tensor([17., 39.]) : row-dot-vector, shape (2,)
A.T              # tensor([[1., 3.], [2., 4.]])
# element-wise (Hadamard) vs matmul are different: a*b != a@b
```

## Autograd: how derivatives get computed

The chain rule (multiply local derivatives backward through the graph) *is* what autograd runs. You build the forward graph by doing ordinary tensor math on tensors that track gradients; `.backward()` walks it in reverse.

| Math concept | PyTorch | Lesson |
|---|---|---|
| "this is a variable to differentiate" | `x = torch.tensor(2., requires_grad=True)` | [→ 25](lessons/module-08/lesson-03.md) |
| the computational graph node | `y.grad_fn` (e.g. `<MulBackward0>`) | [→ 25](lessons/module-08/lesson-03.md) |
| run the backward pass (chain rule) | `loss.backward()` | [→ 25](lessons/module-08/lesson-03.md) |
| the gradient $\partial L/\partial x$ | `x.grad` (same shape as `x`) | [→ 20](lessons/module-07/lesson-02.md) |
| leaf tensor (a parameter/input) | `x.is_leaf`; only leaves keep `.grad` by default | [→ 25](lessons/module-08/lesson-03.md) |
| keep grad on a non-leaf | `h.retain_grad()` before backward | [→ 25](lessons/module-08/lesson-03.md) |
| forward with no graph (inference) | `with torch.no_grad(): ...` | [→ 36](lessons/module-12/lesson-02.md) |
| gradients accumulate (must reset) | `x.grad.zero_()` / `optimizer.zero_grad()` | [→ 25](lessons/module-08/lesson-03.md) |

```python
x = torch.tensor(3., requires_grad=True)
y = x ** 2 + 2 * x          # y = x^2 + 2x, so dy/dx = 2x + 2
y.backward()                # walks the graph in reverse (VJP)
print(x.grad)               # tensor(8.)  matches 2*3 + 2 = 8
print(y.grad_fn)            # <AddBackward0> : the last op in the graph
```

<div class="callout warn">
<p><code>.backward()</code> <em>adds</em> into <code>.grad</code>; it does not overwrite. Forgetting <code>zero_grad()</code> sums this step's gradient onto the last one. This is the single most common training bug. See [→ 25](lessons/module-08/lesson-03.md).</p>
</div>

## nn building blocks

| Math concept | PyTorch | Lesson |
|---|---|---|
| linear layer $\mathbf{y}=\mathbf{W}\mathbf{x}+\mathbf{b}$ | `nn.Linear(in, out)`; stores `.weight` $(out,in)$, `.bias` $(out,)$ | [→ 23](lessons/module-08/lesson-01.md) |
| embedding lookup (select rows) | `nn.Embedding(V, C)`; `emb(token_ids)` | [→ 31](lessons/module-10/lesson-03.md) |
| LayerNorm | `nn.LayerNorm(C)` | [→ 34](lessons/module-11/lesson-03.md) |
| softmax $p_i=e^{z_i}/\sum_j e^{z_j}$ | `F.softmax(z, dim=-1)` | [→ 30](lessons/module-10/lesson-02.md) |
| log-softmax (stable) | `F.log_softmax(z, dim=-1)` | [→ 30](lessons/module-10/lesson-02.md) |
| cross-entropy from logits | `F.cross_entropy(logits, targets)` | [→ 24](lessons/module-08/lesson-02.md) |
| MSE | `F.mse_loss(pred, target)` | [→ 24](lessons/module-08/lesson-02.md) |
| sigmoid $\sigma(x)$ | `torch.sigmoid(x)` | [→ 17](lessons/module-06/lesson-03.md) |
| GELU activation | `F.gelu(x)` | [→ 33](lessons/module-11/lesson-02.md) |

```python
lin = torch.nn.Linear(3, 2)      # weight (2,3), bias (2,)
h = lin(torch.randn(4, 3))       # batch of 4 -> shape (4, 2)
# F.cross_entropy expects RAW logits and integer class targets:
logits = torch.tensor([[2.0, 1.0, 0.0]])   # vocab = 3
target = torch.tensor([1])                  # correct class = index 1
loss = F.cross_entropy(logits, target)      # combines log_softmax + NLL
```

<div class="callout pt">
<p><code>nn.Linear</code> stores its weight as $(out, in)$ and computes <code>x @ weight.T + bias</code>, so the math $\mathbf{y}=\mathbf{W}\mathbf{x}+\mathbf{b}$ matches with $\mathbf{W}$ = <code>weight</code>. Do NOT pass softmax probabilities to <code>F.cross_entropy</code> — it applies log-softmax itself. See [→ 30](lessons/module-10/lesson-02.md).</p>
</div>

## Attention ops

The math is $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_h}+M)\mathbf{V}$ per head. Line by line:

| Math concept | PyTorch | Lesson |
|---|---|---|
| scores $\mathbf{Q}\mathbf{K}^\top/\sqrt{d_h}$ | `q @ k.transpose(-2, -1) / d_h ** 0.5` | [→ 32](lessons/module-11/lesson-01.md) |
| lower-triangular mask | `mask = torch.tril(torch.ones(T, T))` | [→ 32](lessons/module-11/lesson-01.md) |
| set future scores to $-\infty$ | `scores.masked_fill(mask == 0, float('-inf'))` | [→ 32](lessons/module-11/lesson-01.md) |
| softmax over keys | `F.softmax(scores, dim=-1)` | [→ 32](lessons/module-11/lesson-01.md) |
| weighted sum of values | `attn @ v` | [→ 32](lessons/module-11/lesson-01.md) |
| the whole thing, fused & stable | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` | [→ 32](lessons/module-11/lesson-01.md) |

```python
B, nh, T, dh = 1, 2, 3, 4
q = k = v = torch.randn(B, nh, T, dh)
scores = q @ k.transpose(-2, -1) / dh ** 0.5     # (B, nh, T, T)
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float('-inf'))
attn = F.softmax(scores, dim=-1)                 # rows sum to 1
out = attn @ v                                   # (B, nh, T, dh)
```

## Optimizers

Adam keeps a first moment (mean of gradients, math symbol $m$) and a second moment (mean of squared gradients, $v$) per parameter — that is the optimizer *state*.

| Math concept | PyTorch | Lesson |
|---|---|---|
| SGD $\theta\leftarrow\theta-\eta g$ | `torch.optim.SGD(model.parameters(), lr=0.1)` | [→ 26](lessons/module-09/lesson-01.md) |
| momentum | `SGD(..., momentum=0.9)` | [→ 27](lessons/module-09/lesson-02.md) |
| RMSProp | `torch.optim.RMSprop(..., alpha=0.99)` | [→ 27](lessons/module-09/lesson-02.md) |
| Adam ($\beta_1,\beta_2,\epsilon$) | `torch.optim.Adam(..., betas=(0.9, 0.999), eps=1e-8)` | [→ 27](lessons/module-09/lesson-02.md) |
| AdamW (decoupled weight decay $\lambda$) | `torch.optim.AdamW(..., weight_decay=0.1)` | [→ 27](lessons/module-09/lesson-02.md) |
| apply the update / reset grads | `optimizer.step()`, `optimizer.zero_grad()` | [→ 26](lessons/module-09/lesson-01.md) |
| different lr per group (no-decay biases) | `optimizer.param_groups` | [→ 27](lessons/module-09/lesson-02.md) |
| inspect stored moments $m,v$ | `optimizer.state[p]['exp_avg']`, `['exp_avg_sq']` | [→ 27](lessons/module-09/lesson-02.md) |

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
# after a backward pass, PyTorch stores per-parameter moments:
# opt.state[p]['exp_avg']    is m  (first moment)
# opt.state[p]['exp_avg_sq'] is v  (second moment)
```

## Numerical stability

| Math concept | PyTorch | Lesson |
|---|---|---|
| smallest/largest representable float | `torch.finfo(torch.float32).eps`, `.max` | [→ 35](lessons/module-12/lesson-01.md) |
| clip exploding gradients | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` | [→ 35](lessons/module-12/lesson-01.md) |
| mixed precision (bf16) | `with torch.autocast('cuda', dtype=torch.bfloat16): ...` | [→ 35](lessons/module-12/lesson-01.md) |
| why fused CE is stable | `F.cross_entropy` uses log-sum-exp internally, never forms huge $e^{z}$ | [→ 35](lessons/module-12/lesson-01.md) |
| check for NaN/Inf | `torch.isnan(t).any()`, `torch.isinf(t).any()` | [→ 35](lessons/module-12/lesson-01.md) |

<div class="callout warn">
<p>Computing softmax then log yourself can overflow (huge exponentials) or produce log of zero. <code>F.log_softmax</code> and <code>F.cross_entropy</code> subtract the max logit first (the log-sum-exp trick), so prefer them over hand-rolled versions. See [→ 35](lessons/module-12/lesson-01.md).</p>
</div>

## The 4-line training loop

Every GPT-2 training step is these four lines. Mathematically: reset gradients, run the forward composition to get the scalar loss $L$, backprop the chain rule to fill every `.grad`, then let the optimizer apply $\theta\leftarrow\theta-\eta(\dots)$.

```python
optimizer.zero_grad()          # clear last step's accumulated grads
logits, loss = model(x, y)     # forward pass: builds the graph, computes L
loss.backward()                # backward pass: chain rule fills p.grad for every param
optimizer.step()               # update: theta <- theta - lr * (moment-scaled grad)
```

Line-by-line math and what PyTorch stores at each line is walked through in [→ 39](lessons/module-12/lesson-05.md); the full GPT-2 forward/backward is [→ 37](lessons/module-12/lesson-03.md).

## Next

Use this next to the [Math cheat sheet](assets/math-cheatsheet.md): read a formula there, then find the row here that turns it into code. Any unfamiliar term is in the [Glossary](assets/glossary.md).
