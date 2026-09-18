# 38 · Fine-tuning and LoRA

<div class="prereq">
<p><strong>Prerequisites:</strong> matrix multiplication and shapes from <a href="../module-03/lesson-02.md">10 · Matrix multiplication</a>; <strong>rank</strong> from <a href="../module-03/lesson-03.md">11 · Transpose, identity, inverse, rank</a>; the linear layer and its gradients from <a href="../module-08/lesson-01.md">23 · The linear layer</a>; <code>requires_grad</code> and leaf tensors from <a href="../module-08/lesson-03.md">25 · Backpropagation and autograd</a>; and the full training step from <a href="lesson-03.md">37 · Training a GPT-2-style model end to end</a>.</p>
<p><strong>You will learn:</strong> what it means to start from pretrained weights, freeze parameters, and fine-tune; why fine-tuning uses a smaller learning rate; what catastrophic forgetting is; the cost of full fine-tuning versus parameter-efficient methods; and <strong>LoRA</strong> in full mathematical detail — the low-rank update $\Delta W = BA$, its parameter count, its gradients, its initialization, and its PyTorch form.</p>
<p><strong>Why this matters for ML:</strong> you rarely train a GPT from scratch; you take a pretrained model and adapt it. LoRA is how that adaptation is done cheaply — training a $768\times768$ layer by learning only $12{,}288$ numbers instead of $589{,}824$. Understanding it ties together rank, gradients, and the parameter/optimizer-state accounting from the previous lesson.</p>
</div>

## 1. Pretraining vs fine-tuning

**Pretraining** is what [lesson 37](lesson-03.md) described: start from random weights and train on a huge generic corpus until the model is a competent next-token predictor. It is expensive — many GPUs, a long time.

**Fine-tuning** starts from those already-trained **pretrained parameters** and continues training on a smaller, task-specific dataset (say, your company's support tickets) to specialize the model. The crucial difference is the starting point: instead of initializing $\theta$ randomly, you **initialize from the pretrained weights**. The model already knows grammar, facts, and reasoning patterns; fine-tuning only nudges it toward the new task. Everything mechanical — forward pass, `loss.backward()`, optimizer step — is identical to lesson 37. Only *where you start* and *how far you move* change.

## 2. The fine-tuning learning rate is smaller — and why

Fine-tuning almost always uses a **smaller learning rate** $\eta$ than pretraining (often 10–100× smaller). The reason is geometric. The pretrained weights already sit in a good region of the loss landscape ([lesson 26](../module-09/lesson-01.md)). A large step would fling the parameters far from that hard-won region, destroying the general competence before the model has learned anything task-specific. A small $\eta$ takes gentle steps that adapt the model while staying near the pretrained solution.

This connects directly to the next idea.

## 3. Catastrophic forgetting

**Catastrophic forgetting** is what happens when fine-tuning overwrites the pretrained knowledge: the model gets good at the new task but *loses* its general ability — it forgets how to do everything it knew before. It is the failure mode a too-large learning rate, or too many epochs on a narrow dataset, produces. The model's weights move so far to fit the new data that the old capabilities are erased.

The defenses all amount to "move less": a small learning rate (section 2), few epochs, and — most powerfully — **freezing most of the model and only training a tiny part**, which is where parameter-efficient methods come in.

## 4. Freezing parameters

To **freeze** a parameter is to declare that it will not be updated. In PyTorch that is one flag:

```python
for p in model.parameters():
    p.requires_grad = False        # freeze everything
```

Setting `requires_grad = False` has three precise consequences, each tying back to earlier lessons:

- **It still participates in the forward pass.** A frozen weight is used in its matrix multiply exactly as before — freezing changes learning, not computation.
- **It receives no gradient.** With `requires_grad = False`, autograd does not track operations for it, so its `.grad` stays `None` ([lesson 25](../module-08/lesson-03.md)).
- **It gets no optimizer state and no update.** No gradient means AdamW never allocates $m,v$ for it and never steps it. It is a constant.

A **trainable parameter** is the opposite: `requires_grad = True`, gets a gradient, gets optimizer state, gets updated. Fine-tuning is a choice of which parameters are trainable and which are frozen.

## 5. Full fine-tuning is expensive

**Full fine-tuning** leaves every parameter trainable. It works well but carries the full lesson-37 memory bill: for $P$ parameters you need $P$ gradients plus $2P$ AdamW moments — three extra full-model-sized buffers. For GPT-2 small ($P\approx124$M) that is gigabytes; for a 7-billion-parameter model it is impractical on a single GPU. And you must store a *separate complete copy* of the fine-tuned weights for every task. This is the cost that **parameter-efficient fine-tuning (PEFT)** attacks: adapt the model by training *far fewer* parameters, so gradients and optimizer state shrink accordingly.

## 6. LoRA: the idea

**LoRA** (Low-Rank Adaptation) is the dominant PEFT method. The observation behind it: the *change* a task needs to make to a big weight matrix is usually "simple" — it lives in a low-dimensional subspace, i.e. it has low **rank** ([lesson 11](../module-03/lesson-03.md)). So instead of learning a full-rank update, learn a low-rank one.

Take a pretrained weight matrix $\mathbf{W}$ of shape $(d, k)$. **Freeze it.** Add a trainable update $\Delta\mathbf{W}$ that is *factored* into two thin matrices:

$$
\Delta\mathbf{W} = \mathbf{B}\mathbf{A}, \qquad \mathbf{A}\in\mathbb{R}^{r\times k},\ \ \mathbf{B}\in\mathbb{R}^{d\times r},\ \ r \ll \min(d,k).
$$

The effective weight becomes

$$
\mathbf{W}' = \mathbf{W} + \mathbf{B}\mathbf{A} \qquad\text{(sometimes scaled: } \mathbf{W}' = \mathbf{W} + \tfrac{\alpha}{r}\mathbf{B}\mathbf{A}\text{)}.
$$

- $\mathbf{W}$: $(d,k)$, **frozen** — the pretrained knowledge.
- $\mathbf{A}$: $(r,k)$, **trainable** — projects the $k$-dim input *down* to $r$ dimensions.
- $\mathbf{B}$: $(d,r)$, **trainable** — projects those $r$ dimensions *up* to $d$.
- $r$: the **rank**, a small integer like $4$, $8$, or $16$.
- $\alpha$: an optional fixed scaling constant (not learned) that controls the update's strength.

The product $\mathbf{B}\mathbf{A}$ has shape $(d,r)\times(r,k) = (d,k)$ — the same shape as $\mathbf{W}$, so it is a valid update — but because it is squeezed through the width-$r$ bottleneck, its rank is at most $r$. That is the whole trick: a full-shape update built from two tiny matrices.

## 7. Never form $\mathbf{B}\mathbf{A}$: two small matmuls in the forward pass

The naive reading is "compute $\mathbf{W}' = \mathbf{W} + \mathbf{B}\mathbf{A}$, then multiply by $x$." That would form the full $(d,k)$ matrix $\mathbf{B}\mathbf{A}$ and throw away all the savings. Instead, use associativity to keep everything thin. For input $\mathbf{x}$ (shape $k$):

$$
\mathbf{h} = \mathbf{W}'\mathbf{x} = \mathbf{W}\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x} = \underbrace{\mathbf{W}\mathbf{x}}_{\text{frozen path}} + \underbrace{\mathbf{B}(\mathbf{A}\mathbf{x})}_{\text{LoRA path}}.
$$

Read the LoRA path right to left: first $\mathbf{A}\mathbf{x}$ (shape $(r,k)\times k = r$, a tiny vector), then $\mathbf{B}(\mathbf{A}\mathbf{x})$ (shape $(d,r)\times r = d$). Two small matrix-vector products, never the big $(d,k)$ product. The frozen $\mathbf{W}\mathbf{x}$ is computed as always. You add the two $d$-vectors and you are done.

## 8. Why this reduces trainable parameters — the count

Here is the payoff, made concrete with a realistic GPT-2 layer size $d = k = 768$ and rank $r = 8$.

**Full fine-tuning** trains all of $\mathbf{W}$:

$$
d\cdot k = 768\times768 = 589{,}824 \text{ trainable parameters.}
$$

**LoRA** trains only $\mathbf{A}$ and $\mathbf{B}$:

$$
r(d + k) = 8\times(768 + 768) = 8\times1536 = 12{,}288 \text{ trainable parameters.}
$$

The ratio is $\dfrac{589{,}824}{12{,}288} = 48$ — about **48× fewer** trainable parameters. And the savings compound through the training bill from [lesson 37](lesson-03.md): only these $12{,}288$ numbers get gradients, and only they get AdamW's $m,v$ state. The frozen $\mathbf{W}$ needs neither. So gradient memory and optimizer memory both shrink by the same $\sim48\times$, which is what turns a model too big to full-fine-tune into one that fits.

<div class="callout key"><p>LoRA replaces a full-rank $(d,k)$ update with a rank-$r$ factorization $\mathbf{B}\mathbf{A}$, trading $d\cdot k$ trainable numbers for $r(d+k)$. Because only $\mathbf{A}$ and $\mathbf{B}$ are trainable, gradients and optimizer state shrink by the same factor. At $d=k=768,\ r=8$ that is 48× fewer.</p></div>

Why is a rank-$r$ update enough? Because (the empirical finding behind LoRA) the *adaptation* a downstream task requires lies in a tiny subspace of the full weight space — it has intrinsically low rank ([lesson 11](../module-03/lesson-03.md)). You do not need to move the weight in all $d\cdot k$ directions; a handful, $r$ of them, captures the task.

## 9. Gradients flow only through $\mathbf{A}$ and $\mathbf{B}$

Since $\mathbf{W}$ is frozen, `backward()` computes gradients only for $\mathbf{A}$ and $\mathbf{B}$. Let $\mathbf{g} = \partial L/\partial\mathbf{h}$ be the upstream gradient arriving at the layer output (shape $d$). Using the linear-layer gradient rules from [lesson 23](../module-08/lesson-01.md) on the LoRA path $\mathbf{h} = \mathbf{B}(\mathbf{A}\mathbf{x})$:

$$
\frac{\partial L}{\partial \mathbf{B}} = \mathbf{g}\,(\mathbf{A}\mathbf{x})^\top, \qquad
\frac{\partial L}{\partial \mathbf{A}} = (\mathbf{B}^\top\mathbf{g})\,\mathbf{x}^\top.
$$

Check the shapes, which is the whole point of matrix calculus:

- $\dfrac{\partial L}{\partial \mathbf{B}}$: $\mathbf{g}$ is $d$, $\mathbf{A}\mathbf{x}$ is $r$, so $\mathbf{g}(\mathbf{A}\mathbf{x})^\top$ is $(d,r)$ — matches $\mathbf{B}$. ✓
- $\dfrac{\partial L}{\partial \mathbf{A}}$: $\mathbf{B}^\top\mathbf{g}$ is $(r,d)\times d = r$, and $\mathbf{x}^\top$ is $(1,k)$, so the product is $(r,k)$ — matches $\mathbf{A}$. ✓

There is no $\partial L/\partial\mathbf{W}$ term because $\mathbf{W}$ is frozen — autograd simply does not produce it. Every gradient the layer emits is one of these two small matrices.

## 10. Initialization: $\mathbf{A}$ random, $\mathbf{B}$ zero

LoRA initializes $\mathbf{A}$ with small **random** values and $\mathbf{B}$ with **zeros**. The consequence:

$$
\Delta\mathbf{W} = \mathbf{B}\mathbf{A} = \mathbf{0}\cdot\mathbf{A} = \mathbf{0} \quad\text{at step 0,}\qquad\text{so}\qquad \mathbf{W}' = \mathbf{W}.
$$

At the very first step the adapted model is *exactly* the pretrained model — the LoRA update contributes nothing. Training then starts from the pretrained behavior and gradually grows $\mathbf{B}$ away from zero. This matters for two reasons. First, it means fine-tuning begins from a known-good model rather than a randomly perturbed one, so you never damage the pretrained weights on step 0. Second, $\mathbf{A}$ must be *nonzero* random (not also zero), or its gradient $\partial L/\partial\mathbf{A} = (\mathbf{B}^\top\mathbf{g})\mathbf{x}^\top$ would be fine but $\partial L/\partial\mathbf{B} = \mathbf{g}(\mathbf{A}\mathbf{x})^\top$ would start nonzero only because $\mathbf{A}\mathbf{x}\neq 0$ — the asymmetry (one zero, one random) is exactly what breaks the symmetry so both matrices can start learning.

<div class="callout warn"><p>Do not initialize both $\mathbf{A}$ and $\mathbf{B}$ to zero. Then $\mathbf{A}\mathbf{x}=\mathbf{0}$ and $\mathbf{B}^\top\mathbf{g}=\mathbf{0}$, so <em>both</em> gradients vanish and LoRA never learns. The recipe is $\mathbf{B}=\mathbf{0}$ (so $\Delta\mathbf{W}=0$ initially) but $\mathbf{A}$ random (so gradients can flow).</p></div>

## 11. LoRA in PyTorch

A minimal `LoRALinear` that wraps a frozen `nn.Linear` and adds the two trainable matrices:

```python
import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base                       # the pretrained (d, k) Linear
        for p in self.base.parameters():
            p.requires_grad = False            # FREEZE W (and its bias)

        d, k = base.out_features, base.in_features
        self.A = nn.Parameter(torch.randn(r, k) * 0.01)   # random, small
        self.B = nn.Parameter(torch.zeros(d, r))          # zero -> dW = 0 at start
        self.scale = alpha / r                             # the alpha/r factor

    def forward(self, x):                       # x: (..., k)
        frozen = self.base(x)                   # W x  (+ frozen bias)
        lora   = (x @ self.A.t()) @ self.B.t()  # (A x) then B(A x): two small matmuls
        return frozen + self.scale * lora       # W x + (alpha/r) B A x
```

Two details worth reading against the math. The frozen path `self.base(x)` is the untouched pretrained layer. The LoRA path computes `x @ self.A.t()` first (the small $\mathbf{A}\mathbf{x}$, down to width $r$) and only then `@ self.B.t()` (back up to $d$) — associativity keeping both matmuls thin, section 7. Only `self.A` and `self.B` are `nn.Parameter` with `requires_grad=True`, so they are the only **leaf tensors with gradients** and the only ones AdamW allocates state for.

**Merging at inference.** Once training is done you can fold the update back into the weight so there is zero extra latency — LoRA adds nothing to the forward cost after merging:

```python
with torch.no_grad():
    delta = (alpha / r) * (lora.B @ lora.A)     # form B A once, shape (d, k)
    lora.base.weight += delta                    # W  <-  W + (alpha/r) B A
# now lora.base alone reproduces W', and A, B can be discarded
```

You form the full $(d,k)$ product $\mathbf{B}\mathbf{A}$ *only here*, once, offline — never in the training forward pass.

## 12. Under the hood

Everything ties back to the accounting from [lesson 37](lesson-03.md):

- **Only $\mathbf{A}$ and $\mathbf{B}$ are leaves with `requires_grad=True`.** They are the only tensors `backward()` fills a `.grad` for. $\mathbf{W}$'s `.grad` stays `None`.
- **Optimizer state exists only for $\mathbf{A}$ and $\mathbf{B}$.** AdamW allocates $m,v$ per *trainable* parameter, so the moment buffers are $\sim48\times$ smaller than in full fine-tuning — the memory win, restated in the language of optimizer state.
- **The frozen $\mathbf{W}$ is a constant in the graph.** It appears in the forward pass (as $\mathbf{W}\mathbf{x}$) but autograd records no backward function that targets it, so it contributes nothing to memory beyond the weights themselves.

LoRA is, in the end, three ideas you already own stacked together: freezing (`requires_grad=False`), a low-rank factorization (rank, from module 3), and the ordinary linear-layer gradient rules (module 8) applied to two small matrices instead of one big one.

## Check yourself

<details><summary>For $d=k=768$ and $r=8$, how many trainable parameters does LoRA have, versus full fine-tuning, and what is the ratio?</summary>

Full: $d\cdot k = 768\times768 = 589{,}824$. LoRA: $r(d+k) = 8\times1536 = 12{,}288$. Ratio $589{,}824 / 12{,}288 = 48$ — about 48× fewer trainable parameters.

</details>

<details><summary>Why compute $\mathbf{B}(\mathbf{A}\mathbf{x})$ rather than $(\mathbf{B}\mathbf{A})\mathbf{x}$?</summary>

$(\mathbf{B}\mathbf{A})$ is a full $(d,k)$ matrix — forming it throws away all the savings and costs as much as the original weight. $\mathbf{B}(\mathbf{A}\mathbf{x})$ does two thin matmuls: $\mathbf{A}\mathbf{x}$ (down to width $r$), then $\mathbf{B}(\cdot)$ (back to $d$). Same result, tiny cost.

</details>

<details><summary>Why initialize $\mathbf{B}=\mathbf{0}$ but $\mathbf{A}$ random?</summary>

$\mathbf{B}=\mathbf{0}$ makes $\Delta\mathbf{W}=\mathbf{B}\mathbf{A}=\mathbf{0}$ at step 0, so training begins from exactly the pretrained model — no damage. $\mathbf{A}$ must be random (nonzero) so that gradients can flow into both matrices; if both were zero, both gradients would vanish and LoRA would never learn.

</details>

<details><summary>After freezing $\mathbf{W}$, what is <code>W.grad</code> after <code>backward()</code>, and does AdamW keep state for it?</summary>

`W.grad` is `None` — with `requires_grad=False`, autograd never computes a gradient for it. AdamW keeps no $m,v$ state for it and never updates it. Only $\mathbf{A}$ and $\mathbf{B}$ get gradients and optimizer state.

</details>

## Next

You can now adapt a pretrained model cheaply and explain every tensor involved. The final lesson is the capstone: a real training loop read line by line, closing with the checklist that proves you can read GPT-2 code end to end.

Continue to [39 · Read the code: a training loop line by line](lesson-05.md).
