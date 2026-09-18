# 39 · Read the code: a training loop line by line

<div class="prereq">
<p><strong>Prerequisites:</strong> the whole course, and especially the previous four lessons — numerical stability (<a href="lesson-01.md">35</a>), automatic differentiation (<a href="lesson-02.md">36</a>), the end-to-end model (<a href="lesson-03.md">37</a>), and fine-tuning (<a href="lesson-04.md">38</a>). Backprop mechanics from <a href="../module-08/lesson-03.md">25 · Backpropagation and autograd</a> and optimizers from <a href="../module-09/lesson-02.md">27 · AdamW</a> and <a href="../module-09/lesson-03.md">28 · Muon</a> are used constantly.</p>
<p><strong>You will learn:</strong> exactly what each line of a training loop does — <code>optimizer.zero_grad()</code>, the forward call, <code>loss.backward()</code>, <code>optimizer.step()</code> — in terms of which tensors exist, what math runs, where derivatives are computed, where gradients live, and how parameters change. Then the same reading applied to a fuller, realistic loop with autocast, clipping, scheduling, and gradient accumulation.</p>
<p><strong>Why this matters for ML:</strong> this is the finish line the whole course aimed at. When you can read a real GPT-2 training loop and narrate every line as precise mathematics, you can debug it, modify it, and reason about it. The closing checklist is the proof.</p>
</div>

## 1. The four lines

Almost every neural network ever trained has this loop at its heart:

```python
optimizer.zero_grad()
logits, loss = model(x, y)
loss.backward()
optimizer.step()
```

Four lines. They look trivial, but each is a dense mathematical event. We read them in the order they run, and — because `zero_grad`'s purpose only makes sense once you have seen `backward` accumulate — we will loop back to line 1 at the end. Assume the model, optimizer, and a batch `(x, y)` of shape $(B,T)$ from [lesson 37](lesson-03.md).

## 2. `optimizer.zero_grad()` — clear the gradient buffers

**What tensors exist:** every parameter $\theta_i$ and its gradient buffer `θ.grad`, which still holds *last step's* gradient (or `None` on the first call).

**What math happens:** each `θ.grad` is reset — set to `None` (modern default, `set_to_none=True`) or filled with zeros.

**Why it is here:** recall the single most important operational fact about `backward()` from [lesson 25](../module-08/lesson-03.md): **it *accumulates* into `.grad`, it does not overwrite.** That accumulation is correct *within* a backward pass (it implements the branching rule — a parameter used in several places sums its contributions). But between *training steps* it is a bug: without a reset, step 2's gradient would be added on top of step 1's leftover, and the update would use a stale, inflated gradient — as if the learning rate were secretly growing every step.

**Derivatives / gradients / state:** no derivatives computed here, no parameters changed. It only *clears the slots* so the upcoming `backward()` writes into clean buffers. It touches gradients, not parameters and not optimizer state (AdamW's $m,v$ are untouched by `zero_grad`).

## 3. `logits, loss = model(x, y)` — the forward pass

**What tensors exist afterward:** all the activations from [lesson 37](lesson-03.md) — the embedded $x$ of shape $(B,T,C)$, every block's intermediates, `logits` of shape $(B,T,V)$, and the scalar `loss` — plus, crucially, a **computational graph** connecting them, recorded as the code ran ([lesson 36](lesson-02.md)).

**What math happens:** the entire pipeline of lesson 37: embedding lookup and add, $N$ transformer blocks, final LayerNorm, the LM head producing logits, and cross-entropy against the shifted targets `y` producing the scalar loss

$$
L = -\frac{1}{B\,T}\sum_{b,t}\log\text{softmax}(\text{logits}_{b,t,:})_{\,y_{b,t}}.
$$

**Where derivatives come in:** none are *computed* yet — but the graph is *built for* them. Each operation on a tracked tensor attaches a `grad_fn` (its vector-Jacobian-product function, [lesson 36](lesson-02.md)) and records which tensors fed it. PyTorch also keeps the activations these VJPs will need during the backward pass. This is the "record" phase of reverse-mode AD.

**Gradients / state:** `.grad` buffers are still empty (we just cleared them); optimizer state is untouched. The forward pass produces *values and a graph*, nothing else.

## 4. `loss.backward()` — the backward pass

**What tensors exist afterward:** every parameter's `.grad` is now filled — one gradient tensor per parameter, each the same shape as its parameter ([lesson 37](lesson-03.md)). The graph and the activations it held are, by default, **freed**.

**What math happens — the heart of training.** Reverse-mode automatic differentiation ([lesson 36](lesson-02.md)):

1. Seed $\dfrac{\partial L}{\partial L} = 1$ at the loss node.
2. Walk the recorded graph in reverse topological order — each node processed only after everything it fed has been processed ([lesson 25](../module-08/lesson-03.md)).
3. At each node apply its VJP: multiply the incoming upstream gradient by that node's local derivative to get the downstream gradient, summing where a tensor branched.
4. Deposit the result into each leaf parameter's `.grad`.

**Where derivatives are computed:** here, and only here. Every $\partial L/\partial\theta_i$ is produced in this one reverse sweep — the whole gradient of a scalar loss with respect to millions of parameters, in roughly the cost of one forward pass, because it uses vector-Jacobian products and never forms a full Jacobian ([lesson 36](lesson-02.md)).

**Where gradients are stored:** in each parameter's `.grad`, *added* to whatever was there — which is exactly why line 1 cleared them first.

**Gradients / state:** parameters are unchanged (backward computes gradients, it does not update weights); optimizer state is still untouched. After this line, `.grad` is full and ready to be consumed.

<div class="callout key"><p><code>loss.backward()</code> is the entire calculus half of the course executed at once: the chain rule (module 6–7), applied as vector-Jacobian products (module 7), over the graph recorded in the forward pass (module 5, lesson 36), depositing $\partial L/\partial\theta$ into every parameter's <code>.grad</code> (module 8). Nothing here is magic — it is the by-hand backward pass of <a href="../module-08/lesson-01.md">lesson 23</a>, automated.</p></div>

## 5. `optimizer.step()` — update the parameters

**What tensors exist:** parameters $\theta$, their freshly-filled `.grad`, and the optimizer's per-parameter **state** ($m,v$ for AdamW; a momentum buffer for Muon) carried over from previous steps.

**What math happens:** for each parameter the optimizer combines the gradient with its state to produce an update. For AdamW ([lesson 27](../module-09/lesson-02.md)), per parameter:

$$
m \leftarrow \beta_1 m + (1-\beta_1)g,\quad v \leftarrow \beta_2 v + (1-\beta_2)g^2,\quad
\hat m = \tfrac{m}{1-\beta_1^t},\ \hat v = \tfrac{v}{1-\beta_2^t},
$$

$$
\theta \leftarrow \theta - \eta\left(\frac{\hat m}{\sqrt{\hat v}+\varepsilon} + \lambda\theta\right),
$$

where $g$ is the parameter's `.grad`, $\eta$ the learning rate, $\lambda$ the weight decay, and $\varepsilon$ the stability term from [lesson 35](lesson-01.md). For Muon ([lesson 28](../module-09/lesson-03.md)) the 2-D hidden matrices are updated by their orthogonalized gradient instead.

**Where gradients are consumed and state updated:** the `.grad` buffers are *read* (not cleared — that is line 1's job next step), the state buffers $m,v$ are *updated in place*, and the parameters are *modified in place*. This is the only line that changes $\theta$.

**Under the hood:** `step()` runs inside `torch.no_grad()` — the update $\theta \leftarrow \theta - \ldots$ must not be recorded as a graph operation, or it would corrupt the next backward pass ([lesson 25](../module-08/lesson-03.md)). After `step()`, the parameters are new; the loop returns to `zero_grad()` and repeats with the next batch.

## 6. The cycle, in one breath

Put the four lines in a sentence you can now unpack completely: *clear the old gradients, run the forward pass to get the loss and record the graph, replay the chain rule backward to fill every `.grad`, then let the optimizer turn those gradients (plus its state) into new parameters.* Values flow forward; the graph is built forward; derivatives flow backward; parameters move once, at the end. That is training.

## 7. A fuller, realistic loop

Real training loops add a few lines around the core four. Here is a representative one, annotated line by line.

```python
for x, y in dataloader:                              # 1
    x, y = x.to(device), y.to(device)                # 2
    with torch.autocast(device_type="cuda",          # 3
                         dtype=torch.bfloat16):
        logits, loss = model(x, y)                   # 4
        loss = loss / accum_steps                    # 5
    loss.backward()                                  # 6
    if (step + 1) % accum_steps == 0:                # 7
        torch.nn.utils.clip_grad_norm_(              # 8
            model.parameters(), max_norm=1.0)
        optimizer.step()                             # 9
        scheduler.step()                             # 10
        optimizer.zero_grad(set_to_none=True)        # 11
    step += 1                                        # 12
```

**Line 1 — batch fetch.** The dataloader yields a micro-batch `x, y`, each shape $(B,T)$ of token IDs, `y` being `x` shifted by one ([lesson 37](lesson-03.md)). This is input data — no gradient, integer dtype.

**Line 2 — move to device.** Copy the tensors to the GPU so the matmuls run there. Pure data movement, no math.

**Line 3 — autocast / bfloat16.** Inside this block, PyTorch runs the heavy operations (matmuls, attention) in **bfloat16** for speed and memory, while keeping numerically sensitive reductions (the loss, softmax normalizers) in float32. This is exactly the range-vs-precision tradeoff of [lesson 35](lesson-01.md): bf16's float32-sized exponent range avoids overflow, and the loss stays in float32 so log-sum-exp has the precision it needs.

**Line 4 — forward.** The full lesson-37 pipeline: embeddings → blocks → final LayerNorm → head → cross-entropy. Produces `logits` $(B,T,V)$ and the scalar `loss`, and records the graph.

**Line 5 — scale for accumulation.** Divide the loss by `accum_steps` (explained at line 7). Because gradients are linear in the loss, dividing the loss by $N$ divides every gradient by $N$, so that summing gradients over $N$ micro-batches yields their *average* — what a single big batch would have produced.

**Line 6 — backward.** Reverse-mode AD (section 4) fills `.grad`. Called every micro-batch. Because `backward()` accumulates and we do *not* zero between micro-batches, the gradients from several micro-batches **add up** in `.grad` — this is the mechanism of gradient accumulation.

**Line 7 — accumulation gate.** Only every `accum_steps` micro-batches do we actually update. Gradient accumulation fakes a large batch on limited memory: instead of one forward/backward on a huge batch (which may not fit), do `accum_steps` smaller forward/backward passes, letting their gradients sum in `.grad`, then update once. The effective batch size is $B \times \texttt{accum\_steps}$. The division on line 5 is what makes the accumulated sum an average rather than a total.

**Line 8 — gradient clipping.** `clip_grad_norm_` measures the global gradient norm across all parameters and, if it exceeds `max_norm=1.0`, rescales the whole gradient down to that norm — capping length, preserving direction ([lesson 35](lesson-01.md)). This is the standard guard against gradient explosion, placed after all accumulation and before the step.

**Line 9 — optimizer step.** Consume `.grad` and the optimizer state to update parameters (section 5). In a two-optimizer setup ([lesson 28](../module-09/lesson-03.md)) you would call `.step()` on both the AdamW and the Muon optimizer here.

**Line 10 — scheduler step.** The learning-rate **scheduler** adjusts $\eta$ according to a schedule — typically a warmup (ramp $\eta$ up over the first steps to avoid early instability) followed by a cosine decay toward zero. It changes the $\eta$ that line 9 will use next time; it touches no gradients.

**Line 11 — zero the gradients.** Now, *after* the update, clear `.grad` (`set_to_none=True` frees the memory and is slightly faster). This resets for the next accumulation cycle. Placed here, not every micro-batch, precisely because accumulation needs the gradients to persist across micro-batches until the update fires.

**Line 12 — step counter.** Bookkeeping for the accumulation gate and the scheduler.

<div class="callout pt"><p>The extra lines are all refinements of the core four. autocast is <a href="lesson-01.md">lesson 35</a>'s precision tradeoff; clipping is its explosion guard; the scheduler tunes $\eta$; accumulation exploits that <code>backward()</code> sums into <code>.grad</code> (<a href="../module-08/lesson-03.md">lesson 25</a>) to average several micro-batches into one virtual big batch. Strip them away and <code>zero_grad → forward → backward → step</code> remains.</p></div>

## 8. You can now read GPT-2 — the checklist

This is the promise the course opened with, now discharged. Given a real GPT-2 training implementation, you can answer each of the following — with the module that taught it in brackets.

- **What every tensor means.** Tokens are integer IDs; embeddings turn them into vectors; hidden states are $(B,T,C)$ activations; logits are $(B,T,V)$ scores; the loss is one scalar. Each is a **parameter, activation, gradient, or optimizer state** — the four kinds of quantity. *(modules 2–4, [lesson 37](lesson-03.md))*
- **What every dimension means.** $B$ batch, $T$ sequence, $C$ channels/$d_\text{model}$, $V$ vocabulary, $n_h$ heads, $d_h$ head dim with $C=n_h d_h$; reshapes and transposes just re-view the same numbers to expose the head axis. *(module 4, [lesson 32](../module-11/lesson-01.md))*
- **What every major operation means.** Matrix multiply = a linear layer; dot products = attention scores; softmax = turning scores into a distribution; residual add = the skip path; LayerNorm = rescaling activations; cross-entropy = negative log-likelihood of the true next token. *(modules 3, 8, 10, 11)*
- **Where derivatives are computed.** In `loss.backward()`, and nowhere else — one reverse-mode sweep of vector-Jacobian products over the graph the forward pass recorded. *(modules 6–7, [lesson 36](lesson-02.md))*
- **Where gradients are stored.** In each parameter's `.grad`, one tensor per parameter, same shape, *accumulated* (hence `zero_grad`). *(module 8, [lesson 25](../module-08/lesson-03.md))*
- **How gradients flow backward.** From the loss, through the head and every block — including the multiply-by-one residual paths that keep them from vanishing — back to the embeddings, by the chain rule. *(modules 6–7, [lesson 33](../module-11/lesson-02.md), [lesson 35](lesson-01.md))*
- **How optimizer state updates.** AdamW updates first/second moments $m,v$ per parameter; Muon updates a momentum buffer per matrix and orthogonalizes it. State persists across steps; gradients do not. *(module 9, [lesson 27](../module-09/lesson-02.md), [lesson 28](../module-09/lesson-03.md))*
- **How parameters change.** Once per optimizer step, in place, under `no_grad`: $\theta \leftarrow \theta - \eta(\ldots)$ — the only line that moves the weights. *(module 9, [lesson 26](../module-09/lesson-01.md))*

If you can attach those answers to the lines of a training script, you have gone from *"I can program, but I don't understand the mathematics behind neural networks"* to *"I can read the mathematics of GPT-2, follow backpropagation and optimization, and say what PyTorch is doing at every step."* That was the goal.

## Check yourself

<details><summary>In the four-line loop, which line computes derivatives, which line changes parameters, and which line clears gradients?</summary>

`loss.backward()` computes all derivatives (fills `.grad`). `optimizer.step()` is the only line that changes parameters. `optimizer.zero_grad()` clears the gradient buffers. The forward call computes values and records the graph but does neither.

</details>

<details><summary>Why is <code>zero_grad()</code> necessary, and what breaks if you forget it?</summary>

`backward()` *adds* into `.grad`. Without a reset between steps, each step's gradient piles on top of the previous ones, so updates use stale, inflated gradients — the loss diverges as if the learning rate kept growing. `zero_grad()` clears the slots so each step's `backward()` writes clean gradients.

</details>

<details><summary>In gradient accumulation, why divide the loss by <code>accum_steps</code>, and why not call <code>zero_grad()</code> every micro-batch?</summary>

`backward()` sums gradients into `.grad` across the micro-batches (we intentionally don't zero between them), giving the *total* over the micro-batches. Dividing the loss by `accum_steps` divides every gradient by that count, turning the sum into the *average* — matching one big batch. You zero only after the actual `step()`, once per accumulation cycle.

</details>

<details><summary>Why does <code>optimizer.step()</code> run under <code>torch.no_grad()</code>?</summary>

The update $\theta \leftarrow \theta - \eta(\ldots)$ modifies parameters; it is not part of the function being differentiated. Recording it in the graph would corrupt the next backward pass and waste memory. `no_grad` tells autograd not to track it. *(See [lesson 25](../module-08/lesson-03.md).)*

</details>

## Next

You have reached the finish line: a full GPT-2 training loop, read line by line, every tensor and operation named. From here, keep the two reference sheets and the glossary at hand while you read real code — they compress this whole course into lookup tables.

Continue to the [Math cheat sheet](../../assets/math-cheatsheet.md) and the [Glossary](../../assets/glossary.md).
