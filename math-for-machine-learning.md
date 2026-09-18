Create a **complete, static, self-contained curriculum called “Mathematics for Machine Learning and Deep Learning”** for me.

The purpose of this curriculum is to give me the mathematical foundation needed to understand, implement, debug, and reason about modern neural networks, especially **GPT-2/GPT-style Transformers, backpropagation, derivatives, optimization, and fine-tuning**.

## My background

I am an experienced software developer and I am currently learning machine learning by implementing GPT-related code, including:

* GPT-2 architecture
* embeddings
* attention
* matrix operations
* derivatives
* backpropagation
* gradient descent
* Adam/AdamW
* Muon optimizer
* fine-tuning
* PyTorch

I already know programming very well, particularly C#, but **assume that I know absolutely nothing about the mathematics of machine learning**.

This does NOT mean the course should be mathematically simplistic. I actually want the mathematics explained thoroughly. Assume I may know ordinary mathematics, algebra, matrices, etc., but **do not rely on that knowledge**. Introduce every mathematical concept when it becomes relevant.

It is much better to include **too much explanation, too many examples, and too much mathematical detail than to skip something important**.

The goal is not to make me mathematically sophisticated for its own sake. The goal is that when I encounter mathematical notation or PyTorch code in an ML implementation, I understand exactly what it means and why it is there.

---

# Most important requirement: connect mathematics to PyTorch

Every major mathematical concept should eventually be connected to **actual PyTorch behavior**.

I don't want a course that teaches mathematics completely separately from machine learning code.

For example, when teaching derivatives:

1. Explain intuitively what a derivative means.
2. Explain the mathematical definition.
3. Show very small numerical examples.
4. Show the derivative calculation by hand.
5. Explain partial derivatives.
6. Explain gradients.
7. Explain the chain rule.
8. Then show what this means in a neural network.
9. Then show a tiny PyTorch example.
10. Explain what PyTorch's `forward()` is doing.
11. Explain what PyTorch stores during the forward pass.
12. Explain what happens when `.backward()` is called.
13. Explain where the gradients are stored.
14. Show the relationship between the mathematical derivative and `tensor.grad`.
15. Explain what PyTorch's autograd is actually calculating.

Do this throughout the curriculum.

I want to constantly be able to answer:

> “I understand the mathematics. Now what does PyTorch actually do with this?”

---

# Curriculum structure

Build the curriculum as a sequence of static pages/modules.

There should be a clear dependency order. Do not introduce a concept before the prerequisites have been explained.

The curriculum should eventually take me from basic mathematical notation all the way to understanding the mathematics behind:

**GPT-2 → training → backpropagation → AdamW → Muon → fine-tuning**

---

# Part 1 — Mathematical language

Start from extremely basic concepts.

Teach:

* numbers
* variables
* constants
* expressions
* functions
* equations
* notation
* summation notation
* products
* powers
* logarithms
* exponentials
* absolute value
* inequalities
* fractions
* rates of change

Explain notation carefully.

For example, don't simply write:

`f(x)`

Explain:

* what `f` means
* what `x` means
* what `f(x)` means
* why we use functions in ML
* examples of functions
* input/output interpretation

Likewise explain notation such as:

* `∑`
* `∂`
* `∇`
* `||x||`
* subscripts
* superscripts
* indices
* transpose
* inverse
* element-wise operations

---

# Part 2 — Vectors

Teach vectors from scratch.

Cover:

* what a vector is
* vector as a list of numbers
* vector as a point
* vector as a direction
* dimensions
* indexing
* vector addition
* scalar multiplication
* dot product
* magnitude/norm
* unit vectors
* normalization
* cosine similarity
* projections

Use tiny examples such as:

```text
x = [2, 3]
w = [4, 5]
```

Calculate everything manually.

Then show the equivalent PyTorch:

```python
x = torch.tensor([2., 3.])
w = torch.tensor([4., 5.])
```

Show the operations explicitly.

Explain the difference between:

* mathematical vector
* Python list
* PyTorch tensor

---

# Part 3 — Matrices

Teach matrices from scratch.

Cover:

* rows
* columns
* dimensions/shapes
* indexing
* matrix addition
* scalar multiplication
* matrix multiplication
* transpose
* identity matrix
* inverse
* rank
* linear transformations
* matrix-vector multiplication
* matrix-matrix multiplication

Use very small matrices and calculate products by hand.

For example:

```text
A = [[1, 2],
     [3, 4]]

x = [5, 6]
```

Calculate `Ax` manually.

Then show exactly what PyTorch does:

```python
A @ x
```

Explain shapes at every step.

This is particularly important because neural networks are largely tensor operations.

---

# Part 4 — Tensors and shapes

Explain tensors specifically in the context of deep learning.

Cover:

* scalar
* vector
* matrix
* 3D tensor
* 4D tensor
* arbitrary-dimensional tensor
* shape
* dimension
* axis
* batch dimension
* sequence dimension
* feature dimension
* broadcasting
* reshape
* view
* transpose
* permutation
* indexing
* slicing

Use GPT examples.

For example:

```text
(B, T, C)
```

Explain exactly what each dimension means.

Then:

```text
(B, T, n_heads, head_dim)
```

Explain why Transformer code constantly changes tensor shapes.

Show corresponding PyTorch operations.

---

# Part 5 — Functions and composition

Explain functions deeply.

Cover:

* one-input functions
* multi-input functions
* vector-valued functions
* composition
* nested functions
* computational graphs

Use examples such as:

```text
y = f(g(x))
```

Then show how neural networks are compositions of functions:

```text
input
  ↓
embedding
  ↓
linear layer
  ↓
activation
  ↓
linear layer
  ↓
loss
```

Make the connection to forward propagation.

---

# Part 6 — Calculus fundamentals

This should be one of the largest sections.

Teach:

* limits
* continuity
* slope
* derivative
* derivative as local sensitivity
* derivative rules
* power rule
* sum rule
* product rule
* quotient rule
* chain rule
* exponential derivative
* logarithm derivative
* sigmoid derivative
* softmax-related derivatives

Do not just state formulas.

For every important derivative:

1. Explain intuition.
2. Give the formula.
3. Give a numerical example.
4. Calculate it manually.
5. Explain what the number means.
6. Connect it to ML.

---

# Part 7 — Partial derivatives

Teach:

* functions with multiple variables
* partial derivative
* holding other variables constant
* gradient
* directional interpretation
* gradient vector

Example:

```text
f(x,y) = x² + 3xy + y²
```

Calculate:

```text
∂f/∂x
∂f/∂y
```

Then calculate the gradient at specific points.

Show the PyTorch equivalent.

---

# Part 8 — Chain rule and computational graphs

This must be extremely thorough.

Explain why the chain rule is the mathematical foundation of backpropagation.

Use tiny examples such as:

```text
x → a → b → y
```

For example:

```text
a = 2x
b = a²
y = b + 3
```

Calculate:

```text
dy/dx
```

both directly and through the chain rule.

Then construct computational graphs.

Show how each node contributes a local derivative.

Then show how these local derivatives are multiplied backward.

Only after this should you introduce neural-network backpropagation.

---

# Part 9 — Gradients

Explain:

* scalar derivative
* partial derivatives
* gradient
* gradient vector
* gradient with respect to a matrix
* gradient with respect to a tensor

Explain the distinction between:

```text
∂L/∂x
```

and:

```text
∇x L
```

Explain exactly what “gradient of the loss with respect to a parameter” means.

Use tiny neural-network examples.

---

# Part 10 — Jacobians

Teach Jacobians from scratch.

Cover:

* vector-valued functions
* derivatives of vectors
* Jacobian matrix
* shape of a Jacobian
* why Jacobians matter
* chain rule using Jacobians

Connect this to:

* linear layers
* activation functions
* vector operations
* softmax

Explain why we usually don't explicitly construct huge Jacobian matrices in PyTorch.

---

# Part 11 — Matrix calculus

Teach the matrix calculus needed for neural networks.

Cover:

* derivative of scalar with respect to vector
* derivative of scalar with respect to matrix
* vector-Jacobian products
* Jacobian-vector products
* matrix derivatives
* useful identities

Do not overwhelm me with abstract notation without motivation.

For every formula, explain what the shapes are.

This section should prepare me to understand derivatives of:

```text
y = Wx + b
```

and eventually:

```text
L(W,b)
```

---

# Part 12 — Linear layers and neural-network mathematics

Start building an actual neural network mathematically.

Explain:

```text
z = Wx + b
a = activation(z)
```

Then:

```text
L = loss(a, target)
```

Derive gradients with respect to:

* output
* activation
* weights
* bias
* input

Do a complete tiny numerical example.

For example, use a network with:

* 2 inputs
* 2 hidden neurons
* 1 output

Use small numbers.

Calculate the entire forward pass manually.

Then calculate the entire backward pass manually.

Then reproduce the exact same calculation with PyTorch.

This should be a major milestone in the course.

---

# Part 13 — Loss functions

Teach the mathematics behind:

* MSE
* MAE
* cross entropy
* negative log likelihood
* binary cross entropy
* softmax
* log-softmax

Explain why classification uses these functions.

For language models, spend significant time on:

```text
logits
→ softmax
→ probabilities
→ cross entropy
→ loss
```

Use tiny vocabulary examples.

For example:

```text
vocabulary = ["cat", "dog", "fish"]
logits = [2.0, 1.0, 0.0]
target = "dog"
```

Calculate:

1. softmax
2. probability of dog
3. negative log probability
4. loss
5. derivative

Then show the equivalent PyTorch operation.

---

# Part 14 — Backpropagation

Teach backpropagation completely from first principles.

Explain:

* forward pass
* computational graph
* loss
* local derivatives
* chain rule
* backward pass
* gradients
* parameter gradients

Do a complete numerical example.

Then show:

```python
loss.backward()
```

and explain what it actually means.

Explain:

* `requires_grad`
* `grad_fn`
* `.grad`
* leaf tensors
* computational graph
* graph construction
* graph traversal
* gradient accumulation
* `zero_grad()`

Explain what PyTorch is doing conceptually rather than treating `.backward()` as magic.

---

# Part 15 — Gradient descent

Teach:

```text
θ ← θ - η ∇L(θ)
```

from scratch.

Explain every symbol.

Use very small numerical examples.

Show how the parameter changes over several iterations.

Explain:

* learning rate
* gradient direction
* loss landscape
* local minima
* saddle points
* overshooting
* too-small learning rate
* too-large learning rate

Then implement the update manually in PyTorch.

---

# Part 16 — Optimization

Build toward modern optimizers.

Explain:

* SGD
* momentum
* Nesterov momentum
* adaptive learning rates
* RMSProp
* Adam
* AdamW

For each optimizer:

1. Explain the problem it is trying to solve.
2. Show the mathematical equations.
3. Define every variable.
4. Use tiny numerical examples.
5. Calculate several optimization steps manually.
6. Show the corresponding PyTorch implementation.
7. Explain exactly what state the optimizer stores.

For AdamW, explain especially:

* first moment
* second moment
* bias correction
* learning rate
* epsilon
* weight decay
* difference between Adam and AdamW

---

# Part 17 — Muon

Include a substantial section on the Muon optimizer.

I am currently learning Muon, so explain it from absolute basics.

Do not assume I understand why it exists.

Start with:

* what ordinary gradient descent does to a weight matrix
* why the gradient has a matrix structure
* why treating the entire gradient as ordinary element-wise updates may not be ideal
* what Muon is trying to change
* the role of matrix geometry
* orthogonalization
* Newton-Schulz iteration
* the relationship between singular values and matrix updates

Use very small matrices.

I want at least one complete start-to-finish example showing:

```text
W
↓
forward
↓
y
↓
loss
↓
gradient dL/dW
↓
Muon processing of the gradient
↓
updated W
```

with actual small numerical values.

Make the example concrete enough that I can calculate it with a calculator.

Then show how this relates to a real PyTorch optimizer implementation.

---

# Part 18 — Probability and statistics

Teach only the probability/statistics necessary for machine learning.

Cover:

* probability
* random variables
* distributions
* expectation
* variance
* standard deviation
* covariance
* conditional probability
* Bayes' theorem
* entropy
* cross entropy
* KL divergence
* maximum likelihood
* negative log likelihood

Connect everything to language modeling.

Explain why a language model produces a probability distribution over the vocabulary.

---

# Part 19 — Softmax and language modeling mathematics

Go deep on:

```text
hidden state
→ linear projection
→ logits
→ softmax
→ probability distribution
```

Explain:

* vocabulary size
* logits
* probabilities
* normalization
* temperature
* cross entropy
* next-token prediction

Use tiny vocabularies.

Explain why the final linear layer often has dimensions like:

```text
(hidden_size, vocabulary_size)
```

and how the matrix multiplication works.

---

# Part 20 — Embeddings

Explain the mathematics of embeddings.

Cover:

* embedding lookup
* embedding matrix
* token IDs
* one-hot interpretation
* matrix multiplication interpretation
* gradients through embedding lookup
* why only selected rows receive gradients during a token lookup
* embedding dimension

Connect this to GPT-2.

Also explain the difference between:

* token embedding lookup
* contextual representations
* final hidden states

---

# Part 21 — Attention mathematics

Teach the mathematics behind Transformer attention from scratch.

Cover:

* queries
* keys
* values
* dot products
* scaling
* attention scores
* causal masking
* softmax
* weighted sums
* multi-head attention

Use very small matrices.

For example:

```text
T = 3
head_dim = 2
```

Calculate an entire attention operation manually.

Show every matrix shape.

Then reproduce it with PyTorch.

Explain exactly how the mathematics maps onto operations such as:

```python
q @ k.transpose(-2, -1)
softmax(...)
@ v
```

---

# Part 22 — Transformer mathematics

Build from attention into a complete Transformer block.

Explain mathematically:

```text
x
↓
LayerNorm
↓
QKV
↓
attention
↓
projection
↓
residual
↓
LayerNorm
↓
MLP
↓
residual
```

Cover:

* residual connections
* normalization
* linear layers
* activation functions
* GELU
* dimensions
* parameter matrices

Explain how gradients flow through residual connections.

---

# Part 23 — Normalization

Teach:

* mean
* variance
* standard deviation
* normalization
* LayerNorm
* RMSNorm

Explain the formulas.

Use tiny vectors.

Calculate them manually.

Then show PyTorch implementations.

Explain why RMSNorm differs from LayerNorm.

---

# Part 24 — Numerical issues

Teach the mathematical/numerical concepts that matter in actual ML code:

* floating-point numbers
* precision
* rounding
* overflow
* underflow
* NaN
* infinity
* numerical stability
* log-sum-exp
* stable softmax
* epsilon terms
* gradient explosion
* gradient vanishing

Connect these directly to PyTorch and Transformer training.

---

# Part 25 — Automatic differentiation

Explain autograd more deeply.

Cover:

* symbolic differentiation
* numerical differentiation
* automatic differentiation
* forward-mode AD
* reverse-mode AD
* computational graphs
* vector-Jacobian products

Explain why neural-network training generally uses reverse-mode automatic differentiation.

Show simple PyTorch examples and explain what happens internally.

---

# Part 26 — Training a GPT-2-style model

Now bring everything together.

Explain the mathematics of the entire training process:

```text
tokens
↓
embeddings
↓
Transformer
↓
hidden states
↓
logits
↓
cross entropy
↓
loss
↓
backpropagation
↓
gradients
↓
optimizer
↓
updated parameters
```

Use equations throughout.

Explain what each tensor represents.

Explain shapes.

Explain which quantities are parameters, activations, gradients, and optimizer state.

---

# Part 27 — Fine-tuning mathematics

Teach the mathematical concepts needed to understand fine-tuning.

Cover:

* pretrained parameters
* initialization from pretrained weights
* freezing parameters
* trainable parameters
* gradients
* learning rates
* catastrophic forgetting
* full fine-tuning
* parameter-efficient fine-tuning
* LoRA

For LoRA, explain mathematically:

```text
W' = W + BA
```

Explain:

* matrix dimensions
* rank
* why this reduces trainable parameters
* gradients through A and B
* how this appears in PyTorch

---

# Part 28 — Final “read the code” section

End with a practical reference section.

Take a small GPT-2-style PyTorch training loop and explain it mathematically line by line.

For example:

```python
optimizer.zero_grad()

logits, loss = model(x, y)

loss.backward()

optimizer.step()
```

Explain exactly what happens mathematically at each line.

Then explain a complete optimizer step.

The final goal is that I can look at a real PyTorch training implementation and understand:

* what every tensor means
* what every dimension means
* what every major mathematical operation means
* where derivatives are calculated
* where gradients are stored
* how gradients flow backward
* how optimizer state is updated
* how parameters change

---

# Teaching style

Use a **bottom-up teaching approach**.

Never say:

> “Obviously…”

Never assume I understand a mathematical term just because it is common in ML.

When introducing a formula, always explain:

1. What the formula is saying in plain English.
2. What every symbol means.
3. The shapes of the quantities.
4. A tiny numerical example.
5. The calculation step by step.
6. Why ML needs this formula.
7. How it appears in PyTorch.

Use many concrete numerical examples.

Prefer:

```text
x = 2
y = 3x
```

before:

```text
y = f(x)
```

and prefer small matrices like 2×2 or 2×3 before large abstract matrices.

---

# Important distinction: intuition vs mathematics vs implementation

For major concepts, structure the explanation approximately like this:

### 1. Intuition

What is happening conceptually?

### 2. Mathematics

What exactly does the equation say?

### 3. Numerical example

Calculate it with actual numbers.

### 4. Machine-learning application

Why do neural networks need it?

### 5. PyTorch

Show the corresponding code.

### 6. What PyTorch is doing internally

Explain autograd/tensors/gradients/optimizer state where relevant.

This structure is particularly important for derivatives, backpropagation, optimizers, attention, normalization, and loss functions.

---

# Static curriculum requirements

This is a **static educational website/course**, not an interactive coding exercise platform.

Do NOT build:

* automatic exercise checking
* Python-based answer validation
* coding challenges requiring a backend
* progress-tracking systems
* unnecessary interactive applications

Instead, create well-organized static pages containing:

* explanations
* equations
* diagrams where useful
* numerical examples
* PyTorch examples
* tables
* references
* prerequisite relationships

It is fine to include small exercises/questions for self-checking, but they should simply provide the question and optionally the answer/explanation on the page. They do not need to be executed or checked by Python.

---

# Navigation

Organize the curriculum so that every page clearly indicates:

**Prerequisites:** what I should understand first.

**You will learn:** what this page teaches.

**Why this matters for ML:** why I am learning it.

**Next:** what concept this enables.

Also provide a complete curriculum map showing dependencies.

For example:

```text
Algebra
   ↓
Functions
   ↓
Vectors
   ↓
Matrices
   ↓
Tensors
   ↓
Derivatives
   ↓
Partial derivatives
   ↓
Chain rule
   ↓
Gradients
   ↓
Computational graphs
   ↓
Backpropagation
   ↓
Gradient descent
   ↓
AdamW
   ↓
Muon
```

and separately:

```text
Vectors
   ↓
Matrices
   ↓
Linear algebra
   ↓
QKV
   ↓
Attention
   ↓
Transformer
   ↓
GPT-2
```

---

# Important: don't make the course artificially short

The course should be **comprehensive**.

If a concept requires 5 pages, use 5 pages.

If derivatives require 10 pages to explain properly, use 10 pages.

Do not compress important mathematics into one paragraph just to reduce the number of pages.

I explicitly prefer **more educational material rather than less**.

At the same time, avoid filler. Every page should teach something useful.

---

# Mathematical rigor

The curriculum should eventually be rigorous enough that I can understand equations found in:

* GPT papers
* Transformer papers
* optimizer papers
* PyTorch implementations
* fine-tuning papers
* LoRA papers

But introduce the rigor gradually.

Do not start with abstract notation and expect me to decode it.

Build the notation step by step.

---

# Special requirement for GPT/LLM relevance

Whenever possible, explain why a mathematical concept matters specifically for GPT-style models.

For example:

* vectors → token representations
* matrices → neural-network weights
* matrix multiplication → linear layers
* derivatives → training
* chain rule → backpropagation
* probability → next-token prediction
* cross entropy → language-model loss
* softmax → token probabilities
* tensor shapes → Transformer implementation
* normalization → Transformer stability
* optimization → parameter updates
* matrix geometry → Muon
* low-rank matrices → LoRA

The course should ultimately feel like one continuous path toward understanding **how the mathematics makes GPT training work**.

---

# Deliverables

Create:

1. A complete curriculum map.
2. Static pages in logical dependency order.
3. A navigation system.
4. Mathematical notation rendered clearly.
5. Many small numerical examples.
6. PyTorch examples throughout.
7. A final GPT-2 training walkthrough.
8. A final mathematical cheat sheet containing the most important formulas.
9. A final PyTorch cheat sheet mapping mathematical concepts to PyTorch operations.
10. A glossary of mathematical/ML terms.

Do not leave major prerequisites unexplained.

If you think I probably already know something, **still explain it briefly**. I can skip it, but I cannot easily learn something that the curriculum omitted.

The final result should be suitable for someone who wants to go from:

> “I can program, but I don't understand the mathematics behind neural networks”

to:

> “I can read the mathematics of GPT-2, understand backpropagation and optimization, and understand what PyTorch is actually doing when it performs forward and backward passes.”
