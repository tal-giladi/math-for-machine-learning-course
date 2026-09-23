# Glossary

<div class="prereq">
<p><strong>What this is:</strong> every important mathematical and ML term used in the course, in one alphabetical list, each with a plain-English definition and a link to the lesson that teaches it properly.</p>
<p><strong>How to read it:</strong> definitions are deliberately short — one to three sentences. Follow the <code>[→ NN]</code> link for the full treatment with worked numbers. Math renders in KaTeX.</p>
</div>

## A

**Activation** — the output $a$ of a neuron after applying a nonlinearity to the pre-activation $z$, i.e. $a=\text{activation}(z)$. In a running network, activations are the intermediate tensors PyTorch keeps around for the backward pass. [→ 23](lessons/module-08/lesson-01.md)

**Adam** — an optimizer that keeps a running mean (first moment) and running mean-of-squares (second moment) of the gradient per parameter, then steps using the mean scaled by the root of the second moment. Combines momentum with per-parameter adaptive step sizes. [→ 27](lessons/module-09/lesson-02.md)

**AdamW** — Adam with *decoupled* weight decay: the decay term $\lambda\theta$ is subtracted straight from the parameter instead of being folded into the gradient. This is the standard optimizer for training GPT-style models. [→ 27](lessons/module-09/lesson-02.md)

**Attention** — the mechanism that lets each token mix in information from other tokens, computed as $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_h}+M)\mathbf{V}$. It is the heart of the Transformer. [→ 32](lessons/module-11/lesson-01.md)

**Autograd** — PyTorch's automatic differentiation engine. It records the operations of the forward pass as a graph and, on `.backward()`, applies the chain rule in reverse to compute every gradient. [→ 25](lessons/module-08/lesson-03.md)

**Axis** — a synonym for a tensor dimension, especially when naming which one an operation runs over (e.g. `dim=-1` is the last axis). For $(B,T,C)$ the axes are batch, time, and channel. [→ 13](lessons/module-04/lesson-02.md)

## B

**Backpropagation** — the algorithm that computes gradients of the loss with respect to every parameter by applying the chain rule backward through the computational graph. It is reverse-mode automatic differentiation applied to a neural network. [→ 25](lessons/module-08/lesson-03.md)

**Backward pass** — the second phase of a training step: starting from the scalar loss, propagate derivatives back through the graph to fill each parameter's gradient. Triggered by `loss.backward()`. [→ 25](lessons/module-08/lesson-03.md)

**Batch / mini-batch** — a group of training examples processed together in one forward/backward step; the batch dimension $B$ is usually the first axis. A mini-batch is a small random subset of the full dataset, which is what makes SGD "stochastic". [→ 37](lessons/module-12/lesson-03.md)

**Bayes' theorem** — the rule for updating a probability given evidence: $P(A\mid B)=P(B\mid A)P(A)/P(B)$. It formalizes how a prior belief becomes a posterior after seeing data. [→ 29](lessons/module-10/lesson-01.md)

**Bias correction** — Adam's fix for the fact that its moment estimates start at zero and are biased low early on; dividing by $1-\beta^t$ rescales them, with the correction fading as $t$ grows. [→ 27](lessons/module-09/lesson-02.md)

**Broadcasting** — the rule that lets tensors of different shapes combine by virtually stretching size-1 dimensions, so a $(C,)$ bias can add to a whole $(B,T,C)$ batch without copying. Shapes are matched right-to-left. [→ 13](lessons/module-04/lesson-02.md)

## C

**Catastrophic forgetting** — when fine-tuning on new data overwrites the knowledge a model learned during pretraining. Small learning rates, freezing, and LoRA all help avoid it. [→ 38](lessons/module-12/lesson-04.md)

**Causal mask** — the lower-triangular mask that sets attention scores for future positions to $-\infty$ so a token can only attend to itself and earlier tokens. It is what makes a language model predict left-to-right. [→ 32](lessons/module-11/lesson-01.md)

**Chain rule** — the calculus rule for differentiating a composition: $\frac{d}{dx}f(g(x))=f'(g(x))\,g'(x)$. Multiplying local derivatives along a chain is exactly what backpropagation does. [→ 16](lessons/module-06/lesson-02.md)

**Composition** — building a function by feeding one function's output into another, $y=f(g(x))$. A neural network is a deep composition of layers. [→ 14](lessons/module-05/lesson-01.md)

**Computational graph** — a directed graph whose nodes are operations and whose edges carry tensors; it records how the output was computed so gradients can flow backward. PyTorch builds one automatically during the forward pass. [→ 14](lessons/module-05/lesson-01.md)

**Continuity** — a function has no jumps or breaks; you can trace it without lifting the pen. Continuity (and smoothness) is what lets derivatives exist. [→ 15](lessons/module-06/lesson-01.md)

**Cosine similarity** — the cosine of the angle between two vectors, $\frac{\mathbf{x}\cdot\mathbf{y}}{\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert}$, ranging from $-1$ to $1$; it measures direction agreement independent of length. [→ 08](lessons/module-02/lesson-03.md)

**Cross-entropy** — the loss $-\sum_i y_i\log p_i$ that measures how surprised the true labels are by the predicted probabilities. It is the standard loss for classification and next-token prediction. [→ 24](lessons/module-08/lesson-02.md)

## D

**Derivative** — the instantaneous rate of change of a function, $f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}$; read it as "how sensitive is the output to a tiny nudge in the input". [→ 15](lessons/module-06/lesson-01.md)

**Dimension** — either the number of axes a tensor has (its rank as an array) or the size along one axis, depending on context. For $(B,T,C)$ there are three dimensions with sizes $B$, $T$, $C$. [→ 12](lessons/module-04/lesson-01.md)

**Dot product** — combine two equal-length vectors into one number, $\mathbf{x}\cdot\mathbf{y}=\sum_i x_i y_i$; it measures how much they point the same way and is the building block of every linear layer. [→ 07](lessons/module-02/lesson-02.md)

## E

**Embedding** — a learned vector representation of a discrete item such as a token; the embedding matrix has one row per vocabulary entry, and a lookup selects rows. [→ 31](lessons/module-10/lesson-03.md)

**Entropy** — the average surprise of a probability distribution, $H(p)=-\sum_i p_i\log p_i$; it is highest when outcomes are equally likely. [→ 29](lessons/module-10/lesson-01.md)

**Epoch** — one full pass over the entire training dataset. Training usually takes many epochs, each made of many mini-batch steps. [→ 37](lessons/module-12/lesson-03.md)

**Expectation** — the probability-weighted average of a random variable, $\mathbb{E}[X]=\sum_x x\,P(x)$; the value you would get on average over many draws. [→ 29](lessons/module-10/lesson-01.md)

**Exploding gradients** — gradients that grow uncontrollably as they propagate back through many layers, causing unstable updates or NaNs; gradient clipping is the usual remedy. [→ 35](lessons/module-12/lesson-01.md)

## F

**Fine-tuning** — continuing to train a pretrained model on a narrower dataset so it specializes, starting from the pretrained weights rather than random ones. [→ 38](lessons/module-12/lesson-04.md)

**First moment** — in Adam, the running exponential average of the gradient itself (its mean), stored as $m$; it acts like momentum. [→ 27](lessons/module-09/lesson-02.md)

**Floating point** — the finite binary approximation computers use for real numbers (e.g. `float32`, `bfloat16`); it has limited precision and range, which is the source of rounding, overflow, and underflow. [→ 35](lessons/module-12/lesson-01.md)

**Forward pass** — running inputs through the network to produce outputs (and, in training, the loss), building the computational graph along the way. [→ 25](lessons/module-08/lesson-03.md)

**Freezing** — marking parameters as non-trainable (`requires_grad = False`) so they receive no gradient and stay fixed during fine-tuning. [→ 38](lessons/module-12/lesson-04.md)

**Function** — a rule $f$ that maps each input $x$ to exactly one output $f(x)$; in ML, a whole network is one big function from tokens to probabilities. [→ 02](lessons/module-01/lesson-02.md)

## G

**GELU** — the Gaussian Error Linear Unit, a smooth activation used in GPT-2's MLP; it behaves like a soft version of ReLU. [→ 33](lessons/module-11/lesson-02.md)

**GPT-2** — the decoder-only Transformer language model this course builds toward; a stack of Transformer blocks that predicts the next token. [→ 37](lessons/module-12/lesson-03.md)

**grad_fn** — the attribute on a non-leaf tensor pointing to the operation that created it, forming the backward graph (e.g. `<AddBackward0>`). It is how autograd knows the reverse step for each node. [→ 25](lessons/module-08/lesson-03.md)

**Gradient** — the vector (or tensor) of all partial derivatives of a scalar with respect to its inputs, $\nabla_{\mathbf{x}}f$; it points in the direction of steepest increase and has the same shape as $\mathbf{x}$. [→ 18](lessons/module-06/lesson-04.md)

**Gradient clipping** — capping the norm of the gradient before the optimizer step so a single huge gradient cannot destabilize training. [→ 35](lessons/module-12/lesson-01.md)

**Gradient descent** — the optimization rule $\theta\leftarrow\theta-\eta\nabla L$ that repeatedly nudges parameters downhill on the loss. [→ 26](lessons/module-09/lesson-01.md)

## I

**Identity matrix** — the square matrix $\mathbf{I}$ with $1$s on the diagonal and $0$s elsewhere; multiplying by it changes nothing, $\mathbf{A}\mathbf{I}=\mathbf{A}$. [→ 11](lessons/module-03/lesson-03.md)

**Inverse** — the matrix $\mathbf{A}^{-1}$ that undoes $\mathbf{A}$, satisfying $\mathbf{A}\mathbf{A}^{-1}=\mathbf{I}$; it exists only when the matrix is square and full-rank. [→ 11](lessons/module-03/lesson-03.md)

## J

**Jacobian** — the matrix of all partial derivatives of a vector-valued function; entry $(i,j)$ is $\partial y_i/\partial x_j$. The chain rule for vectors is a product of Jacobians. [→ 21](lessons/module-07/lesson-03.md)

**JVP (Jacobian-vector product)** — multiplying a Jacobian by a vector on the right, $\mathbf{J}\mathbf{v}$; this is the operation forward-mode automatic differentiation computes. [→ 22](lessons/module-07/lesson-04.md)

## K

**Key** — in attention, the vector $\mathbf{k}$ each token exposes to be matched against queries; the query-key dot product produces the attention score. [→ 32](lessons/module-11/lesson-01.md)

**KL divergence** — a measure of how one distribution $q$ differs from another $p$, $D_{\mathrm{KL}}(p\,\|\,q)=\sum_i p_i\log\frac{p_i}{q_i}$; always non-negative and zero only when the distributions match. [→ 29](lessons/module-10/lesson-01.md)

## L

**LayerNorm** — normalization that recenters and rescales each token's feature vector to zero mean and unit variance, then applies a learnable gain and shift. It stabilizes Transformer training. [→ 34](lessons/module-11/lesson-03.md)

**Leaf tensor** — a tensor at the edge of the graph that you created directly (a parameter or input) rather than as the result of an operation; only leaves accumulate `.grad` by default. [→ 25](lessons/module-08/lesson-03.md)

**Learning rate** — the step-size scalar $\eta$ that multiplies the gradient in an update; too large overshoots, too small crawls. [→ 26](lessons/module-09/lesson-01.md)

**Limit** — the value a function approaches as its input approaches some point, written $\lim_{h\to0}$; the derivative is defined as a limit. [→ 15](lessons/module-06/lesson-01.md)

**Log-softmax** — the logarithm of the softmax, computed directly as $z_i-\log\sum_j e^{z_j}$ for numerical stability; the log-probabilities that feed the NLL loss. [→ 30](lessons/module-10/lesson-02.md)

**Log-sum-exp** — the numerically stable way to compute $\log\sum_j e^{z_j}$ by first subtracting the max logit, avoiding overflow in the exponentials. It underlies stable softmax and cross-entropy. [→ 35](lessons/module-12/lesson-01.md)

**Logits** — the raw, un-normalized scores a model outputs before softmax; any real numbers, one per class or vocabulary entry. [→ 30](lessons/module-10/lesson-02.md)

**LoRA** — Low-Rank Adaptation: freeze the pretrained weight $\mathbf{W}$ and learn a low-rank update $\mathbf{B}\mathbf{A}$ so that $\mathbf{W}'=\mathbf{W}+\mathbf{B}\mathbf{A}$, training far fewer parameters. [→ 38](lessons/module-12/lesson-04.md)

**Loss function** — the scalar $L$ that measures how wrong the model's output is; training minimizes it. Examples: MSE, cross-entropy. [→ 24](lessons/module-08/lesson-02.md)

**Low-rank** — describing a matrix whose rank is much smaller than its dimensions, so it can be written as a product of two thin matrices; the key idea behind LoRA. [→ 38](lessons/module-12/lesson-04.md)

## M

**Matrix** — a rectangular grid of numbers with rows and columns, shape $(m,n)$; neural-network weights are matrices. [→ 09](lessons/module-03/lesson-01.md)

**Matrix multiplication** — combining an $(m,n)$ and an $(n,p)$ matrix into an $(m,p)$ one where each entry is a row-column dot product; the core operation of every linear layer. [→ 10](lessons/module-03/lesson-02.md)

**Maximum likelihood** — choosing parameters that make the observed data most probable; minimizing negative log-likelihood is the same thing, and is what language-model training does. [→ 29](lessons/module-10/lesson-01.md)

**Momentum** — an optimizer trick that accumulates a running velocity of past gradients, $v\leftarrow\mu v+g$, to smooth the path and speed up along consistent directions. [→ 27](lessons/module-09/lesson-02.md)

**MSE** — mean squared error, $\frac1n\sum_i(\hat{y}_i-y_i)^2$; the standard regression loss, penalizing large errors quadratically. [→ 24](lessons/module-08/lesson-02.md)

**Multi-head attention** — running several attention operations in parallel on slices of the feature vector (splitting $C=n_h\,d_h$) and concatenating the results, letting the model attend to different patterns at once. [→ 32](lessons/module-11/lesson-01.md)

**Muon** — an optimizer that orthogonalizes the momentum matrix before stepping, effectively setting all its singular values to one via a Newton–Schulz iteration, so every direction of a weight matrix updates on an equal footing. [→ 28](lessons/module-09/lesson-03.md)

## N

**NaN** — "not a number", the floating-point result of undefined operations like $0/0$ or $\infty-\infty$; a NaN anywhere usually poisons the whole computation and signals a numerical bug. [→ 35](lessons/module-12/lesson-01.md)

**Newton–Schulz iteration** — an iterative matrix procedure that drives a matrix's singular values toward one using only matrix multiplies (no explicit SVD); Muon uses it to orthogonalize gradients. [→ 28](lessons/module-09/lesson-03.md)

**NLL (negative log-likelihood)** — the loss $-\log p_{\text{correct}}$, the negative log-probability the model assigned to the true answer; equals cross-entropy with a one-hot target. [→ 24](lessons/module-08/lesson-02.md)

**Norm** — the length of a vector, most commonly the Euclidean norm $\lVert\mathbf{x}\rVert=\sqrt{\sum_i x_i^2}$. [→ 08](lessons/module-02/lesson-03.md)

**Numerical stability** — writing computations so that finite floating-point precision does not cause overflow, underflow, or catastrophic rounding; the reason for the log-sum-exp trick and epsilon terms. [→ 35](lessons/module-12/lesson-01.md)

## O

**One-hot** — a vector that is $1$ at a single index and $0$ everywhere else, used to represent a discrete label or token; an embedding lookup is equivalent to multiplying a one-hot by the embedding matrix. [→ 31](lessons/module-10/lesson-03.md)

**Optimizer state** — the extra tensors an optimizer stores per parameter beyond the weights themselves, such as Adam's first and second moments; it is why Adam uses far more memory than plain SGD. [→ 27](lessons/module-09/lesson-02.md)

**Orthogonalization** — turning a matrix into one whose columns are perpendicular unit vectors (an orthogonal matrix); Muon orthogonalizes the gradient so it rescales all directions equally. [→ 28](lessons/module-09/lesson-03.md)

**Overflow / underflow** — overflow is a number too large to represent (becomes $\infty$); underflow is a number too small (rounds to $0$). Both occur with raw exponentials in softmax. [→ 35](lessons/module-12/lesson-01.md)

## P

**Parameter** — a number the model learns during training, such as a weight or bias; gradients are computed with respect to parameters and optimizers update them. [→ 23](lessons/module-08/lesson-01.md)

**Partial derivative** — the derivative of a multi-variable function with respect to one variable while holding the others fixed, written $\partial f/\partial x_i$. The gradient collects all of them. [→ 18](lessons/module-06/lesson-04.md)

**Positional embedding** — a learned or fixed vector added to token embeddings to encode each token's position, since attention itself is order-agnostic. [→ 31](lessons/module-10/lesson-03.md)

**Probability distribution** — an assignment of non-negative probabilities that sum to one over a set of outcomes; a language model outputs one over the vocabulary at each step. [→ 29](lessons/module-10/lesson-01.md)

**Projection** — the component of one vector that lies along another, $\frac{\mathbf{x}\cdot\mathbf{y}}{\lVert\mathbf{y}\rVert^2}\mathbf{y}$; the "shadow" of $\mathbf{x}$ onto $\mathbf{y}$. [→ 08](lessons/module-02/lesson-03.md)

## Q

**Query** — in attention, the vector $\mathbf{q}$ a token uses to search for relevant information; its dot product with each key gives the attention scores. [→ 32](lessons/module-11/lesson-01.md)

## R

**Random variable** — a quantity whose value is determined by a random outcome, described by a probability distribution; the next token is a random variable over the vocabulary. [→ 29](lessons/module-10/lesson-01.md)

**Rank (of a matrix)** — the number of linearly independent rows or columns, i.e. the true dimensionality of the transformation it represents. A low rank means the matrix carries less independent information. [→ 11](lessons/module-03/lesson-03.md)

**requires_grad** — the flag that tells PyTorch to track a tensor's operations and compute its gradient; parameters have it set to `True`, frozen weights to `False`. [→ 25](lessons/module-08/lesson-03.md)

**Residual connection** — adding a block's input back to its output, $y=x+f(x)$, so gradients get a $+1$ highway straight through; this is what lets very deep Transformers train. [→ 33](lessons/module-11/lesson-02.md)

**RMSNorm** — a lighter normalization than LayerNorm that skips mean-subtraction and divides by the root-mean-square of the features, then applies a learnable gain. [→ 34](lessons/module-11/lesson-03.md)

**RMSProp** — an optimizer that keeps a running average of squared gradients and divides the step by its square root, giving each parameter an adaptive learning rate. [→ 27](lessons/module-09/lesson-02.md)

## S

**Scalar** — a single number, a rank-0 tensor; contrast with vectors, matrices, and higher tensors. [→ 12](lessons/module-04/lesson-01.md)

**Scaled dot-product** — attention scores computed as $\mathbf{Q}\mathbf{K}^\top/\sqrt{d_h}$, where dividing by $\sqrt{d_h}$ keeps the values in a range softmax handles well. [→ 32](lessons/module-11/lesson-01.md)

**Second moment** — in Adam, the running exponential average of the squared gradient, stored as $v$; its square root scales the step per parameter. [→ 27](lessons/module-09/lesson-02.md)

**SGD (stochastic gradient descent)** — gradient descent using a random mini-batch each step instead of the whole dataset, trading exactness for speed and useful noise. [→ 27](lessons/module-09/lesson-02.md)

**Shape** — the tuple of sizes along each axis of a tensor, e.g. $(B,T,C)$; tracking shapes is most of the work of reading Transformer code. [→ 13](lessons/module-04/lesson-02.md)

**Sigmoid** — the squashing function $\sigma(x)=\frac{1}{1+e^{-x}}$ mapping any real number into $(0,1)$; its derivative is the tidy $\sigma(x)(1-\sigma(x))$. [→ 17](lessons/module-06/lesson-03.md)

**Singular value** — one of the non-negative scale factors in a matrix's SVD, measuring how much the matrix stretches along each of its principal directions. Muon equalizes them. [→ 28](lessons/module-09/lesson-03.md)

**Softmax** — the function $p_i=e^{z_i}/\sum_j e^{z_j}$ that turns a vector of logits into a probability distribution. [→ 30](lessons/module-10/lesson-02.md)

**Standard deviation** — the square root of the variance, $\sqrt{\operatorname{Var}(X)}$; a measure of spread in the same units as the data. [→ 29](lessons/module-10/lesson-01.md)

**SVD (singular value decomposition)** — factoring any matrix as $\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$, where $\mathbf{U},\mathbf{V}$ are orthogonal and $\mathbf{\Sigma}$ holds the singular values; the geometric lens behind Muon and low-rank ideas. [→ 28](lessons/module-09/lesson-03.md)

## T

**Tensor** — a multi-dimensional array of numbers of any rank (scalar, vector, matrix, or higher); the fundamental data type in PyTorch. [→ 12](lessons/module-04/lesson-01.md)

**Token** — a discrete unit of text (a word or word-piece) identified by an integer ID; the input and output of a language model are sequences of tokens. [→ 31](lessons/module-10/lesson-03.md)

**Transformer block** — one layer of a GPT-style model: LayerNorm, attention, residual add, LayerNorm, MLP, residual add. Stacking these blocks builds the whole model. [→ 33](lessons/module-11/lesson-02.md)

**Transpose** — flipping a matrix over its diagonal so rows become columns, $(\mathbf{A}^\top)_{ij}=A_{ji}$; turns an $(m,n)$ matrix into $(n,m)$. It appears whenever a gradient is pulled back through a weight. [→ 11](lessons/module-03/lesson-03.md)

## U

**Underflow** — a floating-point value too small in magnitude to represent, which rounds to zero; taking its log then gives $-\infty$, a common source of NaN. [→ 35](lessons/module-12/lesson-01.md)

**Unit vector** — a vector of length one, obtained by dividing a vector by its norm, $\hat{\mathbf{x}}=\mathbf{x}/\lVert\mathbf{x}\rVert$; it keeps the direction and discards the magnitude. [→ 08](lessons/module-02/lesson-03.md)

## V

**Value** — in attention, the vector $\mathbf{v}$ carrying the actual content a token contributes; the attention weights combine values into the output. [→ 32](lessons/module-11/lesson-01.md)

**Vanishing gradients** — gradients that shrink toward zero as they propagate back through many layers, so early layers barely learn; residual connections and normalization mitigate it. [→ 35](lessons/module-12/lesson-01.md)

**Variance** — the expected squared deviation from the mean, $\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]$; how spread out a distribution is. [→ 29](lessons/module-10/lesson-01.md)

**Vector** — an ordered list of numbers, which can be read as a point or a direction in space; a token is represented as a vector. [→ 06](lessons/module-02/lesson-01.md)

**VJP (vector-Jacobian product)** — multiplying a row vector by a Jacobian, $\mathbf{v}^\top\mathbf{J}$; this is the operation reverse-mode autodiff (backprop) computes, which is why PyTorch never builds full Jacobians. [→ 22](lessons/module-07/lesson-04.md)

## W

**Weight decay** — a regularization that shrinks parameters toward zero each step by adding a term proportional to the parameter; in AdamW it is applied decoupled from the gradient. [→ 27](lessons/module-09/lesson-02.md)

**Weight tying** — sharing the same matrix for the input token embedding and the output (logit) projection, which saves parameters and often improves language models. [→ 31](lessons/module-10/lesson-03.md)

## Next

Keep this open while you read the lessons. Pair it with the [Math cheat sheet](assets/math-cheatsheet.md) for the formulas and the [PyTorch cheat sheet](assets/pytorch-cheatsheet.md) for the code each term maps to.
