Unlike the [[Probabilistic Generative Model|Generative]] method, the PDM does not calculate the **class conditional probabilities** $p(\mathbf{x}|C_j)$ and instead directly computes the posterior $P(C_j|\mathbf{x})$ 

## Posterior of Two Classes with Shared Covariance Matrix

As we derived in the [[Probabilistic Generative Model|PGM]] notes, the posterior is given as follows:

$$
\begin{align*}
P(C_1|\mathbf{x}) &= \sigma(a(\mathbf{x})) \\
&= \frac{1}{1 + e^{-a(\mathbf{x})}} \\
a(\mathbf{x}) &= \ln{\frac{P(\mathbf{x}|C_1)P(C_1)}{P(\mathbf{x}| C_2)P(C_2)}} \\
&= \mathbf{w}^T \mathbf{x} + w_0
\end{align*}
$$
We've also showed these weights for the case the classes follow a **gaussian distribution**:

$$
\begin{align*}
\mathbf{w} &= \Sigma^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2) \\
w_0 &= \frac{1}{2}\left(\mathbf{\mu}_2^T\Sigma^{-1}\mathbf{\mu}_2 - \mathbf{\mu}_1^T\Sigma^{-1}\mathbf{\mu}_1\right) + \ln{\frac{P(C_1)}{P(C_2)}}
\end{align*}
$$
We can combine the $w_0$ bias term into the $\mathbf{w}$ matrix by adding a constant 1 to the input data: $\mathbf{x} = \begin{bmatrix}1 & x_1 & x_2 & \ldots & x_d\end{bmatrix}^T$ and $\mathbf{w} = \begin{bmatrix}w_0 & w_1 &  \ldots & w_d\end{bmatrix}^T$ 

This also simplifies the posterior for a class to:
$$
P(C_1|\mathbf{x}, \mathbf{w}) = \sigma(\mathbf{w}^T \mathbf{x})
$$
## Why PDM?

- PGM - Computes the weights $\mathbf{w}$ indirectly by:
	- Estimating Gaussian class-conditional PDFs
	- Assuming shared or diagonal $\Sigma$
	- Computes the weights from the decision boundary ($a(\mathbf{x}) = 0.5$)
- PDM - Computes the weights $\mathbf{w}$ directly
	- $\mathbf{w}$ maps the inputs $\mathbf{x}$ to the posterior $P(C|\mathbf{x})$
	- Why don't we just directly compute the weights?

## Determining the Weight Parameters 

### Two Class Case

The decision boundary between two classes are where
$$
\mathbf{w}^T \mathbf{x} = 0
$$

Assuming we're given training data $D = [X,\mathbf{y}]$ with observations $X = \{\mathbf{x}_1, \ldots \mathbf{x}_N\}$ and it's respective classes $\mathbf{y} = \{y_1, \ldots, y_N\}$ , $y_n$ can be either 1 or 0 for class $C_1$ or $C_2$ respectively.

We assume conditional independence of sampled data, i.e. for a known $X$ observations, and weights $\mathbf{w}$, knowing one of the output sample output classes $y_m$ does not change the probability of any other output class $y_n$ for $n \ne m$ 

We also assume the data is sampled independently from the same distribution such that individual observations are independent of each other, knowing prior observations does not change the probability of the current observation being in a specific class.

We can then find the joint probability of this training set given the weights:

$$
\begin{align*}
p(D|\mathbf{w}) &= p(X,\mathbf{y}|\mathbf{w}) \\
&= p(\mathbf{y}|X,\mathbf{w})\cdot p(X|\mathbf{w})\text{ (from product rule)} \\
&= p(\mathbf{y}|X,\mathbf{w}) \cdot p(X)\text{ (since X is independent of w)} \\
&= p(X) \prod_{n=1}^N{p(y_n|X,\mathbf{w})}\text{ (assuming conditional independence, use product rule)} \\
&= p(X)\prod_{n=1}^N{p(y_n|\mathbf{x}_n,\mathbf{w})}\text{ (assuming indepentently sampled observations)} 
\end{align*}
$$

Since the outputs $y_n$ follow a **Bernoulli Distribution** we can write
$$
\begin{align*}
p(y_n|\mathbf{x}_n,\mathbf{w}) &= P(C_1|\mathbf{x}_n,\mathbf{w})^{y_n} \cdot (1 - P(C_1|\mathbf{x}_n,\mathbf{w}))^{1 - y_n}
\end{align*}
$$

**What?! Let's explore how this is derived:**
- If $y_n = 1$ then $p(y_n|\mathbf{x}_n,\mathbf{w}) = P(C_1|\mathbf{x}_n,\mathbf{w})$ (since if $y_n = 1$ then the point is in class $C_1$ and therefore these are equivalent)
- Likewise, if $y_n = 0$ then $p(y_n|\mathbf{x}_n,\mathbf{w}) = P(C_2|\mathbf{x}_n,\mathbf{w}) = 1 - P(C_1|\mathbf{x}_n,\mathbf{w})$ 

We just use some clever tricks with powers to combine these two cases into a single equation! Neat.

We can now use our existing formula for the **posteriors**

$$
\begin{align*}
p(D|\mathbf{w}) &= p(X)\prod_{n=1}^N{P(C_1|\mathbf{x}_n,\mathbf{w})^{y_n} \cdot (1 - P(C_1|\mathbf{x}_n,\mathbf{w}))^{1 - y_n}} \\
&= p(X)\prod_{n=1}^N{\sigma(\mathbf{w}^T\mathbf{x}_n)^{y_n} \cdot (1 - \sigma(\mathbf{w}^T\mathbf{x}_n))^{1 - y_n}}
\end{align*}
$$

We want to now **maximise the likelihood** (get the probability as close to 1 as possible). However, since we're working with the product of fractions, and computers don't do floating points well, it is more numerically stable to use the **negative log-likelihood** for maximisation

### Negative Log-Likelihood

$$
E(\mathbf{w}) = -\ln{P(D|\mathbf{w}) = - \sum_{n=1}^N{\left[y_n \ln{\sigma(\mathbf{w}^T\mathbf{x}_n)}+(1-y_n)\ln{(1 - \sigma(\mathbf{w}^T\mathbf{x}_n))}\right]}} - \ln{p(X)}
$$
To maximise this, we consider the gradient

$$
\nabla E(\mathbf{w}) = \frac{\partial E(\mathbf{w})}{\partial \mathbf{w}}
$$

We know

$$
\frac{\partial \sigma(x)}{\partial x} = \sigma(x)(1 - \sigma(x))
$$

We can use this to calculate the partial derivative above using a bunch of chain rule:

$$
\begin{align*}
\frac{\partial E(\mathbf{w})}{\partial \mathbf{w}} &= -\sum_{n = 1}^N{\left[y_n \frac{\sigma(\mathbf{w}^T\mathbf{x}_n)(1 - \sigma(\mathbf{w}^T\mathbf{x}_n))}{\sigma(\mathbf{w}^T\mathbf{x}_n)}-(1-y_n)\frac{\sigma(\mathbf{w}^T\mathbf{x}_n)(1 - \sigma(\mathbf{w}^T\mathbf{x}_n))}{1 - \sigma(\mathbf{w}^T\mathbf{x}_n)}\right]}\frac{\partial \mathbf{w}^T\mathbf{x}_n)}{\partial \mathbf{w}} \\
&= -\sum_{n = 1}^N{\left[y_n - y_n \sigma(\mathbf{w}^T\mathbf{x}_n) - \sigma(\mathbf{w}^T\mathbf{x}_n) + y_n\sigma(\mathbf{w}^T\mathbf{x}_n)\right]}\mathbf{x}_n \\
&= \sum_{n = 1}^N{\left[\sigma(\mathbf{w}^T\mathbf{x}_n)-y_n\right]}\mathbf{x}_n 
\end{align*}
$$

To **maximise** $E(\mathbf{w})$, we'll **minimise** its gradient $\nabla E(\mathbf{w}) = \mathbf{0}$ 

This is a non-linear system of questions with $\mathbf{w}$ needing to be solved numerically. We'll use **Newton-Raphson**

### Problems During Parameter Estimation

When far from the decision boundary, the **posterior** is almost 1 ($P(C_1|\mathbf{x}_n, \mathbf{w}) \approx \sigma(\mathbf{w}^T\mathbf{x}_n) \approx 1$)  so these points have very little influence.

Points close to the decision boundary have a disproportionately large influence.