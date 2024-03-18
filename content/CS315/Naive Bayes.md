An parameter estimation technique for [[Probabilistic Generative Model]]

This approach makes the **naive** assumption that input features (the components of the input vector $\mathbf{x}$) of each class are **conditionally independent**, resulting in a **diagonal covariance matrix**, $\Sigma_j$, for each class, $C_j$.

## Conditional Independence

We know from stats that random variables $X$ and $Y$ are conditionally independent if the following holds:
$$
p(X,Y|Z) = p(X|Z)p(Y|Z)
$$

Therefore, given feature vector:
$$
\mathbf{x} = \begin{bmatrix}
x_1 & \ldots & x_d
\end{bmatrix}^T
$$

With two different features $x_i$ and $x_j$ being independent we get the class conditionals as the product of separate input feature class conditionals:

$$
p(\mathbf{x}|C) = \prod_{n=1}^d{p(x_n|C)}
$$

**This is known as autoregressive decomposition. Don't need to remember that though :-P**

Since we assumed a **Gaussian Distribution** before, we can do so here as well for a **Gaussian Naive Bayes** estimation:

$$
\begin{align*}
p(\mathbf{x}|C) &= \prod_{n=1}^d{\frac{1}{\sqrt{2 \pi \sigma_n^2}}\exp\left(-\frac{1}{2}\frac{(x_n - \mu_n)^2}{\sigma_n^2}\right)} \\
&= \frac{1}{\sqrt{|2 \pi \Sigma|}}\exp\left(-\frac{1}{2}(\mathbf{x} - \mathbf{\mu})^T \Sigma^{-1} (\mathbf{x} - \mathbf{\mu})\right)
\end{align*}
$$

With $\sigma_n^2$ being the variance of the $n$-th feature and likewise $\mu_n$ the mean. We can write it in matrix form with $\Sigma$ being diagonal with the variances on the diagonal.

From [[Bayes Theorem]], we can find

$$
\begin{align*}
P(C_j|\mathbf{x}) &= \frac{P(C_j)p(\mathbf{x}|C_j)}{p(\mathbf{x})} \\
&= \frac{P(C_j)\prod_{n=1}^d{p(x_n|C_j)}}{\sum_{i = 0}^k{\left[P(C_i)\prod_{n=1}^d{p(x_n|C_i)}\right]}}
\end{align*}
$$

With the bottom term being constant the most probable class is the max of the top term:
$$
C^* = \text{arg }\max_{C_j}{P(C_j)\prod_{n=1}^d{p(x_n|C_j)}}
$$
### Naive Bayes Parameter Estimates

From above, we can find the equations for the parameters as follows:

$$
\begin{align*}
P(C_j) &= \frac{N_j}{N} \\
\mu_{nj} &= \frac{1}{N_j} \sum_{\{i:y_i \in C_j\}}{x_{ni}} \\
\sigma_{nj}^2 &= \frac{1}{N_j} \sum_{\{i:y_i \in C_j\}}{(x_{ni} - \mu_{nj})^2}
\end{align*}
$$
The maximum likelihood is thus:

# TODO:
continue from slide 19 and finish this section off.


## How we'd do it in Python

```python
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X, y)

y_pred = clf.predict(X)
```
