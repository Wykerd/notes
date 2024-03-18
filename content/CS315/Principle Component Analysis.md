PCA optimises $A$ by finding component axes that maximises the variation:
- Each new component axis is orthogonal to the previous, since if not, they'd capture overlapping information.
- Projection onto the component axes produces feature vector with features ordered in descending order of variation.

PCA works by finding euclidean planes onto which projections of the input data produces the largest variation. 
![](https://i.stack.imgur.com/lNHqt.gif)

Principle Components are a linear combination of input attributes, i.e. PCA is a linear projection of the data points.

## Data

Given $N$ observations $\mathbf{x}_n, n = 1, ..., N$ of dimension $d$, we collect the original data into a $d \times N$ matrix $X$ 

The centered  (mean subtracted) matrix is called $D$, which is done my subtracting the sample mean of each attribute from each observation:

$$ 
\begin{gathered}
\mathbf{d}_i = \mathbf{x}_i - \bar{\mathbf{x}} \\
D = \left\{\mathbf{d}_1, ..., \mathbf{d}_N\right\}
\end{gathered}
$$

With $\mathbf{\bar{x}}$ being the sample mean:

$$
\mathbf{\bar{x}} = \frac{1}{N} \sum_{n = 1}^{N}{\mathbf{x}_n} 
$$
## Variance

Since PCA maximises variance, lets recap what **variance** means:

$$
\text{Var}\left[\left\{x_1, ..., x_N\right\}\right] = E\left[\left(x - \bar{x}\right)^2\right] = \frac{1}{N} \sum_{n = 1}^N{\left(x_n - \bar{x}\right)^2}
$$
Since we're working with vector observations, we make use of **covariance** instead:
$$
\text{Cov}\left[\left\{\mathbf{x}_1, ..., \mathbf{x}_N\right\}\right] = \frac{1}{N} \sum_{n = 1}^N{\left(\mathbf{x}_n - \bar{\mathbf{x}}\right)\left(\mathbf{x}_n - \bar{\mathbf{x}}\right)^T} = \frac{1}{N} D D^T
$$
- diagonals: variance
- off-diagonals
	- symmetric
	- co-variance between two pairs of inputs

We call the covariance matrix of the data $S = \frac{1}{N} D D^T$, and it is a $d \times d$ matrix

## Finding the first principle component (PC1)

Consider a potential feature vector $\mathbf{u}_1$ , we know score of $\mathbf{x}$ for this feature to be $\mathbf{u}_1^T \mathbf{x}_i$

This means that the sample mean of the projected data is now $\mathbf{u}_1^T \bar{\mathbf{x}}$

Also, we calculate the new covariance matrix of the projected data: 
(remember the property $(AB)^T = B^T A^T$)
$$
\begin{align}
C_X &= \frac{1}{N} \sum_{n = 1}^N{\left(\mathbf{u}_1^T \mathbf{x}_n - \mathbf{u}_1^T \bar{\mathbf{x}}\right)\left(\mathbf{u}_1^T \mathbf{x}_n - \mathbf{u}_1^T \bar{\mathbf{x}}\right)^T} \\
&= \frac{1}{N} \sum_{n = 1}^N{\mathbf{u}_1^T\left(\mathbf{x}_n - \bar{\mathbf{x}}\right)\left(\mathbf{x}_n -  \bar{\mathbf{x}}\right)^T\mathbf{u}_1} \\
&= \mathbf{u}_1^T S \mathbf{u}_1
\end{align}
$$
Now, since PCA is concerned with maximising variance, we can calculate $\mathbf{u}_1$ as:
$$
\max_{\mathbf{u}_1}{\mathbf{u}_1^T S \mathbf{u}_1} \text{ subject to } \mathbf{u}^T \mathbf{u} = 1 \text{ (u is a unit vector)}
$$
This problem lends itself to be solved through [[Lagrange Multipliers]]:
$$
\begin{gathered}
f(\mathbf{u}_1) = \mathbf{u}_1^T S \mathbf{u}_1 \\
g(\mathbf{u}_1) = 0 = 1 - \mathbf{u}_1^T \mathbf{u}_1
\end{gathered}
$$
Setting up the **Lagrangian** and solving yields:
(See [[Matrix Calculus]] to help with the differentiation, note $S$, the covariance matrix, is symmetrical)
$$
\begin{gathered}
\mathcal{L}(\mathbf{u}_1, \lambda_1) = \mathbf{u}_1^T S \mathbf{u}_1 + \lambda \cdot (1 - \mathbf{u}_1^T \mathbf{u}_1) \\
\nabla_{u}\mathcal{L} = 2S\mathbf{u}_1 - 2\lambda\mathbf{u}_1 = 0 \\
2S\mathbf{u}_1 = 2\lambda \mathbf{u}_1 \\
S\mathbf{u}_1 = \lambda \mathbf{u}_1
\end{gathered}
$$
The solution is clearly an [[Eigenvalue]] problem, with $\mathbf{u}_1$ being an eigenvector of $S$, meaning there are $rank(S)$ possible solutions to $\mathbf{u}_1$, finding the optimal by plugging it into the original function $f(\mathbf{u}_1)$. However:
$$
\mathbf{u}_1^T S \mathbf{u}_1 = \lambda
$$
Shows that you must pick the eigenvector corresponding to the largest eigenvalue to maximise the variance. 

**The eigenvector corresponding to the largest eigenvalue is our first principle component**

## Finding the rest of the principle components

Since PCA requires principle components to be orthogonal, we simply add an additional constraint and **Lagrange Multiplier** to optimise the covariance.
$$
\max_{\mathbf{u}_2}{\mathbf{u}_2^T S \mathbf{u}_2} \text{ subject to } \mathbf{u}_2^T \mathbf{u}_2 = 1 \text{ and also } \mathbf{u}_2^T \mathbf{u}_1 = 0 \text{ (orthogonal)}
$$
This creates the **Lagrangian**:
$$
\begin{gathered}
\mathcal{L}(\mathbf{u_2}, \lambda_1, \lambda_2) = \mathbf{u}_2^T S \mathbf{u}_2 + \lambda_1(1 - \mathbf{u}_2^T \mathbf{u}_2) + \lambda_2 \mathbf{u}_2^T \mathbf{u}_1 \\
\nabla_{\mathbf{u_2}}\mathcal{L} = 0 = 2S\mathbf{u_2} -2\lambda_1\mathbf{u_2} + \lambda_2 \mathbf{u}_1 \\
\frac{\delta \mathcal{L}}{\delta \lambda_2} = 0 = 1 - \mathbf{u_2}^T \mathbf{u_2} \\
\frac{\delta \mathcal{L}}{\delta \lambda_2} = 0 = \mathbf{u_2}^T \mathbf{u}_1 \text{ (orthoganal to eachother)}
\end{gathered}
$$
We can prove that $\lambda_2$ is zero by multiplying $\nabla_{\mathbf{u_2}}\mathcal{L}$ with $\mathbf{u}_1^T$
$$
\begin{gathered}
\mathbf{u}_1^T \nabla_{\mathbf{u_2}}\mathcal{L} = 0 = 2\mathbf{u}_1^T S\mathbf{u_2} -2 \mathbf{u}_1^T \lambda_1\mathbf{u_2} + \lambda_2 \mathbf{u}_1^T \mathbf{u}_1 \\
2\mathbf{u}_1^T S\mathbf{u_2} = 0 \text{ (since } \mathbf{u}_1 \text{ and }\mathbf{u}_2 \text{ are orthogonal)} \\
\mathbf{u}_1^T \lambda_1\mathbf{u_2} = 0  \text{ (since } \mathbf{u}_1 \text{ and }\mathbf{u}_2 \text{ are orthogonal)} \\
\lambda_2 \mathbf{u}_1^T \mathbf{u}_1 = 1 \text{ (since a unit vector)} \\
\therefore \lambda_2 = 0
\end{gathered}
$$
With this:
$$
\begin{gathered}
0 = 2S\mathbf{u_2} -2\lambda_1\mathbf{u_2} \\
S\mathbf{u_2} = \lambda_1\mathbf{u_2}
\end{gathered}
$$
The solutions are once again the eigenvectors of $S$, however, since they must be orthogonal, it is **next biggest eigenvalue's eigenvector**

**Therefore, the nth principle component is eigenvector corresponding to the nth biggest eigenvalue**

## Finding Principle Components In Practice

Now that we know what makes up the principle component matrix $A$, here-forth referred to as $Q$ (the eigenvector matrix) or $U$ (as the matrix containing the principle component vectors $U = \left\{\mathbf{u}_1, ..., \mathbf{u}_v\right\}$ )

We want a way to find the eigenvectors and eigenvalues of the covariance matrix $S$.

There's two approaches:
- [[PCA with Eigendecomposition]]
- [[PCA with Singular Value Decomposition (SVD)]]

## Projecting the data

Now consider the PCA transformation:
$$
\mathbf{y}_n = Q^T (\mathbf{x}_n - \bar{\mathbf{x}})
$$
Or in matrix form
$$
Y = Q^T (X - \bar{X}) = Q^T D
$$
Since $Q$ is orthogonal, we've simply shifted the origin of the coordinate axes to the mean of the data and rotated and/or reflected the coordinate axes around the mean to coincide with the principle components. **I.e. we have only changed the basis and reduction has taken place**

The sample covariance $S_y = \Lambda$, therefore the transformed covariance matrix is diagonal with the eigenvalues of $S$ along its diagonal. **This also means the transformed data's components are uncorrelated, since off diagonals represent correlation (covariance between two features)**

Forming a new $d \times v$ matrix $Q_v$ ($v$ indicating the new number of columns), we are performing dimensionality reduction!
$$
S = Q \Lambda Q^T \approx Q_v \Lambda Q_v^T 
$$
With $\Lambda_v$ ($v \times v$ matrix) containing the $v$ largest eigenvalues and $Q$ the corresponding eigenvectors.

We can also consider transformed data $Y$ in terms of the SVD
$$
Y = U_v^T (U_v \Sigma_v V_v^T) = \Sigma_v V_v^T
$$

### Recover data from projection

With dimensionality reduction, a perfect reconstruction is impossible. However, we can get an approximated reconstruction, and is exact if there was no loss during projection.
$$
\hat{\mathbf{x}}_n = \bar{\mathbf{x}} + Q_v \mathbf{y}_n
$$
Or in matrix form:
$$
\hat{X} = \bar{X} + Q_v Y
$$
Or In terms of SVD:
$$
\hat{X} = \bar{X} + U_v \Sigma_v V_v^T = \bar{X} + D_v
$$
## Data Loss

Since the reconstructed data is of form
$$
\hat{X} = \bar{X} + D_v
$$
The data loss is therefore
$$
D - D_v
$$

The amount of variance retained is equal to
$$
\frac{\sum_{i = 1}^{v}{\lambda_i}}{\sum_{i = i}^{r}{\lambda_r}}
$$
With $r$ being the rank of the matrix usually $r = d$

