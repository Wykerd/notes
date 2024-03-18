See first [[Singular Value Decomposition]] for background on how this all works.

We apply SVD to the **centered data** $D$:
$$
D = U \Sigma_d V_d^T \text{ (thin SVD)}
$$

Solving for S (the covariance matrix) as derived in [[Principle Component Analysis|PCA]]:
$$
\begin{aligned}
S &= \frac{1}{N}D D^T \\
&= \frac{1}{N} U \Sigma_d V_d^T (U \Sigma_d V_d^T)^T \\
&= \frac{1}{N} U \Sigma_d V_d^T V_d \Sigma_d^T U^T \\
&= \frac{1}{N} U \Sigma_d^2 U^T \\
&= U \frac{\Sigma_d^2}{N} U^T
\end{aligned}
$$

Therefore, $Q = U$ and $\Lambda = \frac{\Sigma_d^2}{N}$, so we can use SVD to calculate variance for each PC

**Using SVD over eigendcomposition results in numerical stability as covariance matrix computation is avoided**

The cool thing about the compact SVD is that we reduce dimensionality to $r$ dimensions **without data loss**

Nothing prevents us from removing even more of the singular values such that $v \leq r$ to reduce the data to $v$ dimensions:
$$
D \approx D_v = U_v \Sigma_v V_v^T
$$
- $U_v$ is $d \times v$
- $\Sigma_v$ is $v \times v$
- $V_v$ is $v \times N$

**Note that $D_v$ is now an _approximation_ of $D$

Another cool trick we can do is left multiply the data matrix $D$ by $U^T$ 
$$
U^T D = I \Sigma_d V_d^T
$$
This causes the new $D' = U^T D$ data matrix to be a rotation (and possible reflection) of the data matrix $D$. Note the right hand side of the SVD is now the SVD of $D'$ and its $U$ vector is a identity matrix, meaning the principle components of $D'$ are the original component axes (no more rotation of the axes and the data is now on the original component axes)

Since variation is $\frac{\Sigma_d^2}{N}$ we can also see that standard deviation is $\frac{\Sigma_d}{\sqrt{N}}$

Assuming $\Sigma_d$ is all non-zero singular values, we can multiply by the inverse to get
$$
\Sigma_d^{-1} U^T D = I I V_d^T
$$
For this data ($\Sigma_d^{-1} U^T D$), both $U$ and $\Sigma$ in its SVD are identity matrices. This means:
- The principle directions are aligned with the coordinate axes
- All the singular values are 1, this means the data is reshaped to be spherical. To make the standard deviations unity, we scale by an additional factor $\sqrt{N}$
**The factor $V^T$ is the whitened data (up to a scalar multiplicative factor)

