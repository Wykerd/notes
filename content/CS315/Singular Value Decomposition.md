We want to find a orthogonal set of vectors ($V$), which when transformed by our matrix ($X$), remain orthogonal ($U$): $XV = U \Sigma$ (Transforming orthogonal matrix $V$ with $X$ produces orthogonal matrix $U$ scaled by $\Sigma$ (which is diagonal - only scales))

We can consider a 2D case of a sphere (circle) which is transformed by matrix $X$ (which will **stretch** and **rotate** the vector):

![[images/Pasted image 20240303222759.png]]
The n-dimensional sphere has orthogonal vectors ($v_1, ..., v_d$) which is transformed to orthonormal unit vectors $u_1, ..., u_r$ and stretch amounts $\sigma_1, ..., \sigma_r$ (with $r$ being the matrix rank, typically $r = d$)

- $u_i$ are the principle axes
- $\sigma_i$ are called the singular values

This leads to the following mathematical expression:
$$
X\mathbf{v}_i = \sigma_i \mathbf{u}_i
$$
This is similar to the eigenvalue problem $A \mathbf{v} = \lambda \mathbf{v}$, however, it does not require the same vector $v$ on both sides.

We can write this in matrix form:
$$
XV = U \Sigma
$$

See [the video I got this from](https://www.youtube.com/watch?v=EokL7E6o1AE)

**Terminology: Orthogonal matrix is a matrix $A$ such that $A^T A = A A^T = I$**

In other words, any real $d \times N$ matrix can be decomposed as:
- $X = U \Sigma V^T$
	- $U$ is a $d \times d$ orthonormal matrix (this $U$ is not the same as the optimised $A$ transformation matrix used in PCA)
	- V is a $N \times N$ orthogonal matrix
	- $\Sigma$ is a $d \times N$ diagonal matrix with non-negative entries in non-decreasing order down the diagonal ($\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_r \geq 0$)

**This is called the SVD of $X$**

The non-zero entries of $\Sigma$ are called the singular values of $X$, also the number of non-zero entries in $\Sigma$ is $rank(X)$

### Solving these matrixes
$$
\begin{aligned}
X^T X &= (U \Sigma V^T)^T (U \Sigma V^T) \\
&= V \Sigma^T U^T U \Sigma V^T \\
&= V \Sigma^T \Sigma V^T \\
&= V \Sigma^2 V^T \\
X^T X V &= V \Sigma^2
\end{aligned}
$$
$X^T X V = V \Sigma^2$ is an eigenvalue problem ($A \mathbf{x} = \lambda \mathbf{x}$), with $\mathbf{x} = V$, $A = X^T X$, and $\Sigma^2 = \lambda$, thus a eigendecomposition is used to compute $V$

Likewise for $U$:

$$
\begin{aligned}
X^T X &= (U \Sigma V^T) (U \Sigma V^T)^T \\
&= U \Sigma V^T V \Sigma^T U^T  \\
&= U \Sigma \Sigma^T U^T \\
&= U \Sigma^2 U^T \\
X X^T U &= U \Sigma^2
\end{aligned}
$$
Another problem you can solve with eigendecomposition

Columns of $U$ and $V$ (not $V^T$) are called the left and right singular vectors of $X$ and are associated with a singular value in the same column of $\Sigma$

When $N \gg d$ (much more samples than input dimensions), we may trim $\Sigma$ and $V$ of may zero column to get the **thin SVD**
$$
X = U \Sigma_d V^T_d = U \Sigma_+ V^T_+ \text{ (as in the course notes)}
$$
- $\Sigma_d$ is now $d \times d$ (as opposed to $d \times N$) (square diagonal matrix without zeros padding)
- $V_d$ is now $d \times N$ (as opposed to $N \times N$)
Furthermore, if $r = \text{rank}(X) \neq d$, we get the **compact SVD** which trims more zeros from the matrices (since $\Sigma$ only has $r$ non zero rows)
$$
X = U_r \Sigma_r V_r^T
$$
- $U_r$ is now $d \times r$
- $\Sigma_r$ is now $r \times r$
- $V_r$ is now $r \times N$ 

You can also solve these matrices in python

```python
u, s, vt = numpy.linalg.svd(X)
# s is a 1D-array of singular values, 
# and needs to be converted to a diagonal for Sigma
Sigma = np.diag(s)
```
