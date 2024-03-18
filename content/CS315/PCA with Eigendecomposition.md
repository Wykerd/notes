We have the solution for a single principle component of [[Principle Component Analysis|PCA]]: $S\mathbf{u} = \lambda \mathbf{u}$
If we extend this to the matrix of optimized components $Q$, we find: $SQ = Q \Lambda$

Here, $\Lambda$ is a diagonal matrix of eigenvalues and $Q$ contains linearly independant eigenvectors for the corresponding eigenvalues in $\Lambda$ (since it is orthogonal matrix as required by PCA). 

Therefore, $S$ is diagonalisable (as it meets the conditions thereof) and we can extract the eigenvectors from the eigendecomposition: $S = Q \Lambda Q^T$

Columns of the $d \times d$-matrix  $Q$ are the principle components, and has to be arranged such that the diagonal of $\Lambda$ is decreasing (eigenvectors corresponding to biggest eigenvalues first), such that $Q = A = U$

To do this by hand, we must calculate the eigenvalues and eigenvectors by hand first.

With numpy, we may use:
```python
eigenvalues, eigenvectors = np.linalg.eigh(S)
```
With `eigh` being a special variant of `eig` function for symmetrical matrices (as $S$ is)

