- Models make assumptions about the data **for simplification**
- These assumptions may not be true for the initial data
- Whitening is a **preprocessing technique** that makes such assumptions correct, or at least **more reasonable**
- A common assumption models make is that data input components shares the same variance
- This is not the case when using PCs ([[Principle Component Analysis|Principle Components]]) as features
- Whitening scales the axes to make the covariance matrix the identity matrix (such that the variance is unit n-dimensional sphere)

The original variance of the PCs are $\lambda_i$ and therefore we must multiply each point by the square root (standard deviation) of the inverse (divide by the standard deviation) $\lambda_i^{-0.5}$ 

Extending this to the matrix form yields:
$$
\Lambda_v^{-\frac{1}{2}} = (\frac{\Sigma_v^2}{N})^{-\frac{1}{2}}
$$
The transformed data is $\Sigma_v V_v^T$ such that the whitened data is now:
$$
(\frac{\Sigma_v^2}{N})^{-\frac{1}{2}} \Sigma_v V_v^T = \sqrt{N} V_v^T
$$
