The generative approach to classification (PGM), works by:
1. Estimate the class-conditional probabilities, $p(\mathbf{x}|C_j)$ - _which can be used to generate new data points, hence the name 'generative model'_
2. Calculate the posterior probability using [[Bayes Theorem]]: $P(C_j|\mathbf{x}) \propto p(\mathbf{x}|C_j) \cdot P(C_j)$ 

## Key Steps

To solve for the posterior probabilities $P(C_j|\mathbf{x})$ via PGM, we follow the key steps:
1. **Expand the posterior** using logistic/softmax functions
2. **Determine form** of arguments to the softmax function (with model assumptions)
3. **Estimate parameters** for the resulting form

## Two Class Case - Expanding the Posterior (Step 1)

Let us consider the key steps for the simple case of two classes. 

$$
P(C_2) = 1 - P(C_1)
$$

We know the equation for **joint probability** to be:

$$
p(x, y) = p(x|y) \cdot p(y)
$$

We also know the sum rule (marginalisation):

$$
p(x) = \sum_{y}{p(x, y)}
$$

And the product rule
$$
p(x,y) = p(y|x) \cdot p(x)
$$

With this, we can expand the posterior: 

**NOTE: the lecturer mentioned this being in the A1 likely!**

$$
\begin{align}
P(C_1|\mathbf{x}) &= \frac{P(C_1, \mathbf{x})}{P(\mathbf{x})}\text{ (from the definition of joint probability)} \\
&= \frac{P(\mathbf{x}|C_1)P(C_1)}{P(\mathbf{x})}\text{ (from the product rule)} \\
&= \frac{P(\mathbf{x}|C_1)P(C_1)}{P(\mathbf{x}, C_1) + P(\mathbf{x}, C_2)}\text{ (from the sum rule)} \\
&= \frac{P(\mathbf{x}|C_1)P(C_1)}{P(\mathbf{x}|C_1)P(C_1) + P(\mathbf{x}| C_2)P(C_2)}\text{ (from the product rule)} \\
&\text{We're now going to coerce this into a logistic form} \\
&= \frac{1}{1 + \frac{P(\mathbf{x}| C_2)P(C_2)}{P(\mathbf{x}|C_1)P(C_1)}}\text{ (divide by numerator)} \\
&= \frac{1}{1 + e^{\ln{\frac{P(\mathbf{x}| C_2)P(C_2)}{P(\mathbf{x}|C_1)P(C_1)}}}}\text{ (turn to exponential)} \\
&= \frac{1}{1 + e^{-\ln{\frac{P(\mathbf{x}|C_1)P(C_1)}{P(\mathbf{x}| C_2)P(C_2)}}}}\text{ (flip the fraction)} \\
&\text{Now, let us define }a\text{ to be the ln function}\\
&= \frac{1}{1 + e^{-a(\mathbf{x})}} \\
&\text{We can now notice that this is the sigmoid function} \\
&= \sigma(a(\mathbf{x}))
\end{align}
$$

Let's have a closer look at the implications of the definition of $a(\mathbf{x})$:

Some background, the odds of an event $E$ is defined as:

$$
\text{odds}(E) = \frac{P(E)}{1 - P(E)}
$$
We can use this to say the **prior odds** (odds without an observation) for class $C_1$ is:

$$
\text{odds}(C_1) = \frac{P(C_1)}{1 - P(C_1)} = \frac{P(C_1)}{P(C_2)}
$$
Similarly, the **posterior odds** (odds given an observation) is:
$$
\begin{align}
\text{odds}(C_1|\mathbf{x}) &= \frac{P(C_1|\mathbf{x})}{P(C_2|\mathbf{x})} \\
&= \frac{P(\mathbf{x}, C_1)P(C_1)}{P(\mathbf{x}, C_2)P(C_2)}\text{ (product rule)}
\end{align}
$$

Look familiar? This means that $a$ is the **log-posterior odds** of $C_1|\mathbf{x}$ 
$$
a(\mathbf{x}) = \ln{\text{odds}(C_1|\mathbf{x})} = \ln{\frac{P(\mathbf{x}|C_1)P(C_1)}{P(\mathbf{x}| C_2)P(C_2)}}
$$

**Therefore, the posterior probability** $P(C_1|\mathbf{x})$ **is the logistic function (sigmoid) evaluated at the log-posterior odds of** $C_1|\mathbf{x}$ 

**What does this mean?!**

![[Pasted image 20240318083823.png|300]]
The **logistic sigmoid function** (as above), takes the **log-posterior odds** $a(\mathbf{x})$ and maps it to a value between 0 and 1, thereby assigning a **posterior probability** to $\mathbf{x}$

$\mathbf{x}$ belongs to class $C_1$ if $\text{odds} \gt 1$, otherwise $C_2$

### Expanding this to the k-class case

$$
\begin{align}
P(C_n|\mathbf{x}) &= \frac{P(\mathbf{x}|C_n)P(C_n)}{\sum_{j = 1}^{k}{P(\mathbf{x}|C_j)P(C_j)}} \\
&= \frac{\text{exp}(a_n(\mathbf{x}))}{\sum_{j = 1}^{k}{\text{exp}(a_j(\mathbf{x}))}}
\end{align}
$$

This is the basis of the [[Softmax Function]]

Notice that the $a_n(\mathbf{x})$ function is **not the log-posterior odds** as before

$$
a_n(\mathbf{x}) = \ln(P(\mathbf{x}|C_n)P(C_n)) = \ln{P(\mathbf{x}|C_n)} + \ln{P(C_j)}
$$

Also note, the total posterior probability must add to 1
$$
\sum_{j=1}^{k}{P(C_j|\mathbf{x})} = 1
$$
## What's next?

PGM requires two more things:
- Prior class probabilities $P(C_j)$
- Class conditional densities $p(\mathbf{x}|C_j)$ which we'll assume is Gaussian in this course (`GaussianNB` in `scikit-learn`)

## Gaussian Class-Conditional PDFs

![[Pasted image 20240318090031.png|300]]

$$
p(x|C_j) = \frac{1}{\sqrt{|2\pi\Sigma_j|}} \exp\left( -\frac{1}{2} (\mathbf{x} - \mathbf{\mu}_j)^T\Sigma_j^{-1}(\mathbf{x} - \mathbf{\mu}_j) \right)
$$

Let's break this down:
- $|2\pi\Sigma_j|$ is a determinant
- $\mathbf{\mu}_j$ is the mean vector for each class $C_j$
- $\Sigma_j$ is the class covariance matrix

**We're going to assume that the class covariance matrices are the same across all classes**

$$
\Sigma_j = \Sigma_i = \Sigma
$$
## What's our Log-Posterior Odds?

Since we now know the class conditional PDFs, we can calculate the **log-posterior odds** for the two class case as we did before, assuming shared $\Sigma$:

$$
\begin{align}
a(\mathbf{x}) &= \ln{\frac{P(\mathbf{x}|C_1)P(C_1)}{P(\mathbf{x}| C_2)P(C_2)}} \\
&= \ln{P(\mathbf{x}|C_1)} - \ln{P(\mathbf{x}| C_2)} + \ln{\frac{P(C_1)}{P(C_2)}} \\
&= -\frac{1}{2} (\mathbf{x} - \mathbf{\mu}_1)^T\Sigma^{-1}(\mathbf{x} - \mathbf{\mu}_1) + \frac{1}{2} (\mathbf{x} - \mathbf{\mu}_2)^T\Sigma^{-1}(\mathbf{x} - \mathbf{\mu}_2) + \ln{\frac{P(C_1)}{P(C_2)}} \\
&= -\frac{1}{2}\left[ \mathbf{x}^T\Sigma^{-1}\mathbf{x} - 2\mathbf{\mu}_1^T\Sigma^{-1}\mathbf{x} + \mathbf{\mu}_1^T\Sigma^{-1}\mathbf{\mu}_1 \right] + \frac{1}{2}\left[ \mathbf{x}^T\Sigma^{-1}\mathbf{x} - 2\mathbf{\mu}_2^T\Sigma^{-1}\mathbf{x} + \mathbf{\mu}_2^T\Sigma^{-1}\mathbf{\mu}_2 \right]  + \ln{\frac{P(C_1)}{P(C_2)}} \\
&= \mathbf{\mu}_1^T\Sigma^{-1}\mathbf{x} -\frac{1}{2}\mathbf{\mu}_1^T\Sigma^{-1}\mathbf{\mu}_1 -\mathbf{\mu}_2^T\Sigma^{-1}\mathbf{x} + \frac{1}{2}\mathbf{\mu}_2^T\Sigma^{-1}\mathbf{\mu}_2  + \ln{\frac{P(C_1)}{P(C_2)}} \\
&= (\mathbf{\mu}_1 - \mathbf{\mu}_2)^T\Sigma^{-1}\mathbf{x} + \left[\frac{1}{2}\left(\mathbf{\mu}_2^T\Sigma^{-1}\mathbf{\mu}_2 - \mathbf{\mu}_1^T\Sigma^{-1}\mathbf{\mu}_1\right) + \ln{\frac{P(C_1)}{P(C_2)}}\right]
\end{align}
$$

Wow! It's linear! Let's define some terms to make it more obvious:

- $\mathbf{w}^T = (\mathbf{\mu}_1 - \mathbf{\mu}_2)^T\Sigma^{-1} \Rightarrow \mathbf{w} = \Sigma^{-1}(\mathbf{\mu}_1 - \mathbf{\mu}_2)$ 
	- _we can do $(\Sigma^{-1})^T = \Sigma^{-1}$ _since the covariance matrix is symmetrical _
- $w_0 = \frac{1}{2}\left(\mathbf{\mu}_2^T\Sigma^{-1}\mathbf{\mu}_2 - \mathbf{\mu}_1^T\Sigma^{-1}\mathbf{\mu}_1\right) + \ln{\frac{P(C_1)}{P(C_2)}}$ 
	- Which is the **bias** term (constant)

Making the **log-posterior probability** simplified to:
$$
a(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + w_0
$$
