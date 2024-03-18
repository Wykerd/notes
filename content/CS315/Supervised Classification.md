Given a set of **labelled data** $D : (\mathbf{x}_i, y_i), i = 1,...,N$ (a data set of $N$ observations $\mathbf{x}_i$ each with a class label $y_i = C_i$ if $\mathbf{x}_i \in C_i$)

We wish to construct the posterior probabilities, or **class probabilities**, $P(C_j|\mathbf{x})$ from the given data $D$. There are two methods to do this:

- [[Probabilistic Generative Model|Generative]] - through means of [[Naive Bayes]]
	1. Estimate the class-conditional probabilities, $p(\mathbf{x}|C_j)$ - _which can be used to generate new data points, hence the name 'generative model'_
	2. Calculate the posterior probability using [[Bayes Theorem]]: $P(C_j|\mathbf{x}) \propto p(\mathbf{x}|C_j) \cdot P(C_j)$ 
- [[Probabilistic Discriminative Model|Discriminative]] - through means of [[Logistic Regression]]
	1. Directly compute $P(C_j|\mathbf{x})$ without first calculating class conditionals

**So which one do I choose?**

Well, here's quick comparison:

| Generative                        | Discriminative                    |
| --------------------------------- | --------------------------------- |
| More Flexible                     | Less Flexible                     |
| Less Efficient for Classification | More Efficient for Classification |
| Simpler Training (per class)      | Harder Training                   |
| Class Data                        | All Data                          |
| Models each class                 | Focusses on class differences     |

In other words: The generative model is trained per class **and ignores the properties of the other classes**, while the discriminative model considers **all data** during training.