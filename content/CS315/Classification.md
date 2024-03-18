The basic problem of classification is: given a observation $\mathbf{x}$ assign it to one of $k$ classes $C_j, j = 1,...,k$  
## Introduction

[[Supervised Classification]] involves the use of **labeled data**, i.e. data consisting of a pair of both inputs and outputs $(\mathbf{x}_i, y_i)$.

[[Unsupervised Classification]], conversely, utilises only input data $\mathbf{x}_i$

## Data Sets and Preprocessing

Data is preprocessed using:
- [[Principle Component Analysis|PCA]]
- [[Linear Discriminant Analysis|LDA]]

LDA is a more reasonable choice for [[Supervised Classification]].

Data is also split up into three sets while setting up the model:
- **Training set**: This is the set of data set aside to train the classifier
- **Validation set**: We use this set during training to check if we're using the correct model (model selection), and to tune the model **hyperparameters**
- **Test set**: used to evaluate the final model performance

## Performance

For a **single value** summary of model performance, values such as **accuracy** is used.

For a more detailed view on classification performance, consider using the **confusion matrix**

## Core Principles of Classification

To classify a given input data sample $\mathbf{x}$ into one of $k$ classes:
- Define the classes $C_j$ for all $j = 1,...,k$ 
- Determine the posterior probability $P(C_j | \mathbf{x})$ - the probability of $C_j$ being the correct class **given the input data**

While we can then compute the predicted class as the class with the highest probability $C^* = \operatorname*{arg\,max}_{C_j}{P(C_j|\mathbf{x})}$ , the class probabilities are often more useful than just knowing the class with the highest probability.

**So how dow we classify data like this?!**

There's two methods:
- [[Supervised Classification]] - for when your data set is labelled
- [[Unsupervised Classification]] - for when your data set is unlabelled