## How we'd do it in Python

```python
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X, y)

y_pred = clf.predict(X)
```
