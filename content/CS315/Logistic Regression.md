## How we'd do it in Python

```python
from sklearn.linear_model import LogisticRegression as logis

clf = logis(C=1e5)
clf.fit(X, y)

y_pred = clf.predict(X)
```
