# Privileged_Logistic_Regression
This repo contain model source code for logistic regression under the learning using privileged information paradigm.


### Requirements

The project requires the following Python packages and can be easily installed by `pip install -r requirements.txt`.

```txt
cvxpy==1.1.7
numpy==1.19.5
scikit_learn==1.3.1
scipy==1.4.1
```

### Experiments

The privliged logistic regression have been implemented to have similar interface as scikit-learn. The model can be trained and evaluated using the following code snippets when privilegd information is either all avaiable or only partially available.

#### All Privileged Information Available

```python
```python
# create a simluated dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=12, n_informative=5, n_redundant=0, random_state=0)

# split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# select out 5 informtive columns as privileged information
x_train_star = X_train[:, :5]

# the rest are used as base features
x_train = X_train[:, 5:]

# in the test set, we only keep the base features
x_test = X_test[:, 5:]

# import the model
from privileged_lr import PrivilegedLogisticRegression

# create the model
model = PrivilegedLogisticRegression()

# train the model
model.fit(x_train, y_train, x_train_star)

# evaluate the model
model.score(x_test, y_test)
```

#### Privileged Information Partially Available

```python
# create a simluated dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=12, n_informative=5, n_redundant=0, random_state=0)

# split the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# select out 5 informtive columns as privileged information
x_train_star = X_train[:, :5]

# the rest are used as base features
x_train = X_train[:, 5:]

# in the test set, we only keep the base features
x_test = X_test[:, 5:]

# Randomly selected 80% of the rows
import numpy as np
np.random.seed(0)
idx = np.random.choice(x_train_star.shape[0], int(x_train_star.shape[0] * 0.8), replace=False)

# only keep the selected rows in the privileged information and the label
x_train_star = x_train_star[idx]
y_star = y_train[idx]

# import the model
from privileged_lr import PrivilegedLogisticRegression

# create the model
model = PrivilegedLogisticRegression()

# train the model
model.fit(x_train, y_train, x_train_star, y_star)

# evaluate the model
y_pred = model.predict_proba(x_test)

# elvaluate the model with the AUC score
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


## Compare with the plain logistic regression
from cvxpy_lr import CVXPyLogisticRegression

# create the model
model = CVXPyLogisticRegression()

# train the model
model.fit(x_train, y_train)

# evaluate the model
y_pred = model.predict_proba(x_test)

# elvaluate the model with the AUC score
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)
```