# Privileged_Logistic_Regression
This repo contain model source code for logistic regression under the learning using privileged information paradigm.


### Requirements

The project requires the following Python packages and can be easily installed by `pip install --user -r requirements.txt`.

```txt
cvxpy>=1.1.7
scipy>=1.4.1
numpy>=1.19.5
scikit_learn>=1.3.1
```

### How to use 
The privliged logistic regression (PLR) have been implemented to have similar interface as scikit-learn. To train the PLR model, users can employ the `fit` function, providing both the base domain data and privileged domain data along with their respective labels. Meanwhile, evalutaion is perform by `predict_proba` function. To illustrate the general usage of the PLR model, consider the following code snippet

```python
# import the model
from privileged_lr import PrivilegedLogisticRegression

# Define hyperparameters in model instantiation
model = PrivilegedLogisticRegression(kwarg1=value1, kwarg2=value2, ...)

# Train the model with data from the base domain and the privileged domain, along with the associated labels
model.fit(x_train, y_train, x_train_star, y_train_star)

# Obtain test predictions
y_pred = model.predict_proba(x_test)

# Evaluate the test performance using a chosen metric
score = metric_to_evaluate(y_test, y_pred)
```


### Simulated Experiments

[Example_LUPI.ipynb](/Example_LUPI.ipynb) and [Example_LUPAPI.ipynb](/Example_LUPAPI.ipynb) contain code for simulated experiments, catering to scenarios where privileged information is either fully available or only partially available. This code is ready for execution directly within a Jupyter Notebook environment.
