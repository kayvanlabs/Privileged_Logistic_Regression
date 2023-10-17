import cvxpy as cp
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.linear_model._base import SparseCoefMixin
from scipy.special import expit
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _check_sample_weight
from privileged_util import _check_X_y

class CvxpyLogisticRegression(BaseEstimator, 
    LinearClassifierMixin,SparseCoefMixin):

    def __init__(self, penalty="l2", lambda_=1.0, 
        tol=1e-4, fit_intercept=False, class_weight=None, 
        max_iter=100, verbose=False):

        if penalty == 'None' or penalty == 'none':
            penalty = None
        self.penalty = penalty
        
        self.lambda_ = lambda_
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.verbose = verbose

    def fit_cvxpy(self, X, y, sample_weight):

        m, n = X.shape
        if self.fit_intercept:
            nn = n + 1
            beta = cp.Variable(nn)
            Xbeta = np.c_[X, np.ones(m)] @ beta
        else:
            nn = n
            beta = cp.Variable(nn)
            Xbeta = X @ beta
        # Regularization
        w = beta
        if self.fit_intercept:
            w = beta[:-1]

        if self.penalty == "l1":
            penalty = cp.norm(w, 1)
        elif self.penalty == "l2":
            penalty = 0.5 * cp.sum_squares(w)
        else:
            penalty = 0

        cons = []
        cons.append(penalty <= self.lambda_)

        objective_base = sample_weight @ (cp.logistic(-cp.multiply(y, Xbeta)))
        obj = cp.Minimize(objective_base)
        problem = cp.Problem(obj, cons)
        try:
            problem.solve(max_iters=self.max_iter, verbose=self.verbose, abstol=self.tol)
        except Exception as e:
            # Sometimes would raise "cvxpy.error.SolverError: Solver 'ECOS' failed. 
            # Try another solver, or solve with verbose=True for more information."
            raise ValueError(e)
            
        # Return intercept and coefficient
        if self.fit_intercept:
            intercept_ = beta.value[-1]
            coef_ = beta.value[:-1]
        else:
            intercept_ = 0
            coef_ = beta.value

        return coef_, intercept_

    def fit(self, X, y, sample_weight=None):
        X, y, _ = _check_X_y(X, y)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        coef_, intercept_ = self.fit_cvxpy(X,  y, sample_weight)

        if coef_ is not None:
            self.coef_ = np.asarray([coef_])
            self.intercept_ = np.asarray([intercept_])

        else:
            raise ValueError('The problem is infeasible or unbounded.')

        return self
    
    def predict_proba(self, X):
        check_is_fitted(self)

        proba = np.empty((X.shape[0], 2))
        p0 = expit(-self.decision_function(X))
        proba[:, 0] = p0
        proba[:, 1] = 1 - p0

        return proba
