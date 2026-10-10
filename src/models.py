"""Models compared in this project.

Every model has the same small interface:  model.fit(X, y)  and  model.predict(X).

Group 1 is Lasso, in two forms:
  * LassoRegressionClassifier - Lasso Regression on the ordered class score
    (Low=0, Moderate=1, High=2); predicted scores are rounded to a class label.
  * LassoLogisticClassifier   - L1-penalised logistic regression, the usual
    "Lasso" model for a categorical target.
Group 2 is Random Forest.
"""
import warnings

import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV

warnings.filterwarnings("ignore", category=FutureWarning)


class LassoRegressionClassifier:
    name = "Lasso Regression"

    def __init__(self, random_state=0):
        # Regularisation strength (alpha) is chosen by 5-fold cross-validation.
        self.model = LassoCV(cv=5, max_iter=10000, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        scores = self.model.predict(X)
        return np.clip(np.rint(scores), 0, 2).astype(int)

    def selected_features(self, feature_names):
        """Names of the features whose Lasso coefficient is not zero."""
        return [n for n, c in zip(feature_names, self.model.coef_) if abs(c) > 1e-12]


class LassoLogisticClassifier:
    name = "Lasso Logistic (L1)"

    def __init__(self, random_state=0):
        self.model = LogisticRegressionCV(
            Cs=5, cv=3, penalty="l1", solver="saga",
            max_iter=500, tol=1e-3, random_state=random_state,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)



class RandomForestModel:
    name = "Random Forest"

    def __init__(self, random_state=0):
        from sklearn.ensemble import RandomForestClassifier

        # Stopping criteria (max_depth, min_samples_leaf) limit tree growth.
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


COMPARISON_MODEL = RandomForestModel
