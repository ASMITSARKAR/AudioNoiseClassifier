import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from src.evaluation.metrics import COST_MATRIX


def predict_bayes_optimal(probs, cost_matrix=COST_MATRIX):
    expected_costs = np.matmul(probs, cost_matrix)
    return np.argmin(expected_costs, axis=1)


class CostSensitiveHGBClassifier:

    def __init__(self, hgb_model, cost_matrix=COST_MATRIX):
        self.model = hgb_model
        self.cost_matrix = cost_matrix

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        probs = self.model.predict_proba(X)
        return predict_bayes_optimal(probs, self.cost_matrix)
