"""Model registry for the customer churn prediction pipeline.

Contains logistic regression, random forest, and neural network model definitions.
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import List


def get_logistic_regression(random_state: int = 42) -> LogisticRegression:
    """Return a configured logistic regression model."""
    return LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=random_state, C=1.0
    )


def get_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Return a configured random forest classifier."""
    return RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=random_state, n_jobs=-1
    )


class ChurnNeuralNet(nn.Module):
    """Feedforward neural network for binary churn classification."""

    def __init__(self, input_features: int, hidden_units: List[int]):
        super().__init__()
        all_layers = []
        for hidden_unit in hidden_units:
            all_layers.append(nn.Linear(input_features, hidden_unit))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(p=0.3))
            input_features = hidden_unit
        all_layers.append(nn.Linear(hidden_units[-1], 1))
        self.model = nn.Sequential(*all_layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
