"""Model-agnostic SHAP explanations for churn predictions."""

import pandas as pd
import shap


class ChurnExplainer:
    """Wrap shap.Explainer so callers get the same output shape for any model."""

    def __init__(self, model, background: pd.DataFrame):
        # shap.Explainer picks Tree/Linear/etc. based on the model type
        self.explainer = shap.Explainer(model, background)

    def explain(self, X: pd.DataFrame) -> shap.Explanation:
        sv = self.explainer(X)
        # Tree classifiers return (rows, features, 2); keep the churn class
        if sv.values.ndim == 3:
            sv = sv[:, :, 1]
        return sv

    def top_reasons(
        self, X: pd.DataFrame, raw: pd.DataFrame, original_cols: list[str], k: int = 3
    ) -> list[list[dict]]:
        """Top-k drivers per customer, grouped back to original features.

        X: transformed features (columns = preprocessor feature names)
        raw: the same customers' original fields, for readable values
        """
        sv = self.explain(X)
        # Longest names first so e.g. "StreamingTV" can't match a shorter prefix
        originals = sorted(original_cols, key=len, reverse=True)

        def to_original(name: str) -> str:
            stripped = name.split("__", 1)[-1]
            for col in originals:
                if stripped == col or stripped.startswith(col + "_"):
                    return col
            return stripped

        groups = [to_original(c) for c in X.columns]

        reasons = []
        for i, row in enumerate(sv):
            impact = pd.Series(row.values, index=X.columns).groupby(groups).sum()
            top = impact.reindex(impact.abs().sort_values(ascending=False).index).head(
                k
            )
            row_raw = raw.iloc[i]
            reasons.append(
                [
                    {
                        "feature": feat,
                        "value": getattr(
                            row_raw[feat], "item", lambda: row_raw[feat]
                        )(),
                        "impact": round(float(val), 3),
                        "effect": (
                            "increases churn risk"
                            if val > 0
                            else "decreases churn risk"
                        ),
                    }
                    for feat, val in top.items()
                ]
            )
        return reasons
