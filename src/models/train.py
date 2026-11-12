"""Model training for the customer churn engine."""

import sys
from dataclasses import dataclass
from pathlib import Path

from scipy.stats import randint
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

from src.utils.logger import logger
from src.utils.exception import CustomException
from src.utils.helper import save_object

from src.models.model_registry import get_logistic_regression, get_random_forest


@dataclass
class ModelTrainerConfig:
    model_file_path: Path = Path("artifacts") / "model.pkl"


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_val, y_train, y_val, n_iter=30, cv=5):
        logger.info("Initiated model training")
        try:
            # Baseline: logistic regression
            lr = get_logistic_regression()
            lr.fit(X_train, y_train)
            lr_auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
            logger.info(f"Logistic regression val ROC-AUC: {lr_auc:.4f}")

            # Random forest, tuned with 5-fold CV on train
            param_dist = {
                "n_estimators": randint(100, 500),
                "max_depth": [None, 5, 10, 20],
                "min_samples_leaf": randint(1, 20),
                "max_features": ["sqrt", "log2", 0.5],
            }
            search = RandomizedSearchCV(
                get_random_forest().set_params(n_jobs=1),
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="roc_auc",
                cv=cv,
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            rf = search.best_estimator_
            rf_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
            logger.info(f"Random forest best params: {search.best_params_}")
            logger.info(f"Random forest val ROC-AUC: {rf_auc:.4f}")

            # Keep the better model on validation
            # Prefer the simpler model unless the forest is clearly better
            min_gain = 0.01
            if rf_auc - lr_auc > min_gain:
                best_name, best_model, best_auc = "random_forest", rf, rf_auc
            else:
                best_name, best_model, best_auc = "logistic_regression", lr, lr_auc

            save_object(obj=best_model, file_path=self.config.model_file_path)
            logger.info(f"Saved {best_name} (val ROC-AUC {best_auc:.4f})")
            return best_name, best_auc

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.data.ingestion import DataIngestion
    from src.data.preprocessing import DataPreprocessing

    train_path, val_path, test_path = DataIngestion().initiate_data_ingestion()
    X_train, X_val, X_test, y_train, y_val, y_test = (
        DataPreprocessing().initiate_data_preprocessing(train_path, val_path, test_path)
    )
    print(ModelTrainer().initiate_model_training(X_train, X_val, y_train, y_val))
