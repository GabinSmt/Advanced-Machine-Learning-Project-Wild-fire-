"""
Standalone training script for the WildFire XGBoost classifier.

Usage
-----
Run from the project root::

    PYTHONPATH=. python src/models/train.py

Environment variables (set in .env or shell)::

    S3_ENDPOINT_URL   URL of the MinIO / S3 endpoint
    S3_BUCKET         Name of the S3 bucket containing df_aggregated.parquet
    MLFLOW_TRACKING_URI  (optional) MLflow tracking server URI — defaults to ./mlruns

The script will:
    1. Load and clean data from S3 via src.data.df_aggregated.
    2. Run StratifiedKFold cross-validation with RandomizedSearchCV to tune
       hyperparameters on a scikit-learn compatible wrapper.
    3. Retrain the best model on the full training set.
    4. Log parameters, CV metrics and the serialised model to MLflow.
    5. Save the model as a pickle file in models/.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.data.df_aggregated import FEATURES, TARGET_COLUMN, load_data_from_s3
from src.models.classifier import XGBoostClassifier
from src.models.utils import split_train_test

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# MLflow experiment name
EXPERIMENT_NAME = "wildfire_xgboost"

# Default output directory for saved models
MODELS_DIR = Path("models")

# ---------------------------------------------------------------------------
# Scikit-learn wrapper (required for RandomizedSearchCV)
# ---------------------------------------------------------------------------


class XGBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper around XGBoostClassifier.

    Exposes the same hyperparameter interface as XGBoostClassifier.fit()
    so that sklearn's cross-validation and search utilities can be used.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth (default=5).
    learning_rate : float
        Shrinkage factor per boosting round (default=0.3).
    boosting_rounds : int
        Number of boosting iterations (default=10).
    lambda_ : float
        L2 regularization on leaf weights (default=1.5).
    gamma : float
        Minimum gain required to make a split (default=1.0).
    min_leaf : int
        Minimum samples per leaf node (default=5).
    min_child_weight : float
        Minimum Hessian sum per child node (default=1.0).
    solver : str
        Split strategy: 'greedy', 'global', or 'local' (default='greedy').
    eps : float
        Quantile sketch approximation factor (default=0.1).
    """

    def __init__(
        self,
        max_depth=5,
        learning_rate=0.3,
        boosting_rounds=10,
        lambda_=1.5,
        gamma=1.0,
        min_leaf=5,
        min_child_weight=1.0,
        solver="greedy",
        eps=0.1,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lambda_ = lambda_
        self.gamma = gamma
        self.min_leaf = min_leaf
        self.min_child_weight = min_child_weight
        self.solver = solver
        self.eps = eps

    def fit(self, X, y):
        """Fit XGBoostClassifier with stored hyperparameters."""
        self.model_ = XGBoostClassifier()
        self.model_.fit(
            X,
            y,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            boosting_rounds=self.boosting_rounds,
            lambda_=self.lambda_,
            gamma=self.gamma,
            min_leaf=self.min_leaf,
            min_child_weight=self.min_child_weight,
            solver=self.solver,
            eps=self.eps,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        """Return binary predictions."""
        return self.model_.predict(X)

    def predict_proba(self, X):
        """Return class probabilities as (n_samples, 2) array."""
        proba_pos = self.model_.predict_proba(X)
        return np.column_stack([1 - proba_pos, proba_pos])

    def score(self, X, y):
        """Return F1 score (macro average)."""
        return f1_score(y, self.predict(X), average="macro")


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def prepare_features(df: pd.DataFrame):
    """Split cleaned DataFrame into X, y using FEATURES from df_aggregated."""
    X = df[FEATURES].to_numpy(dtype=np.float64)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    return X, y, FEATURES


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

PARAM_DISTRIBUTIONS = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.05, 0.1, 0.2, 0.3, 0.4],
    "boosting_rounds": [10, 20, 30, 50],
    "lambda_": [0.5, 1.0, 1.5, 2.0],
    "gamma": [0.0, 0.5, 1.0, 2.0],
    "min_leaf": [3, 5, 10],
    "min_child_weight": [1.0, 2.0, 5.0],
}


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    n_iter: int = 20,
    random_state: int = 42,
) -> dict:
    """
    Tune XGBoost hyperparameters via RandomizedSearchCV with StratifiedKFold.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    n_splits : int, optional
        Number of CV folds (default=5).
    n_iter : int, optional
        Number of random parameter combinations to try (default=20).
    random_state : int, optional
        Random seed for reproducibility (default=42).

    Returns
    -------
    dict
        Best hyperparameters found by RandomizedSearchCV.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    estimator = XGBoostClassifierWrapper()

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        n_jobs=1,  # Our custom model is not thread-safe
        random_state=random_state,
        verbose=1,
        refit=True,
    )

    logger.info(
        "Starting RandomizedSearchCV (%d iterations, %d-fold CV)…", n_iter, n_splits
    )
    search.fit(X_train, y_train)

    logger.info("Best CV F1 (macro): %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)

    return search.best_params_, search.best_score_


# ---------------------------------------------------------------------------
# MLflow logging helpers
# ---------------------------------------------------------------------------


def log_run(
    params: dict,
    cv_score: float,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
):
    """
    Log parameters, metrics, and the model artefact to MLflow.

    Parameters
    ----------
    params : dict
        Hyperparameters to log.
    cv_score : float
        Mean CV F1 score (logged as ``cv_f1_macro``).
    model : XGBoostClassifierWrapper
        Fitted estimator.
    X_test : np.ndarray
        Held-out test features for final evaluation.
    y_test : np.ndarray
        Held-out test labels.
    feature_names : list of str
        Feature column names (logged as a tag).
    """
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "cv_f1_macro": cv_score,
                "test_f1_macro": test_f1,
                "test_roc_auc": test_auc,
            }
        )
        mlflow.set_tag("feature_names", str(feature_names))

        # Log model as pickle artifact (aligned with Membre 3)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="wildfire_xgboost",
        )

        logger.info(
            "MLflow run logged — test_f1=%.4f  test_auc=%.4f", test_f1, test_auc
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train the WildFire XGBoost model.")
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of RandomizedSearchCV iterations (default: 20).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits (default: 5).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for final evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Directory to save the pickled model (default: models/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- MLflow setup ---
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", mlflow_uri)

    # --- Load data ---
    df = load_data_from_s3()
    X, y, feature_names = prepare_features(df)

    # --- Train / test split (reproducible) ---
    df_full = pd.DataFrame(X, columns=feature_names)
    df_full[TARGET_COLUMN] = y
    df_train, df_test = split_train_test(
        df_full, test_size=args.test_size, random_state=args.random_state
    )

    X_train = df_train.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=np.float64)
    y_train = df_train[TARGET_COLUMN].to_numpy(dtype=np.int64)
    X_test = df_test.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=np.float64)
    y_test = df_test[TARGET_COLUMN].to_numpy(dtype=np.int64)

    logger.info("Train size: %d  |  Test size: %d", len(X_train), len(X_test))

    # --- Hyperparameter tuning ---
    best_params, best_cv_score = tune_hyperparameters(
        X_train,
        y_train,
        n_splits=args.n_splits,
        n_iter=args.n_iter,
        random_state=args.random_state,
    )

    # --- Retrain on full training set with best params ---
    logger.info("Retraining final model on full training set…")
    final_model = XGBoostClassifierWrapper(**best_params)
    final_model.fit(X_train, y_train)

    # --- Log to MLflow ---
    log_run(best_params, best_cv_score, final_model, X_test, y_test, feature_names)

    # --- Save model as pickle (agreed format with Membre 3) ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_wildfire.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
