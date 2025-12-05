"""Entraînement et évaluation des modèles."""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .models import get_models

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RANDOM_STATE, MODELS_DIR, MODEL_NAME


def evaluate_model(model, X, y):
    """
    Évalue un modèle.
    
    Returns:
        dict avec accuracy, log_loss, roc_auc
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    return {
        "accuracy": accuracy_score(y, y_pred),
        "log_loss": log_loss(y, y_proba),
        "roc_auc": roc_auc_score(y, y_proba)
    }


def train_model(X_train, y_train, model_name=None, cv=5):
    """
    Entraîne un modèle avec GridSearchCV.

    Args:
        X_train: features
        y_train: target
        model_name: nom du modèle (si None, utilise MODEL_NAME de config)
        cv: nombre de folds pour la cross-validation

    Returns:
        best_model, info_dict
    """
    if model_name is None:
        model_name = MODEL_NAME

    models = get_models()
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")

    model, param_grid = models[model_name]

    print(f"Training {model_name} with GridSearchCV (cv={cv})...")
    print(f"Training on {len(X_train)} samples")

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True  # Réentraîne automatiquement sur tout X_train avec les meilleurs params
    )
    search.fit(X_train, y_train)  # GridSearchCV gère la validation en interne

    # Scores de cross-validation
    cv_score = search.best_score_
    cv_std = search.cv_results_['std_test_score'][search.best_index_]

    print(f"\nBest params: {search.best_params_}")
    print(f"CV Accuracy: {cv_score:.4f} (+/- {cv_std:.4f})")

    info = {
        "name": model_name,
        "best_params": search.best_params_,
        "cv_accuracy": cv_score,
        "cv_std": cv_std
    }

    return search.best_estimator_, info


def train_with_automl(X_train, y_train, time_budget=300):
    """
    Entraîne avec FLAML AutoML.

    Args:
        X_train: features
        y_train: target
        time_budget: temps max en secondes

    Returns:
        best_model, info_dict
    """
    from flaml import AutoML

    print(f"Running AutoML (time_budget={time_budget}s)...")
    print(f"Training on {len(X_train)} samples")

    automl = AutoML()
    automl.fit(
        X_train, y_train,  # FLAML gère la validation en interne
        task="classification",
        metric="accuracy",
        time_budget=time_budget,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
        seed=RANDOM_STATE,
        verbose=1
    )

    print(f"\nBest estimator: {automl.best_estimator}")
    print(f"Best config: {automl.best_config}")
    print(f"Best validation score: {automl.best_loss:.4f}")

    info = {
        "name": f"AutoML_{automl.best_estimator}",
        "best_params": automl.best_config,
        "best_val_score": -automl.best_loss  # FLAML minimise la loss, donc on inverse
    }

    return automl, info


def save_model(model, path=None):
    """Sauvegarde un modèle."""
    if path is None:
        path = MODELS_DIR / "model.pkl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path=None):
    """Charge un modèle."""
    if path is None:
        path = MODELS_DIR / "model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
