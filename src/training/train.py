"""Entraînement et évaluation des modèles."""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
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
        cv: nombre de folds
    
    Returns:
        best_model, info_dict
    """
    if model_name is None:
        model_name = MODEL_NAME
    
    models = get_models()
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    
    model, param_grid = models[model_name]
    
    # Split pour évaluation finale
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"Training {model_name} with GridSearchCV...")
    
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_tr, y_tr)
    
    # Évaluer sur validation
    metrics = evaluate_model(search.best_estimator_, X_val, y_val)
    
    print(f"Best params: {search.best_params_}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    info = {
        "name": model_name,
        "best_params": search.best_params_,
        **metrics
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
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"Running AutoML (time_budget={time_budget}s)...")
    
    automl = AutoML()
    automl.fit(
        X_tr, y_tr,
        task="classification",
        metric="accuracy",
        time_budget=time_budget,
        estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
        seed=RANDOM_STATE,
        verbose=1
    )
    
    metrics = evaluate_model(automl, X_val, y_val)
    
    print(f"\nBest estimator: {automl.best_estimator}")
    print(f"Best config: {automl.best_config}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    info = {
        "name": f"AutoML_{automl.best_estimator}",
        "best_params": automl.best_config,
        **metrics
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
