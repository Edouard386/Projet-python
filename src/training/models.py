"""Définition des modèles et leurs hyperparamètres."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_models():
    """
    Retourne les modèles et leurs grilles d'hyperparamètres.
    
    Returns:
        dict: {nom: (modèle, param_grid)}
    """
    models = {
        "LogisticRegression": (
            LogisticRegression(random_state=42, max_iter=1000),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            }
        ),
        
        "RandomForest": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        ),
        
        "XGBoost": (
            XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, eval_metric="logloss"),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        )
    }
    
    return models
