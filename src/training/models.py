"""Définition des modèles et leurs hyperparamètres."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class StatsLogitClassifier:
    """Wrapper statsmodels.Logit compatible sklearn avec p-values et summary."""

    def __init__(self):
        self.model = None
        self.results = None
        self.feature_names = None

    def fit(self, X, y):
        """Fit le modèle Logit avec constante."""
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
            # Convertir bool en int pour statsmodels
            X = X.astype({col: int for col in X.select_dtypes(include='bool').columns})
        else:
            self.feature_names = [f"x{i}" for i in range(X.shape[1])]

        X_const = sm.add_constant(X, has_constant='add')
        self.model = sm.Logit(y, X_const)
        self.results = self.model.fit(disp=0)
        return self

    def predict(self, X):
        """Prédit les classes (0 ou 1)."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        """Retourne les probabilités [[p0, p1], ...]."""
        if hasattr(X, 'select_dtypes'):
            X = X.astype({col: int for col in X.select_dtypes(include='bool').columns})
        X_const = sm.add_constant(X, has_constant='add')
        p1 = self.results.predict(X_const)
        p0 = 1 - p1
        return np.column_stack([p0, p1])

    def summary(self):
        """Retourne le summary statsmodels complet."""
        return self.results.summary()

    def get_coefficients(self):
        """Retourne DataFrame avec coefficients, p-values, intervalles."""
        return pd.DataFrame({
            'coef': self.results.params,
            'std_err': self.results.bse,
            'z': self.results.tvalues,
            'p_value': self.results.pvalues,
            'ci_lower': self.results.conf_int()[0],
            'ci_upper': self.results.conf_int()[1]
        })

    def get_significant_features(self, alpha=0.05):
        """Retourne les features avec p-value < alpha."""
        coefs = self.get_coefficients()
        significant = coefs[coefs['p_value'] < alpha]
        return significant.drop('const', errors='ignore')


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
        ),

        "LinearStats": (
            StatsLogitClassifier(),
            {}  # Pas de grid search, entraîné directement
        )
    }

    return models
