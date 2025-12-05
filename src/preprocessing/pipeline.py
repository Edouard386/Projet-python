"""Pipeline de preprocessing complet."""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

from .cleaning import load_matches, clean_matches
from .features import (
    build_player_history, create_features, 
    get_player_stats, get_surface_win_rate, get_h2h, get_default_stats
)

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RANDOM_STATE, N_HISTORICAL_MATCHES, N_SURFACE_MATCHES


class TennisPreprocessor:
    """Preprocesseur pour les données de tennis."""
    
    def __init__(self, n_hist=N_HISTORICAL_MATCHES, n_surf=N_SURFACE_MATCHES):
        self.n_hist = n_hist
        self.n_surf = n_surf
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.history = None
        self.categorical_cols = ["surface", "tourney_level", "round"]
        self.feature_cols = None
        self.fitted = False
    
    def fit_transform(self, matches_df):
        """Fit et transforme le training set."""
        # Construire historique
        self.history = build_player_history(matches_df)
        
        # Créer features
        features_df = create_features(matches_df, self.history, self.n_hist, self.n_surf, self.rng)
        
        # Encoder catégorielles
        features_df = self._encode_categorical(features_df, fit=True)
        
        # Sauvegarder colonnes
        drop_cols = ["target"]
        self.feature_cols = [c for c in features_df.columns if c not in drop_cols]
        self.fitted = True
        
        return features_df
    
    def transform(self, matches_df, all_matches_df):
        """Transforme le test set (utilise all_matches pour l'historique complet)."""
        if not self.fitted:
            raise ValueError("Preprocessor not fitted")
        
        # Reconstruire historique avec toutes les données
        full_history = build_player_history(all_matches_df)
        
        features_df = create_features(matches_df, full_history, self.n_hist, self.n_surf, self.rng)
        features_df = self._encode_categorical(features_df, fit=False)
        
        # Aligner colonnes avec train
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        
        return features_df[[c for c in self.feature_cols if c in features_df.columns] + ["target"]]
    
    def transform_upcoming(self, upcoming_df):
        """
        Transforme un DataFrame de matchs à venir (format player_a/player_b).
        
        Args:
            upcoming_df: DataFrame avec colonnes player_a_*, player_b_*, surface, etc.
        
        Returns:
            X: DataFrame de features prêt pour la prédiction
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted")
        
        # Date de prédiction = maintenant
        prediction_date = pd.Timestamp.now()
        
        features_list = []
        for _, row in upcoming_df.iterrows():
            features = self._create_upcoming_features(row, prediction_date)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Encoder catégorielles
        features_df = self._encode_categorical(features_df, fit=False)
        
        # Aligner colonnes avec train
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        
        return features_df[self.feature_cols]
    
    def _create_upcoming_features(self, row, prediction_date):
        """Crée les features pour un match à prédire."""
        features = {}
        
        # === Features statiques ===
        rank_a = row["player_a_rank"]
        rank_b = row["player_b_rank"]
        points_a = row["player_a_points"]
        points_b = row["player_b_points"]
        
        features["rank_a"] = rank_a
        features["rank_b"] = rank_b
        features["points_a"] = points_a
        features["points_b"] = points_b
        features["age_a"] = row["player_a_age"]
        features["age_b"] = row["player_b_age"]
        features["height_a"] = row["player_a_ht"]
        features["height_b"] = row["player_b_ht"]
        
        features["rank_diff"] = rank_a - rank_b
        features["rank_ratio"] = rank_a / rank_b if rank_b > 0 else 1
        features["points_diff"] = points_a - points_b
        features["age_diff"] = row["player_a_age"] - row["player_b_age"]
        features["height_diff"] = row["player_a_ht"] - row["player_b_ht"]
        features["is_left_a"] = 1 if row["player_a_hand"] == "L" else 0
        features["is_left_b"] = 1 if row["player_b_hand"] == "L" else 0
        
        # Catégorielles (seront encodées après)
        features["surface"] = row["surface"]
        features["tourney_level"] = row["tourney_level"]
        features["round"] = row["round"]
        features["best_of_5"] = 1 if row["best_of"] == 5 else 0
        
        # === Features historiques ===
        id_a = row["player_a_id"]
        id_b = row["player_b_id"]
        surface = row["surface"]
        
        # Stats joueur A (ordre corrigé: history, player_id, match_date, n_matches)
        stats_a = get_player_stats(self.history, id_a, prediction_date, self.n_hist)
        if stats_a is None:
            stats_a = get_default_stats(rank_a)
        
        # Stats joueur B
        stats_b = get_player_stats(self.history, id_b, prediction_date, self.n_hist)
        if stats_b is None:
            stats_b = get_default_stats(rank_b)
        
        # Ajouter toutes les stats
        for k, v in stats_a.items():
            features[f"{k}_a"] = v
        for k, v in stats_b.items():
            features[f"{k}_b"] = v
        
        # Surface win rates (ordre corrigé: history, player_id, match_date, surface, n_matches)
        surface_wr_a = get_surface_win_rate(self.history, id_a, prediction_date, surface, self.n_surf)
        surface_wr_b = get_surface_win_rate(self.history, id_b, prediction_date, surface, self.n_surf)
        
        features["surface_win_rate_a"] = surface_wr_a
        features["surface_win_rate_b"] = surface_wr_b
        features["surface_win_rate_diff"] = surface_wr_a - surface_wr_b
        
        # Différences historiques
        features["win_rate_diff"] = stats_a["win_rate"] - stats_b["win_rate"]
        features["ace_rate_diff"] = stats_a["ace_rate"] - stats_b["ace_rate"]
        features["bp_save_rate_diff"] = stats_a["bp_save_rate"] - stats_b["bp_save_rate"]
        
        # Head-to-head (ordre corrigé: history, player_a, player_b, match_date)
        h2h = get_h2h(self.history, id_a, id_b, prediction_date)
        features["h2h_win_rate_a"] = h2h
        
        return features
    
    def _encode_categorical(self, df, fit=True):
        """One-hot encode les variables catégorielles."""
        if fit:
            self.categories_ = {}
            for col in self.categorical_cols:
                if col in df.columns:
                    self.categories_[col] = df[col].unique().tolist()
        
        for col in self.categorical_cols:
            if col in df.columns:
                if fit:
                    dummies = pd.get_dummies(df[col], prefix=col)
                else:
                    # Utiliser les catégories du fit
                    dummies = pd.get_dummies(df[col], prefix=col)
                    # Ajouter colonnes manquantes
                    for cat in self.categories_.get(col, []):
                        col_name = f"{col}_{cat}"
                        if col_name not in dummies.columns:
                            dummies[col_name] = 0
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        
        return df
    
    def get_X_y(self, features_df):
        """Sépare features et target."""
        X = features_df.drop(columns=["target"])
        y = features_df["target"]
        return X, y
    
    def save(self, path):
        """Sauvegarde le preprocessor."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {path}")
    
    @staticmethod
    def load(path):
        """Charge un preprocessor sauvegardé."""
        with open(path, "rb") as f:
            return pickle.load(f)


def temporal_split(df, test_size=0.2):
    """Split temporel basé sur la date."""
    df = df.sort_values("tourney_date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_preprocessing(raw_dir, save_dir=None):
    """Pipeline complet de preprocessing."""
    print("Loading data...")
    matches = load_matches(raw_dir)
    print(f"Loaded {len(matches)} matches")
    
    matches = clean_matches(matches)
    print(f"After cleaning: {len(matches)} matches")
    
    train_matches, test_matches = temporal_split(matches)
    print(f"Train: {len(train_matches)}, Test: {len(test_matches)}")
    
    print("Creating features...")
    preprocessor = TennisPreprocessor()
    
    train_features = preprocessor.fit_transform(train_matches)
    print(f"Train features created: {train_features.shape}")
    
    test_features = preprocessor.transform(test_matches, matches)
    print(f"Test features created: {test_features.shape}")
    
    X_train, y_train = preprocessor.get_X_y(train_features)
    X_test, y_test = preprocessor.get_X_y(test_features)
    
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        X_train.to_parquet(save_dir / "X_train.parquet")
        X_test.to_parquet(save_dir / "X_test.parquet")
        y_train.to_frame("target").to_parquet(save_dir / "y_train.parquet")
        y_test.to_frame("target").to_parquet(save_dir / "y_test.parquet")
        preprocessor.save(save_dir / "preprocessor.pkl")
        print(f"Saved to {save_dir}")
    
    return X_train, X_test, y_train, y_test, preprocessor
