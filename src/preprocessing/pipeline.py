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
from config import RANDOM_STATE, N_HISTORICAL_MATCHES, N_SURFACE_MATCHES, HALF_LIFE_DAYS


class TennisPreprocessor:
    """Preprocesseur pour les données de tennis."""

    def __init__(self, n_hist=N_HISTORICAL_MATCHES, n_surf=N_SURFACE_MATCHES, half_life_days=HALF_LIFE_DAYS):
        self.n_hist = n_hist
        self.n_surf = n_surf
        self.half_life_days = half_life_days
        self.rng = np.random.default_rng(RANDOM_STATE)
        self.history = None
        self.categorical_cols = ["surface", "tourney_level", "round"]
        self.feature_cols = None
        self.fitted = False

    def build_history(self, matches_df):
        """Construit l'historique des joueurs à partir des matchs."""
        self.history = build_player_history(matches_df)
        return self.history

    def create_features(self, matches_df, rng=None):
        """
        Crée les features à partir des matchs.
        Nécessite que build_history() ait été appelé avant.
        """
        if self.history is None:
            raise ValueError("Appelez build_history() d'abord")

        if rng is None:
            rng = self.rng

        features_df = create_features(
            matches_df,
            self.history,
            self.n_hist,
            self.n_surf,
            rng,
            half_life_days=self.half_life_days,
        )
        return features_df

    def encode(self, features_df, fit=True):
        """Applique le one-hot encoding aux variables catégorielles."""
        encoded = self._encode_categorical(features_df.copy(), fit=fit)
        if fit:
            meta_cols = ["target", "player_a_name", "player_b_name", "tourney_date"]
            self.feature_cols = [c for c in encoded.columns if c not in meta_cols]
            self.fitted = True
        return encoded

    def fit_transform(self, matches_df):
        """Fit et transforme toutes les données (raccourci)."""
        self.build_history(matches_df)
        features_df = self.create_features(matches_df)
        features_df = self.encode(features_df, fit=True)
        return features_df

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
        
        features["rank_ratio"] = rank_a / rank_b if rank_b > 0 else 1
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
        stats_a = get_player_stats(
            self.history, id_a, prediction_date, self.n_hist, half_life_days=self.half_life_days
        )
        if stats_a is None:
            stats_a = get_default_stats(rank_a)
        
        # Stats joueur B
        stats_b = get_player_stats(
            self.history, id_b, prediction_date, self.n_hist, half_life_days=self.half_life_days
        )
        if stats_b is None:
            stats_b = get_default_stats(rank_b)
        
        # Ajouter toutes les stats
        for k, v in stats_a.items():
            features[f"{k}_a"] = v
        for k, v in stats_b.items():
            features[f"{k}_b"] = v
        
        # Surface win rates (ordre corrigé: history, player_id, match_date, surface, n_matches)
        surface_wr_a = get_surface_win_rate(
            self.history, id_a, prediction_date, surface, self.n_surf, half_life_days=self.half_life_days
        )
        surface_wr_b = get_surface_win_rate(
            self.history, id_b, prediction_date, surface, self.n_surf, half_life_days=self.half_life_days
        )
        
        features["surface_win_rate_a"] = surface_wr_a
        features["surface_win_rate_b"] = surface_wr_b
        
        # Head-to-head (ordre corrigé: history, player_a, player_b, match_date)
        h2h = get_h2h(self.history, id_a, id_b, prediction_date, half_life_days=self.half_life_days)
        features["h2h_win_rate_a"] = h2h
        
        return features
    
    def _encode_categorical(self, df, fit=True):
        """One-hot encode les variables catégorielles avec drop_first pour éviter multicolinéarité."""
        if fit:
            self.categories_ = {}
            for col in self.categorical_cols:
                if col in df.columns:
                    self.categories_[col] = sorted(df[col].unique().tolist())

        for col in self.categorical_cols:
            if col in df.columns:
                if fit:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                else:
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    # Ajouter colonnes manquantes (sauf la première qui est droppée)
                    for cat in self.categories_.get(col, [])[1:]:
                        col_name = f"{col}_{cat}"
                        if col_name not in dummies.columns:
                            dummies[col_name] = 0
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        return df
    
    def get_X_y(self, features_df):
        """Sépare features et target, en enlevant les colonnes meta."""
        meta_cols = ["target", "player_a_name", "player_b_name", "tourney_date"]
        cols_to_drop = [c for c in meta_cols if c in features_df.columns]
        X = features_df.drop(columns=cols_to_drop)
        y = features_df["target"]
        return X, y

    def split_temporal(self, features_df, cutoff_year=2024):
        """
        Split temporel : train ≤ cutoff_year, test > cutoff_year.

        Args:
            features_df: DataFrame avec colonne tourney_date
            cutoff_year: année de coupure (incluse dans train)

        Returns:
            X_train, X_test, y_train, y_test, meta_test
        """
        df = features_df.copy()
        df['_year'] = pd.to_datetime(df['tourney_date']).dt.year

        train_df = df[df['_year'] <= cutoff_year].drop(columns=['_year'])
        test_df = df[df['_year'] > cutoff_year].drop(columns=['_year'])

        # Garder les colonnes meta du test pour la comparaison bookmakers
        meta_cols = ["player_a_name", "player_b_name", "tourney_date"]
        meta_test = test_df[meta_cols].copy().reset_index(drop=True)

        X_train, y_train = self.get_X_y(train_df)
        X_test, y_test = self.get_X_y(test_df)

        return X_train, X_test, y_train, y_test, meta_test
    
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


def run_preprocessing(raw_dir, save_dir=None, test_size=0.2):
    """Pipeline complet de preprocessing."""
    from sklearn.model_selection import train_test_split

    print("Loading data...")
    matches = load_matches(raw_dir)
    print(f"Loaded {len(matches)} matches")

    matches = clean_matches(matches)
    print(f"After cleaning: {len(matches)} matches")

    print("Creating features...")
    preprocessor = TennisPreprocessor()
    features_df = preprocessor.fit_transform(matches)
    print(f"Features created: {features_df.shape}")

    X, y = preprocessor.get_X_y(features_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
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
