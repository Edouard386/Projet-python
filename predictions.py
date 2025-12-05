"""
Script de prédiction pour les matchs à venir.
1. Remplir data/predictions/upcoming_matches.csv
2. Lancer: python predictions.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import MODELS_DIR, PROCESSED_DIR
from src.preprocessing.pipeline import TennisPreprocessor
from src.training.train import load_model


PREDICTIONS_DIR = Path("data/predictions")
INPUT_FILE = PREDICTIONS_DIR / "upcoming_matches.csv"
OUTPUT_FILE = PREDICTIONS_DIR / "predictions_output.csv"


def predict_matches(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    """
    Prédit les résultats des matchs dans le fichier CSV.
    """
    print("=" * 50)
    print("MATCH PREDICTIONS")
    print("=" * 50)
    
    # Charger le modèle
    print("Loading model...")
    model = load_model(MODELS_DIR / "model.pkl")
    
    # Charger le preprocessor
    print("Loading preprocessor...")
    preprocessor = TennisPreprocessor.load(PROCESSED_DIR / "preprocessor.pkl")
    
    # Charger les matchs à prédire
    print(f"Loading matches from {input_file}...")
    upcoming = pd.read_csv(input_file)
    print(f"Found {len(upcoming)} matches to predict")
    
    # Générer les features
    print("Generating features...")
    X = preprocessor.transform_upcoming(upcoming)
    print(f"Features shape: {X.shape}")
    
    # Prédictions
    print("Predicting...")
    proba = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    
    # Résultats
    results = upcoming.copy()
    results["proba_player_a_wins"] = proba
    results["predicted_winner"] = np.where(
        predictions == 1,
        results["player_a_name"],
        results["player_b_name"]
    )
    results["confidence"] = np.where(
        predictions == 1,
        proba,
        1 - proba
    )
    
    # Afficher
    print("\n" + "=" * 50)
    print("PREDICTIONS")
    print("=" * 50)
    
    for _, row in results.iterrows():
        print(f"\n{row['player_a_name']} vs {row['player_b_name']}")
        print(f"  Surface: {row['surface']}, Round: {row['round']}")
        print(f"  → Winner: {row['predicted_winner']} ({row['confidence']:.1%} confidence)")
    
    # Sauvegarder
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    predict_matches()
