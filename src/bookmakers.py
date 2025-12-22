"""
Module pour charger les données des bookmakers.

TODO: À compléter par [ton pote] avec la fonction de téléchargement des cotes.
"""

import pandas as pd


def load_bookmaker_odds(start_year=2024, end_year=2025):
    """
    Charge les cotes des bookmakers pour les matchs ATP.

    TODO: Implémenter cette fonction pour retourner un DataFrame avec les colonnes :
        - tourney_date : date du tournoi (format datetime ou YYYYMMDD)
        - winner_name : nom du gagnant (pour faire le matching avec nos données)
        - loser_name : nom du perdant
        - odds_winner : cote du gagnant avant le match
        - odds_loser : cote du perdant avant le match

    Optionnel mais utile :
        - tourney_name : nom du tournoi
        - winner_id / loser_id : identifiants des joueurs

    Args:
        start_year: année de début
        end_year: année de fin

    Returns:
        DataFrame avec les cotes des bookmakers

    Exemple de format attendu :
        tourney_date | winner_name    | loser_name     | odds_winner | odds_loser
        2024-01-15   | Novak Djokovic | Carlos Alcaraz | 1.65        | 2.30
        2024-01-15   | Jannik Sinner  | Daniil Medvedev| 1.80        | 2.05
    """
    # TODO: Remplacer par le vrai code de téléchargement
    raise NotImplementedError(
        "Cette fonction doit être implémentée pour charger les données bookmakers. "
        "Elle doit retourner un DataFrame avec les colonnes : "
        "tourney_date, winner_name, loser_name, odds_winner, odds_loser"
    )


def merge_with_matches(matches_df, odds_df):
    """
    Fusionne les données de matchs avec les cotes des bookmakers.

    Args:
        matches_df: DataFrame des matchs (depuis load_matches)
        odds_df: DataFrame des cotes (depuis load_bookmaker_odds)

    Returns:
        DataFrame fusionné avec les cotes pour chaque match
    """
    # Matching sur date + noms des joueurs
    merged = matches_df.merge(
        odds_df,
        on=['tourney_date', 'winner_name', 'loser_name'],
        how='inner'
    )

    print(f"Matchs avec cotes : {len(merged)} / {len(matches_df)} "
          f"({len(merged)/len(matches_df)*100:.1f}%)")

    return merged
