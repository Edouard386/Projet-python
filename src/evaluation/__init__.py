"""Module d'évaluation et comparaison avec les bookmakers."""

from .comparison import (
    proba_to_odds,
    odds_to_proba,
    remove_margin,
    calculate_roi,
    calculate_roi_value_bets,
    find_value_bets,
    compare_accuracy,
)
