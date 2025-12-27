"""Fonctions de comparaison entre le modèle et les bookmakers."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

def evaluate_model(model, X, y):
    """
    Évalue un modèle.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    return {
        "accuracy": accuracy_score(y, y_pred),
        "log_loss": log_loss(y, y_proba),
        "roc_auc": roc_auc_score(y, y_proba)
    }


def proba_to_odds(proba):
    """
    Convertit une probabilité en cote décimale.

    """
    proba = np.asarray(proba)
    # Éviter division par zéro
    proba = np.clip(proba, 0.01, 0.99)
    return 1 / proba


def odds_to_proba(odds):
    """
    Convertit une cote décimale en probabilité.
    """
    odds = np.asarray(odds)
    return 1 / odds


def remove_margin(odds_a, odds_b):
    """
    Supprime la marge du bookmaker pour obtenir les probabilités réelles.

    Les bookmakers ajoutent une marge (ex: probas totales = 105% au lieu de 100%).
    Cette fonction normalise pour revenir à 100%.

    """
    odds_a = np.asarray(odds_a)
    odds_b = np.asarray(odds_b)

    # Probabilités implicites (avec marge)
    implied_a = 1 / odds_a
    implied_b = 1 / odds_b

    # Total > 1 à cause de la marge
    total = implied_a + implied_b

    # Normaliser pour avoir total = 1
    proba_a = implied_a / total
    proba_b = implied_b / total

    return proba_a, proba_b


def find_value_bets(model_proba, bookmaker_odds, threshold=0.0):
    """
    Identifie les value bets : matchs où le modèle estime une proba plus haute
    que celle implicite du bookmaker.

    Value bet = cote_bookmaker > cote_modele (on est payé plus que le risque réel)
    """
    model_proba = np.asarray(model_proba)
    bookmaker_odds = np.asarray(bookmaker_odds)

    # Cote juste selon notre modèle
    model_odds = proba_to_odds(model_proba)

    # Expected Value = (proba * cote) - 1
    # Si EV > 0, c'est un value bet
    expected_value = (model_proba * bookmaker_odds) - 1

    # Edge = différence entre proba modèle et proba bookmaker
    bookmaker_proba = odds_to_proba(bookmaker_odds)
    edge = model_proba - bookmaker_proba

    df = pd.DataFrame({
        'model_proba': model_proba,
        'model_odds': model_odds,
        'bookmaker_odds': bookmaker_odds,
        'bookmaker_proba': bookmaker_proba,
        'edge': edge,
        'expected_value': expected_value,
        'is_value_bet': expected_value > threshold
    })

    return df


def calculate_roi(predictions, actual_results, bookmaker_odds, stake=1.0):
    """
    Calcule le ROI si on avait parié selon les prédictions du modèle.

    Stratégie : on parie sur le joueur que le modèle prédit gagnant.
    """
    predictions = np.asarray(predictions)
    actual_results = np.asarray(actual_results)
    bookmaker_odds = np.asarray(bookmaker_odds)

    n_bets = len(predictions)
    total_stake = n_bets * stake

    # Paris gagnés = prédiction correcte
    wins = predictions == actual_results
    n_wins = wins.sum()
    n_losses = n_bets - n_wins

    # Gains = somme des (mise * cote) pour les paris gagnés
    gains = (wins * bookmaker_odds * stake).sum()

    profit = gains - total_stake
    roi = profit / total_stake if total_stake > 0 else 0

    return {
        'n_bets': n_bets,
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': n_wins / n_bets if n_bets > 0 else 0,
        'total_stake': total_stake,
        'total_gains': gains,
        'profit': profit,
        'roi': roi
    }


def calculate_roi_value_bets(model_proba, actual_results, bookmaker_odds,
                              threshold=0.0, stake=1.0):
    """
    Calcule le ROI en ne pariant QUE sur les value bets.
    """
    model_proba = np.asarray(model_proba)
    actual_results = np.asarray(actual_results)
    bookmaker_odds = np.asarray(bookmaker_odds)

    # Identifier les value bets
    expected_value = (model_proba * bookmaker_odds) - 1
    value_bet_mask = expected_value > threshold

    if value_bet_mask.sum() == 0:
        return {
            'n_bets': 0,
            'n_wins': 0,
            'n_losses': 0,
            'win_rate': 0,
            'total_stake': 0,
            'total_gains': 0,
            'profit': 0,
            'roi': 0,
            'avg_expected_value': 0
        }

    # Filtrer sur les value bets uniquement
    vb_results = actual_results[value_bet_mask]
    vb_odds = bookmaker_odds[value_bet_mask]
    vb_ev = expected_value[value_bet_mask]

    n_bets = len(vb_results)
    total_stake = n_bets * stake

    # On parie que le joueur gagne (target=1), donc on gagne si actual=1
    wins = vb_results == 1
    n_wins = wins.sum()

    gains = (wins * vb_odds * stake).sum()
    profit = gains - total_stake
    roi = profit / total_stake if total_stake > 0 else 0

    return {
        'n_bets': n_bets,
        'n_wins': n_wins,
        'n_losses': n_bets - n_wins,
        'win_rate': n_wins / n_bets if n_bets > 0 else 0,
        'total_stake': total_stake,
        'total_gains': gains,
        'profit': profit,
        'roi': roi,
        'avg_expected_value': vb_ev.mean()
    }


def compare_accuracy(model_proba, bookmaker_odds_a, bookmaker_odds_b, actual_results):
    """
    Compare l'accuracy du modèle vs celle des bookmakers.

    Chacun prédit le gagnant = celui avec la plus haute probabilité.
    """
    model_proba = np.asarray(model_proba)
    bookmaker_odds_a = np.asarray(bookmaker_odds_a)
    bookmaker_odds_b = np.asarray(bookmaker_odds_b)
    actual_results = np.asarray(actual_results)

    # Prédiction modèle : player_a gagne si proba > 0.5
    model_pred = (model_proba > 0.5).astype(int)

    # Prédiction bookmaker : player_a gagne si sa cote est plus basse (= favori)
    bookmaker_pred = (bookmaker_odds_a < bookmaker_odds_b).astype(int)

    # Probas implicites du bookmaker (sans marge)
    bk_proba_a, bk_proba_b = remove_margin(bookmaker_odds_a, bookmaker_odds_b)

    # Accuracy
    model_correct = model_pred == actual_results
    bookmaker_correct = bookmaker_pred == actual_results

    model_accuracy = model_correct.mean()
    bookmaker_accuracy = bookmaker_correct.mean()

    # Matchs où les deux divergent
    disagree_mask = model_pred != bookmaker_pred
    n_disagree = disagree_mask.sum()

    if n_disagree > 0:
        # Quand ils divergent, qui a raison ?
        model_wins_when_disagree = model_correct[disagree_mask].sum()
        bookmaker_wins_when_disagree = bookmaker_correct[disagree_mask].sum()
    else:
        model_wins_when_disagree = 0
        bookmaker_wins_when_disagree = 0

    return {
        'model_accuracy': model_accuracy,
        'bookmaker_accuracy': bookmaker_accuracy,
        'accuracy_diff': model_accuracy - bookmaker_accuracy,
        'n_matches': len(actual_results),
        'n_disagree': n_disagree,
        'model_wins_when_disagree': model_wins_when_disagree,
        'bookmaker_wins_when_disagree': bookmaker_wins_when_disagree,
        'model_proba_mean': model_proba.mean(),
        'bookmaker_proba_a_mean': bk_proba_a.mean()
    }


def calculate_roi_evolution(predictions, actual_results, bookmaker_odds, stake=1.0):
    """
    Calcule l'évolution du ROI au fil des paris.

    Retourne un DataFrame avec le profit et ROI cumulés après chaque pari.
    """
    predictions = np.asarray(predictions)
    actual_results = np.asarray(actual_results)
    bookmaker_odds = np.asarray(bookmaker_odds)

    n_bets = len(predictions)
    cumul_stake = np.arange(1, n_bets + 1) * stake

    # Gains par pari : mise * cote si gagné, 0 sinon
    wins = predictions == actual_results
    gains_per_bet = wins * bookmaker_odds * stake

    # Cumulés
    cumul_gains = np.cumsum(gains_per_bet)
    cumul_profit = cumul_gains - cumul_stake
    cumul_roi = cumul_profit / cumul_stake * 100

    return pd.DataFrame({
        'n_bets': np.arange(1, n_bets + 1),
        'cumul_stake': cumul_stake,
        'cumul_gains': cumul_gains,
        'cumul_profit': cumul_profit,
        'cumul_roi': cumul_roi
    })


def analyze_threshold_impact(model_proba, actual_results, bookmaker_odds,
                              thresholds=None, stake=1.0):
    """
    Analyse l'impact du seuil de value bet sur le ROI.

    Retourne un DataFrame avec les métriques pour chaque seuil.
    """
    if thresholds is None:
        thresholds = np.arange(0, 0.30, 0.02)

    results = []
    for thresh in thresholds:
        roi = calculate_roi_value_bets(
            model_proba=model_proba,
            actual_results=actual_results,
            bookmaker_odds=bookmaker_odds,
            threshold=thresh,
            stake=stake
        )
        results.append({
            'threshold': thresh,
            'threshold_pct': thresh * 100,
            'n_bets': roi['n_bets'],
            'n_wins': roi['n_wins'],
            'win_rate': roi['win_rate'] * 100,
            'profit': roi['profit'],
            'roi': roi['roi'] * 100
        })

    return pd.DataFrame(results)
