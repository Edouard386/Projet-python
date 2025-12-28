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

    implied_a = 1 / odds_a
    implied_b = 1 / odds_b

    total = implied_a + implied_b

    proba_a = implied_a / total
    proba_b = implied_b / total

    return proba_a, proba_b


def find_value_bets(model_proba, odds_a, odds_b, threshold=0.0):
    """
    Identifie les value bets sur A et sur B.

    Value bet = Expected Value > threshold
    - EV_a = proba(A) * odds_a - 1
    - EV_b = proba(B) * odds_b - 1 = (1 - proba(A)) * odds_b - 1
    """
    model_proba = np.asarray(model_proba)
    odds_a = np.asarray(odds_a)
    odds_b = np.asarray(odds_b)

    proba_a = model_proba
    proba_b = 1 - model_proba

    # Expected Value pour chaque côté
    ev_a = proba_a * odds_a - 1
    ev_b = proba_b * odds_b - 1

    # Probabilités implicites du bookmaker
    bookmaker_proba_a = odds_to_proba(odds_a)
    bookmaker_proba_b = odds_to_proba(odds_b)

    df = pd.DataFrame({
        'model_proba_a': proba_a,
        'model_proba_b': proba_b,
        'odds_a': odds_a,
        'odds_b': odds_b,
        'bookmaker_proba_a': bookmaker_proba_a,
        'bookmaker_proba_b': bookmaker_proba_b,
        'ev_a': ev_a,
        'ev_b': ev_b,
        'is_value_bet_a': ev_a > threshold,
        'is_value_bet_b': ev_b > threshold,
        'is_value_bet': (ev_a > threshold) | (ev_b > threshold)
    })

    return df


def calculate_roi(predictions, actual_results, odds_a, odds_b, stake=1.0):
    """
    Calcule le ROI si on avait parié selon les prédictions du modèle.

    Stratégie : on parie sur le joueur que le modèle prédit gagnant.
    - Si pred=1, on parie sur A avec odds_a
    - Si pred=0, on parie sur B avec odds_b
    """
    predictions = np.asarray(predictions)
    actual_results = np.asarray(actual_results)
    odds_a = np.asarray(odds_a)
    odds_b = np.asarray(odds_b)

    n_bets = len(predictions)
    total_stake = n_bets * stake

    # Cote sur laquelle on parie selon notre prédiction
    bet_odds = np.where(predictions == 1, odds_a, odds_b)

    # Paris gagnés = prédiction correcte
    wins = predictions == actual_results
    n_wins = wins.sum()
    n_losses = n_bets - n_wins

    # Gains = somme des (mise * cote) pour les paris gagnés
    gains = (wins * bet_odds * stake).sum()

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


def calculate_roi_value_bets(model_proba, actual_results, odds_a, odds_b,
                              threshold=0.0, stake=1.0):
    """
    Calcule le ROI en ne pariant QUE sur les value bets (sur A ou B).
    
    Un value bet sur A est identifié quand EV_a = proba(A) * odds_a - 1 > threshold
    Un value bet sur B est identifié quand EV_b = proba(B) * odds_b - 1 > threshold
    """
    model_proba = np.asarray(model_proba)
    actual_results = np.asarray(actual_results)
    odds_a = np.asarray(odds_a)
    odds_b = np.asarray(odds_b)

    proba_a = model_proba
    proba_b = 1 - model_proba

    # Expected value pour parier sur A ou sur B
    ev_a = proba_a * odds_a - 1
    ev_b = proba_b * odds_b - 1

    # Value bet sur A : EV_a > threshold
    vb_a_mask = ev_a > threshold
    # Value bet sur B : EV_b > threshold
    vb_b_mask = ev_b > threshold

    # Éviter de parier sur les deux côtés du même match
    # Si les deux sont value bets, on prend celui avec la meilleure EV
    both_mask = vb_a_mask & vb_b_mask
    vb_a_mask = vb_a_mask & ~(both_mask & (ev_b > ev_a))
    vb_b_mask = vb_b_mask & ~(both_mask & (ev_a >= ev_b))

    # Parier sur A : on gagne si actual == 1
    wins_a = (actual_results == 1) & vb_a_mask
    gains_a = (wins_a * odds_a * stake).sum()
    n_bets_a = vb_a_mask.sum()

    # Parier sur B : on gagne si actual == 0
    wins_b = (actual_results == 0) & vb_b_mask
    gains_b = (wins_b * odds_b * stake).sum()
    n_bets_b = vb_b_mask.sum()

    # Total
    n_bets = n_bets_a + n_bets_b
    total_stake = n_bets * stake
    gains = gains_a + gains_b
    n_wins = wins_a.sum() + wins_b.sum()

    profit = gains - total_stake
    roi = profit / total_stake if total_stake > 0 else 0

    # EV moyenne sur les value bets sélectionnés
    all_ev = np.concatenate([ev_a[vb_a_mask], ev_b[vb_b_mask]]) if n_bets > 0 else np.array([0])

    return {
        'n_bets': int(n_bets),
        'n_bets_on_a': int(n_bets_a),
        'n_bets_on_b': int(n_bets_b),
        'n_wins': int(n_wins),
        'n_losses': int(n_bets - n_wins),
        'win_rate': n_wins / n_bets if n_bets > 0 else 0,
        'total_stake': total_stake,
        'total_gains': gains,
        'profit': profit,
        'roi': roi,
        'avg_expected_value': all_ev.mean() if len(all_ev) > 0 else 0
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


def calculate_roi_evolution(predictions, actual_results, odds_a, odds_b, stake=1.0):
    """
    Calcule l'évolution du ROI au fil des paris.

    Retourne un DataFrame avec le profit et ROI cumulés après chaque pari.
    """
    predictions = np.asarray(predictions)
    actual_results = np.asarray(actual_results)
    odds_a = np.asarray(odds_a)
    odds_b = np.asarray(odds_b)

    n_bets = len(predictions)
    cumul_stake = np.arange(1, n_bets + 1) * stake

    # Cote sur laquelle on parie selon notre prédiction
    bet_odds = np.where(predictions == 1, odds_a, odds_b)

    # Gains par pari : mise * cote si gagné, 0 sinon
    wins = predictions == actual_results
    gains_per_bet = wins * bet_odds * stake

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


def analyze_threshold_impact(model_proba, actual_results, odds_a, odds_b,
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
            odds_a=odds_a,
            odds_b=odds_b,
            threshold=thresh,
            stake=stake
        )
        results.append({
            'threshold': thresh,
            'threshold_pct': thresh * 100,
            'n_bets': roi['n_bets'],
            'n_bets_on_a': roi['n_bets_on_a'],
            'n_bets_on_b': roi['n_bets_on_b'],
            'n_wins': roi['n_wins'],
            'win_rate': roi['win_rate'] * 100,
            'profit': roi['profit'],
            'roi': roi['roi'] * 100
        })

    return pd.DataFrame(results)