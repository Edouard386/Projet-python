"""Feature engineering pour les matchs de tennis (version optimise)."""

import pandas as pd
import numpy as np
from collections import defaultdict


def _compute_decay_weights(dates, match_date, half_life_days=None):
    """
    Cree des poids decroissants avec le temps (decay exponentiel).
    Si half_life_days est None ou <= 0, poids uniformes.
    """
    if half_life_days is None or half_life_days <= 0:
        return np.ones(len(dates)) / len(dates)

    deltas = (pd.to_datetime(match_date) - pd.to_datetime(dates)).days
    deltas = np.asarray(deltas, dtype=float)
    deltas = np.clip(deltas, a_min=0, a_max=None)
    weights = np.exp(-deltas / half_life_days)
    total = weights.sum()
    return weights / total if total > 0 else np.ones(len(dates)) / len(dates)


def _weighted_sum(recent, weights, key):
    """Somme ponderee en ignorant les NaN."""
    total = 0.0
    weight_total = 0.0
    for m, w in zip(recent, weights):
        val = m.get(key, np.nan)
        if pd.notna(val):
            total += w * val
            weight_total += w
    return total, weight_total


def build_player_history(df):
    """
    Precalcule l'historique de tous les joueurs.
    Retourne un dict: player_id -> list de matchs tries par date.
    """
    history = defaultdict(list)
    
    for _, row in df.iterrows():
        match_data = {
            "date": row["tourney_date"],
            "surface": row["surface"],
            "opponent_id": row["loser_id"],
            "won": 1,
            "ace": row.get("w_ace", np.nan),
            "df": row.get("w_df", np.nan),
            "svpt": row.get("w_svpt", np.nan),
            "1stIn": row.get("w_1stIn", np.nan),
            "1stWon": row.get("w_1stWon", np.nan),
            "2ndWon": row.get("w_2ndWon", np.nan),
            "bpSaved": row.get("w_bpSaved", np.nan),
            "bpFaced": row.get("w_bpFaced", np.nan),
        }
        history[row["winner_id"]].append(match_data)
        
        match_data_loser = {
            "date": row["tourney_date"],
            "surface": row["surface"],
            "opponent_id": row["winner_id"],
            "won": 0,
            "ace": row.get("l_ace", np.nan),
            "df": row.get("l_df", np.nan),
            "svpt": row.get("l_svpt", np.nan),
            "1stIn": row.get("l_1stIn", np.nan),
            "1stWon": row.get("l_1stWon", np.nan),
            "2ndWon": row.get("l_2ndWon", np.nan),
            "bpSaved": row.get("l_bpSaved", np.nan),
            "bpFaced": row.get("l_bpFaced", np.nan),
        }
        history[row["loser_id"]].append(match_data_loser)
    
    # Trier par date
    for player_id in history:
        history[player_id].sort(key=lambda x: x["date"])
    
    return history


def get_player_stats(history, player_id, match_date, n_matches, half_life_days=None):
    """Calcule les stats d'un joueur avant une date donnee avec un decay temporel."""
    if player_id not in history:
        return None
    
    past = [m for m in history[player_id] if m["date"] < match_date]
    if len(past) == 0:
        return None
    
    recent = past[-n_matches:]
    weights = _compute_decay_weights([m["date"] for m in recent], match_date, half_life_days=half_life_days)

    wins = float(np.dot(weights, [m["won"] for m in recent]))

    total_ace, _ = _weighted_sum(recent, weights, "ace")
    total_df, _ = _weighted_sum(recent, weights, "df")
    total_svpt, _ = _weighted_sum(recent, weights, "svpt")
    total_1stIn, _ = _weighted_sum(recent, weights, "1stIn")
    total_1stWon, _ = _weighted_sum(recent, weights, "1stWon")
    total_2ndWon, _ = _weighted_sum(recent, weights, "2ndWon")
    total_bpSaved, _ = _weighted_sum(recent, weights, "bpSaved")
    total_bpFaced, _ = _weighted_sum(recent, weights, "bpFaced")
    
    second_serve_attempts = total_svpt - total_1stIn
    
    return {
        "win_rate": wins,  # deja normalise car poids normalises
        "matches_played": len(recent),
        "ace_rate": total_ace / total_svpt if total_svpt > 0 else 0,
        "df_rate": total_df / total_svpt if total_svpt > 0 else 0,
        "first_serve_pct": total_1stIn / total_svpt if total_svpt > 0 else 0,
        "first_serve_won": total_1stWon / total_1stIn if total_1stIn > 0 else 0,
        "second_serve_won": total_2ndWon / second_serve_attempts if second_serve_attempts > 0 else 0,
        "bp_save_rate": total_bpSaved / total_bpFaced if total_bpFaced > 0 else 0,
    }


def get_surface_win_rate(history, player_id, match_date, surface, n_matches, half_life_days=None):
    """Calcule le win_rate sur une surface avec decay temporel."""
    if player_id not in history:
        return 0.5
    
    past = [m for m in history[player_id] if m["date"] < match_date and m["surface"] == surface]
    if len(past) == 0:
        return 0.5
    
    recent = past[-n_matches:]
    weights = _compute_decay_weights([m["date"] for m in recent], match_date, half_life_days=half_life_days)
    return float(np.dot(weights, [m["won"] for m in recent]))


def get_h2h(history, player_a, player_b, match_date, half_life_days=None):
    """Calcule le H2H entre deux joueurs avec decay temporel."""
    if player_a not in history:
        return 0.5
    
    h2h = [m for m in history[player_a] if m["date"] < match_date and m["opponent_id"] == player_b]
    if len(h2h) == 0:
        return 0.5
    
    weights = _compute_decay_weights([m["date"] for m in h2h], match_date, half_life_days=half_life_days)
    return float(np.dot(weights, [m["won"] for m in h2h]))


def get_default_stats(rank):
    """Stats par defaut basees sur le rang."""
    # Meilleurs joueurs ont de meilleures stats
    if rank <= 10:
        return {"win_rate": 0.7, "ace_rate": 0.08, "df_rate": 0.02, "first_serve_pct": 0.65,
                "first_serve_won": 0.75, "second_serve_won": 0.55, "bp_save_rate": 0.65, "matches_played": 0}
    elif rank <= 50:
        return {"win_rate": 0.55, "ace_rate": 0.06, "df_rate": 0.03, "first_serve_pct": 0.62,
                "first_serve_won": 0.72, "second_serve_won": 0.52, "bp_save_rate": 0.62, "matches_played": 0}
    elif rank <= 100:
        return {"win_rate": 0.50, "ace_rate": 0.05, "df_rate": 0.03, "first_serve_pct": 0.60,
                "first_serve_won": 0.70, "second_serve_won": 0.50, "bp_save_rate": 0.60, "matches_played": 0}
    else:
        return {"win_rate": 0.45, "ace_rate": 0.04, "df_rate": 0.04, "first_serve_pct": 0.58,
                "first_serve_won": 0.68, "second_serve_won": 0.48, "bp_save_rate": 0.58, "matches_played": 0}
    

def compute_elo_ratings(df, base_rating=1500, k=32, surface_factor=0.5):
    """
    Calcule les Elo globaux et par surface pour chaque match (valeur pré-match).
    Retourne un DataFrame avec colonnes winner_elo, loser_elo, winner_surface_elo, loser_surface_elo.
    """
    df_sorted = df.sort_values("tourney_date").reset_index()
    elo_global = defaultdict(lambda: base_rating)
    elo_surface = defaultdict(lambda: base_rating)

    winner_elo = {}
    loser_elo = {}
    winner_surface_elo = {}
    loser_surface_elo = {}

    for _, row in df_sorted.iterrows():
        idx = row["index"]
        w_id = row["winner_id"]
        l_id = row["loser_id"]
        surface = row["surface"]

        rw = elo_global[w_id]
        rl = elo_global[l_id]
        rws = elo_surface[(w_id, surface)]
        rls = elo_surface[(l_id, surface)]

        winner_elo[idx] = rw
        loser_elo[idx] = rl
        winner_surface_elo[idx] = rws
        loser_surface_elo[idx] = rls

        # Mise à jour Elo global
        expected_w = 1 / (1 + 10 ** ((rl - rw) / 400))
        expected_l = 1 - expected_w
        elo_global[w_id] = rw + k * (1 - expected_w)
        elo_global[l_id] = rl + k * (0 - expected_l)

        # Mise à jour Elo surface
        k_surf = k * surface_factor
        expected_w_s = 1 / (1 + 10 ** ((rls - rws) / 400))
        expected_l_s = 1 - expected_w_s
        elo_surface[(w_id, surface)] = rws + k_surf * (1 - expected_w_s)
        elo_surface[(l_id, surface)] = rls + k_surf * (0 - expected_l_s)

    elo_df = pd.DataFrame({
        "winner_elo": pd.Series(winner_elo),
        "loser_elo": pd.Series(loser_elo),
        "winner_surface_elo": pd.Series(winner_surface_elo),
        "loser_surface_elo": pd.Series(loser_surface_elo),
    })
    return elo_df


def create_features(df, history, n_hist, n_surf, rng, half_life_days=None):
    """Cree toutes les features pour un DataFrame de matchs avec decay temporel."""

    elo_df= compute_elo_ratings(df)
    df=pd.concat([df, elo_df], axis=1)
    features_list = []
    
    for _, row in df.iterrows():
        # Randomiser qui est player_a
        if rng.random() < 0.5:
            a_prefix, b_prefix = "winner", "loser"
            target = 1
        else:
            a_prefix, b_prefix = "loser", "winner"
            target = 0
        
        player_a_id = row[f"{a_prefix}_id"]
        player_b_id = row[f"{b_prefix}_id"]
        match_date = row["tourney_date"]
        surface = row["surface"]

        # Colonnes meta (pour identifier le match)
        f = {"tourney_date": match_date}

        # Features statiques
        f.update({
            "target": target,
            "rank_a": row[f"{a_prefix}_rank"],
            "rank_b": row[f"{b_prefix}_rank"],
            "points_a": row[f"{a_prefix}_rank_points"],
            "points_b": row[f"{b_prefix}_rank_points"],
            "elo_a": row[f"{a_prefix}_elo"],
            "elo_b": row[f"{b_prefix}_elo"],
            "elo_surface_a": row[f"{a_prefix}_surface_elo"],
            "elo_surface_b": row[f"{b_prefix}_surface_elo"],
            "age_a": row[f"{a_prefix}_age"],
            "age_b": row[f"{b_prefix}_age"],
            "height_a": row[f"{a_prefix}_ht"],
            "height_b": row[f"{b_prefix}_ht"],
            "is_left_a": 1 if row[f"{a_prefix}_hand"] == "L" else 0,
            "is_left_b": 1 if row[f"{b_prefix}_hand"] == "L" else 0,
            "surface": surface,
            "tourney_level": row["tourney_level"],
            "best_of_5": 1 if row["best_of"] == 5 else 0,
            "round": row["round"],
        })

        f["rank_ratio"] = f["rank_a"] / f["rank_b"] if f["rank_b"] > 0 else 1
        
        # Stats historiques joueur A
        stats_a = get_player_stats(history, player_a_id, match_date, n_hist, half_life_days=half_life_days)
        if stats_a is None:
            stats_a = get_default_stats(f["rank_a"])
        for k, v in stats_a.items():
            f[f"{k}_a"] = v
        
        # Stats historiques joueur B
        stats_b = get_player_stats(history, player_b_id, match_date, n_hist, half_life_days=half_life_days)
        if stats_b is None:
            stats_b = get_default_stats(f["rank_b"])
        for k, v in stats_b.items():
            f[f"{k}_b"] = v
        
        # Surface win rate
        f["surface_win_rate_a"] = get_surface_win_rate(
            history, player_a_id, match_date, surface, n_surf, half_life_days=half_life_days
        )
        f["surface_win_rate_b"] = get_surface_win_rate(
            history, player_b_id, match_date, surface, n_surf, half_life_days=half_life_days
        )
        
        # H2H
        f["h2h_win_rate_a"] = get_h2h(history, player_a_id, player_b_id, match_date, half_life_days=half_life_days)
        
        
        features_list.append(f)
    
    return pd.DataFrame(features_list)
