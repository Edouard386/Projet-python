"""Nettoyage des données de matchs."""

import pandas as pd


def load_matches(years=None, raw_dir=None, cache=True):    
    """Charge et concatène les fichiers CSV (local ou GitHub).
    """
    BASE_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/"

    dfs = []
    for year in years or range(2000, 2026):
        cache_file = raw_dir / f"{year}.csv" if raw_dir else None      

        # Utiliser le cache si disponible
        if cache and cache_file and cache_file.exists():
            dfs.append(pd.read_csv(cache_file))
            continue

        # Télécharger depuis GitHub
        url = f"{BASE_URL}{year}.csv"
        try:
            df = pd.read_csv(url)
            if cache and cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)   
                df.to_csv(cache_file, index=False)
            dfs.append(df)
        except Exception:
            continue

    return pd.concat(dfs, ignore_index=True)


def remove_incomplete_matches(df):
    """Supprime walkovers, abandons et matchs sans surface."""
    mask = df["score"].str.contains("W/O|RET|DEF", na=True, regex=True)
    df = df[~mask].copy()
    df = df.dropna(subset=["surface"])
    return df.reset_index(drop=True)


def convert_date(df):
    """Convertit tourney_date en datetime."""
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")
    return df


def handle_missing_ranks(df):
    """Impute les rangs manquants avec max_rank + 50."""
    max_rank = df[["winner_rank", "loser_rank"]].max().max()
    fill_value = max_rank + 50
    df["winner_rank"] = df["winner_rank"].fillna(fill_value)
    df["loser_rank"] = df["loser_rank"].fillna(fill_value)
    df["winner_rank_points"] = df["winner_rank_points"].fillna(0)
    df["loser_rank_points"] = df["loser_rank_points"].fillna(0)
    return df


def handle_missing_physical(df):
    """Impute taille et âge manquants avec la médiane."""
    df["winner_ht"] = df["winner_ht"].fillna(df["winner_ht"].median())
    df["loser_ht"] = df["loser_ht"].fillna(df["loser_ht"].median())
    df["winner_age"] = df["winner_age"].fillna(df["winner_age"].median())
    df["loser_age"] = df["loser_age"].fillna(df["loser_age"].median())
    return df


def filter_rounds(df):
    """Garde uniquement les rounds significatifs pour éviter la colinéarité."""
    valid_rounds = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'RR']
    df = df[df["round"].isin(valid_rounds)]
    return df.reset_index(drop=True)


def clean_matches(df):
    """Pipeline de nettoyage complet."""
    df = remove_incomplete_matches(df)
    df = convert_date(df)
    df = filter_rounds(df)
    df = handle_missing_ranks(df)
    df = handle_missing_physical(df)
    return df
