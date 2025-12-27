"""Chargement des cotes bookmakers depuis tennis-data.co.uk."""

import pandas as pd


def load_bookmaker_odds(start_year=2025, end_year=2025, bookmaker='B365'):
    """Charge les cotes depuis tennis-data.co.uk avec les classements."""
    dfs = []
    for year in range(start_year, end_year + 1):
        url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
        dfs.append(pd.read_excel(url))

    data = pd.concat(dfs, ignore_index=True)

    cols = ['Date', 'WRank', 'LRank', f'{bookmaker}W', f'{bookmaker}L']
    result = data[cols].copy()
    result.columns = ['tourney_date', 'winner_rank', 'loser_rank', 'odds_winner', 'odds_loser']
    result['tourney_date'] = pd.to_datetime(result['tourney_date'])

    return result.dropna()


def merge_odds_with_features(meta_test, odds_df, y_test):
    """Fusionne les cotes avec les métadonnées via les classements."""
    meta = meta_test.reset_index(drop=True).copy()
    meta['_idx'] = meta.index
    meta['tourney_date'] = pd.to_datetime(meta['tourney_date'])
    odds_df['tourney_date'] = pd.to_datetime(odds_df['tourney_date'])

    # Merge 1: player_a = winner (rank_a = winner_rank)
    m1 = meta.merge(odds_df, left_on=['tourney_date', 'rank_a', 'rank_b'],
                    right_on=['tourney_date', 'winner_rank', 'loser_rank'], how='inner')
    m1['odds_a'], m1['odds_b'] = m1['odds_winner'], m1['odds_loser']

    # Merge 2: player_a = loser (rank_a = loser_rank)
    m2 = meta.merge(odds_df, left_on=['tourney_date', 'rank_a', 'rank_b'],
                    right_on=['tourney_date', 'loser_rank', 'winner_rank'], how='inner')
    m2['odds_a'], m2['odds_b'] = m2['odds_loser'], m2['odds_winner']

    merged = pd.concat([m1, m2]).drop_duplicates(subset=['_idx']).sort_values('_idx')
    idx = merged['_idx'].tolist()

    result = merged[['tourney_date', 'rank_a', 'rank_b', 'odds_a', 'odds_b']].reset_index(drop=True)
    y_matched = y_test.iloc[idx].reset_index(drop=True)

    return result, y_matched, idx
