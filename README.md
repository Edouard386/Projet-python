# ATP Tennis Match Prediction

A machine learning pipeline for predicting ATP tennis match outcomes using historical data, benchmarked against bookmaker odds.

## Overview

This project predicts the winner of ATP tennis matches using player statistics, rankings, and historical performance data. The model is trained on 25 years of match data (2000-2025) from the Jeff Sackmann tennis database and incorporates time-decayed features to weight recent performance more heavily.

## Key Features

- Historical data processing for 25 years of ATP matches
- Time-decayed statistics using exponential weighting
- Player-specific features: rankings, serve stats, surface performance
- Head-to-head record calculation between players
- Support for multiple models: Logistic Regression, Random Forest, XGBoost, AutoML

## Work in Progress

- Benchmarking model predictions against bookmaker odds to evaluate profitability
- Improving prediction accuracy through feature engineering and model tuning
- Completing dataset statistics visualization and exploratory analysis

## Project Structure

```
├── src/
│   ├── config.py                 # Paths and parameters
│   ├── preprocessing/
│   │   ├── cleaning.py           # Data loading and cleaning
│   │   ├── features.py           # Feature engineering with decay
│   │   └── pipeline.py           # Preprocessing orchestration
│   └── training/
│       ├── models.py             # Model definitions
│       └── train.py              # Training loop, GridSearchCV, AutoML
├── data/
│   ├── raw/                      # ATP match CSVs (2000-2025)
│   ├── processed/                # Train/test parquets, preprocessor
│   └── predictions/              # Model outputs
├── models/                       # Saved models (.pkl)
├── notebooks/                    # Exploration notebooks
├── main.py                       # Entry point
└── predictions.py                # Prediction script for upcoming matches
```

## Methods

### Feature Engineering
- Player rankings and ranking points difference
- Physical attributes: age, height, handedness
- Historical win rate (last N matches with exponential decay)
- Surface-specific win rate (clay, hard, grass)
- Head-to-head record between players
- Serve statistics: ace rate, double fault rate, first serve percentage, break point save rate
- Match context: tournament level, round, best of 3/5

### Time Decay
Statistics are weighted using exponential decay with configurable half-life (default: 180 days). Recent matches have more influence on predictions than older ones.

### Models
- Logistic Regression
- Random Forest
- XGBoost
- FLAML AutoML (automatic model selection)

### Fine tuning
- GridsearchCV

### Evaluation
- Accuracy, Log Loss, ROC AUC
- Baseline comparison: "higher-ranked player wins"
- Bookmaker odds comparison (in progress)

## Technologies

- Python
- Scikit-learn
- XGBoost / LightGBM
- FLAML (AutoML)
- Pandas / NumPy