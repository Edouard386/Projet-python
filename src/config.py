"""Configuration du projet."""

from pathlib import Path

# Chemins
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Paramètres
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_HISTORICAL_MATCHES = 10
N_SURFACE_MATCHES = 20
HALF_LIFE_DAYS = 180

# Training
MODEL_NAME = "LinearStats"  # "LogisticRegression", "RandomForest", "XGBoost", "AutoML", "LinearStats"
AUTOML_TIME_BUDGET = 300  # secondes (si MODEL_NAME = "AutoML")
SKIP_PREPROCESSING = False  # True pour charger directement les parquets
USE_FEATURE_SELECTION = True  # True = RFECV pour LinearStats

# Colonnes
PLAYER_COLS = ["id", "name", "hand", "ht", "age", "rank", "rank_points"]
STAT_COLS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "bpSaved", "bpFaced"]
