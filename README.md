# Prédiction des matchs de tennis ATP

**Projet Python ENSAE**  
*Auteurs : Dejean William, Sala Satorre Daniel, Auvepre Edouard*

Le notebook final se trouve dans `notebooks/rendu_final.ipynb`.

## Table des matières

1. [Sujet et problématique](#sujet-et-problématique)
2. [Base de données](#base-de-données)
3. [Modélisation](#modélisation)
4. [Comparaison avec les bookmakers](#comparaison-avec-les-bookmakers)
5. [Résultats principaux](#résultats-principaux)
6. [Structure du projet](#structure-du-projet)
7. [Installation et reproductibilité](#installation-et-reproductibilité)

## Sujet et problématique

L'objectif de ce projet est de **prédire le vainqueur d'un match de tennis** en utilisant l'historique des données du circuit ATP (2000-2025).

Contrairement aux approches "boîte noire", nous avons privilégié une **approche statistique interprétable** basée sur la régression logistique. Cela nous permet de quantifier précisément l'impact de chaque variable (classement, Elo, forme récente, surface) sur l'issue d'un match.

L'objectif final est de confronter notre modèle aux cotes des bookmakers pour évaluer sa pertinence.

## Base de données

Les données proviennent du repository [TML-Database](https://github.com/Tennismylife/TML-Database).

- **Volume :** ~78 000 matchs (2000-2025)
- **Nettoyage :** Suppression des matchs incomplets (abandons, forfaits) représentant ~3.8% du dataset

### Feature Engineering

Pour éviter le data leakage, nous avons :
1. **Randomisé** l'attribution des joueurs en `Player A` et `Player B`
2. Calculé les features uniquement avec les données **antérieures** à chaque match

**Features créées :**
- **Classement :** Rang ATP, Points ATP, Ratio de classement
- **Elo :** Score Elo global et par surface (mis à jour dynamiquement)
- **Forme récente :** Win rate, stats de service avec decay temporel (demi-vie 180 jours)
- **Contexte :** Surface win rate, Head-to-Head, niveau du tournoi
- **Physique :** Âge, taille, main dominante

## Modélisation

Nous avons utilisé une **régression logistique** via `statsmodels` pour obtenir les p-values et interpréter les coefficients.

**Variables significatives :** Elo, classement (rank, points), win rate, surface win rate, H2H, stats de service, âge.

**Variables non significatives :** Taille, main dominante, tour du tournoi.

## Comparaison avec les bookmakers

Nous avons comparé nos prédictions aux cotes Bet365 sur les matchs de **2025** (316 matchs matchés).

| Métrique | Modèle | Bookmakers |
|----------|--------|------------|
| Accuracy | ~67% | ~72% |
| ROI (tous les paris) | ~-1.5% | - |

**Conclusion :** Notre modèle a une accuracy inférieure aux bookmakers. Le ROI négatif confirme qu'il est difficile de battre le marché avec des données publiques uniquement.

## Résultats principaux

| Modèle | Accuracy | ROC AUC |
|--------|----------|---------|
| Baseline (meilleur classement) | 65.8% | - |
| **Régression Logistique** | **67.7%** | **0.7465** |

- Le modèle bat la baseline de +1.9 points
- L'**Elo** et le **classement** sont les meilleurs prédicteurs
- Les variables physiques (taille, main) ne sont pas significatives

## Structure du projet
```
├── config.py                     # Paramètres et chemins
├── src/
│   ├── preprocessing/
│   │   ├── cleaning.py           # Chargement et nettoyage
│   │   ├── features.py           # Feature engineering
│   │   └── pipeline.py           # Classe TennisPreprocessor
│   ├── training/
│   │   └── models.py             # StatsLogitClassifier
│   └── evaluation/
│       ├── bookmakers.py         # Chargement des cotes
│       └── comparison.py         # Calculs ROI, value bets
├── notebooks/
│   └── rendu_final.ipynb         # Notebook principal
├── data/
│   └── raw/                      # CSVs (téléchargés automatiquement)
└── requirements.txt
```

## Installation et reproductibilité
```bash
# Cloner le projet
git clone https://github.com/Edouard386/Projet-python.git

# Se placer dans le dossier
cd Projet-python

# Installer les dépendances
pip install -r requirements.txt
```

Puis exécuter `notebooks/rendu_final.ipynb`. Les données sont téléchargées automatiquement au premier lancement.

Puis exécuter `notebooks/rendu_final.ipynb`. Les données sont téléchargées automatiquement au premier lancement.