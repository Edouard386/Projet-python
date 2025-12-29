# Prédiction des matchs de tennis ATP

**Projet Python ENSAE**
*Auteurs : Dejean William, Sala Satorre Daniel, Auvepre Edouard*

Le notebook final se trouve dans *notebooks/rendu_final*.

## Table des matières

 1. [Sujet et problématique](#subheading-1)
 2. [Bases de données](#subheading-2)
 3. [Modélisation](#subheading-3)
 4. [Comparaison avec les bookmakers](#subheading-4)
 5. [Résultats principaux et conclusions](#subheading-5)
 6. [Structure du projet](#subheading-6)


 ## Sujet et problématique <a name="subheading-1">

L'objectif de ce projet est de **prédire le vainqueur d'un match de tennis** en utilisant l'historique des données du circuit ATP (2000-2025). Pour cela nous avons obtenu un historique fiable et exhaustif sur TML-Database. 

Contrairement aux approches "boîte noire", nous avons privilégié une **approche statistique interprétable** basée sur la régression logistique. Cela nous permet de quantifier précisément l'impact de chaque variable (classement, forme du moment, surface, historique) sur l'issue d'un match.

L'objectif final est de confronter notre modèle aux cotes des bookmakers pour évaluer sa capacité à générer un retour sur investissement (ROI).

 ## Base de données <a name="subheading-2">

Les données proviennent du repository [TML-Database](https://github.com/Tennismylife/TML-Database).
*   **Volume :** ~78 000 matchs (2000-2025).
*   **Nettoyage :** Suppression des matchs incomplets (Abandons, Forfaits) représentant ~3.8% du dataset.

Nous disposions d'environ une vingtaine de variables avant de procéder au features engineering et au preprocessing.
*   Pour éviter le biais de "target leakage" (où le gagnant est identifié par sa position dans le dataset), nous avons randomisé l'attribution des joueurs en `Player A` et `Player B`.
Nous avons généré des features basées uniquement sur l'information disponible **avant** le match :
*   **Classement :** Rang ATP, Points ATP, Ratio de classement.
*   **Elo :** Score Elo global et Score Elo spécifique à la surface (calculés dynamiquement avec un facteur K=32).
*   **Forme (Decay Temporel) :** Statistiques (Win Rate, Aces, Double Fautes, Break Points) pondérées par un decay exponentiel (demi-vie de 180 jours) pour privilégier les performances récentes.
*   **Contexte :** Surface (Dur, Terre battue, Gazon), Niveau du tournoi, Head-to-Head.
*   **Physique :** Âge, Taille, Main dominante.

 ## Modélisation <a name="subheading-3">
Avant de procéder à la modélisation, nous avons analysé les features crées afin d'étudier leur colinéarité, et pour pouvoir vérifier la cohérence du modèle ultérieurement. Comme l'on pouvait facilement supposer, le classement du joueur est déjà une bonne indication des probabilités de victoire et nous nous attendons à ce que certaines features soient non significatives. 

## Le modèle 

## Comparaison avec les bookmakers <a name="subheading-4">
Nous avons extrait les cotes des bookmakers de Bet365 sur l'année 2025 et avons fait matché ces cotes pour 316 matchs de notre dataset initial afin de les comparer avec les probabilités de victoires issues de nos prédictions. Cela en splittant nos données pour tester le modèle sur 2026.

*   **Résultat :** Nous avons une Accuracy plus faible que celle des probabilités issues des cotes de bookmakers, mais en pariant suivant nos prédictions nous parvenons à obtenir un retour sur investissement d'environ 26% si l'on parie sur tous les matchs.

 ## Résultats principaux et conclusions <a name="subheading-5">
### Baseline vs Modèle
Nous avons comparé notre modèle à une baseline naïve (le joueur le mieux classé gagne toujours).

| Modèle | Accuracy (Test Set) | ROC AUC |
| :--- | :---: | :---: |
| **Baseline (Rang ATP)** | 65.8% | - |
| **Régression Logistique** | **67.7%** | **0.7465** |

Ainsi, notre modèle est parvenu à battre la baseline du classement mais reste moins bon que celui que l'on peut déduire des cotes des bookmakers. Nous avons aussi pu identifier la significativité des différentes variables et confirmer le rôle majeur du classement ou encore de la surface dans l'issue du match. 

 ## Structure du projet <a name="subheading-6">

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

