# 🎾 Projet Tennis : Analyse et Prédiction de Matchs ATP/WTA

**Projet ENSAE - 2ème année**  
**Durée**: 2 mois | **Équipe**: DEJEAN William, AUVEPRE Édouard, SALA SATORRE Daniel

Objectif : Calculer des probabilités de victoire de matchs de tennis et les comparer à des cotes de bookmakers
-->Si temps, créer un portefeuille et parier sur des matchs à venir à partir de nos cotes et observer les gains. 

---

## 🎯 Idée Générale

Créer un modèle qui prédit le vainqueur d'un match de tennis en utilisant Python et machine learning.
**Pourquoi le tennis ?**
- ✅ Données gratuites et complètes (50 ans d'historique)
- ✅ Résultats binaires simples (victoire/défaite)
- ✅ On peut comparer nos prédictions aux bookmakers
- ✅ Plein de variables intéressantes (classement, surface, forme)

---

## 📊 Les Données 

### Source Principale : Jeff Sackmann (GitHub)
**Repository**: `github.com/JeffSackmann/tennis_atp` (et `tennis_wta`)

**Ce qu'on a** :
- Tous les matchs ATP depuis 1968 (format CSV)
- Statistiques détaillées depuis 1991 (aces, doubles fautes, % service, etc.)
- Classements des joueurs par semaine
- Infos joueurs (âge, taille, main, pays)

**Licence** : CC BY-NC-SA 4.0 

### Source Complémentaire : Tennis-Data.co.uk
**Site**: `http://www.tennis-data.co.uk/`

**Ce qu'on a** :
- Résultats de matchs avec cotes de bookmakers (2000-2024)
- ATP et WTA
- Plusieurs bookmakers par match
- Téléchargement direct CSV/Excel

**Pourquoi c'est important ?** Les cotes des bookmakers = prédictions d'experts. On pourra comparer notre modèle aux pros du pari !

Données :
-CSV  Jeff Sackman
- API Rapid API
- Web scrapping : tennis abstract
---

## 🚀 Approche Progressive 

### Phase 1 : 
**Objectif** : Avoir un modèle qui marche, même basique

**Données** :
- Résultats de matchs ATP (Jeff Sackmann)
- Classements des joueurs
- Surface du court (terre, dur, gazon)

**Features simples** :
- Différence de classement entre les 2 joueurs
- Taux de victoire par surface
- Nombre de matchs joués récemment

**Modèle** :
- Régression logistique (le plus simple)
- Évaluation : accuracy, comparaison avec "toujours prédire le mieux classé"

**Livrables** :
- Notebook Jupyter avec analyse exploratoire
- Premier modèle qui tourne
- Quelques visualisations (distributions, taux de victoire)

---

### Phase 2 : 
**Objectif** : Améliorer le modèle avec plus de features

**Nouvelles features** :
- Forme récente (victoires sur les 10 derniers matchs)
- Head-to-head historique entre les 2 joueurs
- Performance dans le tournoi en cours
- Statistiques de service (aces, % première balle)

**Nouveau modèle** :
- Random Forest ou XGBoost
- Feature importance (quelles variables comptent le plus ?)

**Livrables** :
- Modèle amélioré avec meilleures performances
- Graphiques de feature importance
- Analyse des erreurs du modèle

---

### Phase 3 : 
**Objectif** : Voir si notre modèle bat les bookmakers

**Ajout des cotes** :
- Intégrer les données Tennis-Data.co.uk
- Convertir les cotes en probabilités
- Comparer nos prédictions vs bookmakers

**Analyses** :
- Sur quels types de matchs notre modèle est meilleur ?
- Où se trompe-t-on par rapport aux bookmakers ?
- ROI simulé : si on avait parié avec notre modèle, combien on aurait gagné/perdu ?

**Livrables** :
- Tableau comparatif modèle vs bookmakers
- Analyse des forces/faiblesses
- Visualisations interactives (plotly)

---

### Phase 4 : 
**Options à ajouter si on a le temps** :

**Option A - Clustering** :
- Identifier des styles de jeu (serveur-volleyeur, baseliners, etc.)
- Visualisation avec PCA

**Option B - Dashboard** :
- Interface Streamlit simple
- Sélectionner 2 joueurs → voir prédiction

**Option C - Météo** :
- Ajouter données météo (température, vent) via Open-Meteo API gratuite
- Voir si ça améliore les prédictions

**Option D - NLP** :
- Analyser des articles de presse ou Reddit
- Sentiment autour des joueurs



## 📁 Structure du Projet


```
tennis-project/
│
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances Python
│
├── data/
│   ├── raw/                    # Données téléchargées
│   └── processed/              # Données nettoyées
│
├── notebooks/                   # Jupyter notebooks
│   ├── 1_exploration.ipynb     # Phase 1 : analyse de base
│   ├── 2_features.ipynb        # Phase 2 : features avancées
│   ├── 3_modeling.ipynb        # Phase 3 : modèles ML
│   └── 4_bookmakers.ipynb      # Phase 3 : comparaison cotes
│
├── src/                        # Code Python réutilisable
│   ├── data_loading.py
│   ├── features.py
│   └── models.py
│
└── reports/                    # Visualisations et rapport final
    ├── figures/
    └── rapport_final.pdf
```





