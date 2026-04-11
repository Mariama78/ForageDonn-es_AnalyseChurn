# Analyse du Churn Client Telecom

Projet de classification supervisee visant a predire le risque de desabonnement (`Churn`) de clients telecom a partir du dataset **Telco Customer Churn**. Le projet combine preparation des donnees, modelisation, evaluation de plusieurs classificateurs et integration finale dans une application **Streamlit**.

## Objectif

L'objectif principal est de comparer plusieurs modeles de classification afin d'identifier les clients les plus susceptibles de quitter le service, puis de rendre ces predictions exploitables via une interface simple.

## Dataset

- Source : [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Variable cible : `Churn`
- Taille : environ `7043` observations

## Contenu du projet

Le projet couvre les grandes etapes suivantes :

- preparation et nettoyage des donnees
- preprocessing et selection de variables
- entrainement de plusieurs modeles de classification
- evaluation avec des metriques adaptees
- integration dans une application Streamlit

## Technologies

- `Python`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `streamlit`
- `xgboost`

## Structure du depot

```text
ForageDonn-es_AnalyseChurn/
├── README.md
├── requirements.txt
├── app.py
├── data/
├── models/
├── notebooks/
└── src/
```

## Lancement local

pour lancer lapplication :

```bash
streamlit run app.py
```

## Note

le fichier requirements.txt contient toutes les librairies utilisées pour ce projet, pour les installer on execute 'pip install -r  requirements.txt' dans le terminal.

## Préparation des données

Pour commencer, nous avons fait:
- le chargement du dataset qui contient 7043 lignes, 21 variables.
- Correction du type de la variable TotalCharges, initialement au format texte.
- Traitement des valeurs manquantes présentes dans la colonne TotalCharges en remplaçant par la médiane.
- Suppression de la 'customerID' qui n'était pas nécessaire à notre analyse
- Séparation des variables Catégorielles et des Numériques
**Un fichier csv propre a été enregistré dans le dossier data pour utilisation future**

## Analyse exploratoire

Pour chaque variable catégorielle, la distribution a été analysée et affichée avec le pourcentage de chaque modalité.

- Nous constatons qu'environ 73 % des clients ne churnent pas, contre 27 % qui churnent, indiquant un déséquilibre des classes.
- La majorité des clients ont un contrat month‑to‑month
- Les services Internet Fibre Optique sont largement utilisés
- Le moyen de paiement le plus courant est le Electronic Check

En faisant l'analyse bivariée on remarque que:

- Les clients avec un contrat month‑to‑month ont un taux de churn nettement plus élevé
- Les contrats Two‑year présentent un churn très faible
- Les clients utilisant la Fibre Optic churnent davantage
- Le mode de paiement Electronic Check est associé à un taux de churn plus élevé
- Les clients sans services additionnels (TechSupport, OnlineSecurity) sont plus enclins à churner
- Les clients qui churnent ont en moyenne une tenure plus faible
- Les MonthlyCharges sont plus élevées chez les churners
- Les TotalCharges sont plus faibles pour les clients churnant (relation logique avec la tenure)

## Analyse de corrélation

Une matrice de corrélation a été calculée entre les variables numériques et la variable cible (encodée).
On remarque que:

- tenure vs Churn : corrélation négative ce qui signifie que plus la durée est longue, moins le churn est probable
- MonthlyCharges vs Churn : corrélation positive, les charges élevées sont associées au churn
- tenure vs TotalCharges : très forte corrélation, plus la tenure augmente plus la charge augmente.

---

## Preprocessing et Sélection de Variables

Cette section reprend les données nettoyées et prépare les jeux de données finaux pour la modélisation.

Pour commencer, les 15 variables catégorielles ont été encodées en variables numériques via One-Hot Encoding, ce qui a porté le nombre de variables de 19 à 30. Ensuite, les données ont été découpées en trois ensembles : 70% pour l'entraînement (4930 observations), 15% pour la validation (1056 observations) et 15% pour le test (1057 observations), avec stratification pour garantir que la proportion de churn reste identique dans les trois ensembles. La validation sert à comparer et ajuster les modèles sans toucher au test, qui est réservé à l'évaluation finale. Une analyse du déséquilibre des classes a montré un ratio de 2.77:1, considéré comme modéré, il n'a donc pas été nécessaire d'appliquer de rééchantillonnage artificiel. Les variables numériques continues `tenure`, `MonthlyCharges` et `TotalCharges` ont été normalisées via StandardScaler, fitté uniquement sur le train pour éviter le data leakage.

Pour la sélection de variables, la méthode retenue est l'importance des features par Random Forest. Cette méthode a été choisie car elle est robuste, non paramétrique et capture les relations non-linéaires sans suppositions sur la distribution des données. En appliquant un seuil d'importance de 1%, on est passé de 30 à 21 variables finales. Les 9 variables éliminées correspondaient toutes à des modalités redondantes de type "No internet service" ou "No phone service" qui n'apportaient pas d'information utile pour prédire le churn.

Les fichiers produits pour l'application Streamlit sont : `models/scaler.pkl` pour reproduire la normalisation sur de nouvelles données et `models/selected_features.json` pour connaître les 21 variables à utiliser en entrée des modèles.

## Conclusion de la modélisation

Trois modèles de classification ont été entraînés et évalués sur le jeu
de validation puis confirmés sur le jeu de test final.

### Métriques retenues

Deux métriques ont guidé l'évaluation et la comparaison des modèles :

**AUC-ROC** utilisée pour comparer les modèles entre eux de façon neutre. Elle mesure la capacité de discrimination sur tous les seuils de décision possibles, indépendamment du seuil par défaut de 0.5.

**Recall sur la classe Churn** qui est la métrique la plus importante du point devue métier. Dans un contexte télécom, manquer un churner (faux négatif) est plus coûteux que générer une fausse alerte (faux positif) : un client non identifié part sans qu'on intervienne, une fausse alerte génère simplement une offre de rétention inutile. Maximiser le recall revient donc à minimiser les clients perdus sans intervention.

### Classement des modèles

**1. Régression Logistique**
- AUC-ROC : 0.84
- Recall Churn : 0.78
- F1 Churn : 0.61

**2. XGBoost**
- AUC-ROC : 0.81
- Recall Churn : 0.63
- F1 Churn : 0.58

**3. Random Forest**
- AUC-ROC : 0.82
- Recall Churn : 0.46
- F1 Churn : 0.54

### Interprétation

La régression logistique, modèle le plus simple, obtient les meilleures performances sur les deux métriques retenues. Ce résultat contre-intuitif
s'explique par la nature des données : les relations entre les features et la probabilité de churn sont majoritairement linéaires. La complexité supplémentaire de XGBoost et du Random Forest n'apporte pas d'amélioration significative.

Le Random Forest présente le recall le plus faible sur les churneurs (0.46), ce qui signifie qu'il rate plus de la moitié des vrais churneurs. Dans un contexte télécom où chaque client perdu représente un coût, ce comportement le rend inadapté à l'objectif principal.

Les performances sont stables entre validation et test pour les trois modèles, confirmant l'absence de surapprentissage et une bonne capacité de
généralisation sur des données inconnues.

### Modèle recommandé

La **régression logistique** est retenue comme modèle principal pour l'application Streamlit. Le fichier est disponible dans `models/model_logistic_regression.pkl`.
