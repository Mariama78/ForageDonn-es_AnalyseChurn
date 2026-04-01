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