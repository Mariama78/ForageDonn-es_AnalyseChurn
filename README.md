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
├── app.py
├── data/
├── models/
├── notebooks/
├── reports/
└── src/
```

## Lancement local

Quand l'application sera disponible :

```bash
streamlit run app.py
```

## Note

La documentation de travail, la repartition des taches et le suivi interne de l'equipe sont geres separement du depot public.
