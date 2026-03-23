# Projet Final - Forage de Donnees

## Analyse du Churn Client Telecom

Projet de groupe realise dans le cadre du cours de forage de donnees. L'objectif est de construire un systeme de classification supervisee capable de predire si un client telecom risque de quitter le service (`Churn`), puis d'integrer les resultats dans une application Streamlit.

- Date de remise : `19 avril 2026 a 23h55`
- Dataset : [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Cible : `Churn`
- Taille du dataset : environ `7043` observations

## Objectifs du projet

Le projet suit les etapes du devoir :

1. Preparation des donnees
2. Reduction ou selection de dimensions
3. Entrainement de 3 modeles de classification
4. Evaluation des modeles avec les bonnes metriques
5. Integration dans une application Streamlit
6. Rapport final et evaluation par les pairs

## Methode de travail

L'equipe travaille de facon sequentielle sur le pipeline principal :

1. La personne 1 termine la preparation des donnees
2. La personne 2 reprend le dataset propre pour le preprocessing avance
3. La personne 3 entraine et compare les 3 modeles
4. La personne 4 integre les resultats, construit Streamlit et finalise l'evaluation

Chaque etape doit etre validee avant le passage a la suivante. Le but est d'eviter les blocages, les versions incoherentes et les changements de schema en cours de route.

## Repartition proposee des taches

### Personne 1 - Preparation des donnees

Responsabilites :

- charger le dataset
- analyser les colonnes et les types
- traiter les valeurs manquantes
- corriger les types de donnees
- nettoyer `TotalCharges`
- retirer les colonnes inutiles comme `customerID` si necessaire
- encoder la cible
- documenter les choix de nettoyage

Livrables :

- notebook ou script de preparation
- dataset nettoye et fige
- description des transformations appliquees
- liste finale des colonnes

### Personne 2 - Preprocessing et selection de variables

Responsabilites :

- faire le `train/test split`
- encoder les variables categorielles
- normaliser les variables numeriques si necessaire
- traiter eventuellement le desequilibre de classes
- appliquer une reduction ou selection de dimensions
- justifier la methode choisie

Livrables :

- pipeline de preprocessing
- variables retenues
- notebook ou script de transformation
- comparaison avant/apres selection

### Personne 3 - Modelisation

Responsabilites :

- entrainer 3 modeles de classification differents
- utiliser une strategie de validation coherente
- comparer les performances
- sauvegarder les modeles retenus
- preparer une fonction commune de prediction

Livrables :

- scripts ou notebooks d'entrainement
- tableau comparatif des modeles
- modeles sauvegardes
- fonction reutilisable du type `predict_all_models()`

### Personne 4 - Evaluation finale, Streamlit et integration

Responsabilites :

- calculer les metriques finales
- produire les matrices de confusion
- afficher les performances des 3 modeles
- construire l'application Streamlit
- permettre la saisie ou le televersement d'un nouvel objet
- afficher les predictions des 3 modeles
- afficher les metriques dans l'interface
- integrer les captures et la synthese dans le rapport

Livrables :

- application `Streamlit`
- ecran de resultats avec les 3 predictions
- metriques finales des modeles
- captures d'ecran et section finale du rapport

## Bonnes pratiques

- Ne pas modifier le travail de la personne precedente sans validation.
- Garder des noms de colonnes stables pendant tout le projet.
- Documenter les transformations et les decisions a chaque etape.
- Rendre chaque script ou notebook executable de bout en bout.
- Utiliser les memes jeux de validation pour comparer les modeles.
- Garder une trace claire du travail de chacun pour l'evaluation par les pairs.

## Organisation GitHub

Le depot GitHub sert de reference unique pour le projet.

### Regles

- `main` contient uniquement la version stable
- aucune modification directe sur `main`
- chaque membre travaille sur sa branche
- chaque tache importante passe par une Pull Request
- une autre personne relit avant fusion

### Branches conseillees

- `feature/01-preparation-donnees`
- `feature/02-preprocessing-selection`
- `feature/03-modeles-classification`
- `feature/04-streamlit-evaluation`

### Convention de commits

- `feat: ajout du nettoyage des donnees`
- `feat: ajout du pipeline de preprocessing`
- `feat: entrainement des 3 modeles`
- `feat: ajout de l'application streamlit`
- `fix: correction du traitement de TotalCharges`
- `docs: mise a jour du README`

## Structure recommandee du projet

```text
ForageDonn-es_AnalyseChurn/
├── README.md
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_preparation.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelisation.ipynb
├── reports/
└── src/
```

## Technologies pressenties

- `Python`
- `pandas`
- `scikit-learn`
- `matplotlib` / `seaborn`
- `streamlit`

## Lancement local de l'application

Lorsque l'application sera prete :

```bash
streamlit run app.py
```

## Suivi des livrables

- [ ] Preparation des donnees terminee
- [ ] Reduction ou selection de variables terminee
- [ ] Trois modeles entraines et compares
- [ ] Evaluation finale des modeles terminee
- [ ] Application Streamlit fonctionnelle
- [ ] Rapport final complete

## Evaluation par les pairs

Chaque membre doit avoir une contribution visible sur GitHub via :

- commits
- branches
- Pull Requests
- revue de code
- participation a la redaction du rapport

Une bonne trace GitHub aidera a justifier l'implication de chacun.
