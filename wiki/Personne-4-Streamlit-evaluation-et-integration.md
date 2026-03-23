# Personne 4 - Streamlit, evaluation et integration

## Role

Cette personne prend en charge :

- l'etape 4 du devoir : evaluation des modeles
- l'etape 5 du devoir : interface utilisateur
- l'integration finale de l'ensemble du projet

## Entree attendue

Elle repart des modeles valides par la personne 3 et des sorties stabilisees par les etapes precedentes.

## Taches obligatoires

- calculer les metriques finales pour chaque modele
- produire les matrices de confusion
- afficher au minimum `accuracy`, `precision`, `recall`, `F1-score`
- afficher `ROC-AUC` si pertinent
- choisir le meilleur modele et expliquer ce choix
- construire l'application Streamlit
- permettre la saisie manuelle ou le televersement d'un nouvel objet
- afficher la prediction de chacun des trois modeles
- afficher les metriques dans l'application
- integrer les captures et resultats dans le rapport final

## Livrables obligatoires

- `app.py`
- interface Streamlit fonctionnelle
- affichage des predictions des trois modeles
- tableau ou bloc de metriques comparatives
- captures d'ecran pour le rapport

## Definition de termine

L'etape est terminee seulement si :

- l'application s'ouvre sans erreur
- les trois modeles sont interroges correctement
- les resultats affiches sont coherents avec les metriques calculees
- la partie evaluation du rapport est redigee

## Points d'attention

- ne pas recoder un preprocessing different dans Streamlit
- reutiliser les objets produits par les etapes precedentes
- verifier les entrees utilisateur avant la prediction
- garder l'application simple, stable et demonstrable
