# Wiki du projet

## Objectif

Ce wiki sert de reference commune pour l'equipe. Il definit :

- les etapes du devoir
- la repartition des responsabilites
- les livrables obligatoires
- les regles de passage entre les etapes
- les bonnes pratiques GitHub

Le projet doit suivre une logique sequentielle stricte :

1. Personne 1 termine la preparation des donnees
2. Personne 2 reprend uniquement la version validee
3. Personne 3 construit et compare les modeles
4. Personne 4 integre, evalue et finalise l'application Streamlit

Une personne ne commence pas son etape principale tant que l'etape precedente n'a pas ete validee.

## Pages du wiki

- [Processus de travail](Processus-de-travail)
- [Personne 1 - Preparation des donnees](Personne-1-Preparation-des-donnees)
- [Personne 2 - Preprocessing et selection](Personne-2-Preprocessing-et-selection)
- [Personne 3 - Modelisation](Personne-3-Modelisation)
- [Personne 4 - Streamlit, evaluation et integration](Personne-4-Streamlit-evaluation-et-integration)
- [GitHub et livrables](GitHub-et-livrables)

## Regle centrale

Le projet ne doit pas fonctionner en parallele sur le pipeline principal. Le passage d'une etape a la suivante se fait uniquement apres :

- verification du code
- verification des sorties attendues
- validation par l'equipe
- commit propre sur GitHub

## Definition minimale d'une etape terminee

Une etape est consideree comme terminee seulement si :

- le code s'execute
- les fichiers de sortie sont clairement identifies
- les choix sont documentes
- le membre suivant peut reprendre sans deviner ce qui a ete fait

## Cible du projet

Dataset : Telco Customer Churn

Objectif : predire la variable `Churn` a l'aide de trois modeles de classification, puis integrer les resultats dans une application Streamlit avec affichage des predictions et des metriques.
