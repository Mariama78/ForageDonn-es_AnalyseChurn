# Personne 1 - Preparation des donnees

## Role

Cette personne prend en charge l'etape 1 du devoir : la preparation des donnees.

## Objectif

Livrer une base de donnees propre, coherente et exploitable par la personne 2.

## Taches obligatoires

- charger le dataset d'origine
- examiner toutes les colonnes et leurs types
- verifier les doublons
- traiter les valeurs manquantes
- convertir `TotalCharges` en variable numerique si necessaire
- analyser si `customerID` doit etre retire
- encoder la cible `Churn`
- realiser une petite EDA utile
- noter tous les choix de nettoyage

## Livrables obligatoires

- un notebook ou script de preparation
- un dataset nettoye et fige
- une liste finale des colonnes
- un resume des transformations
- une note sur les problemes detectes dans le dataset

## Definition de termine

L'etape est terminee seulement si :

- le dataset nettoye est genere sans erreur
- la cible est exploitable
- les colonnes finales sont stables
- la personne 2 peut commencer sans refaire le nettoyage

## Points d'attention

- ne pas laisser `TotalCharges` dans un type incorrect
- ne pas utiliser `customerID` comme variable predictive si elle n'a pas de sens
- documenter clairement les lignes retirees ou imputees
