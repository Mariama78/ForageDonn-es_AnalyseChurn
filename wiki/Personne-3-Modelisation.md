# Personne 3 - Modelisation

## Role

Cette personne prend en charge l'etape 3 du devoir : utilisation de trois modeles de classification.

## Entree attendue

Elle repart uniquement du preprocessing valide par la personne 2.

## Taches obligatoires

- entrainer trois modeles de classification differents
- utiliser la meme strategie de validation pour tous
- preciser clairement la methode de validation
- comparer les performances de maniere juste
- sauvegarder les modeles finaux
- preparer une fonction commune de prediction pour l'integration

## Modeles conseilles

- regression logistique
- random forest
- SVM ou XGBoost

## Livrables obligatoires

- un notebook ou script d'entrainement
- les trois modeles entraines
- un tableau comparatif des performances
- une fonction du type `predict_all_models()`
- des fichiers de modeles sauvegardes si l'architecture retenue le permet

## Definition de termine

L'etape est terminee seulement si :

- les trois modeles sont comparables
- la strategie de validation est documentee
- la personne 4 peut reutiliser les sorties dans Streamlit

## Points d'attention

- ne pas comparer des modeles sur des jeux differents
- ne pas changer le preprocessing sans repasser par l'etape precedente
- livrer une interface de prediction simple a brancher dans l'application
