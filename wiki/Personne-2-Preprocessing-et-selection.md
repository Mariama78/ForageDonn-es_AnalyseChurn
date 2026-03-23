# Personne 2 - Preprocessing et selection

## Role

Cette personne prend en charge l'etape 2 du devoir : reduction ou selection de dimensions, avec preprocessing propre.

## Entree attendue

Elle repart uniquement du dataset valide par la personne 1.

## Taches obligatoires

- separer les variables explicatives et la cible
- faire un `train/test split` avec stratification
- encoder correctement les variables categorielles
- normaliser les variables numeriques si necessaire
- verifier le desequilibre de classes
- appliquer une reduction ou une selection de variables
- justifier la methode retenue

## Methodes possibles

- `PCA`
- `SelectKBest`
- `RFE`
- importance des variables

## Livrables obligatoires

- un notebook ou script de preprocessing
- un pipeline clair et reutilisable
- la liste finale des variables retenues
- une justification methodologique
- une comparaison simple avant/apres reduction ou selection

## Definition de termine

L'etape est terminee seulement si :

- la chaine de preprocessing est stable
- les variables finales sont connues
- la personne 3 peut entrainer les modeles sur une base fixe

## Points d'attention

- ne pas modifier les colonnes d'entree sans le documenter
- ne pas appliquer un preprocessing different entre train et test
- ne pas ajouter de technique de reequilibrage sans verification de son utilite
