# GitHub et livrables

## Regles GitHub

- `main` doit rester stable
- aucune modification directe sur `main`
- chaque membre travaille sur sa propre branche
- chaque etape importante doit passer par une Pull Request
- une autre personne relit avant fusion si possible

## Branches conseillees

- `feature/01-preparation-donnees`
- `feature/02-preprocessing-selection`
- `feature/03-modelisation`
- `feature/04-streamlit-evaluation`

## Convention de commits

- `feat: ajout du nettoyage des donnees`
- `feat: ajout du pipeline de preprocessing`
- `feat: entrainement des 3 modeles`
- `feat: ajout de l'application streamlit`
- `fix: correction du traitement de TotalCharges`
- `docs: mise a jour du wiki`

## Livrables par etape

### Etape 1

- code de nettoyage
- dataset nettoye
- colonnes finales

### Etape 2

- pipeline de preprocessing
- variables retenues
- justification de la methode

### Etape 3

- trois modeles
- comparaison
- fonction de prediction

### Etape 4 et 5

- metriques finales
- application Streamlit
- captures
- partie finale du rapport

## Ce que GitHub doit montrer

GitHub doit permettre de voir clairement :

- qui a fait quoi
- quelles decisions ont ete prises
- a quel moment une etape a ete validee
- quels fichiers servent de sortie officielle pour l'etape suivante

## Regle pratique

Personne ne doit livrer uniquement du code local. Tout ce qui valide une etape doit exister dans le depot avec un commit clair.
