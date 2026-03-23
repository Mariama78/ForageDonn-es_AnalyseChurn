# Processus de travail

## Logique generale

Le projet suit exactement les etapes du devoir :

1. preparation des donnees
2. reduction ou selection de dimensions
3. entrainement de trois modeles
4. evaluation avec les metriques adequates
5. integration dans Streamlit

## Regle de passage

Le passage entre les etapes est strict.

- Personne 2 attend la validation de la Personne 1
- Personne 3 attend la validation de la Personne 2
- Personne 4 attend la validation de la Personne 3 pour l'integration finale

## Validation obligatoire a chaque etape

Avant de passer a l'etape suivante, il faut verifier :

- que le notebook ou script principal s'execute
- que les sorties attendues existent
- que les noms de colonnes sont stables
- que les decisions importantes sont notees
- que le commit GitHub correspondant est propre

## Reunion de passage

Chaque fin d'etape doit se terminer par un mini point d'equipe.

Contenu minimum :

- ce qui a ete fait
- fichiers produits
- problemes rencontres
- decisions a conserver
- ce que la personne suivante doit utiliser

## Ce qu'il faut eviter

- modifier le schema de donnees sans prevenir
- renommer des colonnes au dernier moment
- lancer des modeles sur des donnees non validees
- integrer Streamlit avec des sorties instables
- melanger plusieurs taches dans un seul commit
