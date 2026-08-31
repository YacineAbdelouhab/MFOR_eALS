# eALS en PySpark: deux implémentations

Ce repo présente deux implémentations différentes du papier de He et al.,
"Fast Matrix Factorization for Online Recommendation with Implicit
Feedback" en PySpark. Le papier ne fournit aucune implémentation MapReduce, alors,
deux conceptions RDD ont été développées indépendamment, avec des choix
d'architecture différents.

## `implementation1/`: RDD uniquement

`P` et `Q` restent des RDD d'une itération à l'autre et la structure des
interactions est rejointe à chaque itération. Tout tourne dans une image
Docker unique (`eals-1`), aucune installation locale nécessaire. Trois scripts
d'expérience (qualité à K = 8/16/32, initialisation aléatoire vs. SVD,
traitement online de 100 interactions) et un job de figures.

Voir [`implementation1/README.md`](implementation1/README.md)

## `implementation2/`: P/Q broadcastés, structure distribuée une seule fois

`P` et `Q` vivent comme des tableaux NumPy sur le driver, broadcastés à chaque itération. La structure des interactions est distribuée une seule fois,
en amont de la boucle. S'exécute sous WSL2 ou via l'image Docker fournie
(`eals`). Quatre scripts d'expérience portant sur le passage à l'échelle (K
jusqu'à 1024, nombre de cœurs, volume de données), la sensibilité aux
hyperparamètres de pondération et le protocole online, plus deux notebooks
(preuve d'exactitude, démo sur un échantillon).

Voir [`implementation2/README.md`](implementation2/README.md)

## Les différences, très brièvement

| | `implementation1` | `implementation2` |
|---|---|---|
| `P`, `Q` | RDD, rejoints à chaque itération | NumPy, broadcastés à chaque itération |
| Structure des interactions | reconstruite par jointure | distribuée une seule fois |
| Exécution | Docker uniquement | WSL2 ou Docker |
| Expériences | qualité (K=8/16/32), initialisation, online | passage à l'échelle (K jusqu'à 1024, cœurs, volume), pondération, online |
| Notebooks | — | preuve d'exactitude + démo petites données |

