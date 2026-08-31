# eALS en PySpark, RDD uniquement

Implémentation MapReduce de *Fast Matrix Factorization for Online
Recommendation with Implicit Feedback* (He et al., SIGIR 2016), apprentissage
offline et mise à jour online.

Tout passe par des RDD de paires.

---

## Démarrage

Rien à installer d'autre que Docker.

```bash
docker build -t eals-1 .
```

```bash
docker run --rm eals-1 pytest -q
```

---

## Reproduire les expériences du rapport

Placer `yelp.rating` dans un dossier `data`, puis lancer les quatre commandes.

```bash
docker run --rm -v "$PWD/data:/data" eals-1 python scripts/xp1_qualite.py --input /data/yelp.rating
docker run --rm -v "$PWD/data:/data" eals-1 python scripts/xp2_initialisation.py --input /data/yelp.rating
docker run --rm -v "$PWD/data:/data" eals-1 python scripts/xp3_online.py --input /data/yelp.rating
docker run --rm -v "$PWD/data:/data" eals-1 python scripts/figures_job.py
```

| Commande | Ce qu'elle mesure | Produit | Durée |
|---|---|---|---|
| `xp1_qualite` | HR@100 et NDCG@100, popularité et eALS a K = 8, 16, 32 | `xp1.json` | 42 min |
| `xp2_initialisation` | initialisation aléatoire contre SVD randomisée, K = 16 | `xp2.json` | 25 min |
| `xp3_online` | traitement de 100 nouvelles interactions | `xp3.json` | 14 min |
| `figures_job` | les deux figures du rapport | `figures/` | 5 s |

Chaque commande affiche son tableau sur la sortie standard et écrit le detail
dans son JSON. Le seul argument obligatoire est le fichier d'entree.

Les réglages sont fixes dans `CONFIG_RAPPORT`, en haut de
`src/pyspark_eals_rdd/jobs/experiences.py`. Vingt itérations, graine 42, sans
arrêt anticipé. Le classement porte sur tout le catalogue, et non sur un
échantillon de négatifs, ce qui rend le seuil de cent significatif.

> **Sous Git Bash**, préfixer par `MSYS_NO_PATHCONV=1`, sinon le chemin `/data`
> du conteneur est réécrit et Docker refuse le montage.

---

## Les autres scripts

| Script | A quoi il sert |
|---|---|
| `smoke_job.py` | voir l'algorithme tourner en une minute, sur un jeu synthétique |
| `generate_job.py` | fabriquer un jeu synthétique JSON, via `--users` et `--items` |
| `train_job.py` | un entraînement unique, via `--config config/paper_yelp.json` |

Ils se lancent de la même façon, `docker run --rm eals-1 python scripts/<nom>.py`.

Les formats d'entrée acceptés par `--input-format` sont `paper-rating`, celui du
depôt des auteurs, `json`, une interaction par ligne, et `csv`.

---

## Organisation des fichiers

```
Dockerfile              l'image, tout tourne dedans
config/paper_yelp.json  les reglages du rapport
src/pyspark_eals_rdd/
    config.py           hyperparametres, lecture du JSON
    data.py             lecture, deduplication, decoupage temporel
    math_core.py        les formules eALS en NumPy pur, sans Spark
    model.py            le plan RDD, les caches, la boucle d'apprentissage
    online.py           mise a jour incrementale par micro-lots
    evaluate.py         HR@k et NDCG@k contre tout le catalogue
    jobs/               un module par commande
scripts/                un fichier par commande, qui appelle jobs/
tests/                  formules, découpage, pipeline Spark
```

Le fichier à lire en premier est `math_core.py`, il contient les équations sans
la couche distribuée. Vient ensuite `model.py`, qui les distribue.
