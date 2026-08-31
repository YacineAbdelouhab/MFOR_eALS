# eALS en PySpark, RDD uniquement

Implementation MapReduce de *Fast Matrix Factorization for Online
Recommendation with Implicit Feedback* (He et al., SIGIR 2016), apprentissage
offline et mise a jour online.

Tout passe par des RDD de paires. Aucun DataFrame, aucun MLlib.

---

## Demarrage

Rien a installer d'autre que Docker.

```bash
docker build -t eals-1 .
```

```bash
docker run --rm eals-1 pytest -q
```

La seconde doit afficher **5 passed**. Elle couvre les formules de mise a jour,
l'agregation des caches, le decoupage temporel, l'apprentissage offline et la
mise a jour online.

---

## Reproduire les experiences du rapport

Placer `yelp.rating` dans un dossier `data`, puis lancer les quatre commandes.

```bash
docker run --rm -v "$PWD/data:/data" eals-1 python scripts/xp1_qualite.py --input /data/yelp.rating
docker run --rm -v "$PWD/data:/data" eals-1 python scripts/xp2_initialisation.py --input /data/yelp.rating
docker run --rm -v "$PWD/data:/data" eals-1 python scripts/xp3_online.py --input /data/yelp.rating
docker run --rm -v "$PWD/data:/data" eals-1 python scripts/figures_job.py
```

| Commande | Ce qu'elle mesure | Produit | Duree |
|---|---|---|---|
| `xp1_qualite` | HR@100 et NDCG@100, popularite et eALS a K = 8, 16, 32 | `xp1.json` | 42 min |
| `xp2_initialisation` | initialisation aleatoire contre SVD randomisee, K = 16 | `xp2.json` | 25 min |
| `xp3_online` | traitement de 100 nouvelles interactions | `xp3.json` | 14 min |
| `figures_job` | les deux figures du rapport | `figures/` | 5 s |

Chaque commande affiche son tableau sur la sortie standard et ecrit le detail
dans son JSON. Le seul argument obligatoire est le fichier d'entree.

Les reglages sont fixes dans `CONFIG_RAPPORT`, en haut de
`src/pyspark_eals_rdd/jobs/experiences.py`. Vingt iterations, graine 42, sans
arret anticipe. Le classement porte sur tout le catalogue, et non sur un
echantillon de negatifs, ce qui rend le seuil de cent significatif.

> **Sous Git Bash**, prefixer par `MSYS_NO_PATHCONV=1`, sinon le chemin `/data`
> du conteneur est reecrit et Docker refuse le montage.

---

## Les autres scripts

| Script | A quoi il sert |
|---|---|
| `smoke_job.py` | voir l'algorithme tourner en une minute, sur un jeu synthetique |
| `generate_job.py` | fabriquer un jeu synthetique JSON, via `--users` et `--items` |
| `train_job.py` | un entrainement unique, via `--config config/paper_yelp.json` |

Ils se lancent de la meme facon, `docker run --rm eals-1 python scripts/<nom>.py`.

Les formats d'entree acceptes par `--input-format` sont `paper-rating`, celui du
depot des auteurs, `json`, une interaction par ligne, et `csv`.

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
tests/                  formules, decoupage, pipeline Spark
```

Le fichier a lire en premier est `math_core.py`, il contient les equations sans
la couche distribuee. Vient ensuite `model.py`, qui les distribue.
