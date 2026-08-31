# eALS en PySpark (v2)

Seconde implémentation de **eALS** (He et al., SIGIR'16, *Fast Matrix Factorization for
Online Recommendation with Implicit Feedback*) en PySpark.

Intuition de cette deuxième implémentation: `P` et `Q` vivent comme des tableaux NumPy
sur le driver et sont diffusés à chaque itération, tandis que la structure des
interactions n'est distribuée qu'une seule fois (`groupByKey`, en amont de la
boucle). L'apprentissage offline (algorithme 1) et la mise à jour incrémentale
online (algorithme 2, via Spark Structured Streaming) sont toutes les deux
implémentées.

## Structure

- `src/eals/` — le code : `numpy_reference.py` (sert de référence de correction), `spark_eals.py` (algorithme 1 distribué),
  `spark_eals_online.py` (algorithme 2, micro-batches), `evaluation.py`
  (HR@N / NDCG@N), `weighting.py` (poids par popularité, Eq. 8),
  `data_prep.py`, `baseline_als.py` (référence Spark MLlib ALS).
- `notebooks/` — `00_correctness_check.ipynb` (preuve d'exactitude sur un
  problème synthétique) et `01_demo_small_data.ipynb` (pipeline complète sur un
  échantillon).
- `scripts/` — expériences reproductibles (passage à l'échelle, sensibilité
  aux hyperparamètres, protocole online).
- `data/sample/` — échantillon Yelp (300 utilisateurs) pour les notebooks.
- `results/` — sorties des expériences.

## Installation

> Toutes les commandes ci-dessous doivent être lancées depuis **ce dossier** — y compris
> `docker build`, dont le contexte doit pointer ici
> (`docker build -t eals implementation2/`).

Nécessite Python 3.12 et Java 17 ou 21.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group dev
```

**Sous Windows**: exécuter le
projet sous **WSL2** (recommandé), ou via l'image **Docker** fournie :

```bash
docker build -t eals .
docker run --rm eals                # suite de tests
docker run --rm eals jupyter nbconvert --to notebook --execute --stdout \
    notebooks/01_demo_small_data.ipynb
```

## Données

`data/sample/yelp_sample.rating` est déjà présent dans le dépôt et suffit pour
les notebooks. Pour reproduire les expériences sur le dataset complet,
télécharger le fichier prétraité par les auteurs de eALS eux-mêmes (dépôt
`hexiangnan/sigir16-eals`, format `user\titem\trating\ttimestamp`, ids déjà
denses) et le placer dans `data/raw/yelp.rating`.

## Tests

```bash
uv run pytest -q
```

## Notebooks

```bash
uv run jupyter lab
```

`01_demo_small_data.ipynb` exécute toute la pipeline (Spark,
comparaison avec MLlib ALS, mise à jour
online) sans dépendre du
dataset complet.

## Expériences

Nécessitent `data/raw/yelp.rating`. Les résultats déjà obtenus figurent dans
`results/`.

```bash
uv run python scripts/run_offline_experiment.py --algo eals --K 64
uv run python scripts/run_scalability_sweep.py
uv run python scripts/run_weighting_sweep.py
uv run python scripts/run_online_experiment.py
```
