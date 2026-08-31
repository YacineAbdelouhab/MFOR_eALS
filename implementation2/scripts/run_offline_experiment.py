"""One offline fit + evaluate run, params via CLI — the reusable form of the ad hoc
scripts used throughout the project's sweeps (K=128 confirmation, convergence checks).

Usage:
  python scripts/run_offline_experiment.py --algo eals --K 64 --c0 2048 --alpha 0.5 --n-iters 30
  python scripts/run_offline_experiment.py --algo mllib --K 64 --n-iters 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eals import experiment as xp  # noqa: E402
from eals.spark_utils import build_session  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "yelp.rating"
RESULTS = ROOT / "results" / "offline_runs.csv"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["eals", "mllib"], default="eals")
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--c0", type=float, default=2048.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--n-iters", type=int, default=30)
    p.add_argument("--n-parts", type=int, default=8)
    p.add_argument("--cores", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=0, help="track HR/NDCG every N iters (eALS only, 0=off)")
    p.add_argument("--data", type=Path, default=DATA_PATH)
    args = p.parse_args()

    spark = build_session(f"offline_{args.algo}", n_cores=args.cores)
    sc = spark.sparkContext
    data = xp.load_yelp_paper_ratings(args.data)

    if args.algo == "eals":
        row = xp.run_eals(
            sc, data, K=args.K, c0=args.c0, alpha=args.alpha, lam=args.lam,
            n_iters=args.n_iters, n_parts=args.n_parts, eval_every=args.eval_every,
        )
    else:
        row = xp.run_mllib_als(spark, data, K=args.K, lam=args.lam, n_iters=args.n_iters, n_parts=args.n_parts)

    xp.append_csv(RESULTS, row)
    print(f"{row['algo']} K={row['K']}: HR@10={row['HR@10']:.4f} NDCG@10={row['NDCG@10']:.4f} "
          f"HR@100={row['HR@100']:.4f} t_per_iter={row['t_per_iter']:.2f}s")
    spark.stop()
