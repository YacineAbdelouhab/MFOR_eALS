"""Phase 5 accuracy sweep: c0 then alpha, reproducing the paper's Fig. 1 on our own Yelp data.

Unlike run_scalability_sweep.py this needs real convergence, so it pays for the full
iteration budget at every point (30 iterations, per the paper's own Fig. 2 showing Yelp
essentially converged well before its 100-iteration ceiling).

The c0 grid auto-extends rightward if the best point lands on its current right edge —
found by experience at K=64 (notes/04_weighting.md): the initial {8,32,128,512,2048} grid
was still climbing at 2048, and only extending to {4096,8192,16384} revealed the real
interior peak. Reporting a boundary value as "the optimum" without checking this would be
indistinguishable from just not having searched far enough.

Usage: python scripts/run_weighting_sweep.py [--K 64]
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
RESULTS = ROOT / "results" / "weighting.csv"
N_ITERS, N_PARTS = 30, 8

C0_GRID = [8.0, 32.0, 128.0, 512.0, 2048.0]
ALPHA_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]
MAX_EXTENSIONS = 3


def run_point(sc, data, K: int, stage: str, c0: float, alpha: float):
    row = xp.run_eals(sc, data, K=K, c0=c0, alpha=alpha, n_iters=N_ITERS, n_parts=N_PARTS, eval_every=10)
    row["stage"] = stage
    xp.append_csv(RESULTS, row)
    print(f"  [{stage}] c0={c0:<8} alpha={alpha:<4} HR@10={row['HR@10']:.4f}  NDCG@10={row['NDCG@10']:.4f}")
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=64)
    K = p.parse_args().K
    stage_c0, stage_alpha = f"c0_K{K}", f"alpha_K{K}"

    spark = build_session(f"weighting_sweep_K{K}", n_cores=8)
    sc = spark.sparkContext
    data = xp.load_yelp_paper_ratings(DATA_PATH)

    print(f"Stage 1 (K={K}): c0 grid at alpha=0 (uniform weighting, Hu et al. special case)")
    grid = list(C0_GRID)
    c0_rows = [run_point(sc, data, K, stage_c0, c0, alpha=0.0) for c0 in grid]
    for _ in range(MAX_EXTENSIONS):
        best = max(c0_rows, key=lambda r: r["HR@10"])
        if best["c0"] != grid[-1]:
            break
        next_c0 = grid[-1] * 2
        print(f"  best c0 ({best['c0']}) is at the grid's right edge -> extending to {next_c0}")
        grid.append(next_c0)
        c0_rows.append(run_point(sc, data, K, stage_c0, next_c0, alpha=0.0))
    best_c0 = max(c0_rows, key=lambda r: r["HR@10"])["c0"]
    print(f"  best c0 = {best_c0}")

    print(f"Stage 2 (K={K}): alpha grid at c0={best_c0}")
    alpha_rows = [run_point(sc, data, K, stage_alpha, best_c0, alpha) for alpha in ALPHA_GRID]
    best_alpha = max(alpha_rows, key=lambda r: r["HR@10"])["alpha"]
    print(f"  best alpha = {best_alpha}")

    print(f"\nFinal K={K} optimum: c0={best_c0}, alpha={best_alpha}, "
          f"HR@10={max(alpha_rows, key=lambda r: r['HR@10'])['HR@10']:.4f}")

    spark.stop()
