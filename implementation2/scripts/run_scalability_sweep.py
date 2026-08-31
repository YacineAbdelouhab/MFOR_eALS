"""Phase 5 timing sweeps: K, Spark parallelism (local[n]), data size.

Each point uses a short, fixed iteration count: this measures wall-clock cost per
iteration, not convergence, so there is no reason to pay for a full training run at
every grid point. Accuracy at convergence is a separate concern (run_weighting_sweep.py).

Usage: python scripts/run_scalability_sweep.py {K|cores|data_size} v1,v2,v3,...
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eals import experiment as xp  # noqa: E402
from eals.spark_utils import build_session  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "yelp.rating"
RESULTS = ROOT / "results" / "scalability.csv"
N_ITERS = 5  # median of these gives a stable per-iteration cost
DEFAULT_K, DEFAULT_CORES, DEFAULT_PARTS = 64, 8, 8
C0, ALPHA = 512.0, 0.4  # Yelp's own optimum per the paper (Fig. 1); accuracy isn't the axis here


ALGOS = {
    "eals": ("eALS", lambda sc, spark, data, K, n_parts: xp.run_eals(sc, data, K=K, c0=C0, alpha=ALPHA, n_iters=N_ITERS, n_parts=n_parts, eval_every=0)),
    "mllib": ("MLlib-ALS", lambda sc, spark, data, K, n_parts: xp.run_mllib_als(spark, data, K=K, n_iters=N_ITERS, n_parts=n_parts)),
}


def run_point(sc, spark, data, axis: str, value, K: int, cores: int, n_parts: int, algos: list[str]):
    for key in algos:
        algo, make = ALGOS[key]
        fn = lambda make=make: make(sc, spark, data, K, n_parts)
        t0 = time.perf_counter()
        try:
            row = fn()
        except Exception as exc:  # noqa: BLE001 - report and keep sweeping other points
            print(f"  [{axis}={value}][{algo}] FAILED after {time.perf_counter()-t0:.1f}s: {exc}")
            xp.append_csv(RESULTS, {"axis": axis, "value": value, "algo": algo, "K": K, "cores": cores, "n_parts": n_parts, "error": str(exc)[:200]})
            continue
        row["axis"], row["value"] = axis, value
        row["cores"] = cores
        xp.append_csv(RESULTS, row)
        print(f"  [{axis}={value}][{algo}] t_per_iter={row['t_per_iter']:.3f}s  HR@10={row.get('HR@10', float('nan')):.3f}  (wall {time.perf_counter()-t0:.1f}s)")


def sweep_K(values: list[int], algos: list[str]):
    spark = build_session("sweep_K", n_cores=DEFAULT_CORES)
    sc = spark.sparkContext
    data = xp.load_yelp_paper_ratings(DATA_PATH)
    for v in values:
        run_point(sc, spark, data, "K", v, K=v, cores=DEFAULT_CORES, n_parts=DEFAULT_PARTS, algos=algos)
    spark.stop()


def sweep_cores(values: list[int], algos: list[str]):
    data_ref = xp.load_yelp_paper_ratings(DATA_PATH)  # loading doesn't need Spark
    for v in values:
        spark = build_session(f"sweep_cores_{v}", n_cores=v)
        sc = spark.sparkContext
        run_point(sc, spark, data_ref, "cores", v, K=DEFAULT_K, cores=v, n_parts=max(v, DEFAULT_PARTS), algos=algos)
        spark.stop()


def sweep_data_size(values: list[float], algos: list[str]):
    spark = build_session("sweep_data_size", n_cores=DEFAULT_CORES)
    sc = spark.sparkContext
    full = xp.load_yelp_paper_ratings(DATA_PATH)
    for v in values:
        data = xp.subsample(full, v)
        run_point(sc, spark, data, "data_size", v, K=DEFAULT_K, cores=DEFAULT_CORES, n_parts=DEFAULT_PARTS, algos=algos)
    spark.stop()


if __name__ == "__main__":
    axis = sys.argv[1]
    raw_values = sys.argv[2].split(",")
    algos = sys.argv[3].split(",") if len(sys.argv) > 3 else list(ALGOS)
    if axis == "K":
        sweep_K([int(v) for v in raw_values], algos)
    elif axis == "cores":
        sweep_cores([int(v) for v in raw_values], algos)
    elif axis == "data_size":
        sweep_data_size([float(v) for v in raw_values], algos)
    else:
        raise SystemExit(f"unknown axis {axis!r}, expected K|cores|data_size")
