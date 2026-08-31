"""Phase 6: online eALS, reproducing the paper's online-protocol experiments (§5.3, Figs. 5-8)
on our own Yelp chronological stream.

Three stages sharing one offline-trained model:
  A. n_iters_online sweep (Fig. 6 analogue) on a fixed stream subset — cheap.
  B. w_new sweep (Fig. 7 analogue) at the best n_iters_online — cheap.
  C. one full-stream pass at the best (n_iters_online, w_new) — the only expensive run,
     feeding both the Fig. 5 analogue (HR/NDCG evolution) and the Fig. 8 analogue
     (cold-start breakdown by prior interaction count).

Scope note (agreed in the plan): no RCD/BPR online baselines — n_iters_online=0 (frozen
offline model, no incremental update at all) is stage A's own baseline point and already
gives the paper's central online-protocol comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eals import experiment as xp  # noqa: E402
from eals import spark_eals_online as online  # noqa: E402
from eals.spark_utils import build_session  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "yelp.rating"
STREAM_DIR = ROOT / "data" / "stream"
CKPT_DIR = ROOT / ".spark-checkpoints" / "online"
RESULTS_SWEEP = ROOT / "results" / "online_sweep.csv"
RESULTS_FULL = ROOT / "results" / "online_stream_full.csv"
RESULTS_COLDSTART = ROOT / "results" / "online_coldstart.csv"

K, C0, ALPHA, LAM = 64, 2048.0, 0.5, 0.01  # our own weighting-sweep optimum (notes/04)
OFFLINE_ITERS, N_PARTS = 20, 8
BATCH_ROWS = 1000
SWEEP_EVENTS = 8000

N_ITERS_GRID = [0, 1, 2, 3, 5]
W_NEW_GRID = [1.0, 2.0, 4.0, 8.0, 16.0]
DEFAULT_W_NEW = 4.0


def aggregate(results: list[dict]) -> dict:
    ranks = np.array([r["rank"] for r in results], dtype=np.float64)
    hit10, hit100 = ranks <= 10, ranks <= 100
    return {
        "n": int(ranks.shape[0]),
        "mean_rank": float(ranks.mean()),
        "HR@10": float(hit10.mean()),
        "NDCG@10": float(np.where(hit10, 1.0 / np.log2(ranks + 1.0), 0.0).mean()),
        "HR@100": float(hit100.mean()),
        "NDCG@100": float(np.where(hit100, 1.0 / np.log2(ranks + 1.0), 0.0).mean()),
    }


def fresh_state(P, Q, c, split, w_new: float) -> online.OnlineState:
    return online.OnlineState.from_offline_fit(P, Q, c, lam=LAM, w_new=w_new, train_users=split.offline.users, train_items=split.offline.items)


if __name__ == "__main__":
    spark = build_session("online_experiment", n_cores=8)
    sc = spark.sparkContext

    split = xp.load_yelp_online_split(DATA_PATH, train_frac=0.9)
    print(f"offline: {split.offline.users.shape[0]} interactions, n_users={split.n_users}, n_items={split.n_items}")
    print(f"stream: {split.stream_users.shape[0]} events (raw, undeduped)")

    print(f"Fitting offline model: K={K}, c0={C0}, alpha={ALPHA}, {OFFLINE_ITERS} iterations...")
    P, Q, c = xp.fit_eals_for_online(sc, split, K=K, c0=C0, alpha=ALPHA, lam=LAM, n_iters=OFFLINE_ITERS, n_parts=N_PARTS)

    # --- Stage A: n_iters_online sweep, Fig. 6 analogue ---
    sub_u, sub_i, sub_ts = split.stream_users[:SWEEP_EVENTS], split.stream_items[:SWEEP_EVENTS], split.stream_ts[:SWEEP_EVENTS]
    online.write_stream_batches(sub_u, sub_i, sub_ts, STREAM_DIR / "sweep", BATCH_ROWS)

    print(f"Stage A: n_iters_online sweep on {SWEEP_EVENTS} events, w_new={DEFAULT_W_NEW}")
    stage_a = []
    for n_iters_online in N_ITERS_GRID:
        state = fresh_state(P, Q, c, split, DEFAULT_W_NEW)
        results = online.run_streaming(spark, state, STREAM_DIR / "sweep", CKPT_DIR / f"niter_{n_iters_online}", n_iters_online=n_iters_online)
        row = {"stage": "n_iters_sweep", "n_iters_online": n_iters_online, "w_new": DEFAULT_W_NEW, **aggregate(results)}
        xp.append_csv(RESULTS_SWEEP, row)
        stage_a.append(row)
        print(f"  n_iters_online={n_iters_online}  HR@10={row['HR@10']:.4f}  NDCG@10={row['NDCG@10']:.4f}")
    best_n_iters = max(stage_a, key=lambda r: r["HR@10"])["n_iters_online"]
    print(f"  best n_iters_online = {best_n_iters}")

    # --- Stage B: w_new sweep at best n_iters_online, Fig. 7 analogue ---
    print(f"Stage B: w_new sweep at n_iters_online={best_n_iters}")
    stage_b = []
    for w_new in W_NEW_GRID:
        state = fresh_state(P, Q, c, split, w_new)
        results = online.run_streaming(spark, state, STREAM_DIR / "sweep", CKPT_DIR / f"wnew_{w_new}", n_iters_online=best_n_iters)
        row = {"stage": "w_new_sweep", "n_iters_online": best_n_iters, "w_new": w_new, **aggregate(results)}
        xp.append_csv(RESULTS_SWEEP, row)
        stage_b.append(row)
        print(f"  w_new={w_new}  HR@10={row['HR@10']:.4f}  NDCG@10={row['NDCG@10']:.4f}")
    best_w_new = max(stage_b, key=lambda r: r["NDCG@10"])["w_new"]  # paper reports this sweep via NDCG
    print(f"  best w_new = {best_w_new}")

    # --- Stage C: one full-stream pass, Fig. 5 + Fig. 8 analogues ---
    print(f"Stage C: full stream ({split.stream_users.shape[0]} events) at n_iters_online={best_n_iters}, w_new={best_w_new}")
    online.write_stream_batches(split.stream_users, split.stream_items, split.stream_ts, STREAM_DIR / "full", BATCH_ROWS)
    state = fresh_state(P, Q, c, split, best_w_new)
    full_results = online.run_streaming(spark, state, STREAM_DIR / "full", CKPT_DIR / "full", n_iters_online=best_n_iters)

    import csv

    with RESULTS_FULL.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["batch_id", "user", "item", "ts", "prior_count", "rank"])
        w.writeheader()
        w.writerows(full_results)

    buckets = [0, 1, 2, 3, 4, 5]
    with RESULTS_COLDSTART.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prior_count_bucket", "n", "mean_rank", "HR@10", "NDCG@10", "HR@100", "NDCG@100"])
        w.writeheader()
        for b in buckets:
            sub = [r for r in full_results if r["prior_count"] == b] if b < 5 else [r for r in full_results if r["prior_count"] >= 5]
            if not sub:
                continue
            agg = aggregate(sub)
            label = str(b) if b < 5 else "5+"
            w.writerow({"prior_count_bucket": label, **agg})
            print(f"  prior_count={label}: n={agg['n']} HR@10={agg['HR@10']:.4f} NDCG@10={agg['NDCG@10']:.4f}")

    full_agg = aggregate(full_results)
    print(f"Full stream aggregate: HR@10={full_agg['HR@10']:.4f} NDCG@10={full_agg['NDCG@10']:.4f}")

    spark.stop()
