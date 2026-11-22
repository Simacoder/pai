#!/usr/bin/env python3
"""
How sure are we? Cluster bootstrap for the fitted Elo-style model.

Games by the same player are not independent, so we resample PLAYERS, not games:
each bootstrap round draws white players with replacement and refits the model on
the matching (re-weighted) train games. This is an approximation: it groups games
by the white player only, so a player's games as black still count as independent.
Say so in the writeup.

Needs params.json from `python fit_calibration.py --data <folder>` first.
Never reads test.csv.

Usage:
    python bootstrap_ci.py --data data_blitz              # 500 rounds
    python bootstrap_ci.py --data data_blitz --reps 200   # quicker

Writes <folder>/bootstrap_results.json (also handy for the website).
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fit_calibration import predict, EPS


def per_game_loss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def weighted_fit(gap, y, w, x0):
    """x0 = [log s, h]. h is optimised in units of 100 rating points so the
    gradient is not vanishingly small (otherwise BFGS stops at the start)."""
    def nll(theta):
        p = predict(gap, np.exp(theta[0]), theta[1] * 100.0)
        return float(np.sum(w * per_game_loss(p, y)) / w.sum())
    res = minimize(nll, [x0[0], x0[1] / 100.0], method="BFGS", options={"gtol": 1e-9})
    return float(np.exp(res.x[0])), float(res.x[1] * 100.0)


def cluster_weights(codes, n_clusters, rng):
    """Bootstrap weight per game = how many times its player's cluster was drawn."""
    counts = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters))
    return counts[codes].astype(float)


def interval(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def load(data_dir, split):
    df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    gap = (df["white_elo"] - df["black_elo"]).to_numpy(float)
    y = df["white_score"].to_numpy(float)
    codes, uniques = pd.factorize(df["white"])
    return gap, y, codes, len(uniques)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_blitz")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    params = json.load(open(os.path.join(args.data, "params.json")))
    s0, h0 = params["fitted"]
    rng = np.random.default_rng(args.seed)

    tgap, ty, tcodes, tn = load(args.data, "train")
    vgap, vy, vcodes, vn = load(args.data, "val")
    print(f"train: {len(ty):,} games from {tn:,} white players; val: {len(vy):,} games from {vn:,}")

    # 1) uncertainty in the fitted parameters (train players resampled)
    t0 = time.time()
    fits = np.empty((args.reps, 2))
    x0 = [np.log(s0), h0]
    for i in range(args.reps):
        w = cluster_weights(tcodes, tn, rng)
        fits[i] = weighted_fit(tgap, ty, w, x0)
        if (i + 1) % 100 == 0:
            print(f"  ...{i + 1}/{args.reps} refits ({time.time() - t0:.0f}s)", flush=True)
    s_ci, h_ci = interval(fits[:, 0]), interval(fits[:, 1])

    # 2) is the improvement over Elo-400 real? (val players resampled, parameters fixed)
    p_elo = predict(vgap, 400.0, 0.0)
    p_fit = predict(vgap, s0, h0)
    d_ll = per_game_loss(p_elo, vy) - per_game_loss(p_fit, vy)   # > 0 means fitted is better
    d_br = (p_elo - vy) ** 2 - (p_fit - vy) ** 2
    boot_ll, boot_br = [], []
    for _ in range(args.reps):
        w = cluster_weights(vcodes, vn, rng)
        boot_ll.append(np.sum(w * d_ll) / w.sum())
        boot_br.append(np.sum(w * d_br) / w.sum())

    # 3) win-chance table for white, with a band for the fitted model
    gaps = list(range(-400, 401, 50))
    table = []
    for g in gaps:
        ps = predict(float(g), fits[:, 0], fits[:, 1])
        lo, hi = np.percentile(ps, [2.5, 97.5])
        table.append({"gap": g,
                      "elo400": float(predict(float(g), 400.0, 0.0)),
                      "fitted": float(predict(float(g), s0, h0)),
                      "fitted_lo": float(lo), "fitted_hi": float(hi)})

    out = {
        "reps": args.reps,
        "clustering": "train/val games grouped by white player (approximation)",
        "point_estimate": {"scale": s0, "white_advantage": h0},
        "ci95": {"scale": s_ci, "white_advantage": h_ci},
        "val_improvement_over_elo400": {
            "logloss": {"mean": float(d_ll.mean()), "ci95": interval(boot_ll)},
            "brier": {"mean": float(d_br.mean()), "ci95": interval(boot_br)},
        },
        "win_prob_table_white": table,
    }
    path = os.path.join(args.data, "bootstrap_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nscale s          {s0:7.1f}   95% CI [{s_ci[0]:.1f}, {s_ci[1]:.1f}]")
    print(f"white advantage  {h0:7.1f}   95% CI [{h_ci[0]:.1f}, {h_ci[1]:.1f}]  (rating points)")
    for name, key in (("log loss", "logloss"), ("Brier", "brier")):
        r = out["val_improvement_over_elo400"][key]
        verdict = "excludes 0" if r["ci95"][0] > 0 else "includes 0"
        print(f"val {name:8s} improvement over Elo-400: {r['mean']:.5f}  95% CI "
              f"[{r['ci95'][0]:.5f}, {r['ci95'][1]:.5f}]  ({verdict})")
    print("\nWhite win chance by rating gap (white minus black):")
    print(f"{'gap':>6}{'elo400':>9}{'fitted':>9}{'  95% band':>16}")
    for r in table:
        if r["gap"] % 100 == 0:
            print(f"{r['gap']:>6}{r['elo400']:>9.3f}{r['fitted']:>9.3f}   [{r['fitted_lo']:.3f}, {r['fitted_hi']:.3f}]")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
