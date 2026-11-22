#!/usr/bin/env python3
"""
Fit and check an Elo-style win model on the CSVs made by lichess_split.py.

Model (white's expected score, where a draw counts as 0.5):
    p = 1 / (1 + 10 ** (-(gap + h) / s)),   gap = white_elo - black_elo
    s = rating scale (Elo uses 400), h = white's first-move advantage in rating points

Three models are compared:
    elo400          s = 400, h = 0           (the textbook Elo formula)
    elo400+white    s = 400, h fitted
    fitted          s and h both fitted      (by maximum likelihood on train.csv only)

Usage:
    pip install numpy pandas scipy matplotlib

    # Day-to-day: fit on train, evaluate on validation. Never touches test.csv.
    python fit_calibration.py --data data_blitz

    # ONCE, at the very end: evaluate the frozen parameters on the locked test set.
    python fit_calibration.py --data data_blitz --final

--final checks test.csv against the hash in MANIFEST.json, uses the parameters
saved from the train fit (params.json), and refuses to run a second time.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

EPS = 1e-12


def predict(gap, scale, h):
    return 1.0 / (1.0 + 10.0 ** (-(gap + h) / scale))


def log_loss(p, y):
    """Cross-entropy; y may be 0, 0.5 or 1 (draws count as half)."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def fit(gap, y, fit_scale, fit_h):
    """Maximum likelihood for (log s, h). Fixed parameters stay at 400 / 0."""
    def unpack(theta):
        i = 0
        scale, h = 400.0, 0.0
        if fit_scale:
            scale = float(np.exp(theta[i])); i += 1
        if fit_h:
            h = float(theta[i])
        return scale, h

    x0 = []
    if fit_scale:
        x0.append(np.log(400.0))
    if fit_h:
        x0.append(0.0)
    if not x0:
        return 400.0, 0.0
    res = minimize(lambda th: log_loss(predict(gap, *unpack(th)), y), x0, method="BFGS")
    return unpack(res.x)


def calibration_table(p, y, bins=20, min_n=50):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if n < min_n:
            continue
        mean_y = y[m].mean()
        se = y[m].std(ddof=1) / np.sqrt(n)
        rows.append({"n": n, "predicted": p[m].mean(), "actual": mean_y,
                     "lo": mean_y - 1.96 * se, "hi": mean_y + 1.96 * se})
    return pd.DataFrame(rows)


def load(data_dir, split):
    df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"))
    gap = (df["white_elo"] - df["black_elo"]).to_numpy(float)
    y = df["white_score"].to_numpy(float)
    avg = ((df["white_elo"] + df["black_elo"]) / 2).to_numpy(float)
    return gap, y, avg


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def report(name, models, gap, y, avg, out_dir):
    print(f"\n=== {name}: {len(y):,} games, observed white score {y.mean():.4f} ===")
    print(f"{'model':<14}{'scale s':>9}{'white h':>9}{'log loss':>11}{'Brier':>9}")
    preds = {}
    for label, (s, h) in models.items():
        p = predict(gap, s, h)
        preds[label] = p
        print(f"{label:<14}{s:>9.1f}{h:>9.1f}{log_loss(p, y):>11.5f}{brier(p, y):>9.5f}")

    # calibration plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, label="perfectly calibrated")
    for label, color in (("elo400", "tab:red"), ("fitted", "tab:blue")):
        tab = calibration_table(preds[label], y)
        ax.errorbar(tab["predicted"], tab["actual"],
                    yerr=[tab["actual"] - tab["lo"], tab["hi"] - tab["actual"]],
                    fmt="o-", ms=4, lw=1, capsize=2, color=color, label=label)
    ax.set_xlabel("predicted white score")
    ax.set_ylabel("actual white score")
    ax.set_title(f"Calibration on {name} (95% intervals)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(out_dir, f"calibration_{name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")

    # by rating level: is calibration the same for weak and strong players?
    bands = pd.cut(avg, [0, 1200, 1600, 2000, 4000], labels=["<1200", "1200-1599", "1600-1999", "2000+"])
    rows = []
    for band in bands.categories:
        m = np.asarray(bands == band)
        if m.sum() < 100:
            continue
        rows.append({"band": band, "n": int(m.sum()), "actual": y[m].mean(),
                     "pred_elo400": preds["elo400"][m].mean(), "pred_fitted": preds["fitted"][m].mean()})
    if rows:
        print("\nBy average rating of the two players:")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def improvement_ci(data_dir, split, gap, y, models, reps=1000, seed=2026):
    """Is the fitted model really better than Elo-400? Resample white players
    (approximation: games grouped by white player) and report 95% intervals."""
    df = pd.read_csv(os.path.join(data_dir, f"{split}.csv"), usecols=["white"])
    codes, uniques = pd.factorize(df["white"])
    n = len(uniques)
    rng = np.random.default_rng(seed)
    s, h = models["fitted"]
    p_elo, p_fit = predict(gap, 400.0, 0.0), predict(gap, s, h)

    def per_game(p):
        p = np.clip(p, EPS, 1 - EPS)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d_ll = per_game(p_elo) - per_game(p_fit)        # > 0 means the fitted model is better
    d_br = (p_elo - y) ** 2 - (p_fit - y) ** 2
    ll, br = [], []
    for _ in range(reps):
        w = rng.multinomial(n, np.full(n, 1.0 / n))[codes].astype(float)
        ll.append(np.sum(w * d_ll) / w.sum())
        br.append(np.sum(w * d_br) / w.sum())
    print(f"\nImprovement over Elo-400 on {split} ({reps} player-resampling rounds):")
    for name, mean, boot in (("log loss", d_ll.mean(), ll), ("Brier", d_br.mean(), br)):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {name:<9}{mean:.5f}   95% CI [{lo:.5f}, {hi:.5f}]  ({'excludes 0' if lo > 0 else 'includes 0'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_blitz", help="folder made by lichess_split.py")
    ap.add_argument("--final", action="store_true", help="evaluate once on the locked test set")
    args = ap.parse_args()

    params_path = os.path.join(args.data, "params.json")
    opened_path = os.path.join(args.data, "test_opened.txt")

    if args.final:
        if os.path.exists(opened_path):
            sys.exit(f"Test set already opened ({open(opened_path).read().strip()}). It can only be opened once.")
        if not os.path.exists(params_path):
            sys.exit("params.json not found. Run without --final first so parameters are fitted on train.")
        manifest = json.load(open(os.path.join(args.data, "MANIFEST.json")))
        actual = sha256_file(os.path.join(args.data, "test.csv"))
        if actual != manifest["sha256"]["test"]:
            sys.exit("test.csv does not match the hash in MANIFEST.json. The lock is broken; not evaluating.")
        models = {k: tuple(v) for k, v in json.load(open(params_path)).items()}
        gap, y, avg = load(args.data, "test")
        with open(opened_path, "w") as f:
            f.write("opened " + datetime.now(timezone.utc).isoformat(timespec="seconds"))
        report("test", models, gap, y, avg, args.data)
        improvement_ci(args.data, "test", gap, y, models)
        print("\nTest set opened. Report these numbers; do not tune anything now.")
        return

    gap, y, avg = load(args.data, "train")
    models = {
        "elo400": (400.0, 0.0),
        "elo400+white": (400.0, fit(gap, y, False, True)[1]),
        "fitted": fit(gap, y, True, True),
    }
    with open(params_path, "w") as f:
        json.dump({k: list(v) for k, v in models.items()}, f, indent=2)
    report("train", models, gap, y, avg, args.data)

    vgap, vy, vavg = load(args.data, "val")
    report("val", models, vgap, vy, vavg, args.data)


if __name__ == "__main__":
    main()
