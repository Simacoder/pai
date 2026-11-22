#!/usr/bin/env python3
"""
Bundle every number the website needs into one small JSON file (site_data.json).

Run AFTER the final test evaluation (it refuses otherwise, to protect the lock):
    python export_site_data.py --data data_blitz

Reads:  params.json, bootstrap_results.json, MANIFEST.json, test_opened.txt, test.csv
Writes: <data>/site_data.json   (a few tens of KB; the site's limit is 5 MB)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from fit_calibration import predict, log_loss, brier, calibration_table, load


def r(x, nd=5):
    return None if x is None else round(float(x), nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_blitz")
    ap.add_argument("--out", default=None, help="default: <data>/site_data.json")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.data, "test_opened.txt")):
        raise SystemExit("The test set has not been opened yet. Run fit_calibration.py --final first.")

    params = json.load(open(os.path.join(args.data, "params.json")))
    boot = json.load(open(os.path.join(args.data, "bootstrap_results.json")))
    man = json.load(open(os.path.join(args.data, "MANIFEST.json")))
    gap, y, avg = load(args.data, "test")

    models, calib = [], {}
    for label in ("elo400", "elo400+white", "fitted"):
        s, h = params[label]
        p = predict(gap, s, h)
        models.append({"model": label, "scale": r(s, 1), "white_advantage": r(h, 1),
                       "log_loss": r(log_loss(p, y)), "brier": r(brier(p, y))})
        if label in ("elo400", "fitted"):
            tab = calibration_table(p, y)
            calib[label] = [{"predicted": r(a, 4), "actual": r(b, 4), "lo": r(c, 4), "hi": r(d, 4), "n": int(n)}
                            for a, b, c, d, n in zip(tab["predicted"], tab["actual"], tab["lo"], tab["hi"], tab["n"])]

    bands = pd.cut(avg, [0, 1200, 1600, 2000, 4000], labels=["under 1200", "1200-1599", "1600-1999", "2000+"])
    s_f, h_f = params["fitted"]
    rating_bands = []
    for band in bands.categories:
        m = np.asarray(bands == band)
        if m.sum() < 100:
            continue
        rating_bands.append({
            "band": band, "n": int(m.sum()), "actual": r(y[m].mean(), 4),
            "elo400": r(predict(gap[m], 400.0, 0.0).mean(), 4),
            "fitted": r(predict(gap[m], s_f, h_f).mean(), 4)})

    edges = list(range(-600, 601, 50))
    counts, _ = np.histogram(np.clip(gap, -599, 599), bins=edges)
    gap_hist = [{"from": a, "to": b, "n": int(n)} for a, b, n in zip(edges[:-1], edges[1:], counts)]

    out = {
        "title": "Are chess ratings calibrated?",
        "data": {
            "source": man.get("source_file"),
            "time_control": man.get("args", {}).get("time_control"),
            "sample_fraction": man.get("args", {}).get("sample"),
            "split_boundaries": man.get("split_boundaries"),
            "rows": man.get("rows"),
            "filter_counts": man.get("filter_counts"),
            "test_sha256": man.get("sha256", {}).get("test"),
        },
        "fitted": {"scale": r(s_f, 1), "white_advantage": r(h_f, 1),
                   "ci95": boot["ci95"], "bootstrap_reps": boot["reps"], "clustering": boot["clustering"]},
        "models_test": models,
        "val_improvement_over_elo400": boot["val_improvement_over_elo400"],
        "calibration_test": calib,
        "rating_bands_test": rating_bands,
        "win_prob_table_white": boot["win_prob_table_white"],
        "gap_histogram_test": gap_hist,
    }
    path = args.out or os.path.join(args.data, "site_data.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {path} ({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
