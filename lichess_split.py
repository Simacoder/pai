#!/usr/bin/env python3
"""
Stream a Lichess monthly PGN dump, filter it, and write time-based
train / validation / test CSVs plus a manifest that "locks" the test set.

Only the PGN headers are read (moves are ignored), so it is fast and uses
almost no memory. Handles .pgn, .pgn.zst, .pgn.bz2, or "-" for stdin.

Example (blitz, 5% sample, split by date):
    python lichess_split.py lichess_db_standard_rated_2016-03.pgn.bz2 \
        --time-control blitz --sample 0.05 --out data_blitz

    # or explicit dates:
    python lichess_split.py dump.pgn.zst --train-end 2024.03.14 --val-end 2024.03.21

Output columns:
    game_id, date, time, white, black, white_elo, black_elo,
    white_score (1 / 0.5 / 0), time_control, tc_class, termination
"""

import argparse
import bz2
import csv
import hashlib
import io
import json
import os
import random
import stat
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

# Lichess speed classes use estimated duration = base + 40 * increment (seconds)
TC_LIMITS = [("ultrabullet", 30), ("bullet", 180), ("blitz", 480), ("rapid", 1500)]

COLUMNS = [
    "game_id", "date", "time", "white", "black", "white_elo", "black_elo",
    "white_score", "time_control", "tc_class", "termination",
]
RESULT_TO_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def tc_class(tc):
    """Map a TimeControl header like '300+3' to a Lichess speed class."""
    if not tc or tc == "-":
        return None
    try:
        base, inc = tc.split("+")
        est = int(base) + 40 * int(inc)
    except ValueError:
        return None
    for name, limit in TC_LIMITS:
        if est < limit:
            return name
    return "classical"


def open_stream(path):
    if path == "-":
        return sys.stdin
    if path.endswith(".zst"):
        try:  # prefer the zstd binary: much faster than pure Python
            proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
            return io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace")
        except FileNotFoundError:
            import zstandard  # pip install zstandard
            fh = open(path, "rb")
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    if path.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_headers(stream, sample, rng, counts):
    """Yield one dict of header tags per game. Skipped games are never parsed."""
    tags = None
    skipping = False
    emitted = True
    for line in stream:
        if line.startswith("[Event "):
            if tags is not None and not emitted:
                yield tags
            counts["games_seen"] += 1
            if counts["games_seen"] % 1_000_000 == 0:
                print(f"  ...{counts['games_seen']:,} games scanned", file=sys.stderr)
            skipping = sample < 1.0 and rng.random() > sample
            tags = None if skipping else {}
            emitted = skipping
            if skipping:
                continue
        if tags is None:
            continue
        if line.startswith("["):
            key, _, val = line[1:].rstrip().rstrip("]").partition(" ")
            tags[key] = val.strip('"')
        elif line.strip() and not emitted:
            emitted = True
            yield tags
    if tags is not None and not emitted:
        yield tags


def to_row(t, args, counts):
    """Apply filters. Returns a row dict, or None (and counts the reason)."""

    def drop(reason):
        counts["dropped_" + reason] += 1
        return None

    if "Rated" not in t.get("Event", ""):
        return drop("not_rated")
    if t.get("Variant", "Standard") != "Standard":
        return drop("not_standard")
    score = RESULT_TO_SCORE.get(t.get("Result", ""))
    if score is None:
        return drop("no_result")
    term = t.get("Termination", "")
    if args.terminations and term not in args.terminations:
        return drop("termination")
    tcc = tc_class(t.get("TimeControl"))
    if tcc is None:
        return drop("no_time_control")
    if args.time_control != "all" and tcc != args.time_control:
        return drop("other_time_control")
    try:  # provisional / missing ratings are not plain integers
        we, be = int(t["WhiteElo"]), int(t["BlackElo"])
    except (KeyError, ValueError):
        return drop("bad_or_provisional_rating")
    if min(we, be) < args.min_rating:
        return drop("below_min_rating")
    if t.get("White", "?") == "?" or t.get("Black", "?") == "?":
        return drop("anonymous_player")

    date = t.get("UTCDate", t.get("Date", ""))
    if len(date) != 10:
        return drop("bad_date")
    return {
        "game_id": t.get("Site", "").rsplit("/", 1)[-1],
        "date": date,
        "time": t.get("UTCTime", ""),
        "white": t["White"],
        "black": t["Black"],
        "white_elo": we,
        "black_elo": be,
        "white_score": score,
        "time_control": t.get("TimeControl"),
        "tc_class": tcc,
        "termination": term,
    }


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="PGN file (.pgn, .pgn.zst, .pgn.bz2) or '-' for stdin")
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    ap.add_argument("--time-control", default="blitz",
                    choices=["ultrabullet", "bullet", "blitz", "rapid", "classical", "all"])
    ap.add_argument("--sample", type=float, default=1.0,
                    help="keep each game with this probability, e.g. 0.05 (default: 1.0)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-rating", type=int, default=0,
                    help="drop games where either player is below this rating")
    ap.add_argument("--terminations", nargs="*", default=["Normal", "Time forfeit"],
                    help="keep only these Termination values (pass none to keep all)")
    ap.add_argument("--train-end", help="last date in train, YYYY.MM.DD (default: 14th of first game's month)")
    ap.add_argument("--val-end", help="last date in validation, YYYY.MM.DD (default: 21st)")
    ap.add_argument("--max-games", type=int, default=0, help="stop after keeping this many (0 = no limit)")
    args = ap.parse_args()

    if not 0 < args.sample <= 1:
        ap.error("--sample must be in (0, 1]")

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    counts = Counter()
    split_counts = Counter()

    paths = {s: os.path.join(args.out, f"{s}.csv") for s in ("train", "val", "test")}
    for p in paths.values():
        if os.path.exists(p):
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)  # allow overwrite of a previous run
    files = {s: open(p, "w", newline="", encoding="utf-8") for s, p in paths.items()}
    writers = {s: csv.DictWriter(f, fieldnames=COLUMNS) for s, f in files.items()}
    for w in writers.values():
        w.writeheader()

    train_end, val_end = args.train_end, args.val_end
    started = time.time()
    stream = open_stream(args.path)
    try:
        for tags in iter_headers(stream, args.sample, rng, counts):
            row = to_row(tags, args, counts)
            if row is None:
                continue
            if train_end is None:  # infer from the first kept game's month
                ym = row["date"][:8]
                train_end = train_end or ym + "14"
                val_end = val_end or ym + "21"
                print(f"Split: train <= {train_end} < val <= {val_end} < test", file=sys.stderr)
            d = row["date"]
            split = "train" if d <= train_end else "val" if d <= val_end else "test"
            writers[split].writerow(row)
            split_counts[split] += 1
            counts["kept"] += 1
            if args.max_games and counts["kept"] >= args.max_games:
                break
    finally:
        for f in files.values():
            f.close()
        if stream is not sys.stdin:
            stream.close()

    # Lock the test set: record its hash and make the file read-only.
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_file": os.path.basename(args.path),
        "args": vars(args),
        "split_boundaries": {"train_end": train_end, "val_end": val_end},
        "rows": dict(split_counts),
        "filter_counts": dict(counts),
        "sha256": {s: sha256_file(p) for s, p in paths.items()},
        "note": "test.csv is read-only; open it once, at the end. If the hash changes, the lock is broken.",
    }
    with open(os.path.join(args.out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    os.chmod(paths["test"], stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

    print(f"\nScanned {counts['games_seen']:,} games, kept {counts['kept']:,} "
          f"in {time.time() - started:.0f}s", file=sys.stderr)
    for s in ("train", "val", "test"):
        print(f"  {s:5s} {split_counts[s]:>10,}", file=sys.stderr)
    drops = {k: v for k, v in counts.items() if k.startswith("dropped_")}
    if drops:
        print("Dropped:", ", ".join(f"{k[8:]}={v:,}" for k, v in sorted(drops.items())), file=sys.stderr)


if __name__ == "__main__":
    main()
