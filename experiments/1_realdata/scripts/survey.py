"""Survey candidate Monash corpora: shape, resolution, missingness, start times.

Run: .venv/bin/python experiments/1_realdata/scripts/survey.py
Writes runs/survey.json and prints one block per candidate dataset.
"""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np

from fbias.realdata import load_monash, monash_cache_dir

CANDIDATES = [
    "solar_10_minutes_dataset",
    "australian_electricity_demand_dataset",
    "wind_4_seconds_dataset",
    "oikolab_weather_dataset",
]

OUT = Path(__file__).resolve().parents[1] / "runs" / "survey.json"

# Monash @frequency strings, in minutes.
FREQ_MINUTES = {
    "secondly": 1 / 60,
    "4_seconds": 4 / 60,
    "10_seconds": 10 / 60,
    "minutely": 1.0,
    "10_minutes": 10.0,
    "half_hourly": 30.0,
    "hourly": 60.0,
    "daily": 1440.0,
    "weekly": 10080.0,
}

# Monash writes timestamps as "2006-01-01 00-00-01" or "2002-01-01 00:00:00".
TS = re.compile(r"(\d{4}-\d{2}-\d{2})(?:[ T](\d{2})[-:](\d{2})(?:[-:](\d{2}))?)?")

# Analysis config the corpus has to serve (PLAN.md Appendix C, S0/S2).
CTX, K = 512, 32
WIN = CTX + K


def zip_path(name):
    """Path of the cached zip for one Monash sub-dataset."""
    matches = list((monash_cache_dir() / "snapshots").glob("*/data/*.zip"))
    hits = [z for z in matches if z.stem == name]
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def first_data_lines(name, n=3):
    """The first ``n`` raw data lines of a sub-dataset, verbatim."""
    with zipfile.ZipFile(zip_path(name)) as zf:
        with zf.open(zf.namelist()[0]) as raw:
            out = []
            for line in io.TextIOWrapper(raw, encoding="latin-1"):
                if line.startswith(("@", "#")) or not line.strip():
                    continue
                out.append(line.rstrip()[:200])
                if len(out) == n:
                    return out
    return out


def parse_ts(s):
    """Timestamp in a start field, or None if the field is not a timestamp."""
    m = TS.search(s)
    if not m:
        return None
    date, hh, mm, ss = m.groups()
    stamp = f"{date}T{hh or '00'}:{mm or '00'}:{ss or '00'}"
    return np.datetime64(stamp)


def survey(name):
    t0 = time.time()
    header, series = load_monash(name)
    lengths = np.array([len(v) for _, _, v in series])
    nan_frac = np.array([np.isnan(v).mean() for _, _, v in series])
    std = np.array([np.nanstd(v) for _, _, v in series])

    starts_field = [s for _, s, _ in series]
    ts_from_field = [parse_ts(s) for s in starts_field]
    ts_from_line = [parse_ts(line) for line in first_data_lines(name, len(series))]
    ts = ts_from_line if all(t is not None for t in ts_from_line) else ts_from_field
    n_ok = sum(t is not None for t in ts)

    freq = header.get("frequency")
    step_min = FREQ_MINUTES.get(freq)
    grid_ok = None
    if n_ok > 1 and step_min:
        step = np.timedelta64(int(round(step_min * 60)), "s")
        diffs = np.diff(np.unique(np.array([t for t in ts if t is not None])))
        grid_ok = bool(np.all(diffs % step == np.timedelta64(0, "s")))

    per_series_windows = lengths // WIN
    info = {
        "n_series": len(series),
        "frequency": freq,
        "step_minutes": step_min,
        "length": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "max": int(lengths.max()),
            "total": int(lengths.sum()),
        },
        "lengths_distinct": int(len(set(lengths.tolist()))),
        "windows_per_series_nonoverlap": {
            "min": int(per_series_windows.min()),
            "median": float(np.median(per_series_windows)),
            "max": int(per_series_windows.max()),
        },
        "windows_total_nonoverlap": int(per_series_windows.sum()),
        "nan_frac_mean": float(nan_frac.mean()),
        "nan_frac_max": float(nan_frac.max()),
        "series_with_nan": int((nan_frac > 0).sum()),
        "per_series_std_min": float(std.min()),
        "per_series_std_max": float(std.max()),
        "scale_ratio_max_over_min": float(std.max() / std.min()),
        "start_field_is_timestamp": f"{sum(t is not None for t in ts_from_field)}"
        f"/{len(ts_from_field)}",
        "start_first": starts_field[0],
        "start_last": starts_field[-1],
        "n_distinct_starts": len(set(starts_field)),
        "starts_on_sampling_grid": grid_ok,
        "start_recovered_from_line": bool(ts_from_line[0] is not None),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"\n=== {name} ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("  raw lines:")
    for line in first_data_lines(name):
        print("   ", line[:120])
    return info


def main():
    out = {name: survey(name) for name in CANDIDATES}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
