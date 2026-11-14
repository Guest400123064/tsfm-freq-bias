from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np


def load_tsf(f):
    r"""Parse a Monash ``.tsf`` file into ``(header, series)``.

    ``header`` is a dict of the ``@key value`` lines (keys lowercased);
    ``series`` is a list of ``(name, start, values)`` with ``values`` a numpy
    array of float64 where missing entries are NaN.
    """
    header = {}
    series = []
    in_data = False
    for raw in f:
        line = raw.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            if line.lower().startswith("@data"):
                in_data = True
                continue
            key, _, value = line[1:].partition(" ")
            header[key.lower()] = value.strip()
            continue
        if not in_data:
            continue
        # Data line: <name> : <start> : <v1>,<v2>,...
        name, _, rest = line.partition(":")
        start, _, vals = rest.partition(":")
        # Some variants carry extra metadata fields (metric labels, ids,
        # company names -- which may themselves contain commas) before the
        # value list; values are numbers/"?" and never contain ":", so the
        # value list is everything after the last colon.
        vals = vals.rsplit(":", 1)[-1]
        values = []
        for v in vals.split(","):
            v = v.strip()
            values.append(float("nan") if v in ("?", "") else float(v))
        series.append(
            (
                name.strip(),
                start.strip(),
                np.asarray(values, dtype=np.float64),
            )
        )
    return header, series


def monash_cache_dir():
    r"""Directory holding the ``Monash-University/monash_tsf`` hub snapshot."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return hf_home / "hub" / "datasets--Monash-University--monash_tsf"


def available_datasets():
    r"""Names of the Monash sub-datasets present in the local HF cache."""
    cache = monash_cache_dir()
    return sorted(z.stem for z in cache.glob("snapshots/*/data/*.zip"))


def load_monash(name):
    r"""Load one Monash sub-dataset from the local HF cache as ``(header, series)``."""
    cache = monash_cache_dir()
    matches = list(cache.glob("snapshots/*/data/*.zip"))
    zips = [z for z in matches if z.stem == name]
    if not zips:
        raise FileNotFoundError(
            f"Monash dataset {name!r} not found under {cache} "
            f"(available: {available_datasets()})"
        )
    with zipfile.ZipFile(zips[0]) as zf:
        tsf_name = zf.namelist()[0]
        with zf.open(tsf_name) as raw:
            return load_tsf(io.TextIOWrapper(raw, encoding="latin-1"))
