#!/usr/bin/env python3
"""Adapter converting Neyman JSON outputs into NPZ arrays for plotting.

Inputs:
  * One or more RooStats Feldman–Cousins JSON files (LRS baseline).
  * The FTS/LRS joint calibration JSON produced by ``neyman_fts_qr.py``.

Outputs:
  * NPZ file containing μ grid, coverage curves, and interval-length tracks for
    the requested (1-α) confidence levels, ready for Fig. 2-style plots.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def match_alpha_key(keys: Iterable[str], alpha: float) -> str:
    best = min(keys, key=lambda k: abs(float(k) - alpha))
    return best


def poisson_pmf(n: int, lam: float) -> float:
    if lam <= 0.0:
        return 0.0
    return math.exp(-lam + n * math.log(lam) - math.lgamma(n + 1.0))


def coverage_from_belt(mu: np.ndarray, lower: List[float], upper: List[float], signal: float, background: float) -> np.ndarray:
    if len(lower) != len(mu) or len(upper) != len(mu):
        raise ValueError("Belt arrays must match μ grid length")
    cov = np.zeros_like(mu, dtype=float)
    for i, (mu_i, lo, hi) in enumerate(zip(mu, lower, upper)):
        if math.isnan(lo) or math.isnan(hi):
            cov[i] = float("nan")
            continue
        n_lo = int(math.ceil(lo))
        n_hi = int(math.floor(hi))
        if n_lo > n_hi:
            cov[i] = 0.0
            continue
        lam = mu_i * signal + background
        cov[i] = sum(poisson_pmf(n, lam) for n in range(n_lo, n_hi + 1))
    return cov


def belt_interval_length(mu: np.ndarray, lower: List[float], upper: List[float], n_obs: int) -> float:
    mu_arr = np.asarray(mu, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    d = np.maximum(lower_arr - n_obs, n_obs - upper_arr)
    accepted = d <= 0.0
    if not np.any(accepted):
        return 0.0

    edges = np.where(accepted[:-1] != accepted[1:])[0]
    points: List[float] = []
    if accepted[0]:
        points.append(mu_arr[0])
    for idx in edges:
        d1, d2 = d[idx], d[idx + 1]
        if (d2 - d1) != 0.0:
            t = d1 / (d1 - d2)
            x = mu_arr[idx] + t * (mu_arr[idx + 1] - mu_arr[idx])
        else:
            x = mu_arr[idx]
        points.append(float(x))
    if accepted[-1]:
        points.append(mu_arr[-1])

    length = 0.0
    for lo, hi in zip(points[0::2], points[1::2]):
        length += max(0.0, hi - lo)
    return float(length)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Neyman JSON outputs to NPZ for plotting")
    parser.add_argument("--fts-json", type=Path, required=True, help="FTS/LRS joint calibration JSON")
    parser.add_argument(
        "--lrs-json",
        type=Path,
        nargs="*",
        default=[],
        help="Optional Feldman-Cousins JSON files (one per confidence level)",
    )
    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=[0.3173, 0.0455],
        help="Alpha values (tail probabilities) to extract",
    )
    parser.add_argument("--out", type=Path, required=True, help="Destination NPZ path")
    args = parser.parse_args()

    fts_payload = load_json(args.fts_json)
    mu_grid = np.asarray(fts_payload["grid"]["mu"], dtype=float)
    signal = float(fts_payload["model"]["signal"])
    background = float(fts_payload["model"]["background"])

    crit_fts = fts_payload["grid"]["critical"]["fts"]
    crit_lrs = fts_payload["grid"]["critical"]["lrs"]
    cov_fts = fts_payload["grid"]["coverage"]["fts"]
    cov_lrs = fts_payload["grid"]["coverage"]["lrs"]
    len_fts = fts_payload["grid"]["asimov_interval_length"]["fts"]
    len_lrs = fts_payload["grid"]["asimov_interval_length"]["lrs"]

    data: Dict[str, np.ndarray] = {"mu": mu_grid}

    for alpha in args.alphas:
        key = match_alpha_key(crit_fts.keys(), alpha)
        data[f"critical_fts_{alpha:.6f}"] = np.asarray(crit_fts[key], dtype=float)
        data[f"critical_lrs_{alpha:.6f}"] = np.asarray(crit_lrs[key], dtype=float)
        cov_key = match_alpha_key(cov_fts.keys(), alpha)
        data[f"coverage_fts_{alpha:.6f}"] = np.asarray(cov_fts[cov_key], dtype=float)
        data[f"coverage_lrs_qr_{alpha:.6f}"] = np.asarray(cov_lrs[cov_key], dtype=float)
        len_key = match_alpha_key(len_fts.keys(), alpha)
        data[f"length_fts_{alpha:.6f}"] = np.asarray(len_fts[len_key], dtype=float)
        data[f"length_lrs_qr_{alpha:.6f}"] = np.asarray(len_lrs[len_key], dtype=float)

    for path in args.lrs_json:
        fc_payload = load_json(path)
        cl = float(fc_payload.get("confidence_level", 0.0))
        alpha = 1.0 - cl
        fc_mu = np.asarray(fc_payload["grid"]["mu"], dtype=float)
        if fc_mu.shape != mu_grid.shape or not np.allclose(fc_mu, mu_grid, rtol=0, atol=1e-9):
            raise ValueError("μ grid in FTS and RooStats payloads differ; rerun with matching grids")
        lower = fc_payload["grid"].get("lower", [])
        upper = fc_payload["grid"].get("upper", [])
        cov = coverage_from_belt(mu_grid, lower, upper, signal, background)
        data[f"coverage_lrs_fc_{alpha:.6f}"] = cov

        lengths = np.zeros_like(mu_grid)
        for idx, mu_true in enumerate(mu_grid):
            n_asimov = int(round(mu_true * signal + background))
            lengths[idx] = belt_interval_length(mu_grid, lower, upper, n_asimov)
        data[f"length_lrs_fc_{alpha:.6f}"] = lengths

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **data)
    print(f"Saved interval metrics to {args.out}")


if __name__ == "__main__":
    main()
