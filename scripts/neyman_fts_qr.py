#!/usr/bin/env python3
"""Quantile-regressed Neyman calibration for FTS vs LRS (counting model).

This script sets up a Poisson counting toy without nuisances, calibrates
FTS and LRS critical curves on a μ grid, optionally smooths them with
quantile regression, and records coverage plus interval-length diagnostics.
The output JSON is designed to feed directly into the plotting utilities.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

import sys
project_root = Path(__file__).resolve().parents[1]
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from fts_core import OptimizedFTSCalculator, ProductionFocusFunction
from publication_plotting import find_intervals_precise

try:
    from neyman import _maybe_fit_qr, _empirical_quantile
except Exception:  # noqa: BLE001
    def _empirical_quantile(a: np.ndarray, alpha: float) -> float:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0,1)")
        q = 1.0 - float(alpha)
        try:
            return float(np.quantile(a, q, method="linear"))
        except TypeError:  # NumPy < 1.22
            return float(np.quantile(a, q, interpolation="linear"))

    def _maybe_fit_qr(mu: np.ndarray, y: np.ndarray, alpha: float):  # noqa: ANN001
        return None


@dataclass
class CalibrationConfig:
    observed: int
    signal: float
    background: float
    mu_grid: np.ndarray
    alphas: Tuple[float, ...]
    n_toys: int
    theta_bounds: Tuple[float, float]
    fts_grid_points: int
    focus_mu: float
    focus_sigma: float
    focus_nsigma: float
    coverage_toys: Optional[int]
    seed: Optional[int]
    smoothing: bool
    output: Path
    model_tag: str
    confidence_level: float


class CountingNLLCalculator:
    """Minimal Poisson NLL calculator compatible with OptimizedFTSCalculator."""

    def __init__(self, signal: float, background: float) -> None:
        self.signal = float(signal)
        self.background = float(background)
        self._data: Dict[str, int] = {}
        self._cache: Dict[Tuple[str, float], float] = {}

    def register_dataset(self, dataset_id: str, count: int) -> None:
        self._data[dataset_id] = int(count)
        keys_to_drop = [k for k in self._cache if k[0] == dataset_id]
        for key in keys_to_drop:
            self._cache.pop(key, None)

    def forget_dataset(self, dataset_id: str) -> None:
        self._data.pop(dataset_id, None)
        keys_to_drop = [k for k in self._cache if k[0] == dataset_id]
        for key in keys_to_drop:
            self._cache.pop(key, None)

    def get_nll_at_mu(self, dataset_id: str, mu: float, use_cache: bool = True) -> float:
        mu = float(mu)
        key = (dataset_id, mu)
        if use_cache and key in self._cache:
            return self._cache[key]

        if dataset_id not in self._data:
            raise KeyError(f"Dataset '{dataset_id}' is not registered")

        n = self._data[dataset_id]
        lam = mu * self.signal + self.background
        if lam <= 0.0:
            raise ValueError(f"Non-positive expectation λ={lam} for μ={mu}")

        nll = lam - n * math.log(lam) + math.lgamma(n + 1.0)
        if use_cache:
            self._cache[key] = float(nll)
        return float(nll)


def parse_args() -> CalibrationConfig:
    parser = argparse.ArgumentParser(
        description="Quantile-regressed Neyman calibration for FTS vs LRS (counting toy)")
    parser.add_argument(
        "--observed",
        type=float,
        default=8.0,
        help="Observed event count (negative value auto-switches to Asimov)",
    )
    parser.add_argument("--signal", type=float, default=5.0, help="Signal yield at μ=1")
    parser.add_argument("--background", type=float, default=3.0, help="Background yield")
    parser.add_argument("--mu-min", dest="mu_min", type=float, default=0.0, help="Minimum μ")
    parser.add_argument("--mu-max", dest="mu_max", type=float, default=5.0, help="Maximum μ")
    parser.add_argument("--mu-n", dest="mu_n", type=int, default=61, help="Number of μ grid points")
    parser.add_argument(
        "--alphas",
        nargs="*",
        type=float,
        default=[0.3173, 0.0455],
        help="Tail probabilities (e.g. 0.3173 for 68% CL)",
    )
    parser.add_argument("--n-toys", type=int, default=2000, help="Toys per μ for calibration")
    parser.add_argument(
        "--coverage-toys",
        type=int,
        default=None,
        help="Optional extra toys per μ for coverage check (defaults to n-toys)",
    )
    parser.add_argument(
        "--theta-min",
        type=float,
        default=0.0,
        help="Lower θ bound passed to the FTS optimiser",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=100.0,
        help="Upper θ bound passed to the FTS optimiser",
    )
    parser.add_argument(
        "--fts-grid-points",
        type=int,
        default=401,
        help="Number of μ points in the FTS integration grid",
    )
    parser.add_argument("--focus-mu", type=float, default=1.0, help="Focus centre μ₀")
    parser.add_argument("--focus-sigma", type=float, default=0.5, help="Focus width σ")
    parser.add_argument(
        "--focus-nsigma",
        type=float,
        default=5.0,
        help="Grid half-width in units of σ around the focus centre",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable quantile-regression smoothing (pure empirical belt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base RNG seed (negative disables explicit seeding)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fts_interval_mu.json"),
        help="Destination JSON file",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="counts_no_nuisance_v1",
        help="Model identifier stored in JSON",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="CL for reporting the observed interval",
    )

    args = parser.parse_args()

    if args.mu_n < 2:
        parser.error("mu-n must be at least 2")
    if args.mu_min >= args.mu_max:
        parser.error("mu-min must be strictly less than mu-max")
    if not args.alphas:
        parser.error("At least one alpha value must be provided")
    if any(a <= 0.0 or a >= 1.0 for a in args.alphas):
        parser.error("All alpha values must lie in (0, 1)")
    if not (0.0 < args.confidence_level < 1.0):
        parser.error("confidence-level must be between 0 and 1")

    mu_grid = np.linspace(args.mu_min, args.mu_max, args.mu_n, dtype=float)
    if args.coverage_toys is not None and args.coverage_toys <= 0:
        parser.error("coverage-toys must be positive")

    if args.observed < 0:
        args.observed = args.background + args.signal

    return CalibrationConfig(
        observed=int(round(args.observed)),
        signal=args.signal,
        background=args.background,
        mu_grid=mu_grid,
        alphas=tuple(float(a) for a in args.alphas),
        n_toys=int(args.n_toys),
        theta_bounds=(args.theta_min, args.theta_max),
        fts_grid_points=int(args.fts_grid_points),
        focus_mu=args.focus_mu,
        focus_sigma=args.focus_sigma,
        focus_nsigma=args.focus_nsigma,
        coverage_toys=None if args.coverage_toys is None else int(args.coverage_toys),
        seed=None if args.seed < 0 else int(args.seed),
        smoothing=not args.no_smoothing,
        output=args.output,
        model_tag=args.model_tag,
        confidence_level=float(args.confidence_level),
    )


def build_focus(cfg: CalibrationConfig) -> ProductionFocusFunction:
    focus = ProductionFocusFunction(
        mu_focus=cfg.focus_mu,
        sigma_focus=cfg.focus_sigma,
        normalize=True,
    )
    focus.n_sigma = cfg.focus_nsigma  # Cool downstream metadata
    return focus


def generate_toys(
    *,
    calc: OptimizedFTSCalculator,
    nll_calc: CountingNLLCalculator,
    focus: ProductionFocusFunction,
    cfg: CalibrationConfig,
    rng: np.random.Generator,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate one α-level belt returning (C_lrs, C_fts, cov_lrs, cov_fts)."""
    mu_grid = cfg.mu_grid
    n_mu = len(mu_grid)

    lrs_toys = np.empty((n_mu, cfg.n_toys), dtype=float)
    fts_toys = np.empty((n_mu, cfg.n_toys), dtype=float)

    for j, mu0 in enumerate(mu_grid):
        for i in range(cfg.n_toys):
            n_obs = rng.poisson(mu0 * cfg.signal + cfg.background)
            dataset_id = f"toy_mu{j}_rep{i}"
            nll_calc.register_dataset(dataset_id, int(n_obs))

            fts_val, lrs_val = calc.compute_fts_and_lrt(
                nll_calc=nll_calc,
                dataset_id=dataset_id,
                mu0=float(mu0),
                focus_obj=focus,
                theta_bounds=cfg.theta_bounds,
                n_grid=cfg.fts_grid_points,
                verbose=False,
            )
            lrs_toys[j, i] = lrs_val
            fts_toys[j, i] = fts_val
            nll_calc.forget_dataset(dataset_id)

    C_lrs = np.array([_empirical_quantile(lrs_toys[j], alpha) for j in range(n_mu)], dtype=float)
    C_fts = np.array([_empirical_quantile(fts_toys[j], alpha) for j in range(n_mu)], dtype=float)

    if cfg.smoothing:
        predictor_lrs = _maybe_fit_qr(cfg.mu_grid, C_lrs, alpha)
        predictor_fts = _maybe_fit_qr(cfg.mu_grid, C_fts, alpha)
        if predictor_lrs is not None:
            C_lrs = predictor_lrs(cfg.mu_grid)
        if predictor_fts is not None:
            C_fts = predictor_fts(cfg.mu_grid)

    cov_lrs = np.mean(lrs_toys <= C_lrs[:, None], axis=1)
    cov_fts = np.mean(fts_toys <= C_fts[:, None], axis=1)
    return C_lrs, C_fts, cov_lrs, cov_fts


def coverage_cross_check(
    *,
    calc: OptimizedFTSCalculator,
    nll_calc: CountingNLLCalculator,
    focus: ProductionFocusFunction,
    cfg: CalibrationConfig,
    rng: np.random.Generator,
    critical_lrs: Mapping[float, np.ndarray],
    critical_fts: Mapping[float, np.ndarray],
    initial_cov_lrs: Mapping[float, np.ndarray],
    initial_cov_fts: Mapping[float, np.ndarray],
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """Optional extra toys to probe coverage using final smoothed curves."""
    if cfg.coverage_toys is None or cfg.coverage_toys <= 0:
        return (
            {alpha: np.array(initial_cov_lrs[alpha], copy=True) for alpha in cfg.alphas},
            {alpha: np.array(initial_cov_fts[alpha], copy=True) for alpha in cfg.alphas},
        )

    coverage_lrs: Dict[float, np.ndarray] = {}
    coverage_fts: Dict[float, np.ndarray] = {}

    for alpha in cfg.alphas:
        cov_lrs = np.zeros_like(cfg.mu_grid, dtype=float)
        cov_fts = np.zeros_like(cfg.mu_grid, dtype=float)
        C_lrs = critical_lrs[alpha]
        C_fts = critical_fts[alpha]

        for j, mu0 in enumerate(cfg.mu_grid):
            lrs_hits = 0
            fts_hits = 0
            for i in range(cfg.coverage_toys):
                n_obs = rng.poisson(mu0 * cfg.signal + cfg.background)
                dataset_id = f"cov_mu{j}_rep{i}"
                nll_calc.register_dataset(dataset_id, int(n_obs))
                fts_val, lrs_val = calc.compute_fts_and_lrt(
                    nll_calc=nll_calc,
                    dataset_id=dataset_id,
                    mu0=float(mu0),
                    focus_obj=focus,
                    theta_bounds=cfg.theta_bounds,
                    n_grid=cfg.fts_grid_points,
                    verbose=False,
                )
                if lrs_val <= C_lrs[j]:
                    lrs_hits += 1
                if fts_val <= C_fts[j]:
                    fts_hits += 1
                nll_calc.forget_dataset(dataset_id)
            cov_lrs[j] = lrs_hits / cfg.coverage_toys
            cov_fts[j] = fts_hits / cfg.coverage_toys

        coverage_lrs[alpha] = cov_lrs
        coverage_fts[alpha] = cov_fts

    return coverage_lrs, coverage_fts


def compute_observed_tracks(
    *,
    calc: OptimizedFTSCalculator,
    nll_calc: CountingNLLCalculator,
    focus: ProductionFocusFunction,
    cfg: CalibrationConfig,
    dataset_id: str,
) -> Tuple[np.ndarray, np.ndarray]:
    T_fts = np.empty_like(cfg.mu_grid)
    T_lrs = np.empty_like(cfg.mu_grid)
    for idx, mu0 in enumerate(cfg.mu_grid):
        fts_val, lrs_val = calc.compute_fts_and_lrt(
            nll_calc=nll_calc,
            dataset_id=dataset_id,
            mu0=float(mu0),
            focus_obj=focus,
            theta_bounds=cfg.theta_bounds,
            n_grid=cfg.fts_grid_points,
            verbose=False,
        )
        T_fts[idx] = fts_val
        T_lrs[idx] = lrs_val
    return T_fts, T_lrs


def interval_segments(mu_grid: np.ndarray, track: np.ndarray, critical: np.ndarray) -> List[List[float]]:
    segments = find_intervals_precise(mu_grid, track, critical)
    return [[float(lo), float(hi)] for lo, hi in segments]


def interval_length(segments: Iterable[Iterable[float]]) -> float:
    return float(sum(max(0.0, hi - lo) for lo, hi in segments))


def compute_asimov_lengths(
    *,
    calc: OptimizedFTSCalculator,
    nll_calc: CountingNLLCalculator,
    focus: ProductionFocusFunction,
    cfg: CalibrationConfig,
    critical_lrs: Mapping[float, np.ndarray],
    critical_fts: Mapping[float, np.ndarray],
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    lengths_lrs: Dict[float, np.ndarray] = {alpha: np.zeros_like(cfg.mu_grid) for alpha in cfg.alphas}
    lengths_fts: Dict[float, np.ndarray] = {alpha: np.zeros_like(cfg.mu_grid) for alpha in cfg.alphas}

    for j, mu_true in enumerate(cfg.mu_grid):
        n_asimov = int(round(mu_true * cfg.signal + cfg.background))
        dataset_id = f"asimov_{j}"
        nll_calc.register_dataset(dataset_id, n_asimov)
        track_fts, track_lrs = compute_observed_tracks(
            calc=calc,
            nll_calc=nll_calc,
            focus=focus,
            cfg=cfg,
            dataset_id=dataset_id,
        )
        for alpha in cfg.alphas:
            seg_lrs = interval_segments(cfg.mu_grid, track_lrs, critical_lrs[alpha])
            seg_fts = interval_segments(cfg.mu_grid, track_fts, critical_fts[alpha])
            lengths_lrs[alpha][j] = interval_length(seg_lrs)
            lengths_fts[alpha][j] = interval_length(seg_fts)
        nll_calc.forget_dataset(dataset_id)

    return lengths_lrs, lengths_fts


def main() -> None:
    cfg = parse_args()

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    focus = build_focus(cfg)
    calc = OptimizedFTSCalculator()
    nll_calc = CountingNLLCalculator(signal=cfg.signal, background=cfg.background)

    nll_calc.register_dataset("obs", cfg.observed)

    critical_lrs: Dict[float, np.ndarray] = {}
    critical_fts: Dict[float, np.ndarray] = {}
    coverage_lrs: Dict[float, np.ndarray] = {}
    coverage_fts: Dict[float, np.ndarray] = {}

    for alpha in cfg.alphas:
        C_lrs, C_fts, cov_lrs, cov_fts = generate_toys(
            calc=calc,
            nll_calc=nll_calc,
            focus=focus,
            cfg=cfg,
            rng=rng,
            alpha=alpha,
        )
        critical_lrs[alpha] = C_lrs
        critical_fts[alpha] = C_fts
        coverage_lrs[alpha] = cov_lrs
        coverage_fts[alpha] = cov_fts

    coverage_lrs, coverage_fts = coverage_cross_check(
        calc=calc,
        nll_calc=nll_calc,
        focus=focus,
        cfg=cfg,
        rng=rng,
        critical_lrs=critical_lrs,
        critical_fts=critical_fts,
        initial_cov_lrs=coverage_lrs,
        initial_cov_fts=coverage_fts,
    )

    obs_track_fts, obs_track_lrs = compute_observed_tracks(
        calc=calc,
        nll_calc=nll_calc,
        focus=focus,
        cfg=cfg,
        dataset_id="obs",
    )

    lengths_lrs, lengths_fts = compute_asimov_lengths(
        calc=calc,
        nll_calc=nll_calc,
        focus=focus,
        cfg=cfg,
        critical_lrs=critical_lrs,
        critical_fts=critical_fts,
    )

    interval_dict_lrs: Dict[str, List[List[float]]] = {}
    interval_dict_fts: Dict[str, List[List[float]]] = {}
    for alpha in cfg.alphas:
        interval_dict_lrs[f"{alpha:.6f}"] = interval_segments(cfg.mu_grid, obs_track_lrs, critical_lrs[alpha])
        interval_dict_fts[f"{alpha:.6f}"] = interval_segments(cfg.mu_grid, obs_track_fts, critical_fts[alpha])

    target_alpha = min(cfg.alphas, key=lambda a: abs(1.0 - a - cfg.confidence_level))
    observed_interval = {
        "confidence_level": cfg.confidence_level,
        "alpha_used": float(target_alpha),
        "lrs": interval_dict_lrs[f"{target_alpha:.6f}"],
        "fts": interval_dict_fts[f"{target_alpha:.6f}"],
    }

    metadata = {
        "schema_version": "1.0",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": "FTS_vs_LRS",
        "calibration": {
            "type": "Neyman",
            "alphas": [float(a) for a in cfg.alphas],
            "n_toys": cfg.n_toys,
            "coverage_toys": cfg.coverage_toys,
            "smoothing": "quantile_regression" if cfg.smoothing else "empirical",
        },
        "model": {
            "tag": cfg.model_tag,
            "observed": cfg.observed,
            "signal": cfg.signal,
            "background": cfg.background,
        },
        "focus": {
            "mu_focus": cfg.focus_mu,
            "sigma": cfg.focus_sigma,
            "n_sigma": cfg.focus_nsigma,
        },
        "settings": {
            "mu_grid": cfg.mu_grid.tolist(),
            "theta_bounds": list(cfg.theta_bounds),
            "fts_grid_points": cfg.fts_grid_points,
            "seed": cfg.seed,
        },
        "grid": {
            "mu": cfg.mu_grid.tolist(),
            "T_obs": {
                "fts": obs_track_fts.tolist(),
                "lrs": obs_track_lrs.tolist(),
            },
            "critical": {
                "fts": {f"{alpha:.6f}": critical_fts[alpha].tolist() for alpha in cfg.alphas},
                "lrs": {f"{alpha:.6f}": critical_lrs[alpha].tolist() for alpha in cfg.alphas},
            },
            "coverage": {
                "fts": {f"{alpha:.6f}": coverage_fts[alpha].tolist() for alpha in cfg.alphas},
                "lrs": {f"{alpha:.6f}": coverage_lrs[alpha].tolist() for alpha in cfg.alphas},
            },
            "asimov_interval_length": {
                "fts": {f"{alpha:.6f}": lengths_fts[alpha].tolist() for alpha in cfg.alphas},
                "lrs": {f"{alpha:.6f}": lengths_lrs[alpha].tolist() for alpha in cfg.alphas},
            },
        },
        "intervals": {
            "fts": interval_dict_fts,
            "lrs": interval_dict_lrs,
        },
        "observed_interval": observed_interval,
    }

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(metadata, indent=2))
    print(f"Saved FTS/LRS calibration summary to {cfg.output}")
    print(
        f"Observed {cfg.confidence_level*100:.1f}% interval (LRS): {interval_dict_lrs[f'{target_alpha:.6f}']}\n"
        f"Observed {cfg.confidence_level*100:.1f}% interval (FTS): {interval_dict_fts[f'{target_alpha:.6f}']}"
    )


if __name__ == "__main__":
    main()
