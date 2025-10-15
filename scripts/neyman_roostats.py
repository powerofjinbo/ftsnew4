#!/usr/bin/env python3
"""Run a Feldman–Cousins Neyman construction with RooStats (no nuisances).

This helper builds a minimal counting experiment (signal + background without
nuisance parameters), runs the RooStats Feldman–Cousins construction for the
likelihood-ratio statistic, and saves the resulting interval and belt to JSON.

Usage example:

    python scripts/neyman_roostats.py \
        --observed 8 \
        --signal 5.0 \
        --background 3.0 \
        --mu-min 0.0 \
        --mu-max 5.0 \
        --nbins 60 \
        --confidence-level 0.95 \
        --output results/fc_interval_mu.json

The JSON output contains the observed interval and (optionally) the confidence
belt points sampled on the μ grid. Adapt the ``build_workspace`` function to use
an existing RooWorkspace/ModelConfig if needed; this script keeps
all nuisance parameters fixed for the no-systematics setup.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import platform

import ROOT  # type: ignore


ROOT.gROOT.SetBatch(True)


@dataclass
class NeymanConfig:
    observed: float
    signal: float
    background: float
    mu_bounds: Tuple[float, float]
    nbins: int
    confidence_level: float
    output: Path
    model_tag: str
    seed: int
    no_belt: bool


def build_workspace(cfg: NeymanConfig) -> Tuple[ROOT.RooWorkspace, ROOT.RooRealVar]:
    """Construct a simple signal+background counting workspace without nuisances."""

    mu_lo, mu_hi = cfg.mu_bounds
    w = ROOT.RooWorkspace("w", "Counting experiment workspace")

    mu = ROOT.RooRealVar("mu", "signal strength", mu_lo, mu_lo, mu_hi)
    mu.setVal((mu_lo + mu_hi) / 2.0)

    signal = ROOT.RooRealVar("s", "signal yield", cfg.signal)
    signal.setConstant(True)

    background = ROOT.RooRealVar("b", "background yield", cfg.background)
    background.setConstant(True)

    lam_hi = cfg.mu_bounds[1] * cfg.signal + cfg.background
    safety = lam_hi + 10.0 * math.sqrt(max(lam_hi, 1.0))
    obs_max = max(50.0, cfg.observed, safety)
    obs = ROOT.RooRealVar("n_obs", "observed events", 0.0, float(int(obs_max + 0.5)))
    obs.setBins(int(obs_max) + 1)

    n_exp = ROOT.RooFormulaVar("n_exp", "@0*@1 + @2", ROOT.RooArgList(mu, signal, background))
    model = ROOT.RooPoisson("model", "signal+background Poisson", obs, n_exp)

    getattr(w, "import")(mu)
    getattr(w, "import")(signal)
    getattr(w, "import")(background)
    getattr(w, "import")(obs)
    getattr(w, "import")(model)

    data = ROOT.RooDataSet("data", "observed data", ROOT.RooArgSet(obs))
    obs.setVal(cfg.observed)
    data.add(ROOT.RooArgSet(obs))
    getattr(w, "import")(data)

    return w, mu


def create_model_config(w: ROOT.RooWorkspace) -> ROOT.RooStats.ModelConfig:
    """Prepare a RooStats ModelConfig with μ as the sole POI."""

    mc = ROOT.RooStats.ModelConfig("ModelConfig", w)
    pdf = w.pdf("model")
    data = w.data("data")
    obs = w.var("n_obs")
    mu = w.var("mu")

    if pdf is None or data is None or obs is None or mu is None:
        raise RuntimeError("Workspace missing required components (mu, n_obs, model, data)")

    mc.SetPdf(pdf)
    mc.SetObservables(ROOT.RooArgSet(obs))
    mc.SetParametersOfInterest(ROOT.RooArgSet(mu))

    # Intentionally leave nuisances unset (no-systematics phase).
    return mc


def run_feldman_cousins(cfg: NeymanConfig) -> Dict[str, object]:
    """Execute Feldman–Cousins and collect interval / belt information."""

    w, mu = build_workspace(cfg)
    mc = create_model_config(w)
    data = w.data("data")
    if data is None:
        raise RuntimeError("Failed to retrieve dataset 'data' from workspace")

    if cfg.seed >= 0:
        ROOT.RooRandom.randomGenerator().SetSeed(cfg.seed)
        if hasattr(ROOT, "gRandom"):
            ROOT.gRandom.SetSeed(cfg.seed)

    fc = ROOT.RooStats.FeldmanCousins(data, mc)
    fc.SetConfidenceLevel(cfg.confidence_level)
    fc.SetTestSize(1.0 - cfg.confidence_level)
    fc.SetNBins(cfg.nbins)
    fc.UseAdaptiveSampling(True)
    fc.FluctuateNumDataEntries(False)
    try:
        fc.SetNEventsPerToy(1)
    except AttributeError:
        pass
    if not cfg.no_belt:
        fc.CreateConfBelt(True)

    mu_lo, mu_hi = cfg.mu_bounds
    try:
        fc.SetMuMinMax(mu_lo, mu_hi)
    except AttributeError:
        # Older ROOT versions do not expose SetMuMinMax; rely on the POI range instead.
        pass

    interval = fc.GetInterval()
    if interval is None:
        raise RuntimeError("FeldmanCousins returned a null interval")

    lower = float(interval.LowerLimit(mu))
    upper = float(interval.UpperLimit(mu))

    mu_values = [mu_lo + (mu_hi - mu_lo) * i / max(cfg.nbins - 1, 1) for i in range(cfg.nbins)]
    belt_points: List[Dict[str, float]] = []
    belt = None if cfg.no_belt else fc.GetConfidenceBelt()
    if belt:
        for value in mu_values:
            mu.setVal(value)
            poi_point = ROOT.RooArgSet(mu)
            try:
                lo = float(belt.GetLowerLimit(poi_point))
                hi = float(belt.GetUpperLimit(poi_point))
            except AttributeError:
                # Fallback if GetLowerLimit is unavailable.
                lo = float("nan")
                hi = float("nan")
            belt_points.append({"mu": value, "lower": lo, "upper": hi})

    grid_mu = mu_values
    grid_lower = [p["lower"] for p in belt_points]
    grid_upper = [p["upper"] for p in belt_points]

    root_version = ROOT.gROOT.GetVersion() if hasattr(ROOT.gROOT, "GetVersion") else "unknown"

    return {
        "schema_version": "1.1",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "method": "LRS",
        "construction": "FeldmanCousins",
        "ordering": "likelihood_ratio",
        "confidence_level": cfg.confidence_level,
        "poi": "mu",
        "grid": {
            "mu": grid_mu,
            "lower": grid_lower,
            "upper": grid_upper,
        },
        "observed_interval": {"lower": lower, "upper": upper},
        "model": {
            "tag": cfg.model_tag,
            "observed": cfg.observed,
            "signal": cfg.signal,
            "background": cfg.background,
        },
        "settings": {
            "mu_bounds": list(cfg.mu_bounds),
            "nbins": cfg.nbins,
            "seed": None if cfg.seed < 0 else cfg.seed,
            "belt_computed": not cfg.no_belt,
            "root_version": root_version,
            "python": platform.python_version(),
        },
    }


def parse_args() -> NeymanConfig:
    parser = argparse.ArgumentParser(description="Feldman-Cousins Neyman construction (no nuisances)")
    parser.add_argument(
        "--observed",
        type=float,
        default=8.0,
        help="Observed event count (negative value auto-switches to Asimov)",
    )
    parser.add_argument("--signal", type=float, default=5.0, help="Expected signal yield at mu=1")
    parser.add_argument("--background", type=float, default=3.0, help="Fixed background yield")
    parser.add_argument("--mu-min", dest="mu_min", type=float, default=0.0, help="Lower bound for mu")
    parser.add_argument("--mu-max", dest="mu_max", type=float, default=5.0, help="Upper bound for mu")
    parser.add_argument("--nbins", "--mu-n", type=int, default=60, help="Number of mu points for the belt")
    parser.add_argument(
        "--confidence-level",
        "--cl",
        type=float,
        default=0.95,
        help="Confidence level for the interval",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fc_interval_mu.json"),
        help="Destination JSON file",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default="counts_no_nuisance_v1",
        help="Metadata label stored in the JSON output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Optional random seed passed to RooRandom (negative disables explicit seeding)",
    )
    parser.add_argument(
        "--no-belt",
        action="store_true",
        help="Skip confidence-belt construction (faster when only the observed interval is needed)",
    )

    args = parser.parse_args()

    if not 0.0 < args.confidence_level < 1.0:
        parser.error("confidence level must be between 0 and 1")
    if args.mu_min >= args.mu_max:
        parser.error("mu-min must be strictly less than mu-max")
    if args.nbins < 2:
        parser.error("nbins must be >= 2")

    if args.observed < 0:
        args.observed = args.background + args.signal

    return NeymanConfig(
        observed=args.observed,
        signal=args.signal,
        background=args.background,
        mu_bounds=(args.mu_min, args.mu_max),
        nbins=args.nbins,
        confidence_level=args.confidence_level,
        output=args.output,
        model_tag=args.model_tag,
        seed=args.seed,
        no_belt=args.no_belt,
    )


def main() -> None:
    cfg = parse_args()
    results = run_feldman_cousins(cfg)

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(results, indent=2))

    print(f"Saved Feldman-Cousins interval to {cfg.output}")
    interval = results["observed_interval"]
    print(
        f"Observed interval at CL={cfg.confidence_level:.3f}: "
        f"[{interval['lower']:.4f}, {interval['upper']:.4f}]"
    )


if __name__ == "__main__":
    main()
