#!/usr/bin/env python3
"""Render publication-ready figures from NPZ summaries.

Two modes are supported automatically based on the NPZ contents:
  1. Critical-curve mode (legacy): expects observed test-statistic tracks and
     critical curves (keys like ``T_lrs_obs`` and ``C_fts_68``) and produces the
     dual-panel TS-vs-critical plot.
  2. Interval-metric mode (new): expects coverage/length arrays (keys like
     ``coverage_fts_0.317300``) and produces Fig. 2-style panels comparing FTS
     and LRS coverage and interval length versus μ.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from publication_plotting import plot_fts_lrs_paper_style


def _collect_alpha_suffixes(data: np.lib.npyio.NpzFile, prefix: str) -> List[str]:
    return sorted({key.split(prefix)[-1] for key in data.files if key.startswith(prefix)})


def _alpha_label(alpha_str: str) -> str:
    alpha = float(alpha_str)
    cl = (1.0 - alpha) * 100.0
    if abs(cl - round(cl)) < 0.25:
        return f"{int(round(cl))}% CL"
    return f"{cl:.1f}% CL"


def plot_interval_metrics(mu, data, out_path, xlim=None):
    length_keys = _collect_alpha_suffixes(data, "length_fts_")
    coverage_keys = _collect_alpha_suffixes(data, "coverage_fts_")
    if not length_keys or not coverage_keys:
        raise KeyError("Interval-metric mode requires length_fts_* and coverage_fts_* keys")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    colors = {"fts": "#1f77b4", "lrs_qr": "#ff7f0e", "lrs_fc": "#2ca02c"}

    for alpha_str in length_keys:
        label = _alpha_label(alpha_str)
        axes[0].plot(mu, data[f"length_fts_{alpha_str}"], color=colors["fts"], lw=2, label=f"FTS {label}")
        if f"length_lrs_qr_{alpha_str}" in data:
            axes[0].plot(
                mu,
                data[f"length_lrs_qr_{alpha_str}"],
                color=colors["lrs_qr"],
                lw=1.6,
                linestyle="--",
                label=f"LRS (QR) {label}",
            )
        if f"length_lrs_fc_{alpha_str}" in data:
            axes[0].plot(
                mu,
                data[f"length_lrs_fc_{alpha_str}"],
                color=colors["lrs_fc"],
                lw=1.4,
                linestyle=":",
                label=f"LRS (FC) {label}",
            )

    axes[0].set_ylabel("Interval length")
    axes[0].set_xlabel("μ")
    axes[0].set_title("Interval length vs μ")
    axes[0].grid(alpha=0.3)

    for alpha_str in coverage_keys:
        label = _alpha_label(alpha_str)
        cl = 1.0 - float(alpha_str)
        axes[1].plot(mu, data[f"coverage_fts_{alpha_str}"], color=colors["fts"], lw=2, label=f"FTS {label}")
        if f"coverage_lrs_qr_{alpha_str}" in data:
            axes[1].plot(
                mu,
                data[f"coverage_lrs_qr_{alpha_str}"],
                color=colors["lrs_qr"],
                lw=1.6,
                linestyle="--",
                label=f"LRS (QR) {label}",
            )
        if f"coverage_lrs_fc_{alpha_str}" in data:
            axes[1].plot(
                mu,
                data[f"coverage_lrs_fc_{alpha_str}"],
                color=colors["lrs_fc"],
                lw=1.4,
                linestyle=":",
                label=f"LRS (FC) {label}",
            )
        axes[1].axhline(cl, color="gray", lw=0.9, linestyle="--", alpha=0.7)

    axes[1].set_ylabel("Coverage")
    axes[1].set_xlabel("μ")
    axes[1].set_title("Coverage vs μ")
    axes[1].grid(alpha=0.3)

    if xlim is not None:
        for ax in axes:
            ax.set_xlim(xlim)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Render publication-ready FTS/LRS figure from NPZ arrays")
    ap.add_argument('--npz', required=True, help='Path to NPZ file containing arrays')
    ap.add_argument('--out', required=True, help='Output image path')
    ap.add_argument('--mu-asimov', type=float, default=1.0, help='Reference μ for TS-panel mode')
    ap.add_argument('--xlim', nargs=2, type=float, default=None, help='Optional x-axis limits: lo hi')
    ap.add_argument('--focus-center', type=float, default=None)
    ap.add_argument('--focus-sigma', type=float, default=None)
    ap.add_argument('--focus-nsigma', type=float, default=1.0)
    args = ap.parse_args()

    data = np.load(args.npz)
    mu_key = 'mu_grid' if 'mu_grid' in data else 'mu'
    mu = data[mu_key]
    xlim = tuple(args.xlim) if args.xlim is not None else None

    legacy_required = {'T_lrs_obs', 'T_fts_obs', 'C_lrs_68', 'C_lrs_95', 'C_fts_68', 'C_fts_95'}
    if legacy_required.issubset(set(data.files)):
        focus_region = None
        if args.focus_center is not None and args.focus_sigma is not None:
            focus_region = (args.focus_center, args.focus_sigma, args.focus_nsigma)
        Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
        plot_fts_lrs_paper_style(
            mu,
            data['T_lrs_obs'],
            data['T_fts_obs'],
            data['C_lrs_68'],
            data['C_lrs_95'],
            data['C_fts_68'],
            data['C_fts_95'],
            mu_asimov=args.mu_asimov,
            focus_region=focus_region,
            xlim=xlim,
            savepath=args.out,
            dpi=300,
        )
        print(f"Saved: {args.out}")
    else:
        plot_interval_metrics(mu, data, args.out, xlim=xlim)


if __name__ == '__main__':
    main()
