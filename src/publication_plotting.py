#!/usr/bin/env python3
"""
Publication-quality plotting utilities for FTS vs LRS
=====================================================

These helpers render dual-panel figures (top: LRS, bottom: FTS) matching
the style requested, without touching any statistical computations.

Key properties:
- Precise acceptance boundaries via linear interpolation (TS_obs ≤ C(μ)).
- Multi-segment acceptance support (if multiple crossings exist).
- No smoothing/monotonic forcing; no artificial clipping or shifts.
- Consistent styling, legends, reference lines, and endpoint labels.

Usage (inside a notebook or script):

    from publication_plotting import plot_fts_lrs_paper_style
    fig, axes = plot_fts_lrs_paper_style(
        mu_grid,
        T_lrs_obs, T_fts_obs,
        C_lrs_68, C_lrs_95,
        C_fts_68, C_fts_95,
        mu_asimov=1.0,
        focus_region=None,     # or (mu_center, sigma, n_sigma)
        xlim=(0, 4),
        savepath='results/physically_correct_fts_gaussian_sanity.png',
        dpi=300
    )

This module is plotting-only by design.
Feed it arrays computed by your own pipeline (observed and toy-calibrated critical curves).
"""

from typing import List, Sequence, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


def find_intervals_precise(mu_grid: Sequence[float],
                           ts_obs: Sequence[float],
                           crit: Sequence[float]) -> List[List[float]]:
    """Return acceptance intervals [mu_lo, mu_hi] with linear-interpolated boundaries.

    - Defines acceptance by test inversion: accept μ if TS_obs(μ) ≤ C(μ).
    - Supports multiple disjoint segments.
    - No smoothing or clipping; uses given arrays as-is.
    """
    mu = np.asarray(mu_grid, float)
    obs = np.asarray(ts_obs, float)
    c = np.asarray(crit, float)

    d = obs - c
    accepted = d <= 0
    if not np.any(accepted):
        return []

    edges = np.where(accepted[:-1] != accepted[1:])[0]
    points: List[float] = []

    # left endpoint if starting inside acceptance
    if accepted[0]:
        points.append(mu[0])

    # crossing points via linear interpolation
    for i in edges:
        d1, d2 = d[i], d[i + 1]
        if (d2 - d1) != 0:
            t = d1 / (d1 - d2)
            x = mu[i] + t * (mu[i + 1] - mu[i])
        else:
            x = mu[i]
        points.append(float(x))

    # right endpoint if ending inside acceptance
    if accepted[-1]:
        points.append(mu[-1])

    points = sorted(points)
    out: List[List[float]] = []
    for i in range(0, len(points), 2):
        if i + 1 < len(points):
            out.append([points[i], points[i + 1]])
    return out


def _draw_interval_bars(ax,
                        segments: List[List[float]],
                        y: float,
                        height: float,
                        color: str,
                        label: Optional[str] = None,
                        annotate: bool = True,
                        text_color: Optional[str] = None) -> None:
    if not segments:
        return
    for lo, hi in segments:
        ax.barh(y, hi - lo, left=lo, height=height, color=color, alpha=0.85,
                edgecolor='white', linewidth=0.6)
        ax.plot([lo, lo], [y - height / 2, y + height / 2], 'k-', lw=0.8, alpha=0.7)
        ax.plot([hi, hi], [y - height / 2, y + height / 2], 'k-', lw=0.8, alpha=0.7)
        if annotate:
            ax.text(lo, y - 0.85 * height - 0.4, f'{lo:.2f}', ha='center', va='top',
                    fontsize=9, color=text_color or color)
            ax.text(hi, y - 0.85 * height - 0.4, f'{hi:.2f}', ha='center', va='top',
                    fontsize=9, color=text_color or color)
    if label:
        cx = 0.5 * (segments[0][0] + segments[0][1])
        ax.text(cx, y, label, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')


def prepare_critical_arrays(
    C_lrs: dict,
    C_fts: dict,
    target_levels: Tuple[float, float] = (0.3173, 0.0455)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[str, str]]:
    """Select critical arrays for two target α levels (default: 68% and 95% CL).

    - Inputs C_lrs, C_fts are dicts mapping α -> array (as produced by your calibration).
    - Chooses the available α keys closest to the requested target levels.
    - Returns arrays in order (lrs_68, lrs_95, fts_68, fts_95) and human-readable labels.
    - If exact target α not present, nearest α is used and labels reflect the actual CL.
    """
    if not isinstance(C_lrs, dict) or not isinstance(C_fts, dict):
        raise TypeError("C_lrs and C_fts must be dicts mapping alpha -> array")

    def nearest_alpha(available: Sequence[float], target: float) -> float:
        av = list(available)
        return min(av, key=lambda a: abs(a - target))

    alphas_lrs = list(C_lrs.keys())
    alphas_fts = list(C_fts.keys())
    if not alphas_lrs or not alphas_fts:
        raise ValueError("Critical dicts have no alpha keys")

    a1_t, a2_t = target_levels
    a1_l = nearest_alpha(alphas_lrs, a1_t)
    a2_l = nearest_alpha(alphas_lrs, a2_t)
    a1_f = nearest_alpha(alphas_fts, a1_t)
    a2_f = nearest_alpha(alphas_fts, a2_t)

    L68 = np.asarray(C_lrs[a1_l], float)
    L95 = np.asarray(C_lrs[a2_l], float)
    F68 = np.asarray(C_fts[a1_f], float)
    F95 = np.asarray(C_fts[a2_f], float)

    def label(alpha: float) -> str:
        cl = (1.0 - float(alpha)) * 100.0
        # Round to nearest integer if close, else one decimal
        if abs(cl - round(cl)) < 0.25:
            return f"{int(round(cl))}% CL"
        return f"{cl:.1f}% CL"

    labels = (label(a1_l), label(a2_l))
    return L68, L95, F68, F95, labels


def plot_fts_lrs_paper_style(
    mu_grid: Sequence[float],
    T_lrs_obs: Sequence[float],
    T_fts_obs: Sequence[float],
    C_lrs_68: Sequence[float],
    C_lrs_95: Sequence[float],
    C_fts_68: Sequence[float],
    C_fts_95: Sequence[float],
    *,
    mu_asimov: float = 1.0,
    focus_region: Optional[Tuple[float, float, float]] = None,  # (center, sigma, n_sigma)
    xlim: Optional[Tuple[float, float]] = None,
    savepath: Optional[str] = None,
    dpi: int = 300,
    level_labels: Tuple[str, str] = ("68% CL", "95% CL"),
):
    """Render dual-panel (LRS, FTS) publication-quality figure.

    All arrays must be real outputs from your pipeline. This function does not
    modify or smooth any inputs; it only draws.
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    mu = np.asarray(mu_grid, float)
    Tl = np.asarray(T_lrs_obs, float)
    Tf = np.asarray(T_fts_obs, float)
    L68, L95 = np.asarray(C_lrs_68, float), np.asarray(C_lrs_95, float)
    F68, F95 = np.asarray(C_fts_68, float), np.asarray(C_fts_95, float)

    seg_l_68 = find_intervals_precise(mu, Tl, L68)
    seg_l_95 = find_intervals_precise(mu, Tl, L95)
    seg_f_68 = find_intervals_precise(mu, Tf, F68)
    seg_f_95 = find_intervals_precise(mu, Tf, F95)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Optional focus shadow
    if focus_region:
        m0, s, k = focus_region
        lo, hi = m0 - k * s, m0 + k * s
        ax1.axvspan(lo, hi, alpha=0.15, color='lightblue', label=f'Focus (μ={m0}, ±{k}σ)')
        ax2.axvspan(lo, hi, alpha=0.15, color='lightblue')

    # LRS panel
    ax1.plot(mu, Tl, color='#CC0000', lw=2.6, label='Observed LRS', zorder=5)
    ax1.plot(mu, L68, color='#CC0000', ls=':', lw=1.8, alpha=0.9, label='68% critical')
    ax1.plot(mu, L95, color='gray', ls='--', lw=1.6, alpha=0.9, label='95% critical')
    ax1.axhline(1.0, color='gray', ls=':', lw=1.0, alpha=0.5, label='χ²₁ 68% (≈1)')
    ax1.axhline(4.0, color='gray', ls=':', lw=1.0, alpha=0.5, label='χ²₁ 95% (≈4)')
    ax1.axvline(mu_asimov, color='gray', ls='-', lw=1.0, alpha=0.5)

    y_top_lrs = max(Tl.max(), L95.max(), 4.0) + 0.8
    bar_h = 0.35
    _draw_interval_bars(ax1, seg_l_68, y_top_lrs, bar_h, '#CC0000', level_labels[0], annotate=True, text_color='darkred')
    _draw_interval_bars(ax1, seg_l_95, y_top_lrs + 0.7, bar_h, '#990000', level_labels[1], annotate=True, text_color='darkred')
    ax1.set_ylabel('T(D_obs; μ)')
    ax1.set_title('Likelihood Ratio Statistic (LRS)')
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(0.0, max(y_top_lrs + 1.5, 1.2 * max(Tl.max(), L95.max(), 4.0)))

    # FTS panel
    ax2.plot(mu, Tf, color='#0066CC', lw=2.6, label='Observed FTS', zorder=5)
    ax2.plot(mu, F68, color='#0066CC', ls=':', lw=1.8, alpha=0.9, label='68% critical')
    ax2.plot(mu, F95, color='gray', ls='--', lw=1.6, alpha=0.9, label='95% critical')
    ax2.axhline(0.0, color='gray', ls=':', lw=1.0, alpha=0.5)
    ax2.axvline(mu_asimov, color='gray', ls='-', lw=1.0, alpha=0.5)

    y_top_fts = max(Tf.max(), F95.max(), 0.0) + 0.8
    _draw_interval_bars(ax2, seg_f_68, y_top_fts, bar_h, '#0066CC', level_labels[0], annotate=True, text_color='darkblue')
    _draw_interval_bars(ax2, seg_f_95, y_top_fts + 0.9, bar_h, '#004499', level_labels[1], annotate=True, text_color='darkblue')
    ax2.set_xlabel('μ')
    ax2.set_ylabel('T_F(D_obs; μ)')
    ax2.set_title('Focused Test Statistic (FTS)')
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(min(Tf.min(), F68.min(), F95.min(), 0.0) - 1.0,
                 max(y_top_fts + 1.5, 1.2 * max(Tf.max(), F95.max(), 0.0)))

    if xlim:
        ax2.set_xlim(*xlim)
    else:
        ax2.set_xlim(mu.min(), mu.max())

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    return fig, (ax1, ax2)
