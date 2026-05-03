"""
Two strengthening experiments invoked after the main pipeline:
  (E8) Continuous beta-sweep -- the smooth obstruction-to-restoration
       transition predicted by Theorems 2 and 3.
  (E9) Universal coin-flip floor at beta=0 -- vary observation noise,
       particle count, and drift family; verify the X-only estimators
       hit 0.5 with tight CIs across the 13-condition grid. Direct
       empirical companion to Theorem 1.

Outputs:
  results/E8_beta_sweep.csv
  results/E9_universal_floor.csv
  figures/fig_extra_beta_sweep.pdf / .png
  figures/fig_extra_universal_floor.pdf / .png

Run from 'make reproduce' (preferred) or directly:
  python scripts/extras.py
"""
from __future__ import annotations
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Re-use simulator + estimators from the canonical pipeline
from reproduce import (
    SimConfig, simulate_drift, measure_x, measure_xy,
    pf_x_only, pf_xy, snapshot_xy, oracle_viterbi,
    bootstrap_ci, MASTER_SEED, OKABE_ITO,
)

RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.markersize": 4,
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
})


def trial_sign_error(estimator: str, cfg: SimConfig,
                      rng: np.random.Generator,
                      n_particles: int) -> float:
    delta = simulate_drift(cfg, rng)
    if estimator == "x_pf":
        obs = measure_x(delta, cfg, rng)
        est = pf_x_only(obs, cfg, n_particles, rng)
    elif estimator == "xy_snapshot":
        obs = measure_xy(delta, cfg, rng)
        est = snapshot_xy(obs, cfg)
    elif estimator == "xy_pf":
        obs = measure_xy(delta, cfg, rng)
        est = pf_xy(obs, cfg, n_particles, rng)
    elif estimator == "oracle_x":
        est = oracle_viterbi(cfg, delta, rng)
    else:
        raise ValueError(estimator)
    burn = 50
    return float(np.mean(np.sign(est["mean"][burn:]) != np.sign(delta[burn:])))


def experiment_E8_beta_sweep(n_trials: int = 100, n_particles: int = 1500
                              ) -> pd.DataFrame:
    """Continuous beta sweep: 9 beta values, 4 estimators, 100 trials each."""
    print(f"[E8] Running continuous beta sweep ({n_trials} trials x 9 beta x 4 est)...")
    import time; t0 = time.time()
    betas = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    estimators = ["x_pf", "oracle_x", "xy_snapshot", "xy_pf"]
    rows = []
    for beta in betas:
        cfg = SimConfig()
        cfg.theta = beta * cfg.sigma_stat()
        for est in estimators:
            for trial in range(n_trials):
                seed = MASTER_SEED + 80000 + int(beta*10000) + trial * 13 + hash(est) % 997
                rng = np.random.default_rng(seed)
                se = trial_sign_error(est, cfg, rng, n_particles)
                rows.append({"beta": beta, "estimator": est, "trial": trial,
                             "sign_error": se})
        print(f"  beta={beta:.2f} done")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "E8_beta_sweep.csv", index=False)
    print(f"[E8] done in {time.time()-t0:.1f}s")
    return df


def experiment_E9_universal_floor(n_trials: int = 100, n_particles: int = 1500
                                   ) -> pd.DataFrame:
    """At beta=0, vary shot noise, particle count, and drift family.
    Show all X-only estimators hit 0.5."""
    print(f"[E9] Running universal-floor panel at beta=0...")
    import time; t0 = time.time()
    rows = []
    rng_master = np.random.default_rng(MASTER_SEED + 90000)

    # Vary observation noise
    for sigma_obs in [0.05, 0.10, 0.15, 0.25, 0.40]:
        cfg = SimConfig(sigma_obs=sigma_obs)
        cfg.theta = 0.0
        for est in ["x_pf", "oracle_x", "xy_snapshot"]:
            for trial in range(n_trials):
                rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
                se = trial_sign_error(est, cfg, rng, n_particles)
                rows.append({"axis": "sigma_obs", "axis_value": sigma_obs,
                             "estimator": est, "trial": trial,
                             "sign_error": se})

    # Vary particle count for x_pf
    cfg = SimConfig(); cfg.theta = 0.0
    for n_part in [300, 800, 1500, 3000]:
        for trial in range(n_trials):
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
            se = trial_sign_error("x_pf", cfg, rng, n_part)
            rows.append({"axis": "n_particles", "axis_value": n_part,
                         "estimator": "x_pf", "trial": trial,
                         "sign_error": se})

    # Vary drift family at beta=0
    for fam in ["ou", "regime_switching", "reflecting_rw", "heavy_tailed"]:
        cfg = SimConfig(drift_family=fam); cfg.theta = 0.0
        for est in ["x_pf", "oracle_x", "xy_snapshot"]:
            for trial in range(n_trials):
                rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
                se = trial_sign_error(est, cfg, rng, n_particles)
                rows.append({"axis": "drift_family", "axis_value": fam,
                             "estimator": est, "trial": trial,
                             "sign_error": se})

    df = pd.DataFrame(rows)
    df.to_csv(RES / "E9_universal_floor.csv", index=False)
    print(f"[E9] done in {time.time()-t0:.1f}s")
    return df


# ---------- FIGURES ----------

def make_figure_beta_sweep(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    label_map = {"x_pf": r"$X$-only PF",
                 "oracle_x": "Viterbi oracle",
                 "xy_snapshot": r"$X+Y$ snapshot",
                 "xy_pf": r"$X+Y$ PF"}
    color_map = {"x_pf": OKABE_ITO[5], "oracle_x": OKABE_ITO[7],
                 "xy_snapshot": OKABE_ITO[1], "xy_pf": OKABE_ITO[3]}
    for est in ["x_pf", "oracle_x", "xy_snapshot", "xy_pf"]:
        sub = df[df.estimator == est]
        rng = np.random.default_rng(0)
        means, los, his = [], [], []
        betas = sorted(sub.beta.unique())
        for b in betas:
            v = sub[np.isclose(sub.beta, b)].sign_error.to_numpy()
            m, lo, hi = bootstrap_ci(v, n_boot=3000, rng=rng)
            means.append(m); los.append(lo); his.append(hi)
        ax.plot(betas, means, "o-", color=color_map[est], label=label_map[est])
        ax.fill_between(betas, los, his, color=color_map[est], alpha=0.2)
    ax.axhline(0.5, color="red", lw=0.7, ls="--", alpha=0.6, label="coin flip")
    ax.set_xlabel(r"Drift asymmetry $\beta = \theta / \sigma_{\rm stat}$")
    ax.set_ylabel("Sign error rate")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xticks([0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0", "0.05", "0.1", "0.2", "0.3", "0.5", "1.0", "2.0"])
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0)
    ax.set_title(r"Obstruction-to-restoration transition in $\beta$")
    plt.tight_layout()
    plt.savefig(FIG / "fig_extra_beta_sweep.pdf")
    plt.savefig(FIG / "fig_extra_beta_sweep.png", dpi=300)
    plt.close()


def make_figure_universal_floor(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    rng = np.random.default_rng(1)
    color_map = {"x_pf": OKABE_ITO[5], "oracle_x": OKABE_ITO[7],
                 "xy_snapshot": OKABE_ITO[1]}
    label_map = {"x_pf": r"$X$-only PF",
                 "oracle_x": "Viterbi oracle",
                 "xy_snapshot": r"$X+Y$ snapshot"}

    # (a) sigma_obs
    ax = axes[0]
    sub = df[df.axis == "sigma_obs"]
    for est in ["x_pf", "oracle_x", "xy_snapshot"]:
        means, los, his, xs = [], [], [], []
        for v in sorted(sub.axis_value.unique()):
            vals = sub[(sub.estimator == est) &
                       (sub.axis_value.astype(float) == float(v))].sign_error.to_numpy()
            if len(vals) == 0: continue
            m, lo, hi = bootstrap_ci(vals, n_boot=3000, rng=rng)
            means.append(m); los.append(lo); his.append(hi); xs.append(float(v))
        ax.errorbar(xs, means, yerr=[np.array(means)-np.array(los),
                                       np.array(his)-np.array(means)],
                    fmt="o-", color=color_map[est],
                    capsize=3, label=label_map[est])
    ax.axhline(0.5, color="red", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlabel(r"Observation noise $\sigma_{\rm obs}$")
    ax.set_ylabel(r"Sign error ($\beta = 0$)")
    ax.set_ylim(0, 0.6)
    ax.set_title("a  Vary observation noise", loc="left", fontweight="bold")

    # (b) particle count
    ax = axes[1]
    sub = df[df.axis == "n_particles"]
    means, los, his, xs = [], [], [], []
    for v in sorted(sub.axis_value.unique()):
        vals = sub[sub.axis_value.astype(float) == float(v)].sign_error.to_numpy()
        m, lo, hi = bootstrap_ci(vals, n_boot=3000, rng=rng)
        means.append(m); los.append(lo); his.append(hi); xs.append(float(v))
    ax.errorbar(xs, means, yerr=[np.array(means)-np.array(los),
                                   np.array(his)-np.array(means)],
                fmt="o-", color=color_map["x_pf"], capsize=3,
                label=label_map["x_pf"])
    ax.axhline(0.5, color="red", lw=0.7, ls="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Particle count")
    ax.set_title("b  Vary particle count", loc="left", fontweight="bold")
    ax.set_ylim(0, 0.6)

    # (c) drift family
    ax = axes[2]
    sub = df[df.axis == "drift_family"]
    fams = ["ou", "regime_switching", "reflecting_rw", "heavy_tailed"]
    fam_lab = ["OU", "Regime", "RRW", "Heavy"]
    width = 0.25
    for i, est in enumerate(["x_pf", "oracle_x", "xy_snapshot"]):
        means, los, his = [], [], []
        for fam in fams:
            v = sub[(sub.estimator == est) & (sub.axis_value == fam)].sign_error.to_numpy()
            m, lo, hi = bootstrap_ci(v, n_boot=3000, rng=rng)
            means.append(m); los.append(lo); his.append(hi)
        xs = np.arange(len(fams)) + (i-1)*width
        ax.bar(xs, means, width=width, color=color_map[est],
               edgecolor="k", lw=0.4, label=label_map[est])
        ax.errorbar(xs, means, yerr=[np.array(means)-np.array(los),
                                       np.array(his)-np.array(means)],
                    fmt="none", color="k", capsize=2, lw=0.6)
    ax.axhline(0.5, color="red", lw=0.7, ls="--", alpha=0.6)
    ax.set_xticks(np.arange(len(fams)))
    ax.set_xticklabels(fam_lab, fontsize=8)
    ax.set_title("c  Vary drift family", loc="left", fontweight="bold")
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper right", fontsize=7, ncol=1)

    plt.tight_layout()
    plt.savefig(FIG / "fig_extra_universal_floor.pdf")
    plt.savefig(FIG / "fig_extra_universal_floor.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    df8 = experiment_E8_beta_sweep(n_trials=100)
    df9 = experiment_E9_universal_floor(n_trials=100)
    make_figure_beta_sweep(df8)
    make_figure_universal_floor(df9)

    # Summary numbers for the manuscript
    rng = np.random.default_rng(2)
    print("\n=== Summary for manuscript text ===")
    for b in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        sub = df8[np.isclose(df8.beta, b)]
        for est in ["x_pf", "oracle_x", "xy_pf"]:
            v = sub[sub.estimator == est].sign_error.to_numpy()
            m, lo, hi = bootstrap_ci(v, n_boot=3000, rng=rng)
            print(f"  beta={b:.2f}  {est:12s}  {m:.3f} [{lo:.3f}, {hi:.3f}]")

    # Universal-floor: pool all X-only beta=0 trials and report the floor
    sub_all = df9[df9.estimator.isin(["x_pf", "oracle_x"])]
    pooled = sub_all.sign_error.to_numpy()
    pm, plo, phi = bootstrap_ci(pooled, n_boot=10000, rng=rng)
    print(f"\nPOOLED X-only floor across all conditions: {pm:.3f} [{plo:.3f}, {phi:.3f}], n={len(pooled)}")
