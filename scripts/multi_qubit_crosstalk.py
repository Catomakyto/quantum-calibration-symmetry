"""
Two-qubit cross-talk experiment for Section S28.

Question: in a 2-qubit Ramsey calibration with product Z2 x Z2 symmetry,
does cross-talk between qubits restore sign observability that would be
permanently hidden in isolated symmetric-drift qubits?

Theory predictions (Theorem 4 / S28):
  - No cross-talk + theta_1 = theta_2 = 0:
      Both signs hidden.  Sign error -> 0.5 on each.
  - Cross-talk c12 = c21 = 0 + theta_2 != 0:
      Qubit 2 sign recoverable, qubit 1 sign hidden.
  - Cross-talk c12, c21 != 0 + theta_2 != 0:
      BOTH signs recoverable -- qubit 1 inherits restoration via
      cross-talk through spectator qubit 2.

Output:
  results/E10_crosstalk.csv
  figures/fig_extra_crosstalk.pdf / .png
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reproduce import bootstrap_ci, MASTER_SEED, OKABE_ITO

RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica"],
    "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.5, "lines.markersize": 4,
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
})


# ----------------------------------------------------------------------
# Two-qubit simulator with cross-talk
# ----------------------------------------------------------------------

def simulate_2qubit_drift(n_steps: int, theta_1: float, theta_2: float,
                          c12: float, c21: float, rho: float,
                          sigma_drift: float, rng: np.random.Generator
                          ) -> np.ndarray:
    """Joint OU-like drift on (delta_1, delta_2) with linear cross-coupling.

    delta_{t+1} = mu + rho * (delta_t - mu) + cross-coupling + noise
    where the cross-coupling is computed at delta_t.

    For c12 = c21 = 0 this reduces to two independent OU processes.
    For c != 0 it adds a deterministic linear coupling.
    """
    delta = np.empty((n_steps, 2))
    sigma_stat = sigma_drift / np.sqrt(1 - rho**2)
    delta[0, 0] = rng.normal(theta_1, sigma_stat)
    delta[0, 1] = rng.normal(theta_2, sigma_stat)
    for t in range(1, n_steps):
        # OU drift around (theta_1, theta_2) plus cross-coupling
        innov = sigma_drift * rng.standard_normal(2)
        delta[t, 0] = theta_1 + rho * (delta[t-1, 0] - theta_1) \
                       + c12 * delta[t-1, 1] + innov[0]
        delta[t, 1] = theta_2 + rho * (delta[t-1, 1] - theta_2) \
                       + c21 * delta[t-1, 0] + innov[1]
    return delta


def measure_x_2qubit(delta: np.ndarray, t_R: float, sigma_obs: float,
                      rng: np.random.Generator) -> np.ndarray:
    """X-quadrature observation per qubit."""
    return np.cos(delta * t_R) + sigma_obs * rng.standard_normal(delta.shape)


# ----------------------------------------------------------------------
# Joint particle filter (per-qubit sign tracking)
# ----------------------------------------------------------------------

def joint_pf_2qubit(obs: np.ndarray, theta_1: float, theta_2: float,
                    c12: float, c21: float, rho: float, sigma_drift: float,
                    sigma_obs: float, t_R: float, n_particles: int,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Joint particle filter on (delta_1, delta_2).
    Returns posterior mean trajectory (T, 2).
    """
    T = len(obs)
    sigma_stat = sigma_drift / np.sqrt(1 - rho**2)
    particles = rng.normal(0.0, 3.0 * sigma_stat, size=(n_particles, 2))
    weights = np.full(n_particles, 1.0 / n_particles)
    means = np.empty((T, 2))
    for t in range(T):
        # observation update on each qubit independently
        pred = np.cos(particles * t_R)  # shape (N, 2)
        log_w = -0.5 * np.sum(((obs[t] - pred) / sigma_obs) ** 2, axis=1)
        log_w -= log_w.max()
        weights = weights * np.exp(log_w)
        s = weights.sum()
        weights = weights / s if s > 0 else np.full(n_particles, 1.0 / n_particles)
        means[t] = np.sum(weights[:, None] * particles, axis=0)
        # ESS resample
        ess = 1.0 / np.sum(weights ** 2)
        if ess < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, p=weights)
            particles = particles[idx]
            weights = np.full(n_particles, 1.0 / n_particles)
        # propagate with the same drift model used to generate
        innov = sigma_drift * rng.standard_normal((n_particles, 2))
        new_p = np.empty_like(particles)
        new_p[:, 0] = theta_1 + rho * (particles[:, 0] - theta_1) \
                       + c12 * particles[:, 1] + innov[:, 0]
        new_p[:, 1] = theta_2 + rho * (particles[:, 1] - theta_2) \
                       + c21 * particles[:, 0] + innov[:, 1]
        particles = new_p
    return means


# ----------------------------------------------------------------------
# Trial harness
# ----------------------------------------------------------------------

def run_trial(theta_1: float, theta_2: float, c12: float, c21: float,
              rng: np.random.Generator, n_steps: int = 200,
              n_particles: int = 1500, t_R: float = 1.0,
              sigma_obs: float = 0.15, rho: float = 0.95,
              sigma_drift: float = 0.2, burn: int = 50) -> dict:
    delta = simulate_2qubit_drift(n_steps, theta_1, theta_2, c12, c21,
                                   rho, sigma_drift, rng)
    obs = measure_x_2qubit(delta, t_R, sigma_obs, rng)
    means = joint_pf_2qubit(obs, theta_1, theta_2, c12, c21, rho,
                             sigma_drift, sigma_obs, t_R, n_particles, rng)
    sign_err_1 = float(np.mean(np.sign(means[burn:, 0]) != np.sign(delta[burn:, 0])))
    sign_err_2 = float(np.mean(np.sign(means[burn:, 1]) != np.sign(delta[burn:, 1])))
    return {"sign_error_q1": sign_err_1, "sign_error_q2": sign_err_2}


def run_experiment(n_trials: int = 60) -> pd.DataFrame:
    """Three regimes:
       A. theta_1 = theta_2 = 0, c = 0  (full obstruction)
       B. theta_1 = 0, theta_2 = 0.5*sigma_stat, c = 0  (only q2 recoverable)
       C. theta_1 = 0, theta_2 = 0.5*sigma_stat, c = 0.05  (q1 inherits via spectator)
       D. theta_1 = 0, theta_2 = 0.5*sigma_stat, c = 0.10  (stronger spectator effect)
    """
    rho = 0.95
    sigma_drift = 0.2
    sigma_stat = sigma_drift / np.sqrt(1 - rho**2)
    theta_off = 0.5 * sigma_stat

    # Deterministic per-regime seed offsets (do NOT use Python's hash()
    # which is salted across processes).
    regimes = [
        ("A: theta=0, c=0",       0.0, 0.0,        0.0,  0.0,  0),
        ("B: theta_2!=0, c=0",    0.0, theta_off,  0.0,  0.0,  1),
        ("C: spectator c=0.05",   0.0, theta_off,  0.05, 0.05, 2),
        ("D: spectator c=0.10",   0.0, theta_off,  0.10, 0.10, 3),
    ]

    rows = []
    for label, t1, t2, c12, c21, regime_idx in regimes:
        for trial in range(n_trials):
            # Deterministic seed: master + regime offset + trial offset
            seed = MASTER_SEED + 100000 + regime_idx * 10000 + trial * 17
            rng = np.random.default_rng(seed)
            r = run_trial(t1, t2, c12, c21, rng)
            rows.append({"regime": label,
                         "theta_1": t1, "theta_2": t2,
                         "c12": c12, "c21": c21,
                         "trial": trial,
                         **r})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "E10_crosstalk.csv", index=False)
    return df


def make_figure(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    rng = np.random.default_rng(0)
    regimes = list(df.regime.unique())
    xs = np.arange(len(regimes))

    # Panel a: qubit 1 sign error
    ax = axes[0]
    pts, los, his = [], [], []
    for r in regimes:
        v = df[df.regime == r].sign_error_q1.to_numpy()
        m, lo, hi = bootstrap_ci(v, n_boot=3000, rng=rng)
        pts.append(m); los.append(lo); his.append(hi)
    colors = [OKABE_ITO[5], OKABE_ITO[5], OKABE_ITO[3], OKABE_ITO[3]]
    ax.bar(xs, pts, color=colors, edgecolor="k", lw=0.5)
    ax.errorbar(xs, pts,
                yerr=[np.array(pts) - np.array(los),
                      np.array(his) - np.array(pts)],
                fmt="none", color="k", capsize=3)
    ax.axhline(0.5, color="red", lw=0.7, ls="--", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(regimes, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(r"Sign error qubit 1 ($\theta_1 = 0$)")
    ax.set_ylim(0, 0.6)
    ax.set_title("a  Spectator-qubit restoration",
                 loc="left", fontweight="bold")

    # Panel b: qubit 2 sign error
    ax = axes[1]
    pts, los, his = [], [], []
    for r in regimes:
        v = df[df.regime == r].sign_error_q2.to_numpy()
        m, lo, hi = bootstrap_ci(v, n_boot=3000, rng=rng)
        pts.append(m); los.append(lo); his.append(hi)
    ax.bar(xs, pts, color=colors, edgecolor="k", lw=0.5)
    ax.errorbar(xs, pts,
                yerr=[np.array(pts) - np.array(los),
                      np.array(his) - np.array(pts)],
                fmt="none", color="k", capsize=3)
    ax.axhline(0.5, color="red", lw=0.7, ls="--", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(regimes, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(r"Sign error qubit 2 (the spectator)")
    ax.set_ylim(0, 0.6)
    ax.set_title("b  Spectator's own sign", loc="left", fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG / "fig_extra_crosstalk.pdf")
    plt.savefig(FIG / "fig_extra_crosstalk.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = run_experiment(n_trials=60)
    print(f"\n=== 2-qubit cross-talk results (n=60 trials/regime) ===")
    rng = np.random.default_rng(1)
    for r in df.regime.unique():
        sub = df[df.regime == r]
        m1, l1, h1 = bootstrap_ci(sub.sign_error_q1.to_numpy(), 3000, rng=rng)
        m2, l2, h2 = bootstrap_ci(sub.sign_error_q2.to_numpy(), 3000, rng=rng)
        print(f"  {r}")
        print(f"    qubit 1 sign error: {m1:.3f} [{l1:.3f}, {h1:.3f}]")
        print(f"    qubit 2 sign error: {m2:.3f} [{l2:.3f}, {h2:.3f}]")
    make_figure(df)
    print(f"\nTotal wall: {time.time()-t0:.1f}s")
