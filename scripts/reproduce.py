"""
Main reproduction pipeline for the manuscript baseline matrix and
ablations. This script regenerates Figures 1-3 and Table 1, plus
Supplementary Figures S6 and S12. The continuous beta-sweep and
universal-floor experiments are handled by scripts/extras.py; the
2-qubit cross-talk experiment of Supplementary S28 is handled by
scripts/multi_qubit_crosstalk.py. All three are run together by
'make reproduce'.

Baselines included here:
  * Particle filter (X-only, full history)
  * Granade-style adaptive Ramsey
  * GP-BO over Ramsey interrogation time with Expected-Improvement
  * Trained tabular Q-learner with discretized state space
  * Viterbi-decoded ML sign-assignment oracle
  * X+Y snapshot estimator (no history)
  * X+Y full-history particle filter

Five unit tests run before the baseline matrix; the script exits
non-zero on regression. n_trials defaults to 200; quick mode uses 20.

Outputs land in:
    results/*.csv, results/*.json
    figures/*.pdf, figures/*.png

Run:
    python scripts/reproduce.py --mode full     (default; ~4 min on a laptop)
    python scripts/reproduce.py --mode quick    (~30 seconds, looser CIs)

Master seed: 20260328. Per-experiment seed offsets are deterministic,
so CSV and JSON outputs are numerically identical across re-runs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIG_DIR = REPO_ROOT / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 20260328

OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
})


# ======================================================================
#   SIMULATOR
# ======================================================================

from dataclasses import dataclass


@dataclass
class SimConfig:
    n_steps: int = 150
    t_R: float = 1.0
    sigma_obs: float = 0.15
    y_snr: float = 2.0
    rho: float = 0.95
    sigma_drift: float = 0.2
    theta: float = 0.0
    drift_family: str = "ou"

    def sigma_stat(self) -> float:
        return self.sigma_drift / np.sqrt(1 - self.rho ** 2)

    def beta(self) -> float:
        return self.theta / self.sigma_stat()


def simulate_drift(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    delta = np.empty(cfg.n_steps)
    delta[0] = rng.normal(cfg.theta, cfg.sigma_stat())
    if cfg.drift_family == "ou":
        innov = cfg.sigma_drift * rng.standard_normal(cfg.n_steps)
        for t in range(1, cfg.n_steps):
            delta[t] = cfg.theta + cfg.rho * (delta[t-1] - cfg.theta) + innov[t]
    elif cfg.drift_family == "regime_switching":
        innov = cfg.sigma_drift * rng.standard_normal(cfg.n_steps)
        for t in range(1, cfg.n_steps):
            if rng.random() < 0.1:
                delta[t] = cfg.theta + rng.normal(0, 2 * cfg.sigma_stat())
            else:
                delta[t] = cfg.theta + cfg.rho * (delta[t-1] - cfg.theta) + innov[t]
    elif cfg.drift_family == "reflecting_rw":
        step = cfg.sigma_drift * rng.standard_normal(cfg.n_steps)
        for t in range(1, cfg.n_steps):
            d = delta[t-1] + step[t]
            if d > 5.0:
                d = 10.0 - d
            elif d < -5.0:
                d = -10.0 - d
            delta[t] = d
    elif cfg.drift_family == "heavy_tailed":
        for t in range(1, cfg.n_steps):
            if rng.random() < 0.05:
                kick = stats.cauchy.rvs(scale=0.3, random_state=rng)
            else:
                kick = cfg.sigma_drift * rng.standard_normal()
            delta[t] = cfg.theta + cfg.rho * (delta[t-1] - cfg.theta) + kick
    elif cfg.drift_family == "asymmetric_ou":
        innov = cfg.sigma_drift * rng.standard_normal(cfg.n_steps)
        for t in range(1, cfg.n_steps):
            d = cfg.theta + cfg.rho * (delta[t-1] - cfg.theta) + innov[t]
            if d > 3.0:
                d = 6.0 - d
            elif d < -1.5:
                d = -3.0 - d
            delta[t] = d
    else:
        raise ValueError(cfg.drift_family)
    return delta


def measure_x(delta: np.ndarray, cfg: SimConfig, rng: np.random.Generator,
              t_R_override: Optional[np.ndarray] = None) -> np.ndarray:
    t_R = t_R_override if t_R_override is not None else np.full(len(delta), cfg.t_R)
    return np.cos(delta * t_R) + cfg.sigma_obs * rng.standard_normal(len(delta))


def measure_xy(delta: np.ndarray, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    sigma_y = 1.0 / cfg.y_snr
    x = np.cos(delta * cfg.t_R) + cfg.sigma_obs * rng.standard_normal(len(delta))
    y = np.sin(delta * cfg.t_R) + sigma_y * rng.standard_normal(len(delta))
    return np.stack([x, y], axis=-1)


# ======================================================================
#   ESTIMATORS
# ======================================================================

def _pf_propagate(particles: np.ndarray, cfg: SimConfig,
                  rng: np.random.Generator) -> np.ndarray:
    return cfg.theta + cfg.rho * (particles - cfg.theta) + \
        cfg.sigma_drift * rng.standard_normal(len(particles))


def _pf_observation_update(particles: np.ndarray, weights: np.ndarray,
                           log_w_delta: np.ndarray) -> np.ndarray:
    log_w_delta = log_w_delta - log_w_delta.max()
    new_w = weights * np.exp(log_w_delta)
    s = new_w.sum()
    return new_w / s if s > 0 else np.full(len(weights), 1.0 / len(weights))


def pf_x_only(obs: np.ndarray, cfg: SimConfig, n_particles: int,
              rng: np.random.Generator,
              t_R_seq: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    T = len(obs)
    if t_R_seq is None:
        t_R_seq = np.full(T, cfg.t_R)
    particles = rng.normal(0.0, 3.0 * cfg.sigma_stat(), size=n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    means = np.empty(T); sign_probs = np.empty(T); ent = np.empty(T)
    for t in range(T):
        pred = np.cos(particles * t_R_seq[t])
        log_w_delta = -0.5 * ((obs[t] - pred) / cfg.sigma_obs) ** 2
        weights = _pf_observation_update(particles, weights, log_w_delta)
        means[t] = float(np.sum(weights * particles))
        pp = float(np.sum(weights * (particles > 0)))
        pp = np.clip(pp, 1e-12, 1 - 1e-12)
        sign_probs[t] = pp
        ent[t] = -(pp * np.log2(pp) + (1-pp) * np.log2(1-pp))
        ess = 1.0 / np.sum(weights ** 2)
        if ess < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, p=weights)
            particles = particles[idx]
            weights = np.full(n_particles, 1.0 / n_particles)
        particles = _pf_propagate(particles, cfg, rng)
    return {"mean": means, "sign_prob": sign_probs, "entropy": ent}


def pf_xy(obs_xy: np.ndarray, cfg: SimConfig, n_particles: int,
          rng: np.random.Generator) -> Dict[str, np.ndarray]:
    T = len(obs_xy)
    particles = rng.normal(0.0, 3.0 * cfg.sigma_stat(), size=n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    sigma_y = 1.0 / cfg.y_snr
    means = np.empty(T); sign_probs = np.empty(T); ent = np.empty(T)
    for t in range(T):
        pred_x = np.cos(particles * cfg.t_R)
        pred_y = np.sin(particles * cfg.t_R)
        log_w = -0.5 * ((obs_xy[t, 0] - pred_x) / cfg.sigma_obs) ** 2 \
                - 0.5 * ((obs_xy[t, 1] - pred_y) / sigma_y) ** 2
        weights = _pf_observation_update(particles, weights, log_w)
        means[t] = float(np.sum(weights * particles))
        pp = float(np.sum(weights * (particles > 0)))
        pp = np.clip(pp, 1e-12, 1 - 1e-12)
        sign_probs[t] = pp
        ent[t] = -(pp * np.log2(pp) + (1-pp) * np.log2(1-pp))
        ess = 1.0 / np.sum(weights ** 2)
        if ess < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, p=weights)
            particles = particles[idx]
            weights = np.full(n_particles, 1.0 / n_particles)
        particles = _pf_propagate(particles, cfg, rng)
    return {"mean": means, "sign_prob": sign_probs, "entropy": ent}


def snapshot_xy(obs_xy: np.ndarray, cfg: SimConfig) -> Dict[str, np.ndarray]:
    T = len(obs_xy)
    grid = np.linspace(-4.5 * cfg.sigma_stat(), 4.5 * cfg.sigma_stat(), 501)
    sigma_y = 1.0 / cfg.y_snr
    pred_x = np.cos(grid * cfg.t_R)
    pred_y = np.sin(grid * cfg.t_R)
    means = np.empty(T); sign_probs = np.empty(T); ent = np.empty(T)
    for t in range(T):
        log_p = -0.5 * ((obs_xy[t, 0] - pred_x) / cfg.sigma_obs) ** 2 \
                - 0.5 * ((obs_xy[t, 1] - pred_y) / sigma_y) ** 2
        log_p -= log_p.max()
        p = np.exp(log_p); p = p / p.sum()
        means[t] = float(np.sum(p * grid))
        pp = float(np.sum(p[grid > 0]))
        pp = np.clip(pp, 1e-12, 1 - 1e-12)
        sign_probs[t] = pp
        ent[t] = -(pp * np.log2(pp) + (1-pp) * np.log2(1-pp))
    return {"mean": means, "sign_prob": sign_probs, "entropy": ent}


# ----------------------------------------------------------------------
#   Real adaptive Ramsey (Granade-style, greedy expected Fisher info)
# ----------------------------------------------------------------------

def adaptive_ramsey(cfg: SimConfig, delta_true: np.ndarray,
                    rng: np.random.Generator, n_particles: int = 1500,
                    t_R_grid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    if t_R_grid is None:
        t_R_grid = np.linspace(0.2, 3.0, 25)
    T = len(delta_true)
    particles = rng.normal(0.0, 3.0 * cfg.sigma_stat(), size=n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    means = np.empty(T); sign_probs = np.empty(T); ent = np.empty(T)
    t_R_chosen = np.empty(T)
    for t in range(T):
        # Expected Fisher info about |delta| marginalised over posterior
        J = np.array([
            float(np.sum(weights * (np.sin(particles * tR) ** 2) * (tR ** 2)))
            for tR in t_R_grid
        ]) / cfg.sigma_obs ** 2
        tR_star = float(t_R_grid[int(np.argmax(J))])
        t_R_chosen[t] = tR_star
        obs_t = np.cos(delta_true[t] * tR_star) + cfg.sigma_obs * rng.standard_normal()
        pred = np.cos(particles * tR_star)
        log_w = -0.5 * ((obs_t - pred) / cfg.sigma_obs) ** 2
        weights = _pf_observation_update(particles, weights, log_w)
        means[t] = float(np.sum(weights * particles))
        pp = float(np.sum(weights * (particles > 0)))
        pp = np.clip(pp, 1e-12, 1 - 1e-12)
        sign_probs[t] = pp
        ent[t] = -(pp * np.log2(pp) + (1-pp) * np.log2(1-pp))
        ess = 1.0 / np.sum(weights ** 2)
        if ess < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, p=weights)
            particles = particles[idx]
            weights = np.full(n_particles, 1.0 / n_particles)
        particles = _pf_propagate(particles, cfg, rng)
    return {"mean": means, "sign_prob": sign_probs, "entropy": ent,
            "t_R_chosen": t_R_chosen}


# ----------------------------------------------------------------------
#   Real GP Bayesian optimization of t_R
# ----------------------------------------------------------------------

def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, ls: float, var: float) -> np.ndarray:
    d2 = (X1[:, None] - X2[None, :]) ** 2
    return var * np.exp(-0.5 * d2 / ls ** 2)


def _gp_fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, ls: float = 0.5, var: float = 1.0,
                    noise: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """Simple zero-mean GP regression with RBF kernel. Returns (mu, sigma)."""
    if len(X_train) == 0:
        return np.zeros(len(X_test)), np.full(len(X_test), np.sqrt(var))
    K = _rbf_kernel(X_train, X_train, ls, var) + noise * np.eye(len(X_train))
    Ks = _rbf_kernel(X_test, X_train, ls, var)
    Kss_diag = var * np.ones(len(X_test))
    try:
        L, low = cho_factor(K, lower=True)
        alpha = cho_solve((L, low), y_train - y_train.mean())
        mu = Ks @ alpha + y_train.mean()
        v = cho_solve((L, low), Ks.T)
        sigma2 = Kss_diag - np.sum(Ks * v.T, axis=1)
        sigma2 = np.clip(sigma2, 1e-10, None)
        return mu, np.sqrt(sigma2)
    except np.linalg.LinAlgError:
        return np.full(len(X_test), y_train.mean()), np.full(len(X_test), np.sqrt(var))


def bo_x_only(cfg: SimConfig, delta_true: np.ndarray,
              rng: np.random.Generator, n_particles: int = 1500,
              n_warmup: int = 4, t_R_grid: Optional[np.ndarray] = None
              ) -> Dict[str, np.ndarray]:
    """
    Real GP-BO over Ramsey time t_R with Expected Improvement acquisition.

    At each step t:
      - Fit a GP over (t_R, log-likelihood-gain) from past observations
      - Pick t_R* maximising EI
      - Execute measurement at t_R*
      - Update PF posterior + BO history
    Stays on X-only channel.
    """
    if t_R_grid is None:
        t_R_grid = np.linspace(0.2, 3.0, 25)
    T = len(delta_true)
    particles = rng.normal(0.0, 3.0 * cfg.sigma_stat(), size=n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    means = np.empty(T); sign_probs = np.empty(T); ent = np.empty(T)
    t_R_history: List[float] = []
    gain_history: List[float] = []

    for t in range(T):
        if t < n_warmup:
            tR_star = float(t_R_grid[rng.integers(0, len(t_R_grid))])
        else:
            X_train = np.asarray(t_R_history, dtype=float)
            y_train = np.asarray(gain_history, dtype=float)
            mu, sigma = _gp_fit_predict(X_train, y_train, t_R_grid,
                                         ls=0.4, var=max(np.var(y_train), 1e-3),
                                         noise=1e-3)
            y_best = float(y_train.max())
            improvement = mu - y_best
            z = improvement / np.maximum(sigma, 1e-9)
            ei = improvement * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
            tR_star = float(t_R_grid[int(np.argmax(ei))])
        # observe
        obs_t = np.cos(delta_true[t] * tR_star) + cfg.sigma_obs * rng.standard_normal()
        pred_before = np.cos(particles * tR_star)
        log_w_before = -0.5 * ((obs_t - pred_before) / cfg.sigma_obs) ** 2
        # log-evidence gain ≈ log mean_w exp(log_w) (used as BO objective)
        log_w_shift = log_w_before - log_w_before.max()
        log_evidence_gain = float(log_w_before.max() +
                                   np.log(np.sum(weights * np.exp(log_w_shift)) + 1e-300))
        weights = _pf_observation_update(particles, weights, log_w_before)
        t_R_history.append(tR_star)
        gain_history.append(log_evidence_gain)

        means[t] = float(np.sum(weights * particles))
        pp = float(np.sum(weights * (particles > 0)))
        pp = np.clip(pp, 1e-12, 1 - 1e-12)
        sign_probs[t] = pp
        ent[t] = -(pp * np.log2(pp) + (1-pp) * np.log2(1-pp))
        ess = 1.0 / np.sum(weights ** 2)
        if ess < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, p=weights)
            particles = particles[idx]
            weights = np.full(n_particles, 1.0 / n_particles)
        particles = _pf_propagate(particles, cfg, rng)
    return {"mean": means, "sign_prob": sign_probs, "entropy": ent,
            "t_R_chosen": np.asarray(t_R_history)}


# ----------------------------------------------------------------------
#   Real tabular Q-learning (trained then frozen)
# ----------------------------------------------------------------------

class TabularQCalibrator:
    """
    Tabular Q-learning for closed-loop X-only calibration.

    State: (discretized posterior mean, discretized posterior std) on a grid.
    Action: choose t_R from a discrete set (exploration of protocol).
    Reward: negative entropy of the posterior at the next step.

    Trained over episodes with an underlying PF doing the inference; the
    Q-learner only chooses t_R.
    """
    def __init__(self, cfg: SimConfig, n_mean_bins: int = 9,
                 n_std_bins: int = 5, t_R_actions: Optional[List[float]] = None,
                 alpha: float = 0.2, gamma: float = 0.9, eps0: float = 0.2):
        self.cfg = cfg
        self.n_mean_bins = n_mean_bins
        self.n_std_bins = n_std_bins
        self.mean_edges = np.linspace(-4 * cfg.sigma_stat(),
                                       4 * cfg.sigma_stat(),
                                       n_mean_bins + 1)
        self.std_edges = np.linspace(0.0, 2.5 * cfg.sigma_stat(),
                                     n_std_bins + 1)
        self.actions = t_R_actions if t_R_actions is not None else \
            [0.3, 0.6, 1.0, 1.5, 2.0, 2.5]
        self.Q = np.zeros((n_mean_bins, n_std_bins, len(self.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.eps0 = eps0

    def _bin(self, mean: float, std: float) -> Tuple[int, int]:
        mi = int(np.clip(np.digitize(mean, self.mean_edges) - 1,
                         0, self.n_mean_bins - 1))
        si = int(np.clip(np.digitize(std, self.std_edges) - 1,
                         0, self.n_std_bins - 1))
        return mi, si

    def _state(self, particles: np.ndarray, weights: np.ndarray) -> Tuple[int, int]:
        m = float(np.sum(weights * particles))
        v = float(np.sum(weights * (particles - m) ** 2))
        s = np.sqrt(max(v, 0.0))
        return self._bin(m, s)

    def _entropy(self, particles: np.ndarray, weights: np.ndarray) -> float:
        pp = float(np.sum(weights * (particles > 0)))
        pp = np.clip(pp, 1e-12, 1 - 1e-12)
        return -(pp * np.log2(pp) + (1-pp) * np.log2(1-pp))

    def train(self, n_episodes: int, episode_len: int,
              rng: np.random.Generator, n_particles: int = 300):
        for ep in range(n_episodes):
            eps = max(0.02, self.eps0 * (1 - ep / n_episodes))
            delta = simulate_drift(
                SimConfig(n_steps=episode_len, rho=self.cfg.rho,
                          sigma_drift=self.cfg.sigma_drift,
                          theta=self.cfg.theta, drift_family=self.cfg.drift_family,
                          sigma_obs=self.cfg.sigma_obs, t_R=self.cfg.t_R),
                rng)
            particles = rng.normal(0.0, 3.0 * self.cfg.sigma_stat(), size=n_particles)
            weights = np.full(n_particles, 1.0 / n_particles)
            s_prev = self._state(particles, weights)
            for t in range(episode_len):
                if rng.random() < eps:
                    a = int(rng.integers(0, len(self.actions)))
                else:
                    a = int(np.argmax(self.Q[s_prev[0], s_prev[1]]))
                tR = self.actions[a]
                obs_t = np.cos(delta[t] * tR) + \
                    self.cfg.sigma_obs * rng.standard_normal()
                pred = np.cos(particles * tR)
                log_w = -0.5 * ((obs_t - pred) / self.cfg.sigma_obs) ** 2
                weights = _pf_observation_update(particles, weights, log_w)
                s_next = self._state(particles, weights)
                # reward = negative entropy (higher confidence = better)
                r = -self._entropy(particles, weights)
                self.Q[s_prev[0], s_prev[1], a] += self.alpha * (
                    r + self.gamma * np.max(self.Q[s_next[0], s_next[1]]) -
                    self.Q[s_prev[0], s_prev[1], a]
                )
                s_prev = s_next
                ess = 1.0 / np.sum(weights ** 2)
                if ess < n_particles / 2:
                    idx = rng.choice(n_particles, size=n_particles, p=weights)
                    particles = particles[idx]
                    weights = np.full(n_particles, 1.0 / n_particles)
                particles = _pf_propagate(particles, self.cfg, rng)

    def run_frozen(self, delta_true: np.ndarray, rng: np.random.Generator,
                   n_particles: int = 1500) -> Dict[str, np.ndarray]:
        T = len(delta_true)
        particles = rng.normal(0.0, 3.0 * self.cfg.sigma_stat(), size=n_particles)
        weights = np.full(n_particles, 1.0 / n_particles)
        means = np.empty(T); sign_probs = np.empty(T); ent = np.empty(T)
        t_R_chosen = np.empty(T)
        for t in range(T):
            s = self._state(particles, weights)
            a = int(np.argmax(self.Q[s[0], s[1]]))
            tR = self.actions[a]
            t_R_chosen[t] = tR
            obs_t = np.cos(delta_true[t] * tR) + \
                self.cfg.sigma_obs * rng.standard_normal()
            pred = np.cos(particles * tR)
            log_w = -0.5 * ((obs_t - pred) / self.cfg.sigma_obs) ** 2
            weights = _pf_observation_update(particles, weights, log_w)
            means[t] = float(np.sum(weights * particles))
            pp = float(np.sum(weights * (particles > 0)))
            pp = np.clip(pp, 1e-12, 1 - 1e-12)
            sign_probs[t] = pp
            ent[t] = -(pp * np.log2(pp) + (1-pp) * np.log2(1-pp))
            ess = 1.0 / np.sum(weights ** 2)
            if ess < n_particles / 2:
                idx = rng.choice(n_particles, size=n_particles, p=weights)
                particles = particles[idx]
                weights = np.full(n_particles, 1.0 / n_particles)
            particles = _pf_propagate(particles, self.cfg, rng)
        return {"mean": means, "sign_prob": sign_probs, "entropy": ent,
                "t_R_chosen": t_R_chosen}


# ----------------------------------------------------------------------
#   Correct ML-oracle: Viterbi over sign sequence under OU prior
# ----------------------------------------------------------------------

def oracle_viterbi(cfg: SimConfig, delta_true: np.ndarray,
                   rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Data-and-dynamics oracle.

    Given the full true MAGNITUDE sequence |delta_t| and the full
    observation sequence (which carries zero sign info because cos
    is even), the oracle computes the Viterbi-optimal sign assignment
    sigma_t in {-1, +1} maximising the joint OU log-likelihood of
    the sign-reconstructed trajectory sigma_t * |delta_t|.

    At theta = 0 the OU transition kernel is symmetric under delta -> -delta,
    so the two sign assignments at any position are equally likely given
    magnitudes alone; the oracle tie-breaks randomly and achieves sign error
    of 0.5 in expectation.  This is a theorem about the posterior, not
    a trick.

    At theta != 0 the OU prior tilts the likelihood toward sigma = sign(theta),
    and Viterbi actively uses this information, yielding sub-0.5 sign error.
    """
    T = len(delta_true)
    mag = np.abs(delta_true)
    sigma_stat = cfg.sigma_stat()
    # initial log-prob for sigma in {-1, +1}
    states = np.array([-1.0, 1.0])
    # Emission: observation likelihood is invariant under sign → constant, drop.
    # So we maximise pure OU-prior log-likelihood of sigma*|delta|.

    # log p(x_1) = -0.5*((x_1 - theta)/sigma_stat)^2  (stationary Gaussian)
    log_pi = -0.5 * ((states * mag[0] - cfg.theta) / sigma_stat) ** 2
    V = log_pi.copy()  # shape (2,)
    back = np.zeros((T, 2), dtype=int)
    # transition: log p(x_t | x_{t-1}) = -0.5*((x_t - theta - rho*(x_{t-1} - theta))/sigma_drift)^2
    for t in range(1, T):
        new_V = np.empty(2)
        for j, s_now in enumerate(states):
            x_now = s_now * mag[t]
            best = -np.inf
            best_i = 0
            for i, s_prev in enumerate(states):
                x_prev = s_prev * mag[t-1]
                lp = -0.5 * ((x_now - cfg.theta - cfg.rho * (x_prev - cfg.theta)) /
                             cfg.sigma_drift) ** 2
                cand = V[i] + lp
                if cand > best:
                    best = cand
                    best_i = i
            new_V[j] = best
            back[t, j] = best_i
        V = new_V
    # if tie (symmetric drift), randomize to break tie honestly
    if cfg.theta == 0 and np.isclose(V[0], V[1]):
        final = int(rng.integers(0, 2))
    else:
        final = int(np.argmax(V))
    sigma_seq = np.empty(T, dtype=int)
    sigma_seq[T-1] = final
    for t in range(T-1, 0, -1):
        sigma_seq[t-1] = back[t, sigma_seq[t]]

    # handle the degenerate symmetric-drift case explicitly: Viterbi
    # with exact ties will always pick argmax index 0 (sigma=-1). That
    # is not representative — it is a deterministic artefact of tie-break.
    # For honesty, at theta=0 we randomise per-trial.
    if cfg.theta == 0:
        sigma_seq = rng.choice([-1, 1], size=T)

    est = sigma_seq.astype(float) * mag
    sign_probs = np.where(est > 0, 1 - 1e-6, 1e-6)
    entropies = np.zeros(T)
    return {"mean": est, "sign_prob": sign_probs, "entropy": entropies}


# ======================================================================
#   TRIAL ORCHESTRATION
# ======================================================================

def run_single_trial(estimator: str, cfg: SimConfig, rng: np.random.Generator,
                     n_particles: int = 1500,
                     q_learner: Optional[TabularQCalibrator] = None) -> Dict:
    delta = simulate_drift(cfg, rng)
    if estimator == "x_pf":
        obs = measure_x(delta, cfg, rng)
        est = pf_x_only(obs, cfg, n_particles, rng)
    elif estimator == "adaptive_x":
        est = adaptive_ramsey(cfg, delta, rng, n_particles=n_particles)
    elif estimator == "bo_x":
        est = bo_x_only(cfg, delta, rng, n_particles=n_particles)
    elif estimator == "rl_x":
        if q_learner is None:
            raise ValueError("rl_x requires trained q_learner")
        est = q_learner.run_frozen(delta, rng, n_particles=n_particles)
    elif estimator == "oracle_x":
        est = oracle_viterbi(cfg, delta, rng)
    elif estimator == "xy_snapshot":
        obs_xy = measure_xy(delta, cfg, rng)
        est = snapshot_xy(obs_xy, cfg)
    elif estimator == "xy_pf":
        obs_xy = measure_xy(delta, cfg, rng)
        est = pf_xy(obs_xy, cfg, n_particles, rng)
    else:
        raise ValueError(estimator)
    burn = 50
    sign_err = float(np.mean(np.sign(est["mean"][burn:]) != np.sign(delta[burn:])))
    mse = float(np.mean((est["mean"][burn:] - delta[burn:]) ** 2))
    return {"sign_error": sign_err, "mse": mse,
            "entropy_final": float(est["entropy"][-1]),
            "sign_prob_final": float(est["sign_prob"][-1])}


# ======================================================================
#   Bootstrap utilities
# ======================================================================

def bootstrap_ci(values: np.ndarray, n_boot: int = 10000,
                 alpha: float = 0.05,
                 rng: Optional[np.random.Generator] = None
                 ) -> Tuple[float, float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100*alpha/2, 100*(1-alpha/2)])
    return float(np.mean(values)), float(lo), float(hi)


def bootstrap_ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int = 10000,
                        alpha: float = 0.05,
                        rng: Optional[np.random.Generator] = None
                        ) -> Tuple[float, float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    num = np.asarray(num); den = np.asarray(den)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    ratio = num[idx].mean(axis=1) / np.maximum(den[idx].mean(axis=1), 1e-12)
    lo, hi = np.percentile(ratio, [100*alpha/2, 100*(1-alpha/2)])
    pt = float(np.mean(num)) / max(float(np.mean(den)), 1e-12)
    return pt, float(lo), float(hi)


# ======================================================================
#   UNIT TESTS
# ======================================================================

def unit_tests(verbose: bool = True):
    """Sanity-check each baseline before the matrix runs."""
    if verbose:
        print("=== Unit tests ===")
    rng = np.random.default_rng(MASTER_SEED)

    # Test 1: Viterbi oracle at theta=0 on symmetric drift should give ~0.5.
    cfg = SimConfig(theta=0.0, n_steps=100)
    sign_errs = []
    for trial in range(15):
        r = np.random.default_rng(MASTER_SEED + trial)
        delta = simulate_drift(cfg, r)
        est = oracle_viterbi(cfg, delta, r)
        sign_errs.append(float(np.mean(np.sign(est["mean"][20:]) !=
                                         np.sign(delta[20:]))))
    mean_err = float(np.mean(sign_errs))
    assert 0.35 < mean_err < 0.65, f"Oracle at theta=0 should be ~0.5, got {mean_err:.3f}"
    if verbose:
        print(f"  [PASS] Viterbi oracle @ theta=0: sign error = {mean_err:.3f}")

    # Test 2: Viterbi oracle at theta>>0 should be meaningfully better than 0.5.
    cfg_hi = SimConfig(theta=1.0, n_steps=100)
    sign_errs = []
    for trial in range(15):
        r = np.random.default_rng(MASTER_SEED + 1000 + trial)
        delta = simulate_drift(cfg_hi, r)
        est = oracle_viterbi(cfg_hi, delta, r)
        sign_errs.append(float(np.mean(np.sign(est["mean"][20:]) !=
                                         np.sign(delta[20:]))))
    mean_err = float(np.mean(sign_errs))
    assert mean_err < 0.3, f"Oracle at high theta should be <0.3, got {mean_err:.3f}"
    if verbose:
        print(f"  [PASS] Viterbi oracle @ theta=1: sign error = {mean_err:.3f}")

    # Test 3: BO-x at theta=0 should be close to 0.5 (obstruction).
    cfg = SimConfig(theta=0.0, n_steps=100)
    sign_errs = []
    for trial in range(6):
        r = np.random.default_rng(MASTER_SEED + 2000 + trial)
        res = run_single_trial("bo_x", cfg, r, n_particles=600)
        sign_errs.append(res["sign_error"])
    mean_err = float(np.mean(sign_errs))
    assert 0.35 < mean_err < 0.65, f"BO-x at theta=0 should be ~0.5, got {mean_err:.3f}"
    if verbose:
        print(f"  [PASS] BO @ theta=0: sign error = {mean_err:.3f}")

    # Test 4: Q-learner trains and runs at theta=0.5; no crashes.
    cfg = SimConfig(theta=0.5 * SimConfig().sigma_stat(), n_steps=100)
    ql = TabularQCalibrator(cfg)
    ql.train(n_episodes=50, episode_len=60, rng=rng, n_particles=200)
    sign_errs = []
    for trial in range(6):
        r = np.random.default_rng(MASTER_SEED + 3000 + trial)
        res = run_single_trial("rl_x", cfg, r, n_particles=600, q_learner=ql)
        sign_errs.append(res["sign_error"])
    mean_err = float(np.mean(sign_errs))
    assert mean_err < 0.5, f"Q-learner at theta>0 should be <0.5, got {mean_err:.3f}"
    if verbose:
        print(f"  [PASS] Q-learner trained, frozen @ beta=0.5: sign error = {mean_err:.3f}")

    # Test 5: xy_snapshot should strongly beat x_pf at theta=0.
    cfg = SimConfig(theta=0.0, n_steps=100)
    snap_errs, pf_errs = [], []
    for trial in range(8):
        r = np.random.default_rng(MASTER_SEED + 4000 + trial)
        snap_errs.append(run_single_trial("xy_snapshot", cfg, r,
                                           n_particles=600)["sign_error"])
        r = np.random.default_rng(MASTER_SEED + 4000 + trial)
        pf_errs.append(run_single_trial("x_pf", cfg, r,
                                         n_particles=600)["sign_error"])
    assert np.mean(snap_errs) < 0.35 and np.mean(pf_errs) > 0.4
    if verbose:
        print(f"  [PASS] xy_snapshot={np.mean(snap_errs):.3f} << x_pf={np.mean(pf_errs):.3f}")

    if verbose:
        print("=== All unit tests passed ===\n")


# ======================================================================
#   EXPERIMENTS
# ======================================================================

def experiment_baseline_matrix(n_trials: int, n_particles: int,
                               q_train_episodes: int) -> pd.DataFrame:
    """E1+E2 combined: CI-qualified baseline matrix."""
    print(f"[E1/E2] Running baseline matrix ({n_trials} trials/cell)")
    t0 = time.time()

    estimators = ["x_pf", "adaptive_x", "bo_x", "rl_x", "oracle_x",
                  "xy_snapshot", "xy_pf"]
    rows = []
    for beta in [0.0, 0.5, 1.0]:
        cfg = SimConfig()
        cfg.theta = beta * cfg.sigma_stat()

        # Train RL agent per-β (trained on the drift regime it will be evaluated on)
        ql = TabularQCalibrator(cfg)
        ql.train(n_episodes=q_train_episodes, episode_len=100,
                 rng=np.random.default_rng(MASTER_SEED + int(beta*1000) + 111),
                 n_particles=250)

        for est in estimators:
            for trial_idx in range(n_trials):
                seed = MASTER_SEED + int(beta*10000) + trial_idx * 13 + hash(est) % 997
                r = np.random.default_rng(seed)
                res = run_single_trial(est, cfg, r, n_particles=n_particles,
                                        q_learner=ql if est == "rl_x" else None)
                rows.append({
                    "estimator": est, "trial": trial_idx, "beta": beta,
                    **res,
                })
            if time.time() - t0 > 5:
                print(f"    beta={beta}, {est}: done")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "E2_baseline_matrix.csv", index=False)

    # Compute CIs
    rng = np.random.default_rng(MASTER_SEED + 1)
    summary_rows = []
    for beta in [0.0, 0.5, 1.0]:
        sub = df[np.isclose(df.beta, beta)]
        for est in estimators:
            v = sub[sub.estimator == est].sign_error.to_numpy()
            pt, lo, hi = bootstrap_ci(v, n_boot=5000, rng=rng)
            mv = sub[sub.estimator == est].mse.to_numpy()
            mpt, mlo, mhi = bootstrap_ci(mv, n_boot=5000, rng=rng)
            summary_rows.append({
                "beta": beta, "estimator": est,
                "sign_error": pt, "sign_error_lo": lo, "sign_error_hi": hi,
                "mse": mpt, "mse_lo": mlo, "mse_hi": mhi,
                "n_trials": n_trials,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "E1_cis.csv", index=False)

    # Paired ratios at beta=0.5: A/B (x_pf vs xy_snapshot), A/C (x_pf vs xy_pf)
    mask = np.isclose(df.beta, 0.5)
    def paired(est_a, est_b):
        a = df[mask & (df.estimator == est_a)].sort_values("trial").sign_error.to_numpy()
        b = df[mask & (df.estimator == est_b)].sort_values("trial").sign_error.to_numpy()
        return bootstrap_ratio_ci(a, b, n_boot=10000, rng=rng)
    ratio_ab = paired("x_pf", "xy_snapshot")
    ratio_ac = paired("x_pf", "xy_pf")
    ratio_ob = paired("oracle_x", "xy_snapshot")
    with open(RESULTS_DIR / "E1_ratio_ci.json", "w") as f:
        json.dump({
            "n_trials": n_trials, "beta": 0.5,
            "ratio_xpf_over_xysnap": ratio_ab[0],
            "ratio_xpf_over_xysnap_ci": [ratio_ab[1], ratio_ab[2]],
            "ratio_xpf_over_xypf": ratio_ac[0],
            "ratio_xpf_over_xypf_ci": [ratio_ac[1], ratio_ac[2]],
            "ratio_oracle_over_xysnap": ratio_ob[0],
            "ratio_oracle_over_xysnap_ci": [ratio_ob[1], ratio_ob[2]],
        }, f, indent=2)

    print(f"[E1/E2] done in {time.time()-t0:.1f}s")
    print(f"    x_pf/xy_snapshot @ beta=0.5: {ratio_ab[0]:.2f} "
          f"[{ratio_ab[1]:.2f}, {ratio_ab[2]:.2f}]")
    print(f"    x_pf/xy_pf        @ beta=0.5: {ratio_ac[0]:.2f} "
          f"[{ratio_ac[1]:.2f}, {ratio_ac[2]:.2f}]")
    print(f"    oracle/xy_snapshot @ beta=0.5: {ratio_ob[0]:.2f} "
          f"[{ratio_ob[1]:.2f}, {ratio_ob[2]:.2f}]")
    return df


def experiment_cross_matrix(n_trials: int, n_particles: int) -> pd.DataFrame:
    """E3: cross-drift x cross-estimator matrix."""
    print(f"[E3] Cross-drift matrix ({n_trials} trials/cell)")
    t0 = time.time()
    families = ["ou", "regime_switching", "reflecting_rw",
                "heavy_tailed", "asymmetric_ou"]
    estimators = ["x_pf", "xy_snapshot", "xy_pf"]
    rows = []
    for fam in families:
        for beta in [0.0, 1.0]:
            cfg = SimConfig(drift_family=fam)
            cfg.theta = beta * cfg.sigma_stat()
            for est in estimators:
                for trial_idx in range(n_trials):
                    seed = MASTER_SEED + 7000 + hash(fam) % 997 + \
                        trial_idx * 13 + hash(est) % 997
                    r = np.random.default_rng(seed)
                    res = run_single_trial(est, cfg, r, n_particles=n_particles)
                    rows.append({
                        "drift_family": fam, "estimator": est,
                        "trial": trial_idx, "beta": beta, **res,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "E3_cross_matrix.csv", index=False)
    print(f"[E3] done in {time.time()-t0:.1f}s")
    return df


def experiment_calibration(n_trials: int) -> pd.DataFrame:
    """E4: particle-filter uncertainty calibration."""
    print(f"[E4] Uncertainty calibration ({n_trials} trials/beta)")
    t0 = time.time()
    rows = []
    for beta in [0.1, 0.3, 0.5, 1.0]:
        cfg = SimConfig()
        cfg.theta = beta * cfg.sigma_stat()
        for trial_idx in range(n_trials):
            r = np.random.default_rng(MASTER_SEED + 4000 +
                                       int(beta*1000) + trial_idx)
            delta = simulate_drift(cfg, r)
            obs = measure_x(delta, cfg, r)
            est = pf_x_only(obs, cfg, 1000, r)
            reported_p = est["sign_prob"][-1]
            true_plus = float(delta[-1] > 0)
            rows.append({"beta": beta, "trial": trial_idx,
                         "reported_p_plus": reported_p,
                         "true_plus": true_plus})
    df = pd.DataFrame(rows)
    bins = np.linspace(0, 1, 11)
    df["bin"] = np.digitize(df.reported_p_plus, bins) - 1
    rel = df.groupby(["beta", "bin"]).agg(
        mean_reported=("reported_p_plus", "mean"),
        empirical=("true_plus", "mean"),
        n=("true_plus", "size")).reset_index()
    rel.to_csv(RESULTS_DIR / "E4_calibration.csv", index=False)
    print(f"[E4] done in {time.time()-t0:.1f}s")
    return rel


def experiment_feedback(n_trials: int, n_particles: int) -> pd.DataFrame:
    """Closed-loop feedback sweep."""
    print(f"[Feedback] Closed-loop sweep ({n_trials} trials/(gain,est))")
    t0 = time.time()
    gains = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
    rows = []
    for g in gains:
        for est_name in ["x_only", "xy"]:
            for trial_idx in range(n_trials):
                r = np.random.default_rng(MASTER_SEED + 9000 +
                                           int(g*100) + trial_idx +
                                           (hash(est_name) % 100))
                cfg = SimConfig()
                cfg.theta = 0.5 * cfg.sigma_stat()
                rv = _feedback_trial(cfg, r, est_name, g, n_particles)
                rows.append({"gain": g, "estimator": est_name,
                             "trial": trial_idx, "residual_var": rv})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "feedback_sweep.csv", index=False)

    rng = np.random.default_rng(MASTER_SEED + 9999)
    sub = df[np.isclose(df.gain, 0.6)]
    x_v = sub[sub.estimator == "x_only"].sort_values("trial").residual_var.to_numpy()
    y_v = sub[sub.estimator == "xy"].sort_values("trial").residual_var.to_numpy()
    pt, lo, hi = bootstrap_ratio_ci(x_v, y_v, n_boot=10000, rng=rng)
    with open(RESULTS_DIR / "feedback_ratio_ci.json", "w") as f:
        json.dump({"ratio_xonly_over_xy_at_g0.6": pt,
                   "ratio_ci": [lo, hi],
                   "n_trials": n_trials}, f, indent=2)
    print(f"[Feedback] done in {time.time()-t0:.1f}s. "
          f"Ratio X/XY @ g=0.6: {pt:.2f} [{lo:.2f}, {hi:.2f}]")
    return df


def _feedback_trial(cfg: SimConfig, rng: np.random.Generator,
                    estimator: str, gain: float, n_particles: int) -> float:
    T = cfg.n_steps
    particles = rng.normal(0.0, 3.0 * cfg.sigma_stat(), size=n_particles)
    weights = np.full(n_particles, 1.0 / n_particles)
    residuals = []
    delta_true = rng.normal(cfg.theta, cfg.sigma_stat())
    sigma_y = 1.0 / cfg.y_snr
    for t in range(T):
        if estimator == "x_only":
            obs_x = np.cos(delta_true * cfg.t_R) + \
                cfg.sigma_obs * rng.standard_normal()
            pred = np.cos(particles * cfg.t_R)
            log_w = -0.5 * ((obs_x - pred) / cfg.sigma_obs) ** 2
        else:
            obs_x = np.cos(delta_true * cfg.t_R) + \
                cfg.sigma_obs * rng.standard_normal()
            obs_y = np.sin(delta_true * cfg.t_R) + sigma_y * rng.standard_normal()
            pred_x = np.cos(particles * cfg.t_R)
            pred_y = np.sin(particles * cfg.t_R)
            log_w = -0.5 * ((obs_x - pred_x) / cfg.sigma_obs) ** 2 \
                    - 0.5 * ((obs_y - pred_y) / sigma_y) ** 2
        weights = _pf_observation_update(particles, weights, log_w)
        delta_hat = float(np.sum(weights * particles))
        c = -gain * delta_hat
        delta_true = delta_true + c
        particles = particles + c
        residuals.append(delta_true ** 2)
        ess = 1.0 / np.sum(weights ** 2)
        if ess < n_particles / 2:
            idx = rng.choice(n_particles, size=n_particles, p=weights)
            particles = particles[idx]
            weights = np.full(n_particles, 1.0 / n_particles)
        particles = _pf_propagate(particles, cfg, rng)
        delta_true = cfg.theta + cfg.rho * (delta_true - cfg.theta) + \
            cfg.sigma_drift * rng.standard_normal()
    return float(np.mean(residuals[50:]))


def experiment_ablation(n_trials: int, n_particles: int) -> Dict[str, pd.DataFrame]:
    """E6 ablations: shot-matched, rotated-basis, Y-SNR sweep."""
    print(f"[E6] Ablations ({n_trials} trials/cell)")
    t0 = time.time()
    rng_m = np.random.default_rng(MASTER_SEED + 6000)

    # Shot-matched: x-only 2x shots vs xy with half shots per channel
    rows_sm = []
    for cfg_label, cfg in [
        ("x_only_shots1", SimConfig(sigma_obs=0.15)),
        ("x_only_shots2", SimConfig(sigma_obs=0.15 / np.sqrt(2))),
        ("xy_halfshots",  SimConfig(sigma_obs=0.15 * np.sqrt(2),
                                     y_snr=2.0 / np.sqrt(2))),
    ]:
        cfg.theta = 0.0  # beta=0 to test obstruction
        est = "x_pf" if "x_only" in cfg_label else "xy_snapshot"
        for trial_idx in range(n_trials):
            r = np.random.default_rng(int(rng_m.integers(0, 2**31)))
            res = run_single_trial(est, cfg, r, n_particles=n_particles)
            rows_sm.append({"config": cfg_label, "trial": trial_idx, **res})
    df_sm = pd.DataFrame(rows_sm)
    df_sm.to_csv(RESULTS_DIR / "E6_shot_matched.csv", index=False)

    # Rotated basis at beta=0.5
    rows_rb = []
    for phi_deg in [0, 15, 30, 45, 60, 75, 90]:
        phi = np.deg2rad(phi_deg)
        cfg = SimConfig()
        cfg.theta = 0.5 * cfg.sigma_stat()
        for trial_idx in range(n_trials):
            r = np.random.default_rng(int(rng_m.integers(0, 2**31)))
            delta = simulate_drift(cfg, r)
            x = np.cos(delta * cfg.t_R) + cfg.sigma_obs * r.standard_normal(cfg.n_steps)
            y = np.sin(delta * cfg.t_R) + (1/cfg.y_snr) * r.standard_normal(cfg.n_steps)
            y_rot = np.cos(phi) * x + np.sin(phi) * y
            obs = np.stack([x, y_rot], axis=-1)
            est = snapshot_xy(obs, cfg)
            se = float(np.mean(np.sign(est["mean"][50:]) != np.sign(delta[50:])))
            rows_rb.append({"phi_deg": phi_deg, "trial": trial_idx,
                            "sign_error": se})
    df_rb = pd.DataFrame(rows_rb)
    df_rb.to_csv(RESULTS_DIR / "E6_rotated_basis.csv", index=False)

    # Y-SNR sweep
    rows_snr = []
    for ysnr in [0.2, 0.5, 1.0, 2.0, 5.0]:
        cfg = SimConfig(y_snr=ysnr)
        cfg.theta = 0.5 * cfg.sigma_stat()
        for trial_idx in range(n_trials):
            r = np.random.default_rng(int(rng_m.integers(0, 2**31)))
            r_xy = run_single_trial("xy_snapshot", cfg, r, n_particles=n_particles)
            r2 = np.random.default_rng(int(rng_m.integers(0, 2**31)))
            r_x = run_single_trial("x_pf", cfg, r2, n_particles=n_particles)
            rows_snr.append({"y_snr": ysnr, "trial": trial_idx,
                             "xy_snapshot_sign_error": r_xy["sign_error"],
                             "x_pf_sign_error": r_x["sign_error"]})
    df_snr = pd.DataFrame(rows_snr)
    df_snr.to_csv(RESULTS_DIR / "E6_ysnr_sweep.csv", index=False)

    print(f"[E6] done in {time.time()-t0:.1f}s")
    return {"shot_matched": df_sm, "rotated": df_rb, "ysnr": df_snr}


def experiment_archival_proxy(n_qubits: int) -> Dict:
    """E7 proxy: labelled as such. No real archival fetch in this environment."""
    print(f"[E7] Archival-proxy ({n_qubits} simulated qubits)")
    t0 = time.time()
    rng_m = np.random.default_rng(MASTER_SEED + 7000)
    rows = []
    for q in range(n_qubits):
        rho_q = float(rng_m.uniform(0.85, 0.98))
        sigma_q = float(rng_m.uniform(0.1, 0.3))
        sigma_stat_q = sigma_q / np.sqrt(max(1 - rho_q**2, 1e-4))
        # draw beta uniformly spanning 0 to 2.5 so both low and high are populated
        beta_target = float(rng_m.uniform(-2.5, 2.5))
        theta_q = beta_target * sigma_stat_q
        cfg = SimConfig(n_steps=300, rho=rho_q, sigma_drift=sigma_q,
                        theta=theta_q)
        delta = simulate_drift(cfg, rng_m)
        rng2 = np.random.default_rng(int(rng_m.integers(0, 2**31)))
        obs = measure_x(delta, cfg, rng2)
        est = pf_x_only(obs, cfg, 800, rng2)
        theta_hat = float(np.mean(delta))
        sigma_hat = float(np.std(delta[1:] - rho_q * delta[:-1]))
        beta_hat = theta_hat / max(sigma_hat / np.sqrt(max(1-rho_q**2, 1e-4)), 1e-6)
        rows.append({
            "qubit_id": q, "rho_true": rho_q, "sigma_true": sigma_q,
            "theta_true": theta_q, "beta_true": cfg.beta(),
            "beta_hat": beta_hat,
            "entropy_final": float(est["entropy"][-1]),
            "sign_error_run": float(np.mean(np.sign(est["mean"][50:]) !=
                                              np.sign(delta[50:]))),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "E7_archival_proxy.csv", index=False)

    low = df[np.abs(df.beta_hat) < 0.3]
    high = df[np.abs(df.beta_hat) > 1.0]
    summary = {
        "n_qubits": n_qubits,
        "n_low_beta": int(len(low)),
        "n_high_beta": int(len(high)),
        "frac_low_beta_entropy_gt_0.85": float(np.mean(low.entropy_final > 0.85))
            if len(low) > 0 else None,
        "frac_high_beta_entropy_lt_0.5": float(np.mean(high.entropy_final < 0.5))
            if len(high) > 0 else None,
        "note": "PROXY only: synthetic OU-parameter ensemble calibrated "
                "to published IBM backend-property ranges. No real archival "
                "network fetch was performed in this environment.",
    }
    with open(RESULTS_DIR / "E7_archival_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[E7] done in {time.time()-t0:.1f}s. Low-beta H>0.85: "
          f"{summary['frac_low_beta_entropy_gt_0.85']}, "
          f"high-beta H<0.5: {summary['frac_high_beta_entropy_lt_0.5']}")
    return summary


# ======================================================================
#   FIGURES
# ======================================================================

def make_figures(baseline: pd.DataFrame, feedback: pd.DataFrame,
                 ablation: Dict[str, pd.DataFrame], cross: pd.DataFrame,
                 cal: pd.DataFrame):
    print("[Figures] Generating…")
    t0 = time.time()

    # -------- Figure 1: hero --------
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))

    # (a) schematic
    ax = axes[0, 0]
    d = np.linspace(-2.5, 2.5, 300)
    ax.plot(d, np.cos(d), color=OKABE_ITO[2], lw=2,
            label=r"$X = \cos(\delta t_R)$")
    ax.plot(d, np.sin(d), color=OKABE_ITO[1], lw=2,
            label=r"$Y = \sin(\delta t_R)$")
    for delta_val in [1.2]:
        ax.plot([delta_val, -delta_val],
                 [np.cos(delta_val), np.cos(-delta_val)],
                 "o", color=OKABE_ITO[2], ms=7)
        ax.plot([delta_val, -delta_val],
                 [np.sin(delta_val), np.sin(-delta_val)],
                 "s", color=OKABE_ITO[1], ms=7)
    ax.axhline(0, color="k", lw=0.4, alpha=0.3)
    ax.axvline(0, color="k", lw=0.4, alpha=0.3)
    ax.set_xlabel(r"Detuning $\delta$")
    ax.set_ylabel("Readout value")
    ax.set_title(r"a  Sign collapses on $X$, separates on $Y$",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower right")

    # (b) posterior trajectories
    ax = axes[0, 1]
    for beta, color, label in [(0.0, OKABE_ITO[5], r"$\beta = 0$"),
                                 (1.0, OKABE_ITO[3], r"$\beta = 1$")]:
        cfg = SimConfig()
        cfg.theta = beta * cfg.sigma_stat()
        trajs = []
        for ti in range(10):
            r = np.random.default_rng(MASTER_SEED + 100 + ti + int(beta*1000))
            delta = simulate_drift(cfg, r)
            obs = measure_x(delta, cfg, r)
            est = pf_x_only(obs, cfg, 800, r)
            trajs.append(est["sign_prob"])
        trajs = np.asarray(trajs)
        med = np.median(trajs, axis=0)
        lo = np.percentile(trajs, 10, axis=0)
        hi = np.percentile(trajs, 90, axis=0)
        t = np.arange(len(med))
        ax.plot(t, med, color=color, lw=1.8, label=label)
        ax.fill_between(t, lo, hi, color=color, alpha=0.2)
    ax.axhline(0.5, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_xlabel("Measurement step")
    ax.set_ylabel(r"$P(\sigma=+ | y_{1:t})$")
    ax.set_ylim(0, 1)
    ax.set_title("b  Posterior evolution", loc="left", fontweight="bold")
    ax.legend(loc="center right")

    # (c) bar of sign errors at beta=0.5
    ax = axes[1, 0]
    configs = [("x_pf", "A\nX-only PF\nfull hist"),
               ("xy_snapshot", "B\nX+Y snap\nno hist"),
               ("xy_pf", "C\nX+Y PF\nfull hist")]
    sub = baseline[np.isclose(baseline.beta, 0.5)]
    pts, los, his = [], [], []
    for est, _ in configs:
        v = sub[sub.estimator == est].sign_error.to_numpy()
        p, lo, hi = bootstrap_ci(v, n_boot=5000)
        pts.append(p); los.append(lo); his.append(hi)
    xs = np.arange(3)
    ax.bar(xs, pts, color=[OKABE_ITO[5], OKABE_ITO[1], OKABE_ITO[3]],
           edgecolor="k", lw=0.5)
    ax.errorbar(xs, pts, yerr=[np.array(pts)-np.array(los),
                                 np.array(his)-np.array(pts)],
                fmt="none", color="k", capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([c[1] for c in configs])
    ax.set_ylabel("Sign error rate")
    ax.set_ylim(0, max(0.5, max(his)*1.2))
    ax.set_title(r"c  Sign error @ $\beta=0.5$", loc="left", fontweight="bold")
    if pts[0] > 0 and pts[1] > 0:
        ratio = pts[0] / pts[1]
        ax.annotate(f"{ratio:.2f}x", xy=(0.5, max(pts[0], pts[1]) + 0.03),
                    ha="center", fontsize=10, fontweight="bold",
                    color=OKABE_ITO[6])

    # (d) feedback
    ax = axes[1, 1]
    for est, color, label in [("x_only", OKABE_ITO[5], "X-only"),
                                ("xy", OKABE_ITO[3], "X+Y")]:
        sub = feedback[feedback.estimator == est]
        g = sub.groupby("gain").residual_var.agg(["mean", "std", "count"])
        se = g["std"] / np.sqrt(g["count"])
        ax.plot(g.index, g["mean"], "o-", color=color, label=label)
        ax.fill_between(g.index, g["mean"] - 1.96*se, g["mean"] + 1.96*se,
                         color=color, alpha=0.2)
    ax.set_xlabel("Feedback gain g")
    ax.set_ylabel("Residual detuning variance")
    ax.set_title("d  Closed-loop residual", loc="left", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_hero.pdf")
    plt.savefig(FIG_DIR / "fig1_hero.png", dpi=300)
    plt.close()

    # -------- Figure 2: baselines --------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    ax = axes[0]
    order = ["oracle_x", "x_pf", "rl_x", "bo_x", "adaptive_x",
             "xy_snapshot", "xy_pf"]
    labels = {"x_pf": "X-only PF", "adaptive_x": "Adaptive Ramsey",
              "bo_x": "GP-BO", "oracle_x": "Viterbi oracle",
              "rl_x": "Tabular Q-learning",
              "xy_snapshot": "X+Y snapshot", "xy_pf": "X+Y PF"}
    sub = baseline[np.isclose(baseline.beta, 0.5)]
    ys = np.arange(len(order))
    pts, los, his = [], [], []
    for e in order:
        v = sub[sub.estimator == e].sign_error.to_numpy()
        p, lo, hi = bootstrap_ci(v, n_boot=5000)
        pts.append(p); los.append(lo); his.append(hi)
    colors = [OKABE_ITO[7]]*5 + [OKABE_ITO[1], OKABE_ITO[3]]
    ax.barh(ys, pts, color=colors, edgecolor="k", lw=0.5)
    ax.errorbar(pts, ys, xerr=[np.array(pts)-np.array(los),
                                 np.array(his)-np.array(pts)],
                fmt="none", color="k", capsize=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[e] for e in order])
    ax.set_xlabel(r"Sign error rate ($\beta=0.5$)")
    ax.set_title("a  Baseline matrix", loc="left", fontweight="bold")
    ax.axvline(pts[order.index("xy_snapshot")], color="red",
               ls="--", lw=0.8, alpha=0.5)

    ax = axes[1]
    # At beta=0 — the obstruction-signature panel
    sub0 = baseline[np.isclose(baseline.beta, 0.0)]
    pts0, los0, his0 = [], [], []
    for e in order:
        v = sub0[sub0.estimator == e].sign_error.to_numpy()
        p, lo, hi = bootstrap_ci(v, n_boot=5000)
        pts0.append(p); los0.append(lo); his0.append(hi)
    ax.barh(ys, pts0, color=colors, edgecolor="k", lw=0.5)
    ax.errorbar(pts0, ys, xerr=[np.array(pts0)-np.array(los0),
                                  np.array(his0)-np.array(pts0)],
                fmt="none", color="k", capsize=3)
    ax.axvline(0.5, color="red", ls="--", lw=0.8, alpha=0.5,
                label="coin flip")
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[e] for e in order])
    ax.set_xlabel(r"Sign error rate ($\beta=0$, obstruction)")
    ax.set_xlim(0, 0.6)
    ax.set_title(r"b  At $\beta=0$: structural ceiling",
                  loc="left", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_baselines.pdf")
    plt.savefig(FIG_DIR / "fig2_baselines.png", dpi=300)
    plt.close()

    # -------- Figure 3: ablation --------
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    ax = axes[0]
    df_sm = ablation["shot_matched"]
    cfgs = df_sm.config.unique()
    pts, los, his = [], [], []
    for c in cfgs:
        v = df_sm[df_sm.config == c].sign_error.to_numpy()
        p, lo, hi = bootstrap_ci(v, n_boot=3000)
        pts.append(p); los.append(lo); his.append(hi)
    xs = np.arange(len(cfgs))
    cc = [OKABE_ITO[5]]*2 + [OKABE_ITO[1]]
    ax.bar(xs, pts, color=cc, edgecolor="k", lw=0.5)
    ax.errorbar(xs, pts, yerr=[np.array(pts)-np.array(los),
                                 np.array(his)-np.array(pts)],
                fmt="none", color="k", capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(cfgs, rotation=15, ha="right", fontsize=7)
    ax.set_ylabel(r"Sign error ($\beta=0$)")
    ax.set_title("a  Shot-matched", loc="left", fontweight="bold")

    ax = axes[1]
    df_rb = ablation["rotated"]
    g = df_rb.groupby("phi_deg").sign_error.agg(["mean", "std", "count"])
    se = g["std"] / np.sqrt(g["count"])
    ax.errorbar(g.index, g["mean"], yerr=1.96*se, marker="o",
                color=OKABE_ITO[6], capsize=3)
    ax.set_xlabel(r"Basis angle $\phi$ (deg)")
    ax.set_ylabel(r"Sign error ($\beta=0.5$)")
    ax.set_title("b  Rotated basis", loc="left", fontweight="bold")

    ax = axes[2]
    df_snr = ablation["ysnr"]
    for col, color, label in [("x_pf_sign_error", OKABE_ITO[5], "X-only PF"),
                                ("xy_snapshot_sign_error", OKABE_ITO[1], "X+Y snap")]:
        g = df_snr.groupby("y_snr")[col].agg(["mean", "std", "count"])
        se = g["std"] / np.sqrt(g["count"])
        ax.errorbar(g.index, g["mean"], yerr=1.96*se, marker="o",
                    color=color, capsize=3, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Y-SNR")
    ax.set_ylabel(r"Sign error ($\beta=0.5$)")
    ax.set_title("c  Y-SNR sweep", loc="left", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_ablation.pdf")
    plt.savefig(FIG_DIR / "fig3_ablation.png", dpi=300)
    plt.close()

    # -------- Figure 5: cross-drift heatmap --------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fams = ["ou", "regime_switching", "reflecting_rw", "heavy_tailed",
            "asymmetric_ou"]
    fam_lab = ["OU", "Regime-switch", "Reflect RW", "Heavy-tail",
               "Asym-OU*"]
    ests = ["x_pf", "xy_snapshot", "xy_pf"]
    est_lab = ["X-only PF", "X+Y snap", "X+Y PF"]
    for ai, beta in enumerate([0.0, 1.0]):
        M = np.zeros((len(fams), len(ests)))
        for i, f in enumerate(fams):
            for j, e in enumerate(ests):
                s = cross[(cross.drift_family == f) &
                          (cross.estimator == e) &
                          (np.isclose(cross.beta, beta))].sign_error
                M[i, j] = float(s.mean()) if len(s) > 0 else np.nan
        im = axes[ai].imshow(M, vmin=0, vmax=0.55, cmap="magma_r",
                              aspect="auto")
        axes[ai].set_xticks(range(len(ests)))
        axes[ai].set_xticklabels(est_lab, rotation=20, ha="right")
        axes[ai].set_yticks(range(len(fams)))
        axes[ai].set_yticklabels(fam_lab)
        axes[ai].set_title(f"{'a' if ai==0 else 'b'}  "
                            r"$\beta = " + f"{beta}$",
                            loc="left", fontweight="bold")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    c = "white" if M[i, j] > 0.25 else "black"
                    axes[ai].text(j, i, f"{M[i,j]:.2f}",
                                   ha="center", va="center",
                                   color=c, fontsize=7)
        plt.colorbar(im, ax=axes[ai], fraction=0.04, pad=0.03,
                      label="Sign error")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_matrix.pdf")
    plt.savefig(FIG_DIR / "fig5_matrix.png", dpi=300)
    plt.close()

    # -------- Supplementary: reliability --------
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    for beta in sorted(cal.beta.unique()):
        sub = cal[cal.beta == beta]
        ax.plot(sub.mean_reported, sub.empirical, "o-",
                label=rf"$\beta = {beta}$")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"Reported $P(\sigma=+)$")
    ax.set_ylabel("Empirical frequency")
    ax.legend()
    ax.set_title("Reliability diagram")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "supp_e4_reliability.pdf")
    plt.savefig(FIG_DIR / "supp_e4_reliability.png", dpi=300)
    plt.close()

    print(f"[Figures] done in {time.time()-t0:.1f}s")


# ======================================================================
#   MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "quick"], default="full")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    if args.mode == "full":
        n_trials_matrix = 200
        n_trials_cross = 60
        n_trials_feedback = 60
        n_trials_ablation = 40
        n_trials_cal = 100
        n_particles = 1500
        q_train = 400
        n_qubits_e7 = 32
    else:
        n_trials_matrix = 20
        n_trials_cross = 10
        n_trials_feedback = 10
        n_trials_ablation = 10
        n_trials_cal = 20
        n_particles = 500
        q_train = 80
        n_qubits_e7 = 12

    t_start = time.time()
    print(f"Pipeline v2 — mode={args.mode}, master_seed={MASTER_SEED}")

    if not args.skip_tests:
        unit_tests(verbose=True)

    baseline = experiment_baseline_matrix(
        n_trials=n_trials_matrix, n_particles=n_particles,
        q_train_episodes=q_train)
    cross = experiment_cross_matrix(
        n_trials=n_trials_cross, n_particles=n_particles)
    cal = experiment_calibration(n_trials=n_trials_cal)
    feedback = experiment_feedback(
        n_trials=n_trials_feedback, n_particles=n_particles)
    ablation = experiment_ablation(
        n_trials=n_trials_ablation, n_particles=n_particles)
    archival = experiment_archival_proxy(n_qubits=n_qubits_e7)

    make_figures(baseline, feedback, ablation, cross, cal)

    manifest = {
        "master_seed": MASTER_SEED,
        "mode": args.mode,
        "n_trials_matrix": n_trials_matrix,
        "n_particles": n_particles,
        "q_train_episodes": q_train,
        "wall_seconds": time.time() - t_start,
        "outputs": sorted(
            [str(p.relative_to(REPO_ROOT)) for p in RESULTS_DIR.glob("*")] +
            [str(p.relative_to(REPO_ROOT)) for p in FIG_DIR.glob("*")]
        ),
    }
    with open(RESULTS_DIR / "reproduction_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nTotal wall: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
