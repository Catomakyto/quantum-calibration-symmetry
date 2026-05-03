# REVIEWER.md

If you are reviewing this manuscript, this file is the shortest path to reproducing every figure and number. If anything below fails, please email the corresponding author at `hongyili12345@gmail.com`; it is a bug on our side, not a configuration issue on yours.

## Three commands

```bash
pip install -r requirements.txt
make reproduce
ls results/  figures/
```

`make reproduce` executes `scripts/reproduce.py --mode full` followed by `scripts/extras.py` (continuous β-sweep + universal-floor pool) and `scripts/multi_qubit_crosstalk.py` (the 2-qubit cross-talk experiment for Supplementary Section S28). Master seed is `20260328`; per-experiment seed offsets are deterministic, so CSV/JSON outputs are numerically identical across re-runs. Five unit tests run before the baseline matrix; the script exits non-zero on regression. Expected total runtime: ~4 minutes on a laptop with 4 cores.

## 30-second sanity check

```bash
make quick
```

`make quick` runs the same code path at 20 trials per cell instead of 200. Every directional comparison in the paper is preserved at this trial count.

## Auditing a specific manuscript claim

Every numeric claim traces to a row of one of these files. After `make reproduce`:

| Manuscript claim | File |
|---|---|
| Table 1 sign-error CIs | `results/E1_cis.csv` |
| Abstract / Results: 1.52× and 2.53× ratios | `results/E1_ratio_ci.json` |
| Closed-loop residual ratio 3.99× | `results/feedback_ratio_ci.json` |
| Pooled X-only floor 0.502 [0.496, 0.508], n = 2,200 | `results/E9_universal_floor.csv` |
| Continuous β-sweep (Fig 4) | `results/E8_beta_sweep.csv` |
| Cross-drift × cross-estimator matrix | `results/E3_cross_matrix.csv` |
| Retrospective proxy ensemble | `results/E7_archival_summary.json` |

The figure-to-file map is in the project README.

## What you will not find here

- New hardware data. The Phase B1 Rigetti Ankaa-3 protocol is pre-registered in Supplementary Section S25 of the manuscript but not yet executed.
- A pinned environment with hashes. Dependencies are minimal (`numpy`, `scipy`, `pandas`, `matplotlib`); the pipeline is robust across recent versions. A pinned `requirements.txt` with hashes will be archived with the Zenodo DOI on acceptance.

## What we would most like you to check

1. **The universal-floor result.** `results/E9_universal_floor.csv`. Pool all `axis_value`-by-`estimator` rows for `estimator in {x_pf, oracle_x}` and compute the mean and a 95% bootstrap CI. The result should be 0.502 with CI excluding 0.49 and 0.51.
2. **The Viterbi-oracle CI at β = 0.** `results/E1_cis.csv`, row `oracle_x` at `beta = 0.0`. The 95% CI should be approximately [0.491, 0.505]. This is the sharpest empirical demonstration of Theorem 1 in the paper; a wider CI would weaken the claim.
3. **The β-sweep monotonicity.** `results/E8_beta_sweep.csv`. The X+Y particle filter should be strictly best at every β. The X-only baselines should track each other within ±0.03 across the full β range.

If you can find an X-only estimator we missed that crosses the universal-floor at $\beta = 0$, the paper is wrong.

## Contact

Hongyi Li — `hongyili12345@gmail.com`
