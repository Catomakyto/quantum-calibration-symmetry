# quantum-calibration-symmetry

Reproduction code, raw outputs, and figures for the manuscript
**"A measurement symmetry obstruction to quantum calibration removed by sensor augmentation"**
by Hongyi Li (under review, npj Quantum Information).

The Ramsey calibration measurement is even in the qubit detuning, so the
sign of a frequency offset cannot be recovered from the standard
$X$-quadrature readout in any single shot. The repository contains the
pipeline that quantifies how this sign-blindness extends to the full
temporal record under various drift dynamics, the baseline benchmarks
that establish the information-theoretic floor (including a
Viterbi-decoded oracle), and the figure-generating code.

A single command reproduces every number and figure in the paper from a
fixed master seed (`20260328`).

## Quick start

```bash
# 1. Install dependencies (numpy, scipy, pandas, matplotlib).
pip install -r requirements.txt

# 2. Run the full pipeline (~4 minutes on a laptop with 4 cores).
make reproduce
```

`make reproduce` executes `scripts/reproduce.py` (the main 7-experiment
matrix) followed by `scripts/extras.py` (the continuous β-sweep and
13-condition universal-floor pool) and `scripts/multi_qubit_crosstalk.py`
(the 2-qubit cross-talk experiment for Supplementary Section S28).
Outputs land in `results/` and `figures/`. All three scripts use
deterministic per-configuration seed offsets derived from master
seed `20260328`, so re-runs produce numerically identical CSV and
JSON outputs.

To check the pipeline in 30 seconds with looser CIs:

```bash
make quick
```

This regenerates the same outputs at $n = 20$ trials per cell instead
of 200; signs of every comparison are preserved but confidence
intervals widen.

## Repository layout

```
quantum-calibration-symmetry/
├── README.md              this file
├── REVIEWER.md            reviewer-facing reproduction recipe
├── LICENSE                MIT
├── Makefile               make reproduce / make quick / make clean
├── requirements.txt       pinned dependency list
├── scripts/
│   ├── reproduce.py       main pipeline (E1–E7 + closed-loop)
│   └── extras.py          E8 β-sweep + E9 universal-floor pool
├── results/               raw trial-level CSVs and CI JSONs
└── figures/               published figures (PDF + 300 dpi PNG)
```

## What reproduces what

The mapping from manuscript claim to file is deterministic.

### Numerical claims (results/)

| File | Manuscript claim |
|------|---|
| `E1_cis.csv` | Table 1 — sign-error rates with 95% bootstrap CIs at $\beta \in \{0, 0.5, 1.0\}$ |
| `E1_ratio_ci.json` | Abstract / §2.3 — paired-bootstrap ratios 1.52× and 2.53× at $\beta = 0.5$ |
| `E2_baseline_matrix.csv` | Raw trial-level data underlying Table 1 |
| `E3_cross_matrix.csv` | §2.5 / Fig 5 — five drift families × three estimators |
| `E4_calibration.csv` | Supplementary S23 — particle-filter reliability diagram |
| `E6_shot_matched.csv`, `E6_rotated_basis.csv`, `E6_ysnr_sweep.csv` | §2.4 / Fig 3 — three ablations |
| `feedback_sweep.csv`, `feedback_ratio_ci.json` | §2.6 / Fig 1d — closed-loop residual at gain 0.6, ratio 3.99× |
| `E7_archival_proxy.csv`, `E7_archival_summary.json` | §2.7 — retrospective parameter-ensemble proxy |
| `E8_beta_sweep.csv` | Fig 4 — continuous obstruction-to-restoration transition |
| `E9_universal_floor.csv` | Fig 3 — pooled 2,200-trial universal-floor result, 0.502 [0.496, 0.508] |

### Figures (figures/)

| File | Manuscript figure |
|------|---|
| `fig1_hero.pdf` | Figure 1 — schematic + sign-posterior trajectories + sign-error bars + closed-loop sweep |
| `fig2_baselines.pdf` | Figure 2 — five-baseline matrix at $\beta = 0$ and $\beta = 0.5$ |
| `fig3_ablation.pdf` | Figure 3 — shot-matched + rotated-basis + Y-SNR ablations |
| `fig5_matrix.pdf` | Figure 5 — cross-drift heatmap |
| `fig_extra_universal_floor.pdf` | Figure 3 (universal floor) |
| `fig_extra_beta_sweep.pdf` | Figure 4 (β-sweep curve) |
| `supp_e4_reliability.pdf` | Supplementary Fig S12 |

PNG copies at 300 dpi accompany each PDF.

## Reproducibility guarantees

- Master seed: **20260328**, set in `scripts/reproduce.py` and inherited by all experiments.
- Five unit tests run before the baseline matrix and exit non-zero on regression. They check the Viterbi oracle against its analytic limits, that the GP-BO baseline reproduces the obstruction at $\beta = 0$, that the trained Q-learner converges, and that the $X{+}Y$ snapshot beats the $X$-only filter at $\beta = 0$.
- Every numeric claim in the abstract, Table 1, and the manuscript Results traces to one row of `results/E1_cis.csv`, `results/E1_ratio_ci.json`, `results/feedback_ratio_ci.json`, `results/E7_archival_summary.json`, `results/E8_beta_sweep.csv`, or `results/E9_universal_floor.csv`.
- Pipeline determinism has been verified across runs on the same machine; cross-machine determinism is expected modulo numpy floating-point summation order.

## Citation

Manuscript currently under review. Once the DOI is issued, a
`CITATION.cff` entry will be added here. In the meantime, please cite as:

> Hongyi Li, *A measurement symmetry obstruction to quantum calibration removed by sensor augmentation*, manuscript under review (2026). Code: github.com/Catomakyto/quantum-calibration-symmetry.

A Zenodo DOI will be issued upon acceptance.

## License

MIT (see `LICENSE`).

## Contact

Hongyi Li — `hongyili12345@gmail.com`. Issues and pull requests welcome.
