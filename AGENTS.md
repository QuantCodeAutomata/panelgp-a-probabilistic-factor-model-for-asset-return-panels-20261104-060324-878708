# AGENTS.md

Guide for contributors and coding agents working in this repository.

## What this repository implements

A reproduction of selected experiments from *PanelGP: A Probabilistic Factor
Model for Asset-Return Panels* — a sparse variational Gaussian-process model
of asset returns in which both the factor-loading map (over asset
characteristics) and the factor-forecasting map (over date context) are
learned jointly from the panel (paper Sections 3-4, Appendices A-C).

Six experiments are reproduced, one per module in `exp/`:

| Module | Paper section | What it reproduces |
|---|---|---|
| `exp/exp1.py` | 5.1-5.2, Table 1 | Simulation recovery of the true loading/factor maps and innovation variances |
| `exp/exp2.py` | 6.1-6.4, Table 2 | Kernel comparison (affine/RBF loadings x constant/RBF factors) vs. IPCA on the S&P 500 |
| `exp/exp3.py` | 6.5, Figure 3 / S2 | Effect of the number of factors H on predictive performance |
| `exp/exp4.py` | 6.6, Tables 3-4 | Fixed vs. annual rolling re-estimation, turnover and net-of-cost Sharpe ratios |
| `exp/exp5.py` | 6.7, Appendix B.3 | Log-score gain of the full bilinear predictive density over a Gaussian |
| `exp/exp6.py` | 6.8 | Predicted variance decomposition and risk calibration |

See `results/RESULTS.md` for the numbers produced by the latest run next to
the paper's reported values, with a verdict for each metric, and for the
full list of deviations from the paper (data source, universe, sample
period, and training-budget reductions made to fit the time available).

## Layout

- `src/` — flat library modules, the single implementation each experiment
  calls: `gp_model.py` (sparse variational PanelGP estimator), `kernels.py`
  (RBF/affine/three-way-softmax factor kernels), `ipca.py` (IPCA by
  alternating least squares), `alignment.py` (least-squares factor/loading
  rotation for the simulation study), `characteristics.py` (price-based
  characteristics and context variables built from OHLCV bars),
  `evaluation.py` (predictive R², rank correlation, Gaussian log score,
  portfolio construction, Newey-West/Holm tests, variance decomposition),
  `sp500_pipeline.py` (shared data-to-evaluation pipeline used by
  `exp2`-`exp6`), `predictive_distribution.py` (bilinear CGF and Fourier
  inversion for Experiment 5), `config.py` (every experiment parameter as a
  module-level, commented constant; no classes or dicts).
- `exp/` — one script per experiment, calling only the `src/` functions
  above with their documented signatures.
- `data/` — `data_loader.py` loads OHLCV bars through the QuantX Dataloader
  REST API (see the comment above each call); `generate_synthetic.py`
  generates Experiment 1's simulated panel. Downloaded data is cached under
  `data/cache/` (git-ignored) and never committed.
- `tests/` — all pytest tests, unit tests on synthetic/small fixtures plus
  market-data tests on a small cached S&P 500 slice.
- `results/` — every table, plot and log written by the experiment scripts,
  plus `results/RESULTS.md` comparing them with the paper.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env   # DATALOADER_API_KEY, MASSIVE_API_KEY, for exp2-exp6

python -m pytest -m "not market_data"   # unit tests, no network
python -m pytest                        # also exercises a cached real-data slice

python -m exp.exp1              # simulation recovery (paper's own configuration)
python -m exp.exp2              # kernel comparison            [needs API keys]
python -m exp.exp3              # number-of-factors sweep      [needs API keys]
python -m exp.exp4              # annual re-estimation         [needs API keys]
python -m exp.exp5              # higher moments               [needs API keys]
python -m exp.exp6              # variance calibration          [needs API keys]
python -m exp.exp2 --full       # paper's full universe/epoch/seed configuration
```

`reproduce.sh` runs the test suite and all six experiments with these same
commands.

## Conventions

- Every experiment parameter lives in `src/config.py` as an
  `UPPER_CASE` module-level variable with a comment stating its meaning,
  unit and source (a paper section/equation, or "not specified in the
  paper"). Where a run was reduced to fit the time available, both the
  paper value (`..._PAPER`) and the executed value (`..._RUN`) are defined
  side by side, and `--full` switches an experiment to the paper values.
  Do not add parameters anywhere else.
- Market data is loaded only through `data/data_loader.py`
  (`DATALOADER_API_KEY`, `MASSIVE_API_KEY` from the environment via
  `.env`, never hardcoded or logged); synthetic data only through
  `data/generate_synthetic.py`. No experiment script or `src/` module
  fetches or generates data on its own.
- `tests/test_src.py` and `tests/conftest.py` hold all tests.
  Market-data tests are marked `@pytest.mark.market_data`, use a
  session-scoped, cached, reduced slice of the universe, and are skipped
  automatically when `DATALOADER_API_KEY` is unset. Run
  `python -m pytest -m "not market_data"` for the network-free subset.
- API keys live only in a local, uncommitted `.env` (see `.env.example`
  for the two placeholders); they must never appear in code, results, or
  commit messages.
- Re-running an experiment command overwrites its files in `results/`.
  `results/RESULTS.md` is not regenerated automatically: after a rerun,
  check its numbers and verdicts against the new files in `results/` and
  update it by hand.
