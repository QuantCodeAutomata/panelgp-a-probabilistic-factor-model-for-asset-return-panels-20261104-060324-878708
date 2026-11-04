# PanelGP: A Probabilistic Factor Model for Asset-Return Panels

Reproduction of selected experiments from *PanelGP: A Probabilistic Factor
Model for Asset-Return Panels*.

This repository implements the sparse variational Gaussian-process factor
model described in the paper (Sections 3-4, Appendices A-C) and reproduces
six of its experiments:

1. **Simulation recovery of smooth maps and variance parameters** (Section
   5.1-5.2, Table 1) — Monte Carlo recovery of the true loading/factor
   functions and innovation variances from a simulated panel.
2. **Kernel comparison on the S&P 500 with fixed training** (Section 6.1-6.4,
   Table 2) — affine/RBF loading and constant/RBF factor kernels compared
   against IPCA and IPCA+intercept.
3. **Number of factors analysis** (Section 6.5, Figure 3 / Figure S2) — how
   performance varies with the number of factors H.
4. **Annual re-estimation and trading costs** (Section 6.6, Tables 3-4) —
   fixed vs. rolling re-estimation, turnover and net-of-cost Sharpe ratios.
5. **Higher moments of the bilinear predictive distribution** (Section 6.7,
   Appendix B.3) — log-score gain of the full bilinear density over a
   Gaussian with the same mean/variance.
6. **Predicted variance terms and risk calibration** (Section 6.8) — variance
   decomposition and rank correlation with realized squared forecast errors.

## Data

Experiment 1 uses only simulated data (`data/generate_synthetic.py`).
Experiments 2-6 use monthly S&P 500 stock prices loaded through the QuantX
Dataloader REST API (`data/data_loader.py`); see `results/RESULTS.md`,
section "Deviations from the paper", for why price-based characteristics
replace the paper's 137 WRDS/Jensen et al. (2023) characteristics and a
reduced universe/period replace the paper's full 1960-2024 CRSP/S&P 500
panel, and for the exact configuration that was executed.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY for experiments 2-6
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests on synthetic/small fixtures
python -m pytest                        # also exercises a small real-data slice (needs API keys)
```

## Experiments

Each experiment can be run at the time-budgeted configuration described in
`results/RESULTS.md`, or at the paper's full configuration with `--full`:

```bash
python -m exp.exp1              # simulation recovery (Table 1)
python -m exp.exp2              # kernel comparison (Table 2)        [requires API keys]
python -m exp.exp3              # number-of-factors sweep (Figure 3) [requires API keys]
python -m exp.exp4              # annual re-estimation (Tables 3-4)  [requires API keys]
python -m exp.exp5              # higher moments (Section 6.7)       [requires API keys]
python -m exp.exp6              # variance terms (Section 6.8)       [requires API keys]
python -m exp.exp1 --full       # paper's 5,000-epoch, 5-replication configuration
```

All scripts write their tables, metrics and plots directly into `results/`.
`reproduce.sh` runs the full test and experiment sequence.

## Layout

```
src/     shared library: GP model, kernels, IPCA, evaluation, alignment, predictive distribution, config
exp/     one script per experiment, calling only src/ and data/
data/    data_loader.py (QuantX Dataloader OHLCV) and generate_synthetic.py (simulation panel)
tests/   pytest unit and market-data tests
results/ tables, plots, logs and RESULTS.md
```
