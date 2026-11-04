"""Paper-derived parameters for every PanelGP experiment.

Each variable states its meaning, unit, and source (a paper section or
equation, or "not specified in the paper"). No classes or dicts: every
parameter is a module-level constant so experiment scripts can import exactly
what they need.
"""

# ---------------------------------------------------------------------------
# Experiment 1: simulation recovery (paper Section 5.1 "Design")
# ---------------------------------------------------------------------------
SIM_N_ASSETS = 80          # assets in the simulated panel, Section 5.1
SIM_N_MONTHS = 50          # months in the simulated panel, Section 5.1
SIM_N_CHARACTERISTICS = 2  # P, the 9x9 grid in [0,1]^2 has 2 coordinates, Section 5.1
SIM_N_CONTEXT = 1          # Q, scalar context c_t=(t-1)/49, Section 5.1
SIM_H_TRUE = 2             # true number of factors, Section 5.1
SIM_SIGMA_BETA2 = 0.05     # loading-innovation variance sigma_beta^2, Section 5.1
SIM_SIGMA_PHI2 = 0.05      # factor-innovation variance sigma_phi^2, Section 5.1
SIM_SIGMA_R2 = 0.05        # observation-noise variance sigma_r^2, Section 5.1

SIM_M_INDUCING = 64        # trainable inducing inputs per component, Section 5.1
SIM_EPOCHS_PAPER = 5000    # training epochs used in the paper, Section C.4
SIM_EPOCHS_RUN = 5000      # the N=80, T=50 simulated panel is cheap on CPU, so the full epoch count is used by default
SIM_N_REPLICATIONS_PAPER = 5   # Monte Carlo replications reported in Table 1, Section 5.1
SIM_N_REPLICATIONS_RUN = 5     # full replication count, affordable at this panel size; --full is a no-op for exp_1
SIM_DENSE_GRID_POINTS = 40     # points per axis for the R^2 evaluation grids, not specified in the paper

# Optimization settings shared with the application (Section C.4)
LR_VARIATIONAL = 0.01      # Adam learning rate for variational parameters, Section C.4
LR_MODEL = 0.005           # Adam learning rate for model (kernel/inducing) parameters, Section C.4
GRAD_CLIP_NORM = 1.0       # gradient clipping to unit Euclidean norm, Section C.4
BATCH_SIZE_PAPER = 2048    # minibatch size in rows, Section C.4
N_FACTOR_SAMPLES = 64      # Monte Carlo factor draws per likelihood evaluation (32 antithetic pairs), Section C.4
LR_HALVING_FRACTIONS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)  # fractions of training at which lr halves, Section C.4
JITTER_INIT = 1e-7         # initial Cholesky jitter, increased tenfold on failure, Section C.3
GAMMA_BETA = 0.1           # gamma_beta in eta_h initialization, Equation C.6
SIGMA_R2_INIT = 0.1        # initial observation-noise variance in scaled units, Equation C.6
RHO_INIT_RBF = 0.1         # initial rho_h share for the RBF factor kernel, Section C.4
B_INIT_RBF = 0.1           # initial b_h share for the RBF factor kernel, Section C.4
A_INIT_RBF = 0.8           # initial a_h share for the RBF factor kernel, Section C.4
VARIATIONAL_MEAN_INIT_STD = 0.001  # s.d. of the initial whitened variational mean, Section C.4

# ---------------------------------------------------------------------------
# Experiments 2-4: S&P 500 application (paper Section 6.1)
# ---------------------------------------------------------------------------
SP500_TRAIN_START_PAPER = "1960-01-01"   # fixed-training start, Section 6.1
SP500_TRAIN_END_PAPER = "1989-12-31"     # fixed-training end, Section 6.1
SP500_VALID_START = "1990-01-01"         # validation-period start, Section 6.1
SP500_VALID_END = "1994-12-31"           # validation-period end, Section 6.1
SP500_EVAL_START = "1995-01-01"          # evaluation-period start, Section 6.1
SP500_EVAL_END = "2024-12-31"            # evaluation-period end, Section 6.1

N_CHARACTERISTICS_PAPER = 137   # characteristics selected by the paper from Jensen et al. (2023), Section 6.1
N_CONTEXT_PAPER = 6             # macro-financial context variables from Goyal et al. (2024), Section 6.1

# The paper's 137 WRDS characteristics and historical CRSP/S&P 500 membership
# are not obtainable through the QuantX Dataloader or Massive REST APIs used
# here (see results/RESULTS.md, "Deviations from the paper"). This
# implementation instead builds a smaller panel of price-based
# characteristics from Dataloader OHLCV bars for a reduced S&P 500 universe
# and the longest window that fits the available data and compute budget.
N_CHARACTERISTICS_RUN = 12      # price-based characteristics actually computed, not specified in the paper
N_CONTEXT_RUN = 3               # market-wide context variables actually computed from the loaded panel

UNIVERSE_SIZE_PAPER = "full historical S&P 500 membership (465-497 stocks/month)"  # Section 6.1
UNIVERSE_SIZE_RUN = 40          # tickers used in the executed configuration, cut by liquidity (dollar volume) at the sample start
UNIVERSE_SIZE_RUN_SMALL = 25    # tickers used for the H-sweep (exp_3) to keep its larger grid of runs within budget

# Candidate pool for the executed universe: large, long-listed S&P 500
# constituents spanning the main GICS sectors, each trading continuously
# since before SP500_TRAIN_START_RUN. Historical point-in-time CRSP/S&P 500
# membership (paper Section 6.1) is not obtainable through the QuantX
# Dataloader or Massive REST APIs; this fixed pool is a substitute universe,
# cut to UNIVERSE_SIZE_RUN/UNIVERSE_SIZE_RUN_SMALL by trailing dollar-volume
# liquidity as of the training start (see results/RESULTS.md).
SP500_CANDIDATE_TICKERS = [
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "WMT", "JPM", "BAC", "WFC",
    "C", "GE", "IBM", "INTC", "CSCO", "ORCL", "HD", "MCD", "DIS", "VZ",
    "T", "PFE", "MRK", "ABT", "CVX", "COP", "PEP", "MMM", "CAT", "BA",
    "UPS", "UNH", "MO", "SO", "DUK", "AXP", "GS", "USB", "TGT", "LOW",
    "COST", "SBUX", "MDT", "HON", "EMR", "ADBE", "TXN", "QCOM", "AMGN", "NKE",
]

SP500_TRAIN_START_RUN = "2005-01-01"  # executed training start (see RESULTS.md for the runtime-driven window choice)
SP500_TRAIN_END_RUN = "2014-12-31"    # executed training end
SP500_VALID_START_RUN = "2015-01-01"  # executed validation start
SP500_VALID_END_RUN = "2016-12-31"    # executed validation end
SP500_EVAL_START_RUN = "2017-01-01"   # executed evaluation start
SP500_EVAL_END_RUN = "2024-12-31"     # executed evaluation end

H_FACTORS_PAPER = 5             # number of factors used after the validation compromise, Section 6.1/6.5
M_INDUCING_PAPER = 160           # trainable inducing inputs per component, Section 6.1
M_INDUCING_RUN = 20               # inducing inputs per component in the executed configuration
EPOCHS_APPLICATION_PAPER = 500    # training epochs in the application, Section C.4
EPOCHS_APPLICATION_RUN = 60       # epochs executed here; --full selects EPOCHS_APPLICATION_PAPER
N_SEEDS_PAPER = 5                 # optimization seeds averaged in Tables 2-3, Section 6.1
N_SEEDS_RUN = 2                   # seeds executed here; --full selects N_SEEDS_PAPER

H_GRID_PAPER = list(range(1, 9))  # H = 1,...,8 swept in Section 6.5
H_GRID_RUN = [1, 2, 3, 5]         # reduced grid executed here for the H-sweep (exp_3)

NEWEY_WEST_LAGS = 12               # Bartlett/Newey-West lags for pairwise tests, Section 6.1/D.3
HOLM_N_COMPARISONS = 15            # pairwise comparisons per criterion among the six kernel/benchmark rows, Table 2/S1

# ---------------------------------------------------------------------------
# Experiment 4: annual re-estimation and trading costs (paper Section 6.6)
# ---------------------------------------------------------------------------
ROLLING_WINDOW_MONTHS_PAPER = 360   # months per rolling training window, Section 6.1/6.6
ROLLING_WINDOW_MONTHS_RUN = 60      # executed rolling window length (reduced universe/period)
ROLLING_FIRST_FORECAST_YEAR_PAPER = 1995  # first annually re-estimated forecast year, Section 6.6
ROLLING_LAST_FORECAST_YEAR_PAPER = 2024   # last annually re-estimated forecast year, Section 6.6
ROLLING_REESTIMATION_STRIDE_RUN = 2       # re-estimate every N years in the executed configuration, not specified in the paper
TRADING_COST_BPS = 10               # proportional cost c, 10 basis points per dollar traded, Section 6.6/Equation 34

# ---------------------------------------------------------------------------
# Experiment 5: higher moments of the bilinear predictive distribution (B.3)
# ---------------------------------------------------------------------------
CGF_QUAD_TOL = 1e-9          # absolute/relative tolerance for scipy.integrate.quad_vec, Section 6.7
SADDLEPOINT_BRACKET = 50.0   # bracket half-width for the saddlepoint root search, not specified in the paper
STD_ERROR_BINS = (1.0, 2.0, 3.0)  # bins of |standardized error| reported in Section 6.7

# ---------------------------------------------------------------------------
# Shared input/return scaling (paper Section C.1)
# ---------------------------------------------------------------------------
CHARACTERISTIC_RANK_RANGE = (-0.5, 0.5)  # monthly cross-sectional rank range, Equation C.1 context
RANDOM_SEED_BASE = 20261104               # base seed for reproducibility, not specified in the paper
