# Regime-Aware Rare-Event Momentum Dashboard

This Build 2 dashboard investigates this question:

> Do rare positive and negative price events exhibit stronger directional momentum when they align with a Gaussian-Mixture-Model market regime, and does that improve a simulated long/short sleeve relative to trading all rare events?

## Run

```bash
cd /home/daniel/ml4t_files/research_dashboard/pythonProject1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dashboard.txt
python -m streamlit run trading_dashboard.py
```

The virtual environment is intentional: TensorFlow 2.11 requires an older
`protobuf` release than current Streamlit packages. Installing dashboard
dependencies into `.venv` isolates them from TensorFlow and avoids changing
the shared Python environment.

## Data

The default source is a local ML4T-style folder containing one CSV per ticker.
The dashboard includes a local `util.py` copy with the course `get_data` pattern,
but uses an explicit folder input rather than relying on `MARKET_DATA_DIR`.
Its default location is:

```text
/home/daniel/ml4t_files/assess_learners_2026Fall/ml4t_env/data
```

Enter tickers such as `AAPL, MSFT, JPM, XOM`; `SPY` is added automatically as
the market benchmark. The bundled course `SPY.csv` runs from 2000-02-01 through
2012-09-12, so the default dashboard range is 2008-01-01 through 2012-09-12.
You can also upload a combined CSV with a `Date` column and adjusted-close prices.

```text
Date,SPY,AAPL,MSFT,JPM
2020-01-02,324.87,75.09,160.62,139.14
2020-01-03,322.41,74.36,158.62,137.91
```

The application also has deterministic synthetic demo data. It is exclusively for testing the user interface and must not be used as a research result.

## Saved experiment workflow

Use the dashboard in this order:

1. Select chronological **Training end** and **Validation end** dates, leaving later dates for testing.
2. Click **Train and save model**. This writes a GMM/scaler artifact and metadata to `artifacts/`.
3. Select that artifact, then click **Validate and lock settings**. This writes the settings and validation metrics to `experiment_locks/` and `results/`.
4. Select the lock and click **Run final test**. The dashboard loads the locked model/settings without refitting, and saves test trades and a summary in `results/`.

After locking settings, **Run bootstrap robustness check** optionally runs 20–250
multivariate block-bootstrap simulations. Each simulation resamples consecutive
80-, 100-, or 120-day return blocks, rebuilds the price path, refits a GMM only
on its simulated training segment, and evaluates the locked rules on its
simulated test segment. The saved distribution is an uncertainty check; it must
not be used to tune the locked configuration.

The intended split is chronological: training is used to learn regimes, validation chooses and locks settings, and the final test is not used to tune the model.

## Build 2 mapping

The dashboard shows the current probabilistic GMM regime, learned cluster evidence, historical conditional event performance, a regime-aware versus unconditional baseline, transaction-cost effects, current/recent signals, confidence, and methodological limitations.
