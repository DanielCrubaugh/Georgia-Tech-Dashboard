"""Regime-aware rare-event momentum research dashboard.

Run: streamlit run trading_dashboard.py
Upload a CSV with Date and adjusted-close columns (for example SPY, AAPL).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from util import get_data


st.set_page_config(page_title="Regime-Aware Trading Research", layout="wide")

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
LOCK_DIR = APP_DIR / "experiment_locks"
RESULT_DIR = APP_DIR / "results"
for directory in (ARTIFACT_DIR, LOCK_DIR, RESULT_DIR):
    directory.mkdir(exist_ok=True)


def demo_prices() -> pd.DataFrame:
    """Deterministic synthetic data for demonstrating the dashboard only."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2012-01-03", periods=3200)
    states = np.repeat([0, 1, 2, 0, 2, 1], [650, 450, 500, 650, 500, 450])[:len(dates)]
    market_mu = np.choose(states, [0.00045, 0.00005, -0.00055])
    market_sigma = np.choose(states, [0.008, 0.013, 0.021])
    market_return = market_mu + market_sigma * rng.standard_normal(len(dates))
    result = {"SPY": 100 * np.exp(np.cumsum(market_return))}
    for ticker, beta, noise in [("AAPL", 1.15, 0.011), ("MSFT", 0.95, 0.009),
                                ("JPM", 1.10, 0.012), ("XOM", 0.85, 0.014)]:
        returns = beta * market_return + rng.standard_normal(len(dates)) * noise
        rows = rng.choice(np.arange(100, len(dates) - 25), size=14, replace=False)
        returns[rows] += rng.choice([-1, 1], size=14) * rng.uniform(0.06, 0.11, len(rows))
        result[ticker] = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(result, index=dates)


def load_prices(file) -> pd.DataFrame:
    frame = pd.read_csv(file)
    date_col = next((c for c in frame if c.lower() in {"date", "datetime"}), frame.columns[0])
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.set_index(date_col).sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return frame.loc[~frame.index.isna()]


def load_ml4t_prices(symbol_text: str, start_date, end_date, data_dir: str) -> pd.DataFrame:
    """Load a dashboard-ready price frame from the class one-CSV-per-ticker data."""
    symbols = [symbol.strip().upper() for symbol in symbol_text.split(",") if symbol.strip()]
    if not symbols:
        raise ValueError("Enter at least one ticker in addition to SPY.")
    dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    return get_data(symbols, dates, addSPY=True, base_dir=data_dir).dropna(axis=1, how="all")


def make_features(benchmark: pd.Series) -> pd.DataFrame:
    ret = benchmark.pct_change()
    return pd.DataFrame({
        "return_20d": benchmark.pct_change(20),
        "return_60d": benchmark.pct_change(60),
        "volatility_20d": ret.rolling(20).std() * np.sqrt(252),
        "drawdown_60d": benchmark / benchmark.rolling(60).max() - 1,
        "ma_spread": benchmark.rolling(20).mean() / benchmark.rolling(60).mean() - 1,
    }).dropna()


def fit_training_model(features: pd.DataFrame, train_end: pd.Timestamp, clusters: int) -> dict:
    """Learn the scaler/GMM exclusively from the configured training period."""
    training = features.loc[features.index <= train_end]
    if len(training) < max(90, clusters * 20):
        raise ValueError("Training period is too short; use at least 90 complete feature rows.")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(training)
    gmm = GaussianMixture(n_components=clusters, covariance_type="full", n_init=10, random_state=42)
    gmm.fit(x_train)
    cluster = gmm.predict(x_train)
    cluster_stats = training.assign(cluster=cluster).groupby("cluster").mean(numeric_only=True)

    # Numeric cluster labels have no inherent meaning. Name them from learned properties.
    score = cluster_stats["return_60d"] - cluster_stats["volatility_20d"]
    ordered = score.sort_values().index.tolist()
    names = {ordered[0]: "Bear / high volatility", ordered[-1]: "Bull / low volatility"}
    for raw_cluster in ordered[1:-1]:
        names[raw_cluster] = "Sideways / uncertain"
    return {"scaler": scaler, "gmm": gmm, "names": names, "cluster_stats": cluster_stats,
            "features": list(features.columns), "train_end": str(train_end.date())}


def apply_saved_model(features: pd.DataFrame, artifact: dict, validation_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a frozen training artifact; no scaler or GMM fitting occurs here."""
    if list(features.columns) != artifact["features"]:
        raise ValueError("Feature columns differ from the saved training model.")
    probs = artifact["gmm"].predict_proba(artifact["scaler"].transform(features))
    cluster = probs.argmax(axis=1)
    train_end = pd.Timestamp(artifact["train_end"])
    names = {int(key): value for key, value in artifact["names"].items()}

    regime = pd.DataFrame(probs, index=features.index, columns=[f"p_{x}" for x in range(probs.shape[1])])
    regime["cluster"] = cluster
    regime["regime"] = pd.Series(cluster, index=features.index).map(names)
    regime["confidence"] = probs.max(axis=1)
    regime["phase"] = np.where(regime.index <= train_end, "Training",
                         np.where(regime.index <= validation_end, "Validation", "Test"))
    cluster_stats = artifact["cluster_stats"].copy()
    cluster_stats["regime"] = pd.Series(cluster_stats.index, index=cluster_stats.index).map(names)
    return regime, cluster_stats


def find_events(prices: pd.DataFrame, benchmark: str, regime: pd.DataFrame, z_cutoff: float,
                holding_days: int, one_way_cost_bps: float, confidence_floor: float) -> pd.DataFrame:
    returns = prices.pct_change()
    output = []
    for ticker in prices:
        if ticker == benchmark:
            continue
        z = (returns[ticker] - returns[ticker].rolling(60).mean()) / returns[ticker].rolling(60).std()
        # Entry is next-day close; exit is holding_days later. No same-day execution assumption.
        forward = prices[ticker].shift(-holding_days) / prices[ticker].shift(-1) - 1
        event = pd.DataFrame({"ticker": ticker, "daily_return": returns[ticker], "z_score": z,
                              "forward_return": forward})
        event = event.loc[event["z_score"].abs() >= z_cutoff]
        event["side"] = np.where(event["z_score"] > 0, "Long", "Short")
        output.append(event)
    if not output:
        return pd.DataFrame()
    trades = pd.concat(output).join(regime[["regime", "confidence", "phase"]], how="inner")
    trades.index.name = "event_date"
    trades["gross_return"] = np.where(trades["side"].eq("Long"), trades["forward_return"], -trades["forward_return"])
    trades["net_return"] = trades["gross_return"] - 2 * one_way_cost_bps / 10_000
    trades["aligned"] = ((trades["side"].eq("Long") & trades["regime"].eq("Bull / low volatility")) |
                         (trades["side"].eq("Short") & trades["regime"].eq("Bear / high volatility")))
    trades["actionable"] = trades["aligned"] & (trades["confidence"] >= confidence_floor)
    return trades.dropna(subset=["forward_return"])


def stats(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {"trades": 0, "mean": np.nan, "win": np.nan, "sharpe": np.nan, "drawdown": np.nan}
    wealth = (1 + returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    return {
        "trades": len(returns),
        "mean": returns.mean(),
        "win": (returns > 0).mean(),
        "sharpe": np.sqrt(252) * returns.mean() / returns.std() if returns.std() else np.nan,
        "drawdown": dd.min(),
    }


def block_bootstrap_prices(prices: pd.DataFrame, block_days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Resample consecutive *multivariate* return blocks and rebuild price paths.

    Sampling complete rows preserves the contemporaneous relationship among the
    benchmark and tickers. Sampling contiguous rows preserves short-run serial
    behavior better than independently sampling daily returns.
    """
    complete_prices = prices.dropna(how="any")
    returns = complete_prices.pct_change().dropna()
    if len(returns) < block_days:
        raise ValueError("Block length exceeds the available complete return history.")
    row_positions: list[int] = []
    while len(row_positions) < len(returns):
        start = int(rng.integers(0, len(returns) - block_days + 1))
        row_positions.extend(range(start, start + block_days))
    sampled = returns.iloc[row_positions[:len(returns)]].copy()
    sampled.index = returns.index
    initial = complete_prices.iloc[0]
    rebuilt = (1 + sampled).cumprod().mul(initial, axis="columns")
    return pd.concat([initial.to_frame().T.rename(index={initial.name: complete_prices.index[0]}), rebuilt])


def bootstrap_robustness(
    prices: pd.DataFrame,
    benchmark: str,
    lock: dict,
    simulations: int,
    block_days: int,
) -> pd.DataFrame:
    """Refit the frozen workflow on block-resampled histories for uncertainty estimates."""
    results = []
    rng = np.random.default_rng(2026)
    for simulation in range(simulations):
        boot_prices = block_bootstrap_prices(prices, block_days, rng)
        boot_features = make_features(boot_prices[benchmark])
        try:
            boot_artifact = fit_training_model(boot_features, pd.Timestamp(lock["train_end"]),
                                               len(joblib.load(ARTIFACT_DIR / lock["model_file"])["names"]))
            boot_regime, _ = apply_saved_model(boot_features, boot_artifact, pd.Timestamp(lock["validation_end"]))
            boot_trades = find_events(boot_prices.reindex(boot_features.index).ffill(), benchmark, boot_regime,
                                      float(lock["z_cutoff"]), int(lock["holding_days"]),
                                      float(lock["cost_bps"]), float(lock["confidence_floor"]))
            test = boot_trades.loc[boot_trades["phase"].eq("Test")]
            aligned = stats(test.loc[test["actionable"], "net_return"])
            baseline = stats(test["net_return"])
            results.append({"simulation": simulation + 1, "aligned_mean": aligned["mean"],
                            "aligned_sharpe": aligned["sharpe"], "aligned_drawdown": aligned["drawdown"],
                            "aligned_trades": aligned["trades"], "baseline_mean": baseline["mean"],
                            "baseline_sharpe": baseline["sharpe"]})
        except (ValueError, np.linalg.LinAlgError):
            # A rare degenerate resample is recorded as missing rather than used to improve results.
            results.append({"simulation": simulation + 1, "aligned_mean": np.nan, "aligned_sharpe": np.nan,
                            "aligned_drawdown": np.nan, "aligned_trades": 0, "baseline_mean": np.nan,
                            "baseline_sharpe": np.nan})
    return pd.DataFrame(results)


st.title("Regime-Aware Rare-Event Momentum")
st.caption("Research dashboard: simulated long/short event positions, not investment advice or trade execution.")

with st.sidebar:
    st.header("Data and settings")
    source = st.radio("Data source", ["Class ML4T CSV folder", "Upload combined CSV", "Synthetic demo data"])
    class_dir = "/home/daniel/ml4t_files/assess_learners_2026Fall/ml4t_env/data"
    if source == "Class ML4T CSV folder":
        data_dir = st.text_input("ML4T data folder", value=class_dir)
        symbols = st.text_input("Tickers to scan (SPY is added automatically)", value="AAPL, MSFT, JPM, XOM")
        # The bundled ML4T SPY history runs through 2012-09-12.
        start_date = st.date_input("Start date", value=pd.Timestamp("2008-01-01"))
        end_date = st.date_input("End date", value=pd.Timestamp("2012-09-12"))
        upload = None
    elif source == "Upload combined CSV":
        upload = st.file_uploader("Adjusted-close price CSV", type="csv")
        data_dir = None
        symbols = ""
        start_date, end_date = None, None
    else:
        upload = None
        data_dir = None
        symbols = ""
        start_date, end_date = None, None
    z_cutoff = st.slider("Rare-event z-score threshold", 2.0, 4.5, 3.0, 0.1)
    holding_days = st.select_slider("Holding period", [5, 10, 20, 40], value=20, format_func=lambda x: f"{x} days")
    confidence_floor = st.slider("Minimum regime confidence", 0.34, 0.95, 0.55, 0.01)
    cost_bps = st.number_input("One-way transaction cost (bps)", 0.0, 100.0, 10.0, 1.0)
    core_weight = st.slider("Index-core allocation", 0.60, 0.95, 0.80, 0.05)

try:
    if source == "Class ML4T CSV folder":
        prices = load_ml4t_prices(symbols, start_date, end_date, data_dir)
    elif source == "Upload combined CSV":
        if upload is None:
            st.info("Upload a combined adjusted-close CSV, or choose the class data folder.")
            st.stop()
        prices = load_prices(upload)
    else:
        prices = demo_prices()
except Exception as error:
    st.error(f"Could not load prices: {error}")
    st.stop()

if prices.shape[1] < 2 or len(prices) < 180:
    st.error("Provide at least two price columns and 180 trading days.")
    st.stop()

benchmark = st.sidebar.selectbox("Market benchmark", prices.columns, index=0)
features = make_features(prices[benchmark].dropna())
if len(features) < 180:
    st.error("The benchmark has insufficient complete price history for a train/validation/test workflow.")
    st.stop()

date_min, date_max = features.index.min().date(), features.index.max().date()
default_train = features.index[int(len(features) * 0.50)].date()
default_validate = features.index[int(len(features) * 0.75)].date()
st.sidebar.header("Experiment layers")
train_end = pd.Timestamp(st.sidebar.date_input("Training ends", value=default_train, min_value=date_min, max_value=date_max))
validation_end = pd.Timestamp(st.sidebar.date_input("Validation ends", value=default_validate, min_value=date_min, max_value=date_max))
cluster_count = st.sidebar.selectbox("GMM components", [2, 3, 4], index=1)
if not train_end < validation_end < features.index.max():
    st.error("Set Training end before Validation end, and leave later dates for the final test.")
    st.stop()

def model_paths() -> list[Path]:
    return sorted(ARTIFACT_DIR.glob("model_*.joblib"), reverse=True)


def lock_paths() -> list[Path]:
    return sorted(LOCK_DIR.glob("validation_lock_*.json"), reverse=True)


def save_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, default=str), encoding="utf-8")


st.subheader("1. Research setup and saved layers")
st.caption("Training learns the GMM from early dates only. Validation locks settings. The final-test stage loads the lock and does not refit.")
train_button, validate_button, test_button = st.columns(3)

with train_button:
    if st.button("Train and save model", use_container_width=True):
        try:
            artifact = fit_training_model(features, train_end, cluster_count)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            artifact_path = ARTIFACT_DIR / f"model_{run_id}.joblib"
            joblib.dump(artifact, artifact_path)
            save_json(ARTIFACT_DIR / f"model_{run_id}.json", {
                "run_id": run_id, "model_file": artifact_path.name, "benchmark": benchmark,
                "tickers": list(prices.columns), "feature_start": str(features.index.min().date()),
                "train_end": str(train_end.date()), "gmm_components": cluster_count,
                "features": artifact["features"], "created_utc": datetime.now(timezone.utc).isoformat(),
            })
            st.success(f"Saved training model: {artifact_path.name}")
        except Exception as error:
            st.error(f"Training was not saved: {error}")

available_models = model_paths()
if not available_models:
    st.info("Train and save a model to enable validation and final testing.")
    st.stop()

selected_model = st.selectbox("Saved training model", available_models, format_func=lambda path: path.name)
artifact = joblib.load(selected_model)
regime, cluster_stats = apply_saved_model(features, artifact, validation_end)
trades = find_events(prices.reindex(features.index).ffill(), benchmark, regime, z_cutoff,
                     holding_days, cost_bps, confidence_floor)

with validate_button:
    if st.button("Validate and lock settings", use_container_width=True):
        validation_trades = trades.loc[trades["phase"].eq("Validation")]
        aligned_validation = validation_trades.loc[validation_trades["actionable"]]
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lock = {
            "run_id": run_id, "model_file": selected_model.name, "benchmark": benchmark,
            "train_end": artifact["train_end"], "validation_end": str(validation_end.date()),
            "z_cutoff": z_cutoff, "holding_days": holding_days,
            "confidence_floor": confidence_floor, "cost_bps": cost_bps,
            "validation_regime_aligned": stats(aligned_validation["net_return"]),
            "validation_all_events": stats(validation_trades["net_return"]),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        lock_path = LOCK_DIR / f"validation_lock_{run_id}.json"
        save_json(lock_path, lock)
        validation_trades.reset_index().to_csv(RESULT_DIR / f"validation_trades_{run_id}.csv", index=False)
        st.success(f"Validation settings locked: {lock_path.name}")

available_locks = lock_paths()
if not available_locks:
    st.info("Validate and lock a configuration before running the final test.")
    st.stop()

selected_lock = st.selectbox("Locked validation configuration", available_locks, format_func=lambda path: path.name)
lock = json.loads(selected_lock.read_text(encoding="utf-8"))

with test_button:
    if st.button("Run final test", type="primary", use_container_width=True):
        locked_artifact = joblib.load(ARTIFACT_DIR / lock["model_file"])
        locked_regime, _ = apply_saved_model(features, locked_artifact, pd.Timestamp(lock["validation_end"]))
        locked_trades = find_events(prices.reindex(features.index).ffill(), benchmark, locked_regime,
                                    float(lock["z_cutoff"]), int(lock["holding_days"]),
                                    float(lock["cost_bps"]), float(lock["confidence_floor"]))
        final_test = locked_trades.loc[locked_trades["phase"].eq("Test")].copy()
        final_test["experiment_lock"] = selected_lock.name
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        final_test.reset_index().to_csv(RESULT_DIR / f"final_test_trades_{run_id}.csv", index=False)
        save_json(RESULT_DIR / f"final_test_summary_{run_id}.json", {
            "experiment_lock": selected_lock.name, "test_start": str(pd.Timestamp(lock["validation_end"]) + pd.Timedelta(days=1)),
            "regime_aligned": stats(final_test.loc[final_test["actionable"], "net_return"]),
            "all_events": stats(final_test["net_return"]), "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        st.session_state["final_test_file"] = str(RESULT_DIR / f"final_test_trades_{run_id}.csv")
        st.success("Final test saved. Its settings are inherited from the selected validation lock.")

latest = regime.iloc[-1]
prob_cols = [column for column in regime if column.startswith("p_")]
probability_table = pd.DataFrame({"cluster": [int(x.split("_")[1]) for x in prob_cols],
                                  "probability": latest[prob_cols].to_numpy(dtype=float)})
probability_table["regime"] = probability_table["cluster"].map(cluster_stats["regime"])
probability_table = probability_table.groupby("regime", as_index=False)["probability"].sum()

st.subheader("2. Frozen model indication")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current regime", latest["regime"])
col2.metric("GMM confidence", f"{latest['confidence']:.0%}")
multiplier = {"Bull / low volatility": 1.0, "Sideways / uncertain": 0.5, "Bear / high volatility": 0.25}
satellite_cap = (1 - core_weight) * sum(row.probability * multiplier[row.regime] for row in probability_table.itertuples())
col3.metric("Simulated tactical cap", f"{satellite_cap:.1%}")
col4.metric("Index core", f"{core_weight:.0%}")
left, right = st.columns(2)
with left:
    st.bar_chart(probability_table.set_index("regime"))
with right:
    display_stats = cluster_stats[["regime", "return_20d", "return_60d", "volatility_20d", "drawdown_60d"]]
    st.dataframe(display_stats.style.format({x: "{:.2%}" for x in display_stats.columns if x != "regime"}), use_container_width=True)

st.subheader("3. Final test results")
test_file = st.session_state.get("final_test_file")
if test_file and Path(test_file).exists():
    test = pd.read_csv(test_file, parse_dates=["event_date"]).set_index("event_date")
    aligned, baseline = test.loc[test["actionable"]], test
    for column, heading, values in zip(st.columns(2), ["Regime-aligned strategy", "Baseline: all rare events"],
                                       [stats(aligned["net_return"]), stats(baseline["net_return"])]):
        with column:
            st.markdown(f"**{heading}**")
            st.metric("Trades", int(values["trades"]))
            st.metric("Mean net trade return", "—" if pd.isna(values["mean"]) else f"{values['mean']:.2%}")
            st.metric("Win rate", "—" if pd.isna(values["win"]) else f"{values['win']:.1%}")
            st.metric("Trade-sequence Sharpe", "—" if pd.isna(values["sharpe"]) else f"{values['sharpe']:.2f}")

    baseline_daily = baseline.groupby(level=0)["net_return"].mean()
    aligned_daily = aligned.groupby(level=0)["net_return"].mean().reindex(baseline_daily.index, fill_value=0)
    st.line_chart(pd.DataFrame({"Regime-aligned": (1 + aligned_daily).cumprod(),
                                "All-events baseline": (1 + baseline_daily).cumprod()}))
else:
    st.info("No final test has been run in this session. The final-test files will be written to the results folder.")

st.subheader("4. Optional block-bootstrap robustness check")
st.caption("This does not select settings or replace the final test. It estimates how sensitive the locked workflow is to the sampled historical path.")
boot_a, boot_b, boot_c = st.columns([1, 1, 2])
with boot_a:
    bootstrap_count = st.select_slider("Bootstrap simulations", options=[20, 50, 100, 250], value=50)
with boot_b:
    bootstrap_block_days = st.select_slider("Block length", options=[80, 100, 120], value=100,
                                            help="100 days covers the 60-day feature lookback and 20-day holding period with buffer.")
with boot_c:
    run_bootstrap = st.button("Run bootstrap robustness check", use_container_width=True)

if run_bootstrap:
    with st.spinner("Resampling return blocks, refitting GMMs, and evaluating simulated test periods..."):
        try:
            bootstrap = bootstrap_robustness(prices[[column for column in prices.columns if column in prices]], benchmark,
                                             lock, bootstrap_count, bootstrap_block_days)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            bootstrap_path = RESULT_DIR / f"bootstrap_{run_id}.csv"
            bootstrap.to_csv(bootstrap_path, index=False)
            st.session_state["bootstrap_file"] = str(bootstrap_path)
            st.success(f"Saved {len(bootstrap)} bootstrap simulations to {bootstrap_path.name}")
        except Exception as error:
            st.error(f"Bootstrap run could not complete: {error}")

bootstrap_file = st.session_state.get("bootstrap_file")
if bootstrap_file and Path(bootstrap_file).exists():
    bootstrap = pd.read_csv(bootstrap_file)
    valid = bootstrap.dropna(subset=["aligned_sharpe"])
    if valid.empty:
        st.warning("All bootstrap simulations were degenerate. Try a shorter block length or more complete ticker histories.")
    else:
        p025, p50, p975 = valid["aligned_sharpe"].quantile([0.025, 0.50, 0.975])
        b1, b2, b3 = st.columns(3)
        b1.metric("Valid simulations", f"{len(valid)} / {len(bootstrap)}")
        b2.metric("Median simulated Sharpe", f"{p50:.2f}")
        b3.metric("95% simulated Sharpe interval", f"{p025:.2f} to {p975:.2f}")
        st.bar_chart(valid["aligned_sharpe"])
        st.dataframe(valid, use_container_width=True, hide_index=True)

with st.expander("Confidence, limits, and saved records", expanded=True):
    st.markdown("""
    - Training artifacts save the scaler, GMM parameters, cluster names, dates, and feature list.
    - Validation locks settings before testing. A final test loads that lock and does not refit the model.
    - Low GMM confidence is a no-trade condition. A GMM describes historical feature similarity, not future certainty.
    - Extreme-event samples can be small; costs, short availability, survivorship bias, and overlapping positions can materially change results.
    - Bootstrap simulations resample historical return blocks to quantify path sensitivity; they do not prove that a strategy will work in future markets.
    - This is a historical educational backtest, not investment advice.
    """)

if source == "Synthetic demo data":
    st.info("Demo mode is for interface testing only. Do not use synthetic results as research evidence.")
