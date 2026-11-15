"""Regime-aware rare-event momentum research dashboard.

Run: streamlit run trading_dashboard.py
Upload a CSV with Date and adjusted-close columns (for example SPY, AAPL).
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from itertools import product
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
    names = {ordered[0]: "Downtrend / higher volatility", ordered[-1]: "Uptrend / lower volatility"}
    for raw_cluster in ordered[1:-1]:
        names[raw_cluster] = "Mixed / uncertain"
    return {"scaler": scaler, "gmm": gmm, "names": names, "cluster_stats": cluster_stats,
            "features": list(features.columns), "train_end": str(train_end.date())}


def apply_saved_model(features: pd.DataFrame, artifact: dict, validation_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a frozen training artifact; no scaler or GMM fitting occurs here."""
    if list(features.columns) != artifact["features"]:
        raise ValueError("Feature columns differ from the saved training model.")
    probs = artifact["gmm"].predict_proba(artifact["scaler"].transform(features))
    cluster = probs.argmax(axis=1)
    train_end = pd.Timestamp(artifact["train_end"])
    legacy_names = {
        "Bull / low volatility": "Uptrend / lower volatility",
        "Sideways / uncertain": "Mixed / uncertain",
        "Bear / high volatility": "Downtrend / higher volatility",
    }
    names = {int(key): legacy_names.get(value, value) for key, value in artifact["names"].items()}

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
    date_series = pd.Series(prices.index, index=prices.index)
    output = []
    for ticker in prices:
        if ticker == benchmark:
            continue
        z = (returns[ticker] - returns[ticker].rolling(60).mean()) / returns[ticker].rolling(60).std()
        # Entry is next-day close; exit is holding_days later. No same-day execution assumption.
        forward = prices[ticker].shift(-holding_days) / prices[ticker].shift(-1) - 1
        event = pd.DataFrame({"ticker": ticker, "daily_return": returns[ticker], "z_score": z,
                              "forward_return": forward, "entry_date": date_series.shift(-1),
                              "exit_date": date_series.shift(-holding_days)})
        event = event.loc[event["z_score"].abs() >= z_cutoff]
        event["side"] = np.where(event["z_score"] > 0, "Long", "Short")
        output.append(event)
    if not output:
        return pd.DataFrame()
    trades = pd.concat(output).join(regime[["regime", "confidence", "phase"]], how="inner")
    trades.index.name = "event_date"
    trades["gross_return"] = np.where(trades["side"].eq("Long"), trades["forward_return"], -trades["forward_return"])
    trades["net_return"] = trades["gross_return"] - 2 * one_way_cost_bps / 10_000
    trades["aligned"] = ((trades["side"].eq("Long") & trades["regime"].eq("Uptrend / lower volatility")) |
                         (trades["side"].eq("Short") & trades["regime"].eq("Downtrend / higher volatility")))
    trades["actionable"] = trades["aligned"] & (trades["confidence"] >= confidence_floor)
    return trades.dropna(subset=["forward_return"])


def stats(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {"days": 0, "mean": np.nan, "win": np.nan, "sharpe": np.nan, "drawdown": np.nan}
    wealth = (1 + returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    return {
        "days": len(returns),
        "mean": returns.mean(),
        "win": (returns > 0).mean(),
        "sharpe": np.sqrt(252) * returns.mean() / returns.std() if returns.std() else np.nan,
        "drawdown": dd.min(),
    }


def data_fingerprint(prices: pd.DataFrame, benchmark: str) -> str:
    """Create a reproducible identity for the loaded data and selected benchmark."""
    ordered = prices.sort_index().sort_index(axis="columns")
    digest = hashlib.sha256()
    digest.update(benchmark.encode())
    digest.update("|".join(map(str, ordered.columns)).encode())
    digest.update(pd.util.hash_pandas_object(ordered, index=True).values.tobytes())
    return digest.hexdigest()


def select_positions(trades: pd.DataFrame, phase: str, aligned_only: bool, max_positions: int,
                     phase_end: pd.Timestamp) -> pd.DataFrame:
    """Admit highest-magnitude event signals while respecting a concurrent-position cap."""
    candidates = trades.loc[(trades["phase"].eq(phase)) & (trades["exit_date"] <= phase_end)].copy()
    if aligned_only:
        candidates = candidates.loc[candidates["actionable"]]
    candidates["_event_date"] = candidates.index
    candidates = candidates.sort_values(["_event_date", "z_score"],
                                        key=lambda value: value.abs() if value.name == "z_score" else value,
                                        ascending=[True, False])
    accepted, open_positions = [], []
    for event_date, group in candidates.groupby(level=0, sort=True):
        entry_date = group.iloc[0]["entry_date"]
        open_positions = [position for position in open_positions if position["exit_date"] > entry_date]
        available = max_positions - len(open_positions)
        if available <= 0:
            continue
        for _, trade in group.head(available).iterrows():
            record = trade.to_dict()
            record["event_date"] = event_date
            open_positions.append(record)
            accepted.append(record)
    return pd.DataFrame(accepted)


def daily_portfolio_backtest(prices: pd.DataFrame, benchmark: str, trades: pd.DataFrame, phase: str,
                             phase_dates: pd.DatetimeIndex, aligned_only: bool, core_weight: float,
                             max_positions: int, one_way_cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark positions to market daily; unallocated tactical capital remains cash."""
    phase_start, phase_end = phase_dates.min(), phase_dates.max()
    positions = select_positions(trades, phase, aligned_only, max_positions, phase_end)
    daily_index = prices.loc[(prices.index >= phase_start) & (prices.index <= phase_end)].index
    price_returns = prices.pct_change().reindex(daily_index).fillna(0.0)
    daily = pd.DataFrame(index=daily_index)
    daily["SPY buy-and-hold"] = price_returns[benchmark]
    daily["Core + tactical sleeve"] = core_weight * price_returns[benchmark]
    tactical_slot = (1 - core_weight) / max_positions
    if not positions.empty:
        for _, position in positions.iterrows():
            active = (daily.index > position["entry_date"]) & (daily.index <= position["exit_date"])
            direction = 1 if position["side"] == "Long" else -1
            daily.loc[active, "Core + tactical sleeve"] += tactical_slot * direction * price_returns.loc[active, position["ticker"]]
            if position["entry_date"] in daily.index:
                daily.loc[position["entry_date"], "Core + tactical sleeve"] -= tactical_slot * one_way_cost_bps / 10_000
            if position["exit_date"] in daily.index:
                daily.loc[position["exit_date"], "Core + tactical sleeve"] -= tactical_slot * one_way_cost_bps / 10_000
    daily["Cumulative portfolio"] = (1 + daily["Core + tactical sleeve"]).cumprod()
    daily["Cumulative SPY"] = (1 + daily["SPY buy-and-hold"]).cumprod()
    return daily, positions


def portfolio_metrics(daily_returns: pd.Series, positions: pd.DataFrame) -> dict[str, float]:
    base = stats(daily_returns)
    # ``stats`` calls its observation count ``days``. Here observations are
    # daily portfolio returns; retain that label and separately count entries.
    base["position_entries"] = len(positions)
    base["total_return"] = (1 + daily_returns).prod() - 1 if not daily_returns.empty else np.nan
    return base


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
                            "aligned_days": aligned["days"], "baseline_mean": baseline["mean"],
                            "baseline_sharpe": baseline["sharpe"]})
        except (ValueError, np.linalg.LinAlgError):
            # A rare degenerate resample is recorded as missing rather than used to improve results.
            results.append({"simulation": simulation + 1, "aligned_mean": np.nan, "aligned_sharpe": np.nan,
                            "aligned_drawdown": np.nan, "aligned_days": 0, "baseline_mean": np.nan,
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
    cost_bps = st.number_input("One-way transaction cost (bps)", 0.0, 100.0, 10.0, 1.0)
    core_weight = st.slider("Index-core allocation", 0.60, 0.95, 0.80, 0.05)
    max_positions = st.slider("Maximum concurrent tactical positions", 1, 12, 5)

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
loaded_data_fingerprint = data_fingerprint(prices, benchmark)
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
candidate_z = st.sidebar.multiselect("Validation z-score candidates", [2.5, 3.0, 3.5], default=[2.5, 3.0, 3.5])
candidate_holding = st.sidebar.multiselect("Validation holding-period candidates", [5, 10, 20, 40], default=[10, 20])
candidate_confidence = st.sidebar.multiselect("Validation confidence candidates", [0.45, 0.55, 0.65], default=[0.45, 0.55, 0.65])
st.sidebar.caption("Validation evaluates every selected threshold, holding-period, and confidence combination, then locks the best one before the final test.")
if not train_end < validation_end < features.index.max():
    st.error("Set Training end before Validation end, and leave later dates for the final test.")
    st.stop()

def model_paths() -> list[Path]:
    return sorted(ARTIFACT_DIR.glob("model_*.joblib"), reverse=True)


def lock_paths() -> list[Path]:
    return sorted(LOCK_DIR.glob("validation_lock_*.json"), reverse=True)


def save_json(path: Path, content: dict) -> None:
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return str(value)
    path.write_text(json.dumps(content, indent=2, default=convert), encoding="utf-8")


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
                "features": artifact["features"], "data_fingerprint": loaded_data_fingerprint,
                "created_utc": datetime.now(timezone.utc).isoformat(),
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
model_metadata_path = selected_model.with_suffix(".json")
model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8")) if model_metadata_path.exists() else {}
regime, cluster_stats = apply_saved_model(features, artifact, validation_end)

st.subheader("Saved-model inspection")
st.caption("Inspect this model before validation. These cluster summaries use the selected saved model; they do not retrain it.")
model_details = pd.DataFrame([{
    "saved model": selected_model.name,
    "training ends": model_metadata.get("train_end", artifact.get("train_end", "unknown")),
    "GMM components": model_metadata.get("gmm_components", artifact["gmm"].n_components),
    "benchmark": model_metadata.get("benchmark", benchmark),
}])
st.dataframe(model_details, use_container_width=True, hide_index=True)
inspection_stats = cluster_stats[["regime", "return_20d", "return_60d", "volatility_20d", "drawdown_60d"]]
st.dataframe(inspection_stats.style.format({x: "{:.2%}" for x in inspection_stats.columns if x != "regime"}),
             use_container_width=True, hide_index=True)

with validate_button:
    if st.button("Validate and lock settings", use_container_width=True):
        if not candidate_z or not candidate_holding or not candidate_confidence:
            st.error("Choose at least one candidate for each validation setting.")
            st.stop()
        metadata = model_metadata
        if metadata.get("data_fingerprint") != loaded_data_fingerprint:
            st.error("This saved model was trained on different data or a different benchmark. Train a new model before validation.")
            st.stop()
        validation_dates = regime.index[regime["phase"].eq("Validation")]
        candidate_rows = []
        for candidate in product(candidate_z, candidate_holding, candidate_confidence):
            candidate_trades = find_events(prices.reindex(features.index).ffill(), benchmark, regime,
                                           float(candidate[0]), int(candidate[1]), cost_bps, float(candidate[2]))
            daily, positions = daily_portfolio_backtest(prices, benchmark, candidate_trades, "Validation",
                                                        validation_dates, True, core_weight, max_positions, cost_bps)
            metrics = portfolio_metrics(daily["Core + tactical sleeve"], positions)
            candidate_rows.append({"z_cutoff": candidate[0], "holding_days": candidate[1],
                                   "confidence_floor": candidate[2], **metrics})
        candidates = pd.DataFrame(candidate_rows)
        eligible = candidates.loc[candidates["position_entries"] > 0].copy()
        if eligible.empty:
            st.error("No candidate produced a validation position. Lower the event or confidence thresholds.")
            st.stop()
        chosen = eligible.sort_values(["sharpe", "total_return"], ascending=False).iloc[0].to_dict()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        lock = {
            "run_id": run_id, "model_file": selected_model.name, "benchmark": benchmark,
            "train_end": artifact["train_end"], "validation_end": str(validation_end.date()),
            "z_cutoff": chosen["z_cutoff"], "holding_days": int(chosen["holding_days"]),
            "confidence_floor": chosen["confidence_floor"], "cost_bps": cost_bps,
            "core_weight": core_weight, "max_positions": max_positions,
            "data_fingerprint": loaded_data_fingerprint,
            "validation_candidates_file": f"validation_candidates_{run_id}.csv",
            "selected_validation_metrics": chosen,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        lock_path = LOCK_DIR / f"validation_lock_{run_id}.json"
        save_json(lock_path, lock)
        candidates.to_csv(RESULT_DIR / f"validation_candidates_{run_id}.csv", index=False)
        st.success(f"Locked the best validation candidate: z={chosen['z_cutoff']}, hold={int(chosen['holding_days'])} days, confidence={chosen['confidence_floor']:.0%}.")

available_locks = lock_paths()
if not available_locks:
    st.info("Validate and lock a configuration before running the final test.")
    st.stop()

selected_lock = st.selectbox("Locked validation configuration", available_locks, format_func=lambda path: path.name)
lock = json.loads(selected_lock.read_text(encoding="utf-8"))
candidate_file = RESULT_DIR / lock.get("validation_candidates_file", f"validation_candidates_{lock.get('run_id', '')}.csv")
if candidate_file.exists():
    with st.expander("Inspect saved validation candidates", expanded=True):
        st.caption("These are the settings compared on validation dates. The selected row is the highest-Sharpe candidate with at least one position entry.")
        st.dataframe(pd.read_csv(candidate_file).style.format({
            "mean": "{:.2%}", "win": "{:.1%}", "sharpe": "{:.2f}",
            "drawdown": "{:.2%}", "total_return": "{:.2%}"
        }), use_container_width=True, hide_index=True)
else:
    st.info("No validation-candidate table is available for this older lock. Create a new validation lock to save one.")

with test_button:
    if st.button("Run final test", type="primary", use_container_width=True):
        if lock.get("data_fingerprint") != loaded_data_fingerprint or lock.get("benchmark") != benchmark:
            st.error("The loaded data or benchmark does not match this validation lock. Reload the original data or create a new experiment.")
            st.stop()
        locked_artifact = joblib.load(ARTIFACT_DIR / lock["model_file"])
        locked_regime, _ = apply_saved_model(features, locked_artifact, pd.Timestamp(lock["validation_end"]))
        locked_trades = find_events(prices.reindex(features.index).ffill(), benchmark, locked_regime,
                                    float(lock["z_cutoff"]), int(lock["holding_days"]),
                                    float(lock["cost_bps"]), float(lock["confidence_floor"]))
        final_test = locked_trades.loc[locked_trades["phase"].eq("Test")].copy()
        final_test["experiment_lock"] = selected_lock.name
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        test_dates = locked_regime.index[locked_regime["phase"].eq("Test")]
        model_daily, model_positions = daily_portfolio_backtest(
            prices, benchmark, final_test, "Test", test_dates, True, float(lock["core_weight"]),
            int(lock["max_positions"]), float(lock["cost_bps"])
        )
        baseline_daily, baseline_positions = daily_portfolio_backtest(
            prices, benchmark, final_test, "Test", test_dates, False, float(lock["core_weight"]),
            int(lock["max_positions"]), float(lock["cost_bps"])
        )
        trade_path = RESULT_DIR / f"final_test_trades_{run_id}.csv"
        daily_path = RESULT_DIR / f"final_test_daily_{run_id}.csv"
        positions_path = RESULT_DIR / f"final_test_positions_{run_id}.csv"
        final_test.reset_index().to_csv(trade_path, index=False)
        pd.DataFrame({"Regime-aligned portfolio": model_daily["Core + tactical sleeve"],
                      "All-events portfolio": baseline_daily["Core + tactical sleeve"],
                      "SPY buy-and-hold": model_daily["SPY buy-and-hold"]}).to_csv(daily_path)
        pd.concat([model_positions.assign(strategy="Regime-aligned"),
                   baseline_positions.assign(strategy="All events")]).to_csv(positions_path, index=False)
        summary_path = RESULT_DIR / f"final_test_summary_{run_id}.json"
        save_json(summary_path, {
            "experiment_lock": selected_lock.name, "test_start": str(pd.Timestamp(lock["validation_end"]) + pd.Timedelta(days=1)),
            "data_fingerprint": loaded_data_fingerprint, "daily_file": daily_path.name,
            "trade_file": trade_path.name, "positions_file": positions_path.name,
            "regime_aligned_portfolio": portfolio_metrics(model_daily["Core + tactical sleeve"], model_positions),
            "all_events_portfolio": portfolio_metrics(baseline_daily["Core + tactical sleeve"], baseline_positions),
            "spy_buy_hold": portfolio_metrics(model_daily["SPY buy-and-hold"], pd.DataFrame()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        st.session_state["final_test_summary"] = str(summary_path)
        st.success("Final daily portfolio backtest saved. Its data, model, and settings all match the validation lock.")

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
multiplier = {"Uptrend / lower volatility": 1.0, "Mixed / uncertain": 0.5, "Downtrend / higher volatility": 0.25}
satellite_cap = (1 - core_weight) * sum(row.probability * multiplier[row.regime] for row in probability_table.itertuples())
col3.metric("Simulated tactical cap", f"{satellite_cap:.1%}")
col4.metric("Index core", f"{core_weight:.0%}")
left, right = st.columns(2)
with left:
    st.bar_chart(probability_table.set_index("regime"))
with right:
    display_stats = cluster_stats[["regime", "return_20d", "return_60d", "volatility_20d", "drawdown_60d"]]
    st.dataframe(display_stats.style.format({x: "{:.2%}" for x in display_stats.columns if x != "regime"}), use_container_width=True)

st.subheader("Training-regime history")
st.caption("The current-regime indicator above is only the most recent date. These charts show the regimes learned across all training dates.")
training_regime = regime.loc[regime["phase"].eq("Training")].copy()
training_probabilities = pd.DataFrame(index=training_regime.index)
for column in prob_cols:
    raw_cluster = int(column.split("_")[1])
    regime_name = cluster_stats.loc[raw_cluster, "regime"]
    training_probabilities[regime_name] = training_probabilities.get(regime_name, 0.0) + training_regime[column]
training_share = (training_regime["regime"].value_counts(normalize=True)
                  .reindex(training_probabilities.columns, fill_value=0.0)
                  .rename("share of training days"))
history_left, history_right = st.columns(2)
with history_left:
    st.caption("GMM regime probabilities by training date")
    st.line_chart(training_probabilities)
with history_right:
    st.caption("Most-likely regime on training dates")
    st.bar_chart(training_share)

st.subheader("3. Final test results")
saved_summaries = sorted(RESULT_DIR.glob("final_test_summary_*.json"), reverse=True)
if saved_summaries:
    default_summary = Path(st.session_state.get("final_test_summary", saved_summaries[0]))
    summary_index = saved_summaries.index(default_summary) if default_summary in saved_summaries else 0
    selected_summary = st.selectbox("Saved final-test run", saved_summaries, index=summary_index,
                                    format_func=lambda path: path.name)
    summary = json.loads(selected_summary.read_text(encoding="utf-8"))
    if summary.get("data_fingerprint") != loaded_data_fingerprint:
        st.warning("This saved result was created with different loaded data or a different benchmark; it is displayed for audit only.")
    metrics_table = pd.DataFrame([
        {"Portfolio": "Regime-aligned core + tactical", **summary["regime_aligned_portfolio"]},
        {"Portfolio": "All-events core + tactical baseline", **summary["all_events_portfolio"]},
        {"Portfolio": "SPY buy-and-hold", **summary["spy_buy_hold"]},
    ])
    st.dataframe(metrics_table.style.format({"mean": "{:.2%}", "win": "{:.1%}", "sharpe": "{:.2f}",
                                              "drawdown": "{:.2%}", "total_return": "{:.2%}"}),
                 use_container_width=True, hide_index=True)
    daily = pd.read_csv(RESULT_DIR / summary["daily_file"], index_col=0, parse_dates=True)
    st.line_chart((1 + daily).cumprod())
    positions_file = RESULT_DIR / summary["positions_file"]
    if positions_file.exists():
        with st.expander("Saved final-test positions"):
            st.dataframe(pd.read_csv(positions_file), use_container_width=True, hide_index=True)
else:
    st.info("Run a final test to create a persistent portfolio comparison against SPY.")

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
