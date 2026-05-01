"""CICIoT2023 IoT IDS — Streamlit live demo.

Loads tuned models from `models/` (defined by `models/manifest.json`) and
replays flows from `data/demo_holdout.parquet` (or an uploaded CSV) one at a
time, mimicking traffic arriving at an IoT gateway. Reports per-flow latency,
running accuracy, a live confusion matrix, and a recent-alerts log.

To add a new model:
    1. Drop the model file into `models/` (e.g. `lightgbm_tuned.joblib`).
    2. Append an entry to `models/manifest.json`:
         {"name": "LightGBM", "file": "lightgbm_tuned.joblib",
          "type": "sklearn", "needs_scaling": false}
    3. Restart Streamlit.
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"
DATA_DIR = APP_DIR / "data"
MANIFEST_PATH = MODELS_DIR / "manifest.json"

st.set_page_config(page_title="CICIoT2023 IDS Demo", layout="wide")


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODELS_DIR / "feature_scaler.joblib")
    fine_label_encoder = joblib.load(MODELS_DIR / "fine_label_encoder.joblib")
    with open(MODELS_DIR / "selected_features.json") as f:
        selected_features = json.load(f)
    with open(MODELS_DIR / "fine_index_to_category_index.json") as f:
        fine_to_cat = {int(k): int(v) for k, v in json.load(f).items()}
    with open(MODELS_DIR / "category_order.json") as f:
        category_order = json.load(f)
    return scaler, fine_label_encoder, selected_features, fine_to_cat, category_order


@st.cache_resource
def load_models():
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    loaded: dict[str, dict] = {}
    for entry in manifest.get("models", []):
        name = entry["name"]
        path = MODELS_DIR / entry["file"]
        if not path.exists():
            st.warning(f"[skip] {name}: file not found at {path}")
            continue
        if entry.get("type") == "keras":
            with tf.device("/CPU:0"):
                model = keras.models.load_model(path, compile=False)

            def make_keras_predict(m):
                def _predict(X):
                    with tf.device("/CPU:0"):
                        return m.predict(X, verbose=0, batch_size=2048)
                return _predict

            predict_fn = make_keras_predict(model)
        else:
            estimator = joblib.load(path)
            predict_fn = estimator.predict_proba
        loaded[name] = {
            "predict_proba": predict_fn,
            "needs_scaling": bool(entry.get("needs_scaling", False)),
        }
    return loaded


@st.cache_data
def load_demo_holdout() -> pd.DataFrame | None:
    parquet_path = DATA_DIR / "demo_holdout.parquet"
    csv_path = DATA_DIR / "demo_holdout.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_iomt_sample() -> pd.DataFrame | None:
    parquet_path = DATA_DIR / "ciciomt2024_sample.parquet"
    csv_path = DATA_DIR / "ciciomt2024_sample.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def predict_one_flow(features_row: np.ndarray, model_entry: dict, scaler) -> tuple[np.ndarray, float]:
    feats = features_row.reshape(1, -1)
    if model_entry["needs_scaling"]:
        feats = scaler.transform(feats)
    t0 = time.perf_counter()
    probs = model_entry["predict_proba"](feats)
    return probs[0], (time.perf_counter() - t0) * 1000.0


def aggregate_to_categories(probs_1d: np.ndarray, fine_to_cat: dict, num_categories: int) -> np.ndarray:
    cat_probs = np.zeros(num_categories)
    for fine_idx, prob in enumerate(probs_1d):
        cat_probs[fine_to_cat[fine_idx]] += prob
    total = cat_probs.sum()
    return cat_probs / total if total > 0 else cat_probs


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("CICIoT2023 IDS Demo")
st.sidebar.caption(
    "Live flow-by-flow replay using tuned models from the CS 258 hyperparameter "
    "tuning notebook."
)

try:
    scaler, fine_label_encoder, selected_features, fine_to_cat, category_order = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Preprocessing artifacts not found. Copy `feature_scaler.joblib`, "
        "`fine_label_encoder.joblib`, `selected_features.json`, "
        "`fine_index_to_category_index.json`, and `category_order.json` into "
        f"`{MODELS_DIR}/` first."
    )
    st.exception(e)
    st.stop()

models = load_models()
if not models:
    st.error(
        f"No models loaded. Drop your tuned model files into `{MODELS_DIR}/` "
        "and update `manifest.json`."
    )
    st.stop()

primary_model_name = st.sidebar.selectbox(
    "Spotlight model (live alerts)", list(models.keys()),
    index=list(models.keys()).index("Random Forest") if "Random Forest" in models else 0,
)

with st.sidebar.expander("Debug: data dir contents"):
    if DATA_DIR.exists():
        entries = sorted(DATA_DIR.iterdir())
        if not entries:
            st.text(f"{DATA_DIR} is empty")
        for p in entries:
            try:
                size = p.stat().st_size
                st.text(f"{p.name}  {size:>12,} bytes")
            except OSError:
                st.text(f"{p.name}  (stat failed)")
    else:
        st.text(f"{DATA_DIR} does not exist")

# Build the flow-source options dynamically: the CICIoMT2024 cross-dataset
# bundle only appears if scripts/prep_iomt_bundle.py has been run AND the
# file is large enough to be the actual content (not a stranded LFS pointer,
# which would only be ~130 bytes).
SOURCE_DEMO = "CICIoT2023 demo holdout (in-distribution)"
SOURCE_IOMT = "CICIoMT2024 sample (cross-dataset, out-of-distribution)"
SOURCE_UPLOAD = "Upload CSV"

iomt_parquet = DATA_DIR / "ciciomt2024_sample.parquet"
iomt_csv = DATA_DIR / "ciciomt2024_sample.csv"
iomt_size = 0
if iomt_parquet.exists():
    iomt_size = iomt_parquet.stat().st_size
elif iomt_csv.exists():
    iomt_size = iomt_csv.stat().st_size

# LFS pointer files are ~130 bytes; real parquet/CSV samples are ≥100 KB.
iomt_available = iomt_size > 50_000

flow_source_options = [SOURCE_DEMO]
if iomt_available:
    flow_source_options.append(SOURCE_IOMT)
flow_source_options.append(SOURCE_UPLOAD)

st.sidebar.caption(
    f"IoMT bundle detected: {'yes' if iomt_available else 'no'} "
    f"({iomt_size:,} bytes)"
)

source_choice = st.sidebar.radio("Flow source", flow_source_options)
flow_source_note: str | None = None

if source_choice == SOURCE_DEMO:
    flows_df = load_demo_holdout()
    if flows_df is None:
        st.sidebar.error(
            f"No demo holdout found. Put `demo_holdout.parquet` (or .csv) in `{DATA_DIR}/`."
        )
        st.stop()
elif source_choice == SOURCE_IOMT:
    flows_df = load_iomt_sample()
    if flows_df is None:
        st.sidebar.error(
            f"No CICIoMT2024 bundle found. Run `python scripts/prep_iomt_bundle.py` "
            f"to generate `{DATA_DIR.name}/ciciomt2024_sample.parquet`."
        )
        st.stop()
    flow_source_note = (
        "Models were trained on CICIoT2023. Predicted categories are compared "
        "against the CICIoMT2024 labels mapped into the CICIoT2023 8-category "
        "space (e.g. MQTT_DDoS_* → DDoS). Lower accuracy here is expected and "
        "is the headline cross-dataset generalization number for the report."
    )
else:
    upload = st.sidebar.file_uploader("Upload a CSV with the same feature columns", type=["csv"])
    if upload is None:
        st.sidebar.info("Upload a CSV to begin.")
        st.stop()
    flows_df = pd.read_csv(upload)

missing_features = [f for f in selected_features if f not in flows_df.columns]
if missing_features:
    st.sidebar.warning(f"Filling {len(missing_features)} missing feature(s) with 0.")
    for f in missing_features:
        flows_df[f] = 0.0

max_flows = st.sidebar.slider("Max flows to replay", min_value=10, max_value=1000, value=200, step=10)
delay_ms = st.sidebar.slider("Inter-arrival delay (ms)", min_value=0, max_value=500, value=40, step=10)

start_replay = st.sidebar.button("Start live replay", type="primary")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Live IoT Intrusion Detection")
st.caption(
    f"Loaded {len(models)} tuned models from `{MODELS_DIR.name}/`. "
    f"Spotlight: **{primary_model_name}**. "
    f"Flow source: **{source_choice}** ({len(flows_df):,} rows available)."
)
if flow_source_note:
    st.info(flow_source_note)

if start_replay:
    sample = flows_df.sample(n=min(max_flows, len(flows_df)), random_state=42).reset_index(drop=True)

    metrics_box = st.empty()
    cm_box = st.empty()
    alerts_box = st.empty()
    progress = st.progress(0.0, text="Replaying...")

    primary_entry = models[primary_model_name]
    cat_to_idx = {c: i for i, c in enumerate(category_order)}
    cm = np.zeros((len(category_order), len(category_order)), dtype=int)
    latencies: list[float] = []
    correct = 0
    alert_log: deque[dict] = deque(maxlen=15)

    # Warm up so the first sample doesn't dominate the latency stats.
    feats0 = sample.loc[0, selected_features].to_numpy(dtype=np.float64)
    _ = predict_one_flow(feats0, primary_entry, scaler)

    for i, row in sample.iterrows():
        feats = row[selected_features].to_numpy(dtype=np.float64)
        true_label = str(row.get("Label", "?"))
        true_category = str(row.get("attack_class", "?"))

        probs_1d, latency_ms = predict_one_flow(feats, primary_entry, scaler)
        cat_probs = aggregate_to_categories(probs_1d, fine_to_cat, len(category_order))
        pred_cat_idx = int(np.argmax(cat_probs))
        pred_category = category_order[pred_cat_idx]
        confidence = float(cat_probs[pred_cat_idx])
        pred_fine_idx = int(np.argmax(probs_1d))
        pred_label = fine_label_encoder.classes_[pred_fine_idx]

        latencies.append(latency_ms)
        is_correct = pred_category == true_category
        correct += int(is_correct)
        true_idx = cat_to_idx.get(true_category, 0)
        cm[true_idx, pred_cat_idx] += 1

        if (not is_correct) or (true_category != "Benign"):
            alert_log.appendleft({
                "flow#": int(i),
                "true": f"{true_label} ({true_category})",
                "pred": f"{pred_label} ({pred_category})",
                "conf": round(confidence, 3),
                "latency_ms": round(latency_ms, 2),
                "match": "OK" if is_correct else "MISS",
            })

        if i % 5 == 0 or i == len(sample) - 1:
            scored = i + 1
            with metrics_box.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Flows scored", f"{scored}/{len(sample)}")
                c2.metric("Accuracy", f"{correct/scored:.1%}")
                c3.metric("Mean latency (ms)", f"{np.mean(latencies):.2f}")
                c4.metric("p99 latency (ms)", f"{np.percentile(latencies, 99):.2f}")
            with cm_box.container():
                st.subheader("Live category confusion matrix")
                cm_df = pd.DataFrame(cm, index=category_order, columns=category_order)
                st.dataframe(cm_df, use_container_width=True)
            with alerts_box.container():
                st.subheader("Recent alerts")
                if alert_log:
                    st.dataframe(pd.DataFrame(list(alert_log)), use_container_width=True, hide_index=True)
                else:
                    st.info("No alerts yet.")
            progress.progress(scored / len(sample), text=f"Replaying {scored}/{len(sample)}")

        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    progress.empty()
    st.success(f"Replay complete. Final accuracy: {correct/max(len(sample),1):.2%}")

# ---------------------------------------------------------------------------
# Full-pass batch comparison across every loaded model.
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Batch comparison across all loaded models")
st.caption("Runs every loaded model on the full sample of flows in one shot and reports accuracy and per-sample latency.")

if st.button("Run batch comparison"):
    sample = flows_df.sample(n=min(max_flows, len(flows_df)), random_state=42).reset_index(drop=True)
    X = sample[selected_features].to_numpy(dtype=np.float64)
    true_categories = sample.get("attack_class", pd.Series([""] * len(sample))).astype(str).values

    rows = []
    for name, entry in models.items():
        X_input = scaler.transform(X) if entry["needs_scaling"] else X
        t0 = time.perf_counter()
        probs = entry["predict_proba"](X_input)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        preds_fine = probs.argmax(axis=1)
        pred_categories = np.array([category_order[fine_to_cat[int(p)]] for p in preds_fine])
        accuracy = float((pred_categories == true_categories).mean()) if true_categories.size else float("nan")
        rows.append({
            "model": name,
            "category_accuracy": accuracy,
            "ms_per_sample": elapsed_ms / max(len(sample), 1),
            "total_ms": elapsed_ms,
            "needs_scaling": entry["needs_scaling"],
        })
    results_df = pd.DataFrame(rows).sort_values("category_accuracy", ascending=False).reset_index(drop=True)
    st.dataframe(results_df, use_container_width=True)
