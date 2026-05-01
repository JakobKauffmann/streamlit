# CS 258 — CICIoT2023 IoT IDS Live Demo (Streamlit)

Live flow-by-flow replay of an IoT intrusion detection system, using the tuned classifiers from the project's hyperparameter tuning notebook. Companion to:

- `CICIoT2023_Research_Pipeline_Colab.ipynb` (offline benchmark)
- `CICIoT2023_Hyperparameter_Tuning_Colab.ipynb` (Optuna tuning + 4-way split)
- `CICIoT2023_IDS_Live_Demo_Colab.ipynb` (Colab equivalent of this app)

## Layout

```
streamlit/
├── streamlit_app.py            # the app
├── requirements.txt            # pinned deps for Streamlit Cloud
├── .gitignore
├── README.md                   # this file
├── models/
│   ├── manifest.json           # the model registry — edit to add models
│   ├── feature_scaler.joblib
│   ├── fine_label_encoder.joblib
│   ├── selected_features.json
│   ├── fine_index_to_category_index.json
│   ├── category_order.json
│   ├── decision_tree_tuned.joblib
│   ├── knn_tuned.joblib
│   ├── random_forest_tuned.joblib
│   ├── xgboost_tuned.joblib
│   └── compact_mlp_tuned.keras
└── data/
    └── demo_holdout.parquet    # 5% holdout from the tuning notebook
```

## Setup

### 1. Copy artifacts out of Drive

Your tuning notebook wrote everything to `outputs/tuning/` on Drive. Download the files into this folder structure:

| Drive path | Local destination |
|---|---|
| `outputs/tuning/feature_scaler.joblib` | `models/feature_scaler.joblib` |
| `outputs/tuning/fine_label_encoder.joblib` | `models/fine_label_encoder.joblib` |
| `outputs/tuning/selected_features.json` | `models/selected_features.json` |
| `outputs/tuning/fine_index_to_category_index.json` | `models/fine_index_to_category_index.json` |
| `outputs/tuning/category_order.json` | `models/category_order.json` |
| `outputs/tuning/best_models/decision_tree_tuned.joblib` | `models/decision_tree_tuned.joblib` |
| `outputs/tuning/best_models/knn_tuned.joblib` | `models/knn_tuned.joblib` |
| `outputs/tuning/best_models/random_forest_tuned.joblib` | `models/random_forest_tuned.joblib` |
| `outputs/tuning/best_models/xgboost_tuned.joblib` | `models/xgboost_tuned.joblib` |
| `outputs/tuning/best_models/compact_mlp_tuned.keras` | `models/compact_mlp_tuned.keras` |
| `outputs/tuning/demo_holdout/demo_holdout.parquet` | `data/demo_holdout.parquet` |

### 2. Run locally

```bash
cd streamlit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Browser opens at `http://localhost:8501`.

### 3. Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo (public is fine for the class demo).
2. If any single file exceeds 100 MB (likely candidates: `random_forest_tuned.joblib`, `knn_tuned.joblib`), use Git LFS:
   ```bash
   git lfs install
   git lfs track "*.joblib" "*.keras" "*.parquet"
   git add .gitattributes
   git commit -m "Track large model artifacts with LFS"
   ```
3. At [share.streamlit.io](https://share.streamlit.io), click **New app**, point at your repo and `streamlit_app.py`. The free tier handles this stack.

## Adding a new model

The app is driven by `models/manifest.json`. To add a model your partner trained (e.g. LightGBM):

1. Drop the file into `models/`, e.g. `models/lightgbm_tuned.joblib`.
2. Append an entry to `models/manifest.json`:
   ```json
   {
     "name": "LightGBM",
     "file": "lightgbm_tuned.joblib",
     "type": "sklearn",
     "needs_scaling": false
   }
   ```
3. Restart Streamlit. The new model appears in the sidebar's **Spotlight model** dropdown and in the batch comparison.

`type` is `"sklearn"` for anything with a `predict_proba` method (sklearn, XGBoost, LightGBM, CatBoost) or `"keras"` for a `.keras` file. `needs_scaling` is `true` when the model expects scaled features (the saved `StandardScaler` is applied automatically before predict).

## What the app does

- **Sidebar**: pick a "spotlight" model, choose flow source (bundled holdout or upload), set max flows + inter-arrival delay, hit **Start live replay**.
- **Live replay**: streams flows one at a time through the spotlight model, showing running accuracy, mean / p99 latency, a live category confusion matrix, and a recent-alerts table.
- **Batch comparison**: runs *every* loaded model on the same sample at once, reports per-model accuracy and ms-per-sample. This is the head-to-head latency/accuracy table for the report.
