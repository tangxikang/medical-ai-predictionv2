from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from ml_project.web_support import (
        FeatureSpec,
        infer_feature_specs,
        infer_feature_specs_from_pipeline,
        resolve_data_file_path,
        resolve_latest_model_artifacts,
    )
except ModuleNotFoundError:
    from web_support import (
        FeatureSpec,
        infer_feature_specs,
        infer_feature_specs_from_pipeline,
        resolve_data_file_path,
        resolve_latest_model_artifacts,
    )

try:
    from ml_project.model_compat import load_joblib_model
except ModuleNotFoundError:
    from model_compat import load_joblib_model


st.set_page_config(page_title="Medical AI Prediction", layout="wide", page_icon="🩺")

st.markdown(
    """
    <style>
      .stButton > button {
        background: #c62828;
        color: white;
        border-radius: 10px;
        border: none;
        font-size: 18px;
        font-weight: 600;
        padding: 0.45rem 1.2rem;
      }
      .stNumberInput label, .stSelectbox label {
        font-size: 16px;
        font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _collapse_onehot_feature_names(transformed_feature_names: list[str], original_feature_names: list[str]) -> list[str]:
    original = list(original_feature_names)
    original_set = set(original)
    by_len = sorted(original, key=len, reverse=True)
    collapsed: list[str] = []
    for name in transformed_feature_names:
        if name in original_set:
            collapsed.append(name)
            continue
        match = None
        for base in by_len:
            if name.startswith(f"{base}_"):
                match = base
                break
        collapsed.append(match if match is not None else name)
    return collapsed


def _extract_binary_shap_values(exp: shap.Explanation) -> shap.Explanation:
    values = getattr(exp, "values", None)
    if isinstance(values, np.ndarray) and values.ndim == 3 and values.shape[-1] >= 2:
        return exp[..., 1]
    return exp


def _aggregate_shap_by_feature(
    shap_values: np.ndarray,
    transformed_feature_names: list[str],
    original_feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    collapsed_names = _collapse_onehot_feature_names(transformed_feature_names, original_feature_names)
    ordered_names: list[str] = [c for c in original_feature_names if c in set(collapsed_names)]
    for n in collapsed_names:
        if n not in ordered_names:
            ordered_names.append(n)

    idx_map: dict[str, list[int]] = {n: [] for n in ordered_names}
    for i, n in enumerate(collapsed_names):
        idx_map[n].append(i)

    agg_values = np.zeros((shap_values.shape[0], len(ordered_names)), dtype=float)
    for j, n in enumerate(ordered_names):
        cols = idx_map[n]
        agg_values[:, j] = np.sum(shap_values[:, cols], axis=1)
    return agg_values, ordered_names


@st.cache_resource(show_spinner=False)
def _load_model(model_path: str):
    return load_joblib_model(model_path)


@st.cache_data(show_spinner=False)
def _load_data(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_excel(path)


@st.cache_data(show_spinner=False)
def _get_feature_specs(data_path: str, features: tuple[str, ...]) -> list[FeatureSpec]:
    df = _load_data(data_path)
    return infer_feature_specs(df=df, feature_cols=list(features))


def _render_inputs(specs: list[FeatureSpec]) -> pd.DataFrame:
    cols = st.columns(3)
    row: dict[str, Any] = {}
    for i, spec in enumerate(specs):
        col = cols[i % 3]
        with col:
            if spec.kind == "categorical":
                choices = spec.choices or [spec.default_value]
                default_idx = choices.index(spec.default_value) if spec.default_value in choices else 0
                row[spec.name] = st.selectbox(spec.name, choices, index=default_idx, key=f"input_{spec.name}")
            else:
                min_v = spec.min_value if spec.min_value is not None else None
                max_v = spec.max_value if spec.max_value is not None else None
                default_v = float(spec.default_value if spec.default_value is not None else 0.0)
                if min_v is None or max_v is None:
                    row[spec.name] = st.number_input(
                        spec.name,
                        value=default_v,
                        step=0.1,
                        key=f"input_{spec.name}",
                    )
                else:
                    min_value = float(min_v)
                    max_value = float(max_v)
                    step = max((max_value - min_value) / 200.0, 0.01)
                    row[spec.name] = st.number_input(
                        spec.name,
                        min_value=min_value,
                        max_value=max_value,
                        value=default_v,
                        step=step,
                        key=f"input_{spec.name}",
                    )
    return pd.DataFrame([row])


def _build_force_plot_html(
    *,
    pipeline,
    input_df: pd.DataFrame,
    background_df: pd.DataFrame,
    selected_features: list[str],
) -> str:
    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    bg = pre.transform(background_df[selected_features])
    row_trans = pre.transform(input_df[selected_features])
    feature_names = list(pre.get_feature_names_out())

    explainer = shap.Explainer(model, bg, feature_names=feature_names)
    exp = _extract_binary_shap_values(explainer(row_trans))
    values = np.asarray(exp.values, dtype=float)
    agg_values, agg_names = _aggregate_shap_by_feature(values, feature_names, selected_features)

    base_values = np.asarray(exp.base_values)
    base_value = float(base_values[0]) if base_values.ndim > 0 else float(base_values)
    display_values = [input_df.iloc[0][name] if name in input_df.columns else "" for name in agg_names]
    force_html = shap.plots.force(
        base_value=base_value,
        shap_values=agg_values[0],
        features=display_values,
        feature_names=agg_names,
        matplotlib=False,
    ).html()
    return f"<head>{shap.getjs()}</head><body>{force_html}</body>"


st.title("Medical AI Prediction")

base_dir = Path(os.getenv("WEB_BASE_DIR", str(ROOT))).resolve()
data_file_name = os.getenv("WEB_DATA_FILE", "data.xlsx")
artifacts = resolve_latest_model_artifacts(base_dir=base_dir)

try:
    pipeline = _load_model(str(artifacts.model_path))
    data_df: pd.DataFrame | None = None
    try:
        data_path = resolve_data_file_path(base_dir=base_dir, app_dir=ROOT, file_name=data_file_name)
        data_df = _load_data(str(data_path))
        specs = _get_feature_specs(str(data_path), tuple(artifacts.selected_features))
    except FileNotFoundError:
        specs = infer_feature_specs_from_pipeline(pipeline=pipeline, feature_cols=artifacts.selected_features)
        st.info("data.xlsx not found. Loaded input specs from model metadata. SHAP force plot is disabled.")
except Exception as exc:
    st.error(f"Load failed: {exc}")
    st.stop()

st.subheader("Input Features")
input_df = _render_inputs(specs)
if st.button("Start Prediction", type="primary", use_container_width=True):
    proba = float(pipeline.predict_proba(input_df[artifacts.selected_features])[0, 1])
    threshold = artifacts.best_threshold if artifacts.best_threshold is not None else 0.5
    pred = int(proba >= float(threshold))
    st.metric("Predicted Positive Probability", f"{proba:.2%}")
    st.metric("Predicted Class", f"{pred} (threshold={threshold:.3f})")

    st.markdown("---")
    st.subheader("SHAP Force Plot")
    if data_df is None:
        st.info("No data.xlsx provided. SHAP force plot skipped.")
    else:
        try:
            bg = data_df[artifacts.selected_features].sample(n=min(120, len(data_df)), random_state=42)
            html = _build_force_plot_html(
                pipeline=pipeline,
                input_df=input_df,
                background_df=bg,
                selected_features=artifacts.selected_features,
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                tmp.write(html.encode("utf-8"))
                tmp_path = Path(tmp.name)
            components.html(tmp_path.read_text(encoding="utf-8"), height=380, scrolling=True)
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            st.warning(f"SHAP plot generation failed: {exc}")
