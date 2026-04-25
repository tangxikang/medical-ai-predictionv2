from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelArtifacts:
    model_path: Path
    metadata_path: Path
    selected_features: list[str]
    best_threshold: float | None
    outcome_col: str | None


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    default_value: Any
    min_value: float | None
    max_value: float | None
    choices: list[Any] | None


def _read_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid metadata JSON object: {path}")
    return data


def resolve_latest_model_artifacts(*, base_dir: Path) -> ModelArtifacts:
    candidates = sorted(base_dir.glob("outputs_*"), key=lambda p: p.name, reverse=True)
    for root in candidates:
        model_path = root / "models" / "best_model.joblib"
        metadata_path = root / "models" / "best_model_metadata.json"
        if not model_path.exists() or not metadata_path.exists():
            continue
        md = _read_metadata(metadata_path)
        selected = [str(x) for x in md.get("selected_features", [])]
        if not selected:
            raise ValueError(f"selected_features is empty in {metadata_path}")
        threshold_raw = md.get("best_threshold")
        threshold = float(threshold_raw) if threshold_raw is not None else None
        outcome = md.get("outcome_col")
        return ModelArtifacts(
            model_path=model_path,
            metadata_path=metadata_path,
            selected_features=selected,
            best_threshold=threshold,
            outcome_col=(str(outcome) if outcome is not None else None),
        )

    # Fallback layout: artifacts are directly under base_dir.
    direct_model_path = base_dir / "best_model.joblib"
    direct_metadata_path = base_dir / "best_model_metadata.json"
    if direct_model_path.exists() and direct_metadata_path.exists():
        md = _read_metadata(direct_metadata_path)
        selected = [str(x) for x in md.get("selected_features", [])]
        if not selected:
            raise ValueError(f"selected_features is empty in {direct_metadata_path}")
        threshold_raw = md.get("best_threshold")
        threshold = float(threshold_raw) if threshold_raw is not None else None
        outcome = md.get("outcome_col")
        return ModelArtifacts(
            model_path=direct_model_path,
            metadata_path=direct_metadata_path,
            selected_features=selected,
            best_threshold=threshold,
            outcome_col=(str(outcome) if outcome is not None else None),
        )

    raise FileNotFoundError(
        "Cannot find model artifacts under "
        f"{base_dir}. Expected either outputs_*/models/best_model.joblib + best_model_metadata.json, "
        "or base_dir/best_model.joblib + base_dir/best_model_metadata.json."
    )


def _is_categorical_series(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce").dropna().unique()
        if len(vals) == 0:
            return False
        return set(np.unique(vals)).issubset({0, 1})
    return True


def infer_feature_specs(*, df: pd.DataFrame, feature_cols: list[str]) -> list[FeatureSpec]:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataframe: {missing}")

    specs: list[FeatureSpec] = []
    for col in feature_cols:
        s = df[col]
        if _is_categorical_series(s):
            raw_choices = s.dropna().unique().tolist()
            choices = sorted(raw_choices)
            default = choices[0] if choices else ""
            specs.append(
                FeatureSpec(
                    name=col,
                    kind="categorical",
                    default_value=default,
                    min_value=None,
                    max_value=None,
                    choices=choices,
                )
            )
        else:
            num = pd.to_numeric(s, errors="coerce").dropna()
            if len(num) == 0:
                default = 0.0
                min_v = 0.0
                max_v = 1.0
            else:
                default = float(num.median())
                min_v = float(num.min())
                max_v = float(num.max())
                if np.isclose(min_v, max_v):
                    min_v -= 1.0
                    max_v += 1.0
            specs.append(
                FeatureSpec(
                    name=col,
                    kind="numeric",
                    default_value=default,
                    min_value=min_v,
                    max_value=max_v,
                    choices=None,
                )
            )
    return specs


def infer_feature_specs_from_pipeline(*, pipeline: Any, feature_cols: list[str]) -> list[FeatureSpec]:
    preprocess = getattr(pipeline, "named_steps", {}).get("preprocess")
    if preprocess is None:
        raise ValueError("Pipeline is missing named step 'preprocess'")

    transformers = getattr(preprocess, "transformers", None)
    if transformers is None:
        raise ValueError("Preprocess step does not expose 'transformers'")

    numeric_defaults: dict[str, float] = {}
    categorical_choices: dict[str, list[Any]] = {}

    named_transformers = getattr(preprocess, "named_transformers_", {})
    for name, transformer, cols in transformers:
        if name == "remainder" or isinstance(cols, slice):
            continue

        fitted_transformer = named_transformers.get(name, transformer)
        col_names = [str(c) for c in cols]
        lower_name = str(name).lower()
        named_steps = getattr(fitted_transformer, "named_steps", {})
        onehot = named_steps.get("onehot")
        imputer = named_steps.get("impute")

        if onehot is not None and hasattr(onehot, "categories_"):
            categories = getattr(onehot, "categories_", [])
            for col, cat_values in zip(col_names, categories):
                choices = list(cat_values)
                categorical_choices[col] = choices
            continue

        if lower_name.startswith("cat"):
            for col in col_names:
                categorical_choices.setdefault(col, [])
            continue

        if imputer is not None and hasattr(imputer, "statistics_"):
            stats = list(getattr(imputer, "statistics_"))
            for col, stat in zip(col_names, stats):
                if pd.isna(stat):
                    numeric_defaults[col] = 0.0
                else:
                    numeric_defaults[col] = float(stat)
        else:
            for col in col_names:
                numeric_defaults.setdefault(col, 0.0)

    specs: list[FeatureSpec] = []
    for col in feature_cols:
        if col in categorical_choices:
            choices = categorical_choices[col]
            default = choices[0] if choices else ""
            specs.append(
                FeatureSpec(
                    name=col,
                    kind="categorical",
                    default_value=default,
                    min_value=None,
                    max_value=None,
                    choices=choices,
                )
            )
            continue

        default_num = float(numeric_defaults.get(col, 0.0))
        specs.append(
            FeatureSpec(
                name=col,
                kind="numeric",
                default_value=default_num,
                min_value=None,
                max_value=None,
                choices=None,
            )
        )
    return specs


def resolve_data_file_path(*, base_dir: Path, app_dir: Path, file_name: str = "data.xlsx") -> Path:
    raw = Path(file_name)
    if raw.is_absolute():
        absolute = raw.resolve()
        if absolute.exists():
            return absolute
        raise FileNotFoundError(f"Data file not found: {absolute}")

    candidates: list[Path] = []
    for path in (
        base_dir / file_name,
        app_dir / file_name,
    ):
        resolved = path.resolve()
        if resolved not in candidates:
            candidates.append(resolved)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Data file not found: {file_name}. Searched: {searched}")
