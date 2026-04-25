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
    raise FileNotFoundError(
        f"Cannot find model artifacts under {base_dir}. Expected outputs_*/models/best_model.joblib and best_model_metadata.json."
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

