from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_PATH = MODELS_DIR / "preprocessing_metadata.json"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.json"


def load_preprocessing_metadata(path: Path | None = None) -> dict[str, Any]:
    metadata_path = path or METADATA_PATH
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_scaler(path: Path | None = None) -> Any:
    return joblib.load(path or SCALER_PATH)


def load_selected_features(path: Path | None = None) -> list[str]:
    selected_features_path = path or SELECTED_FEATURES_PATH
    with selected_features_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_raw_input(raw_df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    missing_columns = [
        column
        for column in metadata["raw_feature_columns"]
        if column not in raw_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans les donnees d'entree : "
            + ", ".join(missing_columns)
        )

    normalized_df = raw_df[metadata["raw_feature_columns"]].copy()

    for column in metadata["numeric_columns"]:
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")
        normalized_df[column] = normalized_df[column].fillna(
            metadata["numeric_fill_values"][column]
        )

    for column in metadata["categorical_columns"]:
        normalized_df[column] = normalized_df[column].fillna(
            metadata["reference_categories"][column]
        )
        normalized_df[column] = normalized_df[column].astype(str)

        unexpected_values = sorted(
            set(normalized_df[column].unique())
            - set(metadata["categorical_values"][column])
        )
        if unexpected_values:
            raise ValueError(
                f"Valeurs inconnues pour '{column}' : {', '.join(unexpected_values)}"
            )

    return normalized_df


def _encode_features(
    normalized_df: pd.DataFrame, metadata: dict[str, Any]
) -> pd.DataFrame:
    encoded_df = pd.get_dummies(
        normalized_df,
        columns=metadata["categorical_columns"],
        drop_first=True,
    )

    for column in metadata["encoded_feature_columns"]:
        if column not in encoded_df.columns:
            encoded_df[column] = 0

    encoded_df = encoded_df[metadata["encoded_feature_columns"]].copy()
    return encoded_df


def _scale_features(
    encoded_df: pd.DataFrame, metadata: dict[str, Any], scaler: Any
) -> pd.DataFrame:
    scaled_df = encoded_df.copy()
    columns_to_scale = metadata["scale_columns"]
    scaled_df[columns_to_scale] = scaler.transform(scaled_df[columns_to_scale])
    return scaled_df


def prepare_model_input(
    raw_df: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    scaler: Any | None = None,
    selected_features: list[str] | None = None,
) -> pd.DataFrame:
    metadata = metadata or load_preprocessing_metadata()
    scaler = scaler or load_scaler()
    selected_features = selected_features or load_selected_features()

    normalized_df = _normalize_raw_input(raw_df, metadata)
    encoded_df = _encode_features(normalized_df, metadata)
    scaled_df = _scale_features(encoded_df, metadata, scaler)

    missing_selected_features = [
        column for column in selected_features if column not in scaled_df.columns
    ]
    if missing_selected_features:
        raise ValueError(
            "Variables selectionnees introuvables apres preprocessing : "
            + ", ".join(missing_selected_features)
        )

    return scaled_df[selected_features].copy()
