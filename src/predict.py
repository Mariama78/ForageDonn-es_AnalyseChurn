from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import json
import pandas as pd

from src.preprocess import (
    MODELS_DIR,
    load_preprocessing_metadata,
    load_scaler,
    load_selected_features,
    prepare_model_input,
)


MODEL_FILES = {
    "Régression logistique": "model_logistic_regression.pkl",
    "Random forest": "model_random_forest.pkl",
    "XGBoost": "model_xgboost.pkl",
}
METRICS_PATH = MODELS_DIR / "model_metrics.json"


def _format_model_load_error(exc: Exception) -> str:
    message = str(exc).strip().replace("\n", " ")
    return f"{type(exc).__name__}: {message}"


def _patch_loaded_model(model: Any) -> Any:
    # Some teammates may load the serialized LogisticRegression model with a
    # slightly different scikit-learn version. Older releases still expect the
    # `multi_class` attribute during prediction.
    if model.__class__.__name__ == "LogisticRegression" and not hasattr(
        model, "multi_class"
    ):
        setattr(model, "multi_class", "auto")

    return model


def load_models_with_errors(
    models_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    models_path = models_dir or MODELS_DIR
    models: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for model_name, filename in MODEL_FILES.items():
        try:
            loaded_model = joblib.load(models_path / filename)
            models[model_name] = _patch_loaded_model(loaded_model)
        except Exception as exc:
            errors[model_name] = _format_model_load_error(exc)

    return models, errors


def load_models(models_dir: Path | None = None) -> dict[str, Any]:
    models, _ = load_models_with_errors(models_dir=models_dir)
    return models


def load_model_metrics(path: Path | None = None) -> dict[str, Any]:
    metrics_path = path or METRICS_PATH
    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def predict_all_models(raw_df: pd.DataFrame) -> pd.DataFrame:
    metadata = load_preprocessing_metadata()
    scaler = load_scaler()
    selected_features = load_selected_features()
    model_input = prepare_model_input(
        raw_df=raw_df,
        metadata=metadata,
        scaler=scaler,
        selected_features=selected_features,
    )

    predictions: list[dict[str, Any]] = []
    models, load_errors = load_models_with_errors()
    if not models:
        raise RuntimeError(
            "Aucun modele de prediction n'a pu etre charge. "
            + "Erreurs : "
            + " | ".join(f"{name}: {error}" for name, error in load_errors.items())
        )

    for model_name, model in models.items():
        predicted_classes = model.predict(model_input)
        churn_probabilities = (
            model.predict_proba(model_input)[:, 1]
            if hasattr(model, "predict_proba")
            else [None] * len(model_input)
        )

        for row_index, (predicted_class, churn_probability) in enumerate(
            zip(predicted_classes, churn_probabilities)
        ):
            predictions.append(
                {
                    "ligne_source": int(model_input.index[row_index]),
                    "modele": model_name,
                    "prediction": int(predicted_class),
                    "label": "Churn" if int(predicted_class) == 1 else "No Churn",
                    "probabilite_churn": (
                        float(churn_probability)
                        if churn_probability is not None
                        else None
                    ),
                }
            )

    return pd.DataFrame(predictions)
