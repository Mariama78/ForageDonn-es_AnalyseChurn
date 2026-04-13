from __future__ import annotations

import hashlib
from io import StringIO
from time import perf_counter, sleep, time

import pandas as pd
import streamlit as st

from src.predict import load_model_metrics, load_models_with_errors, predict_all_models
from src.preprocess import load_preprocessing_metadata


st.set_page_config(
    page_title="Analyse du Churn Telecom",
    page_icon="📉",
    layout="wide",
)


MIN_LOADING_SECONDS = 1.2
NOTICE_DURATION_SECONDS = 6.0


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @keyframes fade-up {
            from {
                opacity: 0;
                transform: translateY(18px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes pulse-ring {
            0% {
                transform: scale(0.88);
                opacity: 0.85;
            }
            70% {
                transform: scale(1.15);
                opacity: 0;
            }
            100% {
                transform: scale(1.15);
                opacity: 0;
            }
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        @keyframes popup-in {
            from {
                opacity: 0;
                transform: translateY(-12px) scale(0.96);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        @keyframes notice-life {
            0% {
                opacity: 0;
                transform: translateY(-8px) scale(0.98);
            }
            8%,
            82% {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
            100% {
                opacity: 0;
                transform: translateY(-10px) scale(0.98);
            }
        }
        :root {
            --card-bg: rgba(255, 255, 255, 0.04);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-1: #33a1fd;
            --accent-2: #8a6cff;
            --ok: #6ee7b7;
            --warn: #f9c74f;
            --text-main: #e9edf5;
            --text-soft: #dfe6f3;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(120% 120% at 0% 0%, #102347 0, #0b0f1f 40%, #060910 100%);
            color: var(--text-main);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }
        .hero {
            background: linear-gradient(135deg, rgba(51,161,253,0.18), rgba(138,108,255,0.14));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 24px 70px rgba(0,0,0,0.35);
            margin-bottom: 1.25rem;
            animation: fade-up 0.65s ease-out both;
        }
        .hero .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.9rem;
            color: #89c4ff;
            font-weight: 700;
        }
        .hero h1 {
            margin: 0.2rem 0 0.3rem 0;
            font-size: 2.2rem;
            color: var(--text-main);
        }
        .hero p {
            margin: 0;
            color: var(--text-soft);
            line-height: 1.5;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255,255,255,0.05);
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 0.95rem;
            color: var(--text-main);
            transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
            animation: fade-up 0.7s ease-out both;
        }
        .pill:hover {
            transform: translateY(-2px);
            border-color: rgba(138,108,255,0.28);
            background: rgba(255,255,255,0.08);
        }
        .status-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1rem;
            min-height: 122px;
            box-shadow: 0 12px 35px rgba(0,0,0,0.28);
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            animation: fade-up 0.72s ease-out both;
        }
        .status-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: linear-gradient(180deg, var(--accent-1), var(--accent-2));
            opacity: 0.9;
        }
        .status-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 18px 45px rgba(0,0,0,0.36);
            border-color: rgba(138,108,255,0.22);
        }
        .status-card h4 {
            margin: 0 0 0.45rem 0;
            color: var(--text-main);
            font-size: 1.05rem;
            padding-left: 0.4rem;
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }
        .status-chip-ok {
            color: #062414;
            background: rgba(110, 231, 183, 0.95);
        }
        .status-chip-warn {
            color: #352100;
            background: rgba(249, 199, 79, 0.95);
        }
        .status-card p {
            margin: 0;
            color: var(--text-soft);
            line-height: 1.45;
            padding-left: 0.4rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1rem 1rem;
            box-shadow: 0 12px 35px rgba(0,0,0,0.28);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            animation: fade-up 0.78s ease-out both;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 18px 45px rgba(0,0,0,0.34);
            border-color: rgba(138,108,255,0.22);
        }
        .stTabs [role="tablist"] {
            gap: 0.4rem;
        }
        .stTabs [role="tab"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid transparent;
            padding: 0.65rem 1rem;
            border-radius: 12px;
            color: var(--text-main) !important;
            transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
        }
        .stTabs [role="tab"] * {
            color: var(--text-main) !important;
        }
        .stTabs [role="tab"]:hover {
            transform: translateY(-1px);
            border-color: rgba(138,108,255,0.18);
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(120deg, var(--accent-1), var(--accent-2));
            color: #0b0f1a !important;
            font-weight: 700;
        }
        .stTabs [role="tab"][aria-selected="true"] * {
            color: #0b0f1a !important;
        }
        .stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(120deg, var(--accent-1), var(--accent-2));
            color: #0b0f1a;
            border: none;
            border-radius: 12px;
            font-weight: 700;
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
            box-shadow: 0 10px 24px rgba(51, 161, 253, 0.22);
        }
        .stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 36px rgba(51, 161, 253, 0.28);
            filter: brightness(1.02);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            overflow: hidden;
            animation: fade-up 0.84s ease-out both;
        }
        div[data-testid="stExpander"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            box-shadow: 0 10px 24px rgba(0,0,0,0.16);
            margin-top: 1.25rem;
            animation: fade-up 0.9s ease-out both;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary * {
            color: var(--text-main) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,0.035);
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 16px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.2);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
            animation: fade-up 0.82s ease-out both;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 18px 38px rgba(0,0,0,0.3);
            border-color: rgba(138,108,255,0.22) !important;
        }
        label, label p, .stMarkdown, .stMarkdown p, .stCaptionContainer, .stCaptionContainer p {
            color: var(--text-main);
        }
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.10);
            color: var(--text-main) !important;
        }
        .floating-popup {
            position: fixed;
            top: 5rem;
            right: 1.5rem;
            z-index: 999999;
            width: min(360px, calc(100vw - 2rem));
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 0.95rem 1rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(10, 15, 28, 0.92);
            backdrop-filter: blur(14px);
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.34);
            animation: popup-in 0.26s ease-out both;
        }
        .floating-popup-loading {
            border-color: rgba(51, 161, 253, 0.28);
        }
        .floating-popup-success {
            border-color: rgba(110, 231, 183, 0.3);
            background: rgba(9, 27, 22, 0.94);
        }
        .floating-popup-icon {
            width: 18px;
            height: 18px;
            border-radius: 999px;
            flex: 0 0 auto;
            position: relative;
        }
        .floating-popup-loading .floating-popup-icon {
            border: 2px solid rgba(233, 237, 245, 0.22);
            border-top-color: var(--accent-1);
            animation: spin 0.85s linear infinite;
        }
        .floating-popup-success .floating-popup-icon {
            background: rgba(110, 231, 183, 0.16);
            border: 1px solid rgba(110, 231, 183, 0.36);
        }
        .floating-popup-success .floating-popup-icon::before {
            content: "✓";
            position: absolute;
            inset: 0;
            display: grid;
            place-items: center;
            color: var(--ok);
            font-size: 0.9rem;
            font-weight: 800;
        }
        .floating-popup-text {
            color: var(--text-main);
            font-size: 0.95rem;
            line-height: 1.4;
        }
        @media (prefers-reduced-motion: reduce) {
            .hero,
            .pill,
            .status-card,
            div[data-testid="stMetric"],
            div[data-testid="stDataFrame"],
            div[data-testid="stExpander"],
            div[data-testid="stVerticalBlockBorderWrapper"],
            .floating-popup {
                animation: none !important;
                transition: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_preprocessing_metadata() -> dict:
    return load_preprocessing_metadata()


@st.cache_data
def get_model_metrics() -> dict:
    return load_model_metrics()


@st.cache_resource
def get_models_status() -> tuple[dict, dict]:
    return load_models_with_errors()


def build_default_input(metadata: dict) -> dict:
    default_input = {}
    for column in metadata["raw_feature_columns"]:
        if column in metadata["categorical_columns"]:
            default_input[column] = metadata["reference_categories"][column]
        else:
            default_input[column] = metadata["numeric_fill_values"][column]
    return default_input


def build_manual_input_form(metadata: dict) -> pd.DataFrame:
    defaults = build_default_input(metadata)
    form_values: dict[str, object] = {}

    st.subheader("Saisie manuelle d'un client")
    st.caption("Les champs sont alignés sur le preprocessing utilisé dans le notebook.")

    with st.form("manual_prediction_form"):
        col1, col2, col3 = st.columns(3)
        columns_cycle = [col1, col2, col3]

        for idx, column in enumerate(metadata["raw_feature_columns"]):
            container = columns_cycle[idx % len(columns_cycle)]
            with container:
                if column in metadata["categorical_columns"]:
                    options = metadata["categorical_values"][column]
                    default_index = options.index(defaults[column])
                    form_values[column] = st.selectbox(
                        column,
                        options=options,
                        index=default_index,
                        key=f"manual_{column}",
                    )
                elif column == "SeniorCitizen":
                    form_values[column] = int(
                        st.selectbox(
                            column,
                            options=[0, 1],
                            index=int(defaults[column]),
                            key=f"manual_{column}",
                        )
                    )
                elif column == "tenure":
                    form_values[column] = int(
                        st.number_input(
                            column,
                            min_value=0,
                            max_value=120,
                            value=int(defaults[column]),
                            step=1,
                            key=f"manual_{column}",
                        )
                    )
                else:
                    form_values[column] = float(
                        st.number_input(
                            column,
                            min_value=0.0,
                            value=float(defaults[column]),
                            step=1.0,
                            key=f"manual_{column}",
                        )
                    )

        submitted = st.form_submit_button("Lancer la prédiction", use_container_width=True)

    if not submitted:
        return pd.DataFrame()

    return pd.DataFrame([form_values])


def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.getvalue().decode("utf-8")
    return pd.read_csv(StringIO(content))


def build_csv_template(metadata: dict) -> pd.DataFrame:
    return pd.DataFrame([build_default_input(metadata)])


def validate_uploaded_dataframe(
    uploaded_df: pd.DataFrame, metadata: dict
) -> tuple[pd.DataFrame, list[str]]:
    if uploaded_df.empty:
        raise ValueError("Le fichier CSV est vide.")

    expected_columns = metadata["raw_feature_columns"]
    missing_columns = [column for column in expected_columns if column not in uploaded_df.columns]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le fichier CSV : " + ", ".join(missing_columns)
        )

    extra_columns = [column for column in uploaded_df.columns if column not in expected_columns]
    warnings: list[str] = []
    if extra_columns:
        warnings.append(
            "Colonnes supplémentaires ignorées : " + ", ".join(extra_columns)
        )

    validated_df = uploaded_df[expected_columns].copy()
    return validated_df, warnings


def process_batch_dataframe(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_df = predict_all_models(batch_df)
    return batch_df, predictions_df


def clear_batch_results_state() -> None:
    st.session_state.pop("batch_results_cache", None)
    st.session_state.pop("batch_results_signature", None)
    st.session_state.pop("batch_prediction_filter", None)


def build_uploaded_file_signature(uploaded_file) -> str:
    file_content = uploaded_file.getvalue()
    digest = hashlib.md5(file_content).hexdigest()
    return f"{uploaded_file.name}:{digest}"


def show_loading_popup(message: str):
    popup_placeholder = st.empty()
    popup_placeholder.markdown(
        f"""
        <div class="floating-popup floating-popup-loading">
            <div class="floating-popup-icon"></div>
            <div class="floating-popup-text">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return popup_placeholder


def ensure_min_loading_time(start_time: float, minimum_seconds: float = MIN_LOADING_SECONDS) -> None:
    elapsed = perf_counter() - start_time
    if elapsed < minimum_seconds:
        sleep(minimum_seconds - elapsed)


def set_completion_notice(message: str, duration: float = NOTICE_DURATION_SECONDS) -> None:
    st.session_state["completion_notice"] = {
        "message": message,
        "expires_at": time() + duration,
    }


def render_completion_notice(container) -> None:
    notice = st.session_state.get("completion_notice")
    if not notice:
        container.empty()
        return

    remaining_seconds = notice["expires_at"] - time()
    if remaining_seconds <= 0:
        st.session_state.pop("completion_notice", None)
        container.empty()
        return

    animation_duration = max(remaining_seconds, 0.5)
    container.markdown(
        f"""
        <div class="floating-popup floating-popup-success" style="animation: popup-in 0.22s ease-out both, notice-life {animation_duration:.2f}s linear both;">
            <div class="floating-popup-icon"></div>
            <div class="floating-popup-text">{notice["message"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_with_loading_feedback(action, *, loading_message: str, success_message: str, notice_container):
    popup_placeholder = show_loading_popup(loading_message)
    start_time = perf_counter()
    try:
        with st.spinner(loading_message):
            result = action()
        ensure_min_loading_time(start_time)
    except Exception:
        popup_placeholder.empty()
        st.session_state.pop("completion_notice", None)
        notice_container.empty()
        raise

    popup_placeholder.empty()
    set_completion_notice(success_message)
    render_completion_notice(notice_container)
    return result


def render_hero(metrics: dict, models: dict, load_errors: dict) -> None:
    pills = [
        f'<div class="pill">Modèle principal retenu : {metrics["modele_recommande"]}</div>',
        f'<div class="pill">Modèles chargés : {len(models)}/3</div>',
        '<div class="pill">Entrées : manuel + CSV</div>',
    ]
    if load_errors:
        pills.append(f'<div class="pill">Modèles indisponibles : {len(load_errors)}</div>')

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Analyse supervisée du churn</div>
            <h1>Plateforme de prédiction client</h1>
            <p>
                Cette interface recharge les artefacts validés du notebook pour produire des
                prédictions de churn sur un client unique ou sur un fichier CSV, sans sortir
                du cadre du projet académique.
            </p>
            <div class="pill-row">
                {"".join(pills)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_status(models: dict, load_errors: dict) -> None:
    st.subheader("État des modèles")
    status_columns = st.columns(3)

    all_model_names = ["Régression logistique", "Random forest", "XGBoost"]
    for idx, model_name in enumerate(all_model_names):
        with status_columns[idx]:
            available = model_name in models
            chip_class = "status-chip-ok" if available else "status-chip-warn"
            status_label = "Disponible" if available else "Indisponible"
            detail = (
                "Le modèle est prêt pour l'inférence."
                if available
                else "Le chargement a échoué sur cette machine."
            )
            st.markdown(
                f"""
                <div class="status-card">
                    <span class="status-chip {chip_class}">{status_label}</span>
                    <h4>{model_name}</h4>
                    <p>{detail}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if load_errors:
        with st.expander("Détails des modèles indisponibles"):
            for model_name, error in load_errors.items():
                st.write(f"**{model_name}**")
                st.code(error)


def render_metrics(metrics: dict) -> None:
    st.subheader("Métriques de test")
    st.caption(
        f"Modèle principal retenu : **{metrics['modele_recommande']}**. "
        "Critère prioritaire : capacité à bien détecter les clients qui risquent réellement de partir."
    )

    metrics_df = pd.DataFrame(metrics["metriques_test"]).T.reset_index()
    metrics_df = metrics_df.rename(
        columns={
            "index": "modèle",
            "accuracy": "accuracy (exactitude)",
            "precision_churn": "precision_churn (précision churn)",
            "recall_churn": "recall_churn (rappel churn)",
            "f1_churn": "f1_churn (score F1 churn)",
            "auc_roc": "auc_roc (aire sous la courbe ROC)",
        }
    )
    metrics_df = metrics_df[
        [
            "modèle",
            "accuracy (exactitude)",
            "precision_churn (précision churn)",
            "recall_churn (rappel churn)",
            "f1_churn (score F1 churn)",
            "auc_roc (aire sous la courbe ROC)",
        ]
    ]
    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
    )


def format_probability(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def sort_predictions_by_priority(predictions_df: pd.DataFrame, recommended_model: str) -> pd.DataFrame:
    display_df = predictions_df.copy()
    display_df["priority"] = display_df["modele"].apply(
        lambda model_name: 0 if model_name == recommended_model else 1
    )
    display_df = display_df.sort_values(["priority", "modele"]).drop(columns=["priority"])
    return display_df


def render_recommended_decision(predictions_df: pd.DataFrame, metrics: dict) -> None:
    recommended_model = metrics["modele_recommande"]
    recommended_row = predictions_df[predictions_df["modele"] == recommended_model].iloc[0]
    recommended_metrics = metrics["metriques_test"][recommended_model]

    st.subheader("Décision recommandée")
    if recommended_row["label"] == "Churn":
        st.warning(
            f"Le modèle recommandé ({recommended_model}) signale un **risque de churn** "
            f"avec une probabilité estimée de **{format_probability(recommended_row['probabilite_churn'])}**."
        )
    else:
        st.success(
            f"Le modèle recommandé ({recommended_model}) signale un profil **No Churn** "
            f"avec une probabilité de churn estimée à **{format_probability(recommended_row['probabilite_churn'])}**."
        )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Modèle", recommended_model)
    metric_col2.metric("Décision", recommended_row["label"])
    metric_col3.metric(
        "Probabilité churn",
        format_probability(recommended_row["probabilite_churn"]),
    )
    metric_col4.metric("Recall churn", f"{recommended_metrics['recall_churn']:.2f}")


def render_single_predictions(predictions_df: pd.DataFrame, metrics: dict) -> None:
    recommended_model = metrics["modele_recommande"]
    display_df = sort_predictions_by_priority(predictions_df, recommended_model)

    render_recommended_decision(display_df, metrics)

    st.subheader("Comparaison des modèles")
    result_columns = st.columns(len(display_df))

    for column_container, (_, row) in zip(result_columns, display_df.iterrows()):
        model_name = row["modele"]
        model_metrics = metrics["metriques_test"].get(model_name, {})
        with column_container:
            with st.container(border=True):
                if model_name == recommended_model:
                    st.caption("Modèle recommandé")
                else:
                    st.caption("Modèle comparatif")

                st.markdown(f"**{model_name}**")
                if row["label"] == "Churn":
                    st.error(f"Décision : {row['label']}")
                else:
                    st.success(f"Décision : {row['label']}")

                st.metric(
                    "Probabilité de churn",
                    format_probability(row["probabilite_churn"]),
                )
                if model_metrics:
                    st.caption(
                        "Accuracy: "
                        f"{model_metrics['accuracy']:.2f} | "
                        "Recall churn: "
                        f"{model_metrics['recall_churn']:.2f} | "
                        "AUC-ROC: "
                        f"{model_metrics['auc_roc']:.2f}"
                    )


def render_batch_predictions(raw_df: pd.DataFrame, predictions_df: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Résultats du fichier CSV")
    recommended_model = metrics["modele_recommande"]
    predictions_pivot = predictions_df.pivot(
        index="ligne_source",
        columns="modele",
        values="label",
    )
    probability_pivot = predictions_df.pivot(
        index="ligne_source",
        columns="modele",
        values="probabilite_churn",
    )

    churn_counts = (
        predictions_df[predictions_df["modele"] == recommended_model]["label"]
        .value_counts()
        .to_dict()
    )
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Lignes analysées", f"{len(raw_df)}")
    summary_col2.metric("Modèle recommandé", recommended_model)
    summary_col3.metric(
        "Clients signalés churn",
        str(churn_counts.get("Churn", 0)),
    )

    probability_pivot = probability_pivot.rename(
        columns={column: f"{column} - probabilité churn" for column in probability_pivot.columns}
    )

    merged_df = raw_df.reset_index().rename(columns={"index": "ligne_source"})
    merged_df = merged_df.merge(
        predictions_pivot.reset_index(),
        on="ligne_source",
        how="left",
    ).merge(
        probability_pivot.reset_index(),
        on="ligne_source",
        how="left",
    )

    recommended_label_col = recommended_model
    recommended_proba_col = f"{recommended_model} - probabilité churn"
    if recommended_label_col in merged_df.columns:
        merged_df = merged_df.rename(
            columns={
                recommended_label_col: "Décision recommandée",
                recommended_proba_col: "Probabilité churn recommandée",
            }
        )

    preferred_columns = [
        "ligne_source",
        "Décision recommandée",
        "Probabilité churn recommandée",
    ]
    remaining_columns = [
        column for column in merged_df.columns if column not in preferred_columns
    ]
    merged_df = merged_df[preferred_columns + remaining_columns]

    if "Probabilité churn recommandée" in merged_df.columns:
        merged_df["Probabilité churn recommandée"] = merged_df[
            "Probabilité churn recommandée"
        ].map(lambda value: round(value, 4) if pd.notna(value) else value)

    if "Décision recommandée" in merged_df.columns:
        distribution_df = (
            merged_df["Décision recommandée"]
            .value_counts()
            .rename_axis("décision")
            .reset_index(name="clients")
            .set_index("décision")
        )
        st.markdown("**Répartition des décisions recommandées**")
        st.bar_chart(distribution_df, use_container_width=True)

    st.markdown("**Filtrer les lignes affichées**")
    filter_choice = st.radio(
        "Afficher",
        options=["Tous", "Churn", "No Churn"],
        horizontal=True,
        label_visibility="collapsed",
        key="batch_prediction_filter",
    )

    filtered_df = merged_df
    if filter_choice != "Tous" and "Décision recommandée" in merged_df.columns:
        filtered_df = merged_df[merged_df["Décision recommandée"] == filter_choice].reset_index(drop=True)

    st.metric("Lignes affichées", str(len(filtered_df)))

    if filtered_df.empty:
        st.info("Aucune ligne ne correspond au filtre sélectionné.")
        return

    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger les résultats",
        data=csv_data,
        file_name=f"predictions_churn_{filter_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )


def render_expected_schema(metadata: dict) -> None:
    with st.expander("Colonnes attendues pour un fichier CSV"):
        schema_df = pd.DataFrame(
            {
                "colonne": metadata["raw_feature_columns"],
                "type": [
                    "catégorielle" if column in metadata["categorical_columns"] else "numérique"
                    for column in metadata["raw_feature_columns"]
                ],
            }
        )
        st.dataframe(schema_df, use_container_width=True, hide_index=True)


def render_csv_template_download(metadata: dict) -> None:
    template_df = build_csv_template(metadata)
    csv_data = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger un fichier modèle CSV",
        data=csv_data,
        file_name="modele_import_churn.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    inject_css()
    notice_container = st.empty()
    render_completion_notice(notice_container)
    metadata = get_preprocessing_metadata()
    metrics = get_model_metrics()
    models, load_errors = get_models_status()

    render_hero(metrics, models, load_errors)
    render_model_status(models, load_errors)
    render_metrics(metrics)

    manual_tab, batch_tab = st.tabs(["Prédiction manuelle", "Prédiction par CSV"])

    with manual_tab:
        manual_input_df = build_manual_input_form(metadata)
        if not manual_input_df.empty:
            try:
                predictions_df = run_with_loading_feedback(
                    lambda: predict_all_models(manual_input_df),
                    loading_message="Analyse du client en cours...",
                    success_message="Analyse terminée. Les prédictions sont prêtes.",
                    notice_container=notice_container,
                )
                render_single_predictions(predictions_df, metrics)
            except Exception as exc:
                st.error(f"Erreur pendant la prédiction : {exc}")

    with batch_tab:
        st.subheader("Chargement d'un fichier CSV")
        render_csv_template_download(metadata)
        render_expected_schema(metadata)
        uploaded_file = st.file_uploader(
            "Téléverser un fichier CSV contenant les colonnes brutes du projet",
            type=["csv"],
        )

        if uploaded_file is not None:
            current_signature = build_uploaded_file_signature(uploaded_file)
            previous_signature = st.session_state.get("batch_results_signature")
            if previous_signature != current_signature:
                st.session_state["batch_results_signature"] = current_signature
                st.session_state.pop("batch_results_cache", None)

            try:
                preview_df = parse_uploaded_file(uploaded_file)
                validated_preview_df, preview_warnings = validate_uploaded_dataframe(
                    preview_df,
                    metadata,
                )
                for warning_message in preview_warnings:
                    st.warning(warning_message)
                st.write("Aperçu des données chargées")
                st.dataframe(validated_preview_df.head(), use_container_width=True, hide_index=True)

                action_col1, action_col2 = st.columns([1.25, 1])
                with action_col1:
                    analyze_csv = st.button(
                        "Lancer l'analyse du fichier",
                        use_container_width=True,
                        key="batch_analysis_button",
                    )
                with action_col2:
                    reset_csv = st.button(
                        "Réinitialiser les résultats",
                        use_container_width=True,
                        key="batch_reset_button",
                    )

                if reset_csv:
                    clear_batch_results_state()
                    st.rerun()

                if analyze_csv:
                    batch_df, predictions_df = run_with_loading_feedback(
                        lambda: process_batch_dataframe(validated_preview_df),
                        loading_message="Analyse du fichier CSV en cours...",
                        success_message="Analyse du fichier terminée. Les résultats sont disponibles.",
                        notice_container=notice_container,
                    )
                    st.session_state["batch_results_cache"] = {
                        "signature": current_signature,
                        "raw_df": batch_df,
                        "predictions_df": predictions_df,
                    }

                cached_batch_results = st.session_state.get("batch_results_cache")
                if (
                    cached_batch_results
                    and cached_batch_results.get("signature") == current_signature
                ):
                    render_batch_predictions(
                        cached_batch_results["raw_df"],
                        cached_batch_results["predictions_df"],
                        metrics,
                    )
            except Exception as exc:
                st.error(f"Erreur pendant le traitement du fichier : {exc}")


if __name__ == "__main__":
    main()
