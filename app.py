from __future__ import annotations

import hashlib
from html import escape
from io import StringIO
from textwrap import dedent
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
                transform: translateY(16px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        @keyframes float-card {
            0% {
                transform: translateY(0);
            }
            100% {
                transform: translateY(-6px);
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
            --bg-main: #f3f7ff;
            --bg-soft: #eaf1ff;
            --surface: rgba(255, 255, 255, 0.96);
            --surface-strong: #ffffff;
            --ink-main: #10233f;
            --ink-soft: #62718a;
            --line: rgba(16, 35, 63, 0.09);
            --line-strong: rgba(16, 35, 63, 0.18);
            --accent-1: #2f6bff;
            --accent-2: #79b4ff;
            --accent-3: #0f8bb8;
            --accent-4: #4db5ff;
            --navy: #102a52;
            --ok: #12936b;
            --warn: #b57600;
            --danger: #c34a36;
            --shadow-lg: 0 28px 60px rgba(28, 36, 61, 0.10);
            --shadow-md: 0 18px 38px rgba(28, 36, 61, 0.08);
            --shadow-sm: 0 10px 22px rgba(28, 36, 61, 0.06);
        }
        [data-testid="stAppViewContainer"] {
            color: var(--ink-main);
            background:
                radial-gradient(circle at 0% 0%, rgba(47, 107, 255, 0.12), transparent 24%),
                radial-gradient(circle at 100% 12%, rgba(77, 181, 255, 0.11), transparent 20%),
                radial-gradient(circle at 20% 100%, rgba(15, 139, 184, 0.08), transparent 22%),
                linear-gradient(180deg, var(--bg-main) 0%, #fcfdff 100%);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3.5rem;
            max-width: 1240px;
        }
        p, li, label, .stMarkdown, .stMarkdown p, .stCaptionContainer, .stCaptionContainer p {
            color: var(--ink-main);
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--ink-main) !important;
            letter-spacing: -0.03em;
        }
        .nav-shell {
            position: sticky;
            top: 1.45rem;
            z-index: 100;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.9rem 1rem;
            margin-bottom: 1.3rem;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.88);
            border-radius: 24px;
            backdrop-filter: blur(14px);
            box-shadow: var(--shadow-sm);
        }
        .anchor-target {
            position: relative;
            top: -5.9rem;
            height: 0;
        }
        .nav-brand {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            min-width: 0;
        }
        .nav-brand-mark {
            width: 48px;
            height: 48px;
            display: grid;
            place-items: center;
            border-radius: 16px;
            color: white;
            background: linear-gradient(135deg, var(--accent-1), var(--accent-4));
            box-shadow: 0 16px 28px rgba(47, 107, 255, 0.24);
            flex: 0 0 auto;
        }
        .nav-brand-copy strong {
            display: block;
            font-size: 1rem;
            line-height: 1.1;
            color: var(--navy);
        }
        .nav-brand-copy span {
            display: block;
            margin-top: 0.1rem;
            color: var(--ink-soft);
            font-size: 0.85rem;
        }
        .nav-links {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            flex-wrap: wrap;
            justify-content: center;
        }
        .nav-center {
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 1 1 auto;
        }
        .nav-menu {
            position: relative;
        }
        .nav-menu summary {
            list-style: none;
        }
        .nav-menu summary::-webkit-details-marker {
            display: none;
        }
        .nav-menu-summary {
            display: inline-flex;
            align-items: center;
            gap: 0.62rem;
            padding: 0.78rem 1rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #ffffff, #f4f8ff);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-sm);
            color: var(--navy);
            font-size: 0.92rem;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }
        .nav-menu-summary:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
            border-color: var(--line-strong);
        }
        .nav-menu-panel {
            position: absolute;
            top: calc(100% + 0.75rem);
            left: 50%;
            min-width: 280px;
            padding: 0.85rem;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.98);
            border: 1px solid rgba(255, 255, 255, 0.95);
            box-shadow: var(--shadow-lg);
            transform: translateX(-50%);
        }
        .nav-menu-copy {
            margin: 0 0 0.72rem 0;
            color: var(--ink-soft);
            font-size: 0.84rem;
            line-height: 1.5;
        }
        .nav-menu-links {
            display: grid;
            gap: 0.55rem;
        }
        .nav-menu-link {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.78rem 0.9rem;
            border-radius: 18px;
            background: linear-gradient(180deg, #ffffff, #f5f9ff);
            border: 1px solid var(--line);
            color: var(--navy) !important;
            font-size: 0.9rem;
            font-weight: 700;
            text-decoration: none !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }
        .nav-menu-link:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-sm);
            border-color: rgba(47, 107, 255, 0.18);
        }
        .nav-menu-link span {
            width: 18px;
            height: 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-1);
        }
        .nav-link {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.7rem 0.95rem;
            border-radius: 999px;
            color: var(--navy);
            font-size: 0.92rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid transparent;
            transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
        }
        .nav-link:hover {
            transform: translateY(-1px);
            background: rgba(255, 255, 255, 0.9);
            border-color: var(--line);
        }
        .nav-link-active {
            background: linear-gradient(135deg, var(--navy), #22385a);
            color: #fffaf7;
            box-shadow: 0 14px 26px rgba(24, 37, 61, 0.16);
        }
        .nav-meta {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }
        .nav-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.72rem 0.95rem;
            border-radius: 999px;
            background: #ffffff;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-sm);
            color: var(--navy);
            font-size: 0.9rem;
            font-weight: 700;
        }
        .hero-shell {
            display: grid;
            grid-template-columns: minmax(0, 1.28fr) minmax(320px, 0.9fr);
            gap: 1.2rem;
            margin-bottom: 1.4rem;
            animation: fade-up 0.55s ease-out both;
        }
        .hero-panel {
            position: relative;
            overflow: hidden;
            min-height: 100%;
            border-radius: 32px;
            border: 1px solid rgba(255, 255, 255, 0.58);
            box-shadow: var(--shadow-lg);
        }
        .hero-copy {
            padding: 2rem 2rem 1.8rem;
            background:
                radial-gradient(circle at top left, rgba(255, 255, 255, 0.22), transparent 22%),
                linear-gradient(135deg, #0f2342 0%, #13355f 55%, #1b4e84 100%);
            color: #fff8f3;
        }
        .hero-copy::after {
            content: "";
            position: absolute;
            inset: auto -32px -38px auto;
            width: 170px;
            height: 170px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(121, 180, 255, 0.26) 0%, rgba(121, 180, 255, 0.05) 55%, transparent 70%);
        }
        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.5rem 0.85rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: rgba(255, 248, 243, 0.92);
        }
        .hero-copy h1 {
            margin: 1rem 0 0.65rem 0;
            max-width: 720px;
            font-size: 3rem;
            line-height: 1.03;
            color: #fff8f3 !important;
        }
        .hero-copy p {
            margin: 0;
            max-width: 760px;
            font-size: 1.04rem;
            line-height: 1.7;
            color: rgba(255, 248, 243, 0.8);
        }
        .hero-feature-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.8rem;
            margin-top: 1.25rem;
        }
        .hero-feature {
            display: inline-flex;
            align-items: center;
            gap: 0.62rem;
            padding: 0.8rem 0.95rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.11);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #fff8f3;
            font-weight: 600;
            min-width: 180px;
        }
        .hero-feature-icon,
        .hero-card-icon,
        .section-icon,
        .status-icon {
            width: 18px;
            height: 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
        }
        .hero-side {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .hero-card {
            padding: 1.35rem 1.35rem 1.2rem;
            border-radius: 30px;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-md);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(244, 248, 255, 0.98));
        }
        .hero-card-primary {
            position: relative;
        }
        .hero-card-primary::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 8px;
            border-radius: 30px 30px 0 0;
            background: linear-gradient(90deg, var(--accent-1), var(--accent-2), var(--accent-3));
        }
        .hero-card-label {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--accent-1);
            font-size: 0.84rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .hero-card-title {
            margin: 0.65rem 0 0.35rem 0;
            font-size: 1.55rem;
            line-height: 1.14;
            color: var(--navy);
            font-weight: 800;
        }
        .hero-card-copy {
            margin: 0;
            color: var(--ink-soft);
            line-height: 1.6;
        }
        .hero-stat-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 1rem;
        }
        .hero-stat {
            padding: 0.95rem 0.9rem;
            border-radius: 20px;
            background: #f8fbff;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-sm);
        }
        .hero-stat strong {
            display: block;
            font-size: 1.05rem;
            color: var(--navy);
        }
        .hero-stat span {
            display: block;
            margin-top: 0.22rem;
            font-size: 0.82rem;
            color: var(--ink-soft);
        }
        .hero-list {
            display: grid;
            gap: 0.7rem;
        }
        .hero-list-item {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
            padding: 0.85rem 0.95rem;
            border-radius: 20px;
            background: rgba(245, 249, 255, 0.92);
            border: 1px solid var(--line);
        }
        .hero-list-item strong {
            display: block;
            color: var(--navy);
            margin-bottom: 0.12rem;
        }
        .hero-list-item span {
            color: var(--ink-soft);
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .section-shell {
            display: flex;
            align-items: flex-start;
            gap: 0.95rem;
            margin: 1.55rem 0 0.9rem;
            animation: fade-up 0.45s ease-out both;
        }
        .section-icon-wrap {
            width: 52px;
            height: 52px;
            display: grid;
            place-items: center;
            border-radius: 18px;
            color: white;
            background: linear-gradient(135deg, var(--accent-1), var(--accent-4));
            box-shadow: 0 16px 28px rgba(47, 107, 255, 0.20);
            flex: 0 0 auto;
        }
        .section-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--accent-1);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .section-shell h2 {
            margin: 0.18rem 0 0.18rem 0;
            font-size: 1.9rem;
            line-height: 1.08;
            color: var(--navy) !important;
        }
        .section-shell p {
            margin: 0;
            color: var(--ink-soft);
            line-height: 1.55;
            max-width: 760px;
        }
        .status-card {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(244, 248, 255, 0.98));
            border: 1px solid var(--line);
            border-radius: 26px;
            padding: 1.15rem 1.05rem 1rem;
            min-height: 170px;
            box-shadow: var(--shadow-md);
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }
        .status-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 8px;
            background: linear-gradient(90deg, var(--accent-1), var(--accent-2), var(--accent-3));
        }
        .status-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 24px 42px rgba(28, 36, 61, 0.12);
            border-color: var(--line-strong);
        }
        .status-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.85rem;
        }
        .status-icon-wrap {
            width: 44px;
            height: 44px;
            display: grid;
            place-items: center;
            border-radius: 16px;
            color: white;
            background: linear-gradient(135deg, var(--accent-1), var(--accent-4));
            box-shadow: 0 12px 22px rgba(47, 107, 255, 0.18);
            flex: 0 0 auto;
        }
        .status-card h4 {
            margin: 0 0 0.25rem 0;
            color: var(--navy) !important;
            font-size: 1.1rem;
        }
        .status-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.72rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .status-chip-ok {
            color: #0f6d51;
            background: rgba(18, 147, 107, 0.12);
            border: 1px solid rgba(18, 147, 107, 0.22);
        }
        .status-chip-warn {
            color: #8e5c00;
            background: rgba(255, 184, 108, 0.16);
            border: 1px solid rgba(255, 184, 108, 0.22);
        }
        .status-card p {
            margin: 0;
            color: var(--ink-soft);
            line-height: 1.55;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(244, 248, 255, 0.98));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1rem 1rem;
            box-shadow: var(--shadow-md);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            overflow: hidden;
            position: relative;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 22px 36px rgba(28, 36, 61, 0.12);
            border-color: var(--line-strong);
        }
        div[data-testid="stMetric"]::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 6px;
            background: linear-gradient(180deg, var(--accent-3), var(--accent-1));
        }
        .stTabs [role="tablist"] {
            gap: 0.55rem;
            padding: 0.35rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(255, 255, 255, 0.88);
            box-shadow: var(--shadow-sm);
            margin: 0.4rem 0 1.1rem;
        }
        .stTabs [role="tab"] {
            background: transparent;
            border: 1px solid transparent;
            padding: 0.84rem 1rem;
            border-radius: 18px;
            color: var(--navy) !important;
            font-weight: 700;
            transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease;
        }
        .stTabs [role="tab"] * {
            color: var(--navy) !important;
        }
        .stTabs [role="tab"]:hover {
            transform: translateY(-1px);
            background: rgba(255, 255, 255, 0.86);
            border-color: var(--line);
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, var(--navy), #243757);
            color: #fff9f6 !important;
            font-weight: 700;
            box-shadow: 0 14px 28px rgba(24, 37, 61, 0.16);
        }
        .stTabs [role="tab"][aria-selected="true"] * {
            color: #fff9f6 !important;
        }
        .stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(135deg, var(--accent-1), var(--accent-4));
            color: #fff9f6;
            border: 1px solid rgba(47, 107, 255, 0.20);
            border-radius: 18px;
            font-weight: 700;
            min-height: 3rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
            box-shadow: 0 18px 28px rgba(47, 107, 255, 0.22);
        }
        .stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 24px 32px rgba(47, 107, 255, 0.28);
            filter: saturate(1.04);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(255, 255, 255, 0.88);
            border-radius: 26px;
            overflow: hidden;
            box-shadow: var(--shadow-md);
            background: var(--surface);
        }
        div[data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.88);
            border-radius: 24px;
            box-shadow: var(--shadow-sm);
            margin-top: 1.5rem;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary * {
            color: var(--navy) !important;
            font-weight: 700 !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(244, 248, 255, 0.98));
            border: 1px solid rgba(255, 255, 255, 0.9) !important;
            border-radius: 28px;
            box-shadow: var(--shadow-md);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-3px);
            box-shadow: 0 22px 36px rgba(28, 36, 61, 0.12);
            border-color: rgba(255, 255, 255, 1) !important;
        }
        div[data-testid="stForm"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(244, 248, 255, 0.98));
            border: 1px solid rgba(255, 255, 255, 0.92);
            border-radius: 32px;
            padding: 1.3rem 1.25rem 1rem;
            box-shadow: var(--shadow-lg);
        }
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            background: white !important;
            border: 1px solid rgba(22, 32, 52, 0.10) !important;
            border-radius: 16px !important;
            color: var(--navy) !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7), 0 10px 18px rgba(28, 36, 61, 0.04) !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        div[data-testid="stNumberInput"] input {
            color: var(--navy) !important;
        }
        div[data-baseweb="select"] > div:focus-within,
        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stTextInput"] input:focus {
            border-color: rgba(47, 107, 255, 0.34) !important;
            box-shadow: 0 0 0 4px rgba(47, 107, 255, 0.10), 0 14px 24px rgba(28, 36, 61, 0.06) !important;
        }
        div[data-testid="stFileUploader"] {
            max-width: 720px;
            margin: 0.9rem auto 0;
        }
        div[data-testid="stFileUploader"] section {
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(244, 248, 255, 0.98));
            border: 2px dashed rgba(47, 107, 255, 0.22);
            border-radius: 28px;
            box-shadow: var(--shadow-md);
            min-height: 180px;
            padding: 1.25rem 1.4rem;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        div[data-testid="stFileUploaderDropzone"] {
            background: transparent;
        }
        div[data-testid="stFileUploaderDropzone"] > div {
            width: 100%;
            max-width: 460px;
            margin: 0 auto;
            text-align: center;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"] {
            text-align: center;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"] span,
        div[data-testid="stFileUploaderDropzoneInstructions"] small {
            color: var(--ink-soft) !important;
        }
        .upload-dropzone-copy {
            max-width: 720px;
            margin: 0.3rem auto 0.1rem;
            text-align: center;
        }
        .upload-dropzone-copy strong {
            display: block;
            color: var(--navy);
            font-size: 1rem;
            margin-bottom: 0.18rem;
        }
        .upload-dropzone-copy span {
            color: var(--ink-soft);
            font-size: 0.92rem;
            line-height: 1.5;
        }
        div[data-testid="stAlert"] {
            border-radius: 22px;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-sm);
        }
        div[data-testid="stAlert"][kind="warning"] {
            background: rgba(255, 184, 108, 0.18);
            border-color: rgba(255, 184, 108, 0.22);
        }
        div[data-testid="stAlert"][kind="error"] {
            background: rgba(195, 74, 54, 0.10);
            border-color: rgba(195, 74, 54, 0.18);
        }
        div[data-testid="stAlert"][kind="success"] {
            background: rgba(18, 147, 107, 0.10);
            border-color: rgba(18, 147, 107, 0.18);
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
            padding: 1rem 1.05rem;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.92);
            background: rgba(255, 255, 255, 0.94);
            backdrop-filter: blur(14px);
            box-shadow: var(--shadow-lg);
            animation: popup-in 0.26s ease-out both;
        }
        .floating-popup-loading {
            border-color: rgba(47, 107, 255, 0.18);
        }
        .floating-popup-success {
            border-color: rgba(18, 147, 107, 0.22);
            background: rgba(247, 255, 252, 0.96);
        }
        .floating-popup-icon {
            width: 20px;
            height: 20px;
            border-radius: 999px;
            flex: 0 0 auto;
            position: relative;
        }
        .floating-popup-loading .floating-popup-icon {
            border: 2px solid rgba(22, 32, 52, 0.14);
            border-top-color: var(--accent-1);
            animation: spin 0.85s linear infinite;
        }
        .floating-popup-success .floating-popup-icon {
            background: rgba(18, 147, 107, 0.12);
            border: 1px solid rgba(18, 147, 107, 0.24);
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
            color: var(--navy);
            font-size: 0.95rem;
            line-height: 1.4;
        }
        .nav-link,
        .nav-link * {
            color: var(--navy) !important;
        }
        .nav-link:hover,
        .nav-link:hover * {
            color: var(--navy) !important;
        }
        .nav-link-active,
        .nav-link-active * {
            color: #fffaf7 !important;
        }
        .nav-pill-demo {
            background: linear-gradient(135deg, var(--navy), #22385a);
            border-color: transparent;
            box-shadow: 0 14px 24px rgba(24, 37, 61, 0.18);
        }
        .nav-pill-demo,
        .nav-pill-demo * {
            color: #fffaf7 !important;
        }
        .section-shell {
            position: relative;
            overflow: hidden;
            align-items: center;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.97), rgba(244, 248, 255, 0.98));
            border: 1px solid rgba(255, 255, 255, 0.92);
            border-radius: 26px;
            box-shadow: var(--shadow-md);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .section-shell::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 7px;
            background: linear-gradient(90deg, var(--accent-1), var(--accent-2), var(--accent-3));
        }
        .section-shell:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }
        .section-copy {
            display: grid;
            gap: 0.22rem;
        }
        .section-kicker {
            color: var(--accent-1);
        }
        div[data-testid="stExpander"] summary {
            background: rgba(255, 255, 255, 0.96);
            border-radius: 18px;
            padding: 0.78rem 1rem;
        }
        div[data-testid="stExpander"] summary:hover {
            background: #ffffff;
        }
        div[data-testid="stExpander"] details > div {
            color: var(--navy) !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stDataFrame"] * {
            color: var(--navy) !important;
        }
        div[data-testid="stDataFrame"] table,
        div[data-testid="stDataFrame"] thead,
        div[data-testid="stDataFrame"] tbody,
        div[data-testid="stDataFrame"] tr,
        div[data-testid="stDataFrame"] td,
        div[data-testid="stDataFrame"] th,
        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stDataFrame"] [role="row"],
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"],
        div[data-testid="stDataFrame"] [role="rowheader"] {
            background: #fffefc !important;
            color: var(--navy) !important;
            border-color: rgba(22, 32, 52, 0.08) !important;
        }
        div[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] * {
            color: var(--navy) !important;
        }
        div[data-testid="stFileUploader"] button {
            background: #ffffff !important;
            color: var(--navy) !important;
            border: 1px solid var(--line) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        .compare-card {
            position: relative;
            overflow: hidden;
            height: 100%;
            padding: 1.1rem 1.05rem 1rem;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(244, 248, 255, 0.98));
            border: 1px solid var(--line);
            border-radius: 28px;
            box-shadow: var(--shadow-md);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .compare-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 8px;
            background: linear-gradient(90deg, var(--accent-1), var(--accent-2), var(--accent-3));
        }
        .compare-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
        }
        .compare-card-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.8rem;
        }
        .compare-card-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.42rem;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--navy);
            background: rgba(47, 107, 255, 0.10);
            border: 1px solid rgba(47, 107, 255, 0.16);
        }
        .compare-card-badge-primary {
            background: rgba(47, 107, 255, 0.12);
            border-color: rgba(47, 107, 255, 0.18);
            color: var(--accent-1);
        }
        .compare-card h4 {
            margin: 0.72rem 0 0.18rem 0;
            color: var(--navy) !important;
            font-size: 1.16rem;
        }
        .compare-card-subtitle {
            color: var(--ink-soft);
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .decision-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 0.72rem 0.9rem;
            border-radius: 18px;
            font-size: 0.94rem;
            font-weight: 700;
            margin-bottom: 0.95rem;
        }
        .decision-chip-safe {
            background: linear-gradient(135deg, rgba(18, 147, 107, 0.12), rgba(30, 154, 140, 0.08));
            border: 1px solid rgba(18, 147, 107, 0.18);
            color: #0f6d51;
        }
        .decision-chip-risk {
            background: linear-gradient(135deg, rgba(255, 107, 61, 0.14), rgba(255, 77, 141, 0.08));
            border: 1px solid rgba(255, 107, 61, 0.18);
            color: #b54f2e;
        }
        .speedometer {
            position: relative;
            width: 100%;
            max-width: 250px;
            margin: 0 auto 0.8rem;
            padding-top: 0.25rem;
        }
        .speedometer-arc {
            position: relative;
            width: 100%;
            aspect-ratio: 2 / 1.08;
            overflow: hidden;
            border-radius: 240px 240px 22px 22px;
        }
        .speedometer-arc::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: inherit;
            background: conic-gradient(
                from 180deg at 50% 100%,
                #d4e5ff 0deg 55deg,
                #77a8ff 55deg 120deg,
                #2154b8 120deg 180deg,
                transparent 180deg 360deg
            );
        }
        .speedometer-arc::after {
            content: "";
            position: absolute;
            left: 50%;
            bottom: -7%;
            width: 64%;
            aspect-ratio: 1 / 1;
            transform: translateX(-50%);
            border-radius: 50%;
            background: linear-gradient(180deg, #ffffff, #f8fbff);
            box-shadow: 0 -12px 26px rgba(28, 36, 61, 0.08);
        }
        .speedometer-needle {
            position: absolute;
            left: 50%;
            bottom: 15px;
            width: 42%;
            height: 5px;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--navy), #3f5478);
            transform-origin: 0% 50%;
            transform: rotate(calc(-180deg + var(--gauge-angle)));
            box-shadow: 0 6px 14px rgba(24, 37, 61, 0.18);
        }
        .speedometer-pin {
            position: absolute;
            left: 50%;
            bottom: 6px;
            width: 20px;
            height: 20px;
            transform: translateX(-50%);
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
            box-shadow: 0 10px 18px rgba(47, 107, 255, 0.22);
            border: 3px solid #fff8f3;
        }
        .speedometer-scale {
            display: flex;
            justify-content: space-between;
            margin-top: -0.15rem;
            font-size: 0.74rem;
            color: var(--ink-soft);
        }
        .speedometer-value {
            margin-top: 0.45rem;
            text-align: center;
            color: var(--navy);
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }
        .speedometer-caption {
            text-align: center;
            color: var(--ink-soft);
            font-size: 0.85rem;
            margin-top: 0.15rem;
        }
        .compare-meta {
            display: grid;
            gap: 0.5rem;
            margin-top: 0.8rem;
        }
        .compare-meta-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            padding: 0.55rem 0.7rem;
            border-radius: 16px;
            background: rgba(248, 251, 255, 0.92);
            border: 1px solid rgba(16, 35, 63, 0.08);
        }
        .compare-meta-row span {
            color: var(--ink-soft);
            font-size: 0.86rem;
        }
        .compare-meta-row strong {
            color: var(--navy);
            font-size: 0.92rem;
        }
        .view-switch-copy {
            margin: 0.2rem 0 0.75rem 0;
            color: var(--ink-soft);
        }
        .stRadio [role="radiogroup"] {
            gap: 0.6rem;
        }
        .stRadio [role="radiogroup"] label {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(244, 248, 255, 0.98));
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 0.4rem 0.9rem;
            box-shadow: var(--shadow-sm);
        }
        .stRadio [role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, var(--accent-1), #1b54d6);
            border-color: transparent;
        }
        .stRadio [role="radiogroup"] label:has(input:checked) p {
            color: #fff9f6 !important;
        }
        @media (prefers-reduced-motion: reduce) {
            .hero-shell,
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
        @media (max-width: 1100px) {
            .nav-shell {
                flex-direction: column;
                align-items: stretch;
            }
            .nav-center,
            .nav-meta {
                justify-content: flex-start;
            }
            .nav-menu-panel {
                left: 0;
                transform: none;
            }
            .hero-shell {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 800px) {
            .block-container {
                padding-top: 0.55rem;
            }
            .nav-shell {
                border-radius: 20px;
                padding: 0.85rem;
            }
            .nav-menu-summary,
            .nav-pill {
                width: 100%;
                justify-content: center;
            }
            .hero-copy,
            .hero-card {
                padding: 1.35rem 1.15rem 1.2rem;
            }
            .hero-copy h1 {
                font-size: 2.25rem;
            }
            .hero-stat-grid {
                grid-template-columns: 1fr;
            }
            .section-shell {
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def icon_svg(name: str) -> str:
    icons = {
        "brand": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="4"></rect>
                <path d="M7 15V9"></path>
                <path d="M12 15V6"></path>
                <path d="M17 15v-3"></path>
            </svg>
        """,
        "spark": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="m12 3 1.7 5.3L19 10l-5.3 1.7L12 17l-1.7-5.3L5 10l5.3-1.7L12 3Z"></path>
            </svg>
        """,
        "dashboard": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="3" width="7" height="7" rx="2"></rect>
                <rect x="14" y="3" width="7" height="7" rx="2"></rect>
                <rect x="14" y="14" width="7" height="7" rx="2"></rect>
                <rect x="3" y="14" width="7" height="7" rx="2"></rect>
            </svg>
        """,
        "model": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="6" cy="12" r="2.5"></circle>
                <circle cx="18" cy="6" r="2.5"></circle>
                <circle cx="18" cy="18" r="2.5"></circle>
                <path d="M8.4 10.8 15.6 7.2"></path>
                <path d="M8.4 13.2 15.6 16.8"></path>
            </svg>
        """,
        "form": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M8 4h8"></path>
                <rect x="5" y="3" width="14" height="18" rx="2"></rect>
                <path d="M8 9h8"></path>
                <path d="M8 13h8"></path>
                <path d="M8 17h5"></path>
            </svg>
        """,
        "upload": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 16V7"></path>
                <path d="m8.5 10.5 3.5-3.5 3.5 3.5"></path>
                <path d="M20 16.5a3.5 3.5 0 0 0-2.2-3.2 5 5 0 0 0-9.6-1.2A4 4 0 0 0 8 20h10a2 2 0 0 0 2-2v-1.5Z"></path>
            </svg>
        """,
        "shield": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 3 5 6v5c0 5 3.3 8.5 7 10 3.7-1.5 7-5 7-10V6l-7-3Z"></path>
                <path d="m9.5 12 1.7 1.7L15 10"></path>
            </svg>
        """,
        "metrics": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 4a8 8 0 1 0 8 8"></path>
                <path d="M12 12 17 7"></path>
                <path d="M12 12h.01"></path>
            </svg>
        """,
        "target": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="8"></circle>
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M12 2v3"></path>
                <path d="M12 19v3"></path>
                <path d="M2 12h3"></path>
                <path d="M19 12h3"></path>
            </svg>
        """,
        "compare": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect x="4" y="5" width="7" height="14" rx="2"></rect>
                <rect x="13" y="5" width="7" height="14" rx="2"></rect>
                <path d="M8 9h0"></path>
                <path d="M17 9h0"></path>
                <path d="M8 13h0"></path>
                <path d="M17 13h0"></path>
            </svg>
        """,
        "table": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="4" width="18" height="16" rx="2"></rect>
                <path d="M3 10h18"></path>
                <path d="M8 4v16"></path>
            </svg>
        """,
        "check": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="m5 13 4 4L19 7"></path>
            </svg>
        """,
        "mail": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <rect x="3" y="5" width="18" height="14" rx="3"></rect>
                <path d="m5 8 7 5 7-5"></path>
            </svg>
        """,
        "menu": """
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 7h16"></path>
                <path d="M4 12h16"></path>
                <path d="M4 17h16"></path>
            </svg>
        """,
    }
    svg = dedent(icons.get(name, icons["spark"])).strip()
    return "".join(line.strip() for line in svg.splitlines())


def render_navbar(metrics: dict, models: dict) -> None:
    st.markdown(
        dedent(
            f"""
        <div class="nav-shell">
            <div class="nav-brand">
                <div class="nav-brand-mark">{icon_svg("brand")}</div>
                <div class="nav-brand-copy">
                    <strong>ChurnScope</strong>
                    <span>Plateforme Streamlit de prédiction client</span>
                </div>
            </div>
            <div class="nav-center">
                <details class="nav-menu">
                    <summary class="nav-menu-summary">
                        {icon_svg("menu")}
                        <span>Accès rapide</span>
                    </summary>
                    <div class="nav-menu-panel">
                        <p class="nav-menu-copy">Choisis directement la zone à afficher pendant la présentation ou pendant les tests.</p>
                        <div class="nav-menu-links">
                            <a class="nav-menu-link" href="#overview-section"><span>{icon_svg("dashboard")}</span>Vue générale</a>
                            <a class="nav-menu-link" href="#metrics-section"><span>{icon_svg("metrics")}</span>Qualité des modèles</a>
                            <a class="nav-menu-link" href="?vue=manuel#manual-entry"><span>{icon_svg("form")}</span>Entrée manuelle</a>
                            <a class="nav-menu-link" href="?vue=csv#csv-entry"><span>{icon_svg("upload")}</span>Entrée CSV</a>
                        </div>
                    </div>
                </details>
            </div>
            <div class="nav-meta">
                <div class="nav-pill">{icon_svg("shield")} {len(models)}/3 modèles prêts</div>
                <div class="nav-pill nav-pill-demo">{icon_svg("mail")} demo.churnscope.app@gmail.com</div>
            </div>
        </div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str, icon_name: str, anchor_id: str | None = None) -> None:
    anchor_html = f'<div id="{anchor_id}" class="anchor-target"></div>' if anchor_id else ""
    st.markdown(
        dedent(
            f"""
        {anchor_html}
        <div class="section-shell">
            <div class="section-icon-wrap">
                <span class="section-icon">{icon_svg(icon_name)}</span>
            </div>
            <div class="section-copy">
                <div class="section-kicker">Lecture rapide</div>
                <h2>{title}</h2>
                <p>{description}</p>
            </div>
        </div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def build_speedometer(probability: float | None) -> str:
    probability = 0.0 if probability is None or pd.isna(probability) else float(probability)
    probability = max(0.0, min(probability, 1.0))
    angle = probability * 180.0
    return dedent(
        f"""
        <div class="speedometer" style="--gauge-angle: {angle:.2f}deg;">
            <div class="speedometer-arc"></div>
            <div class="speedometer-needle"></div>
            <div class="speedometer-pin"></div>
            <div class="speedometer-scale">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
            </div>
            <div class="speedometer-value">{format_probability(probability)}</div>
            <div class="speedometer-caption">Probabilité de churn</div>
        </div>
    """
    ).strip()


def build_comparison_card(row: pd.Series, model_metrics: dict, is_recommended: bool) -> str:
    decision = str(row["label"])
    decision_class = "decision-chip-risk" if decision == "Churn" else "decision-chip-safe"
    badge_class = "compare-card-badge-primary" if is_recommended else ""
    badge_label = "Lecture principale" if is_recommended else "Modèle comparatif"
    subtitle = (
        "Modèle mis en avant pour la lecture finale."
        if is_recommended
        else "Point de comparaison supplémentaire."
    )
    probability = row["probabilite_churn"]

    return dedent(
        f"""
        <div class="compare-card">
            <div class="compare-card-top">
                <div>
                    <span class="compare-card-badge {badge_class}">{icon_svg("compare")} {badge_label}</span>
                    <h4>{escape(str(row["modele"]))}</h4>
                    <div class="compare-card-subtitle">{subtitle}</div>
                </div>
            </div>
            <div class="decision-chip {decision_class}">Décision : {escape(decision)}</div>
            {build_speedometer(probability)}
            <div class="compare-meta">
                <div class="compare-meta-row">
                    <span>Accuracy</span>
                    <strong>{model_metrics.get("accuracy", 0):.2f}</strong>
                </div>
                <div class="compare-meta-row">
                    <span>Recall churn</span>
                    <strong>{model_metrics.get("recall_churn", 0):.2f}</strong>
                </div>
                <div class="compare-meta-row">
                    <span>AUC-ROC</span>
                    <strong>{model_metrics.get("auc_roc", 0):.2f}</strong>
                </div>
            </div>
        </div>
    """
    ).strip()


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

    render_section_header(
        "Saisie manuelle d'un client",
        "Renseigne un profil client avec les colonnes brutes du projet. L'application applique ensuite automatiquement les mêmes transformations que dans le notebook.",
        "form",
        anchor_id="manual-entry",
    )

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
    recommended_metrics = metrics["metriques_test"][metrics["modele_recommande"]]
    unavailable_text = (
        f"{len(load_errors)} indisponible(s)" if load_errors else "Tous chargés"
    )

    st.markdown(
        dedent(
            f"""
        <div id="overview-section" class="anchor-target"></div>
        <div class="hero-shell">
            <div class="hero-panel hero-copy">
                <div class="hero-kicker">{icon_svg("spark")} Analyse supervisée du churn</div>
                <h1>Plateforme d'analyse du churn pour tester un client ou un fichier complet.</h1>
                <p>
                    Cette interface regroupe un parcours manuel, une analyse par lot en CSV
                    et une lecture claire des trois modèles dans un même espace de travail.
                </p>
                <div class="hero-feature-row">
                    <div class="hero-feature">
                        <span class="hero-feature-icon">{icon_svg("form")}</span>
                        <span>Client individuel</span>
                    </div>
                    <div class="hero-feature">
                        <span class="hero-feature-icon">{icon_svg("upload")}</span>
                        <span>Analyse par lot</span>
                    </div>
                    <div class="hero-feature">
                        <span class="hero-feature-icon">{icon_svg("compare")}</span>
                        <span>Lecture comparative</span>
                    </div>
                </div>
            </div>
            <div class="hero-side">
                <div class="hero-card hero-card-primary">
                    <div class="hero-card-label">{icon_svg("target")} Modèle principal</div>
                    <div class="hero-card-title">{metrics["modele_recommande"]}</div>
                    <p class="hero-card-copy">
                        Le choix principal met l'accent sur la détection des clients à risque,
                        avec un rappel churn plus fort pour éviter de manquer les cas sensibles.
                    </p>
                    <div class="hero-stat-grid">
                        <div class="hero-stat">
                            <strong>{len(models)}/3</strong>
                            <span>modèles prêts</span>
                        </div>
                        <div class="hero-stat">
                            <strong>{recommended_metrics["recall_churn"]:.2f}</strong>
                            <span>rappel churn</span>
                        </div>
                        <div class="hero-stat">
                            <strong>{recommended_metrics["auc_roc"]:.2f}</strong>
                            <span>AUC-ROC</span>
                        </div>
                    </div>
                </div>
                <div class="hero-card">
                    <div class="hero-list">
                        <div class="hero-list-item">
                            <span class="hero-card-icon">{icon_svg("shield")}</span>
                            <div>
                                <strong>État des modèles</strong>
                                <span>{unavailable_text} sur cette machine.</span>
                            </div>
                        </div>
                        <div class="hero-list-item">
                            <span class="hero-card-icon">{icon_svg("table")}</span>
                            <div>
                                <strong>Deux modes d'analyse</strong>
                                <span>Un client à la fois ou un fichier CSV complet avec les colonnes attendues.</span>
                            </div>
                        </div>
                        <div class="hero-list-item">
                            <span class="hero-card-icon">{icon_svg("metrics")}</span>
                            <div>
                                <strong>Sortie exploitable</strong>
                                <span>Décision, probabilité de churn, métriques et export des résultats.</span>
                            </div>
                        </div>
                        <div class="hero-list-item">
                            <span class="hero-card-icon">{icon_svg("mail")}</span>
                            <div>
                                <strong>Demande de démo</strong>
                                <span>demo.churnscope.app@gmail.com</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_model_status(models: dict, load_errors: dict) -> None:
    render_section_header(
        "État des modèles",
        "Cette vue confirme quels modèles sont correctement chargés sur la machine courante avant de lancer une prédiction.",
        "shield",
        anchor_id="model-status",
    )
    status_columns = st.columns(3)

    all_model_names = ["Régression logistique", "Random forest", "XGBoost"]
    for idx, model_name in enumerate(all_model_names):
        with status_columns[idx]:
            available = model_name in models
            chip_class = "status-chip-ok" if available else "status-chip-warn"
            status_label = "Disponible" if available else "Indisponible"
            detail = (
                "Le modèle est prêt pour la prédiction."
                if available
                else "Le chargement a échoué sur cette machine."
            )
            st.markdown(
                f"""
                <div class="status-card">
                    <div class="status-head">
                        <div class="status-icon-wrap">
                            <span class="status-icon">{icon_svg("model")}</span>
                        </div>
                        <span class="status-chip {chip_class}">{status_label}</span>
                    </div>
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
    render_section_header(
        "Métriques de test",
        f"Le modèle principal retenu est {metrics['modele_recommande']}. Le critère prioritaire reste la capacité à repérer les clients qui risquent réellement de partir.",
        "metrics",
        anchor_id="metrics-section",
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

    render_section_header(
        "Décision recommandée",
        "La décision affichée ici provient du modèle principal retenu pour le projet. Elle sert de lecture prioritaire, tout en conservant la comparaison avec les deux autres modèles.",
        "target",
        anchor_id="decision-section",
    )
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

    render_section_header(
        "Comparaison des modèles",
        "Cette vue permet de confronter directement les trois sorties sur le même client afin d'expliquer plus facilement le choix du modèle principal.",
        "compare",
        anchor_id="comparison-section",
    )
    result_columns = st.columns(len(display_df))

    for column_container, (_, row) in zip(result_columns, display_df.iterrows()):
        model_name = row["modele"]
        model_metrics = metrics["metriques_test"].get(model_name, {})
        with column_container:
            st.markdown(
                build_comparison_card(
                    row=row,
                    model_metrics=model_metrics,
                    is_recommended=model_name == recommended_model,
                ),
                unsafe_allow_html=True,
            )


def render_batch_predictions(raw_df: pd.DataFrame, predictions_df: pd.DataFrame, metrics: dict) -> None:
    render_section_header(
        "Résultats du fichier CSV",
        "Après traitement du fichier, l'application affiche une synthèse globale, la répartition des décisions recommandées et un tableau exportable.",
        "table",
        anchor_id="csv-results",
    )
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
        st.table(schema_df)


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


def get_query_param(name: str, default: str) -> str:
    if hasattr(st, "query_params"):
        value = st.query_params.get(name, default)
    else:
        value = st.experimental_get_query_params().get(name, [default])
    if isinstance(value, list):
        return value[0] if value else default
    return value or default


def set_query_param(name: str, value: str) -> None:
    if hasattr(st, "query_params"):
        st.query_params[name] = value
    else:
        st.experimental_set_query_params(**{name: value})


def main() -> None:
    inject_css()
    metadata = get_preprocessing_metadata()
    metrics = get_model_metrics()
    models, load_errors = get_models_status()
    render_navbar(metrics, models)
    notice_container = st.empty()
    render_completion_notice(notice_container)

    render_hero(metrics, models, load_errors)
    render_model_status(models, load_errors)
    render_metrics(metrics)

    view_options = {
        "manuel": "Prédiction manuelle",
        "csv": "Prédiction par CSV",
    }
    current_view_key = get_query_param("vue", "manuel")
    if current_view_key not in view_options:
        current_view_key = "manuel"

    st.markdown(
        '<p class="view-switch-copy">Choisis la vue à afficher. Le menu de la barre supérieure pointe aussi directement vers ces deux entrées.</p>',
        unsafe_allow_html=True,
    )
    selected_view_label = st.radio(
        "Choix de la vue",
        options=list(view_options.values()),
        index=list(view_options.keys()).index(current_view_key),
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_view_key = next(
        key for key, label in view_options.items() if label == selected_view_label
    )
    if selected_view_key != current_view_key:
        set_query_param("vue", selected_view_key)

    if selected_view_key == "manuel":
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

    else:
        render_section_header(
            "Chargement d'un fichier CSV",
            "Téléverse un fichier structuré selon les colonnes brutes du projet. L'application contrôle le format avant d'exécuter les prédictions.",
            "upload",
            anchor_id="csv-entry",
        )
        render_csv_template_download(metadata)
        render_expected_schema(metadata)
        st.markdown(
            """
            <div class="upload-dropzone-copy">
                <strong>Dépose ton fichier CSV ici</strong>
                <span>Glisse-dépose le fichier ou clique dans la zone pour le sélectionner.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Dépose un fichier CSV",
            type=["csv"],
            label_visibility="collapsed",
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
