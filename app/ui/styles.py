from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --page-bg: #f4eee4;
          --page-bg-deep: #ede1d0;
          --paper: rgba(255, 250, 243, 0.94);
          --paper-strong: rgba(255, 248, 238, 0.98);
          --ink: #231912;
          --ink-soft: #65584d;
          --ink-faint: #8c7765;
          --line: rgba(111, 86, 60, 0.16);
          --accent: #c96a2b;
          --accent-deep: #a84f20;
          --shadow: 0 20px 55px rgba(76, 54, 31, 0.10);
        }

        .stApp {
          background:
            radial-gradient(circle at top left, rgba(221, 170, 118, 0.20), transparent 30%),
            radial-gradient(circle at right 14%, rgba(86, 136, 164, 0.10), transparent 22%),
            linear-gradient(180deg, var(--page-bg) 0%, var(--page-bg-deep) 100%);
          color: var(--ink);
        }

        .block-container {
          max-width: 1280px;
          padding-top: 2.35rem;
          padding-bottom: 4rem;
        }

        [data-testid="stAppViewContainer"] .main,
        [data-testid="stAppViewContainer"] .main p,
        [data-testid="stAppViewContainer"] .main label,
        [data-testid="stAppViewContainer"] .main span,
        [data-testid="stAppViewContainer"] .main li,
        [data-testid="stAppViewContainer"] .main div {
          color: var(--ink);
        }

        h1, h2, h3 {
          font-family: "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
          color: var(--ink) !important;
          letter-spacing: -0.02em;
        }

        .hero {
          border: 1px solid var(--line);
          border-radius: 34px;
          padding: 2.6rem 2.4rem 2.2rem;
          background: linear-gradient(135deg, rgba(255, 249, 240, 0.96), rgba(244, 232, 215, 0.92));
          box-shadow: var(--shadow);
          margin-bottom: 1.35rem;
        }

        .hero-kicker, .tag, .chip {
          display: inline-flex;
          align-items: center;
          border-radius: 999px;
        }

        .hero-kicker {
          padding: .35rem .75rem;
          background: rgba(255, 255, 255, .72);
          border: 1px solid var(--line);
          font-size: .82rem;
          color: #8b5d3d !important;
          letter-spacing: .08em;
          text-transform: uppercase;
        }

        .hero-title {
          margin: .95rem 0 .55rem;
          font-size: clamp(2.1rem, 3.8vw, 3.5rem);
          line-height: 1.02;
          color: var(--ink) !important;
          font-weight: 700;
          max-width: 860px;
        }

        .hero-copy, .section-copy, .muted {
          color: var(--ink-soft) !important;
          line-height: 1.78;
        }

        .tags {
          display: flex;
          flex-wrap: wrap;
          gap: .65rem;
        }

        .tag {
          padding: .48rem .78rem;
          background: rgba(255, 251, 246, .92);
          border: 1px solid var(--line);
          color: #57493d !important;
          font-size: .9rem;
        }

        .section-kicker {
          color: #8b5b3d !important;
          font-size: .84rem;
          font-weight: 700;
          letter-spacing: .12em;
          text-transform: uppercase;
        }

        .section-title {
          margin: .1rem 0 0;
          font-size: 1.75rem;
          color: var(--ink) !important;
          font-weight: 700;
        }

        .card, .chart-card {
          border: 1px solid var(--line);
          box-shadow: var(--shadow);
        }

        .card {
          border-radius: 24px;
          background: var(--paper);
          padding: 1rem 1rem .95rem;
        }

        .chart-card {
          border-radius: 24px;
          background: var(--paper-strong);
          padding: .95rem 1rem .35rem;
          margin-bottom: 1rem;
        }

        .card h4, .chart-title {
          margin: 0 0 .45rem;
          color: var(--ink) !important;
          font-weight: 700;
        }

        .model-card-title {
          margin: 0.15rem 0 0.45rem;
          color: var(--ink) !important;
          font-size: 1.05rem;
          font-weight: 700;
          line-height: 1.2;
        }

        .model-card-remove-form {
          margin: 0;
          display: flex;
          justify-content: flex-end;
        }

        .model-card-remove-link {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 2.45rem;
          min-width: 2.45rem;
          height: 2.45rem;
          min-height: 2.45rem;
          margin-left: auto;
          border-radius: 18px;
          background: linear-gradient(135deg, rgba(255, 248, 239, 0.98), rgba(244, 223, 198, 0.94));
          color: #8c4d27 !important;
          border: 1px solid rgba(137, 101, 71, 0.22);
          box-shadow: 0 10px 26px rgba(81, 56, 34, 0.12);
          text-decoration: none !important;
          appearance: none;
          cursor: pointer;
          padding: 0;
          font-size: 1.08rem;
          font-weight: 700;
          line-height: 1;
          font-family: inherit;
          opacity: 1 !important;
          transition: transform 120ms ease, background 120ms ease, color 120ms ease, border-color 120ms ease;
        }

        .model-card-remove-link:hover {
          background: linear-gradient(135deg, rgba(255, 244, 229, 1), rgba(240, 213, 183, 0.96));
          color: #773d1e !important;
          border-color: rgba(137, 101, 71, 0.3);
          transform: translateY(-1px);
        }

        .card p, .chart-hint {
          margin: .16rem 0;
          color: var(--ink-soft) !important;
          line-height: 1.65;
        }

        .card strong {
          color: var(--ink) !important;
        }

        .chip {
          padding: .34rem .65rem;
          background: rgba(255, 255, 255, .86);
          border: 1px solid var(--line);
          color: #54483d !important;
          font-size: .82rem;
          margin: .25rem .25rem 0 0;
        }

        .insight {
          border: 1px solid var(--line);
          border-radius: 28px;
          padding: 1.3rem 1.4rem;
          background: linear-gradient(180deg, rgba(255, 249, 241, 0.98), rgba(249, 239, 226, 0.92));
          box-shadow: var(--shadow);
        }

        .insight h3, .insight p, .insight strong {
          color: var(--ink) !important;
        }

        .overview-card {
          border: 1px solid var(--line);
          border-radius: 24px;
          padding: 1rem 1rem 1.1rem;
          background: rgba(255, 249, 241, 0.95);
          box-shadow: var(--shadow);
          min-height: 118px;
        }

        .overview-label {
          font-size: 0.92rem;
          color: var(--ink-faint) !important;
          margin-bottom: 0.55rem;
          font-weight: 600;
        }

        .overview-value {
          font-family: "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
          font-size: clamp(2rem, 3vw, 2.5rem);
          line-height: 1;
          color: var(--ink) !important;
          font-weight: 700;
          margin-bottom: 0.35rem;
          word-break: break-word;
        }

        .overview-meta {
          color: var(--ink-soft) !important;
          font-size: 0.9rem;
          line-height: 1.55;
        }

        .summary-card {
          border: 1px solid var(--line);
          border-radius: 24px;
          padding: 1rem 1rem 1.05rem;
          background: rgba(255, 249, 241, 0.96);
          box-shadow: var(--shadow);
          min-height: 172px;
        }

        .summary-card-label {
          color: var(--ink-faint) !important;
          font-size: .88rem;
          font-weight: 700;
          letter-spacing: .04em;
          margin-bottom: .5rem;
        }

        .summary-card-model {
          color: var(--ink) !important;
          font-size: 1.2rem;
          font-weight: 700;
          line-height: 1.4;
          margin-bottom: .35rem;
          word-break: break-word;
        }

        .summary-card-value {
          font-family: "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
          color: var(--ink) !important;
          font-size: clamp(1.8rem, 2.3vw, 2.35rem);
          line-height: 1.08;
          font-weight: 700;
          margin-bottom: .4rem;
          word-break: break-word;
        }

        .summary-card-meta {
          color: var(--ink-soft) !important;
          font-size: .92rem;
          line-height: 1.6;
        }

        div[data-testid="stMetric"] {
          background: var(--paper-strong);
          border: 1px solid var(--line);
          padding: 1rem 1rem .9rem;
          border-radius: 20px;
          box-shadow: var(--shadow);
        }

        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] div,
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] div {
          color: var(--ink) !important;
        }

        div[data-testid="stMetricValue"] {
          font-family: "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
          font-weight: 700;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button,
        button[data-testid^="stBaseButton-"] {
          border-radius: 999px;
          border: 1px solid var(--line);
          background: linear-gradient(135deg, #fff6ec, #f4dfc6);
          color: #5a3f2a !important;
          font-weight: 700;
          box-shadow: 0 10px 26px rgba(81, 56, 34, 0.12);
        }

        div[data-testid="stButton"] > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
          background: linear-gradient(135deg, var(--accent), var(--accent-deep));
          color: #fff !important;
          border-color: transparent;
        }

        div[data-testid="stDataFrame"] {
          border-radius: 20px;
          overflow: hidden;
          border: 1px solid var(--line);
        }

        [data-testid="stSidebarNav"] {
          display: none;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] * {
          color: var(--ink) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
          background: rgba(252, 247, 239, 0.92);
          border: 1px dashed rgba(155, 121, 90, 0.55);
        }

        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploaderDropzone"] button {
          background: rgba(255, 246, 235, 0.96) !important;
          color: #5a3f2a !important;
          border: 1px solid rgba(137, 101, 71, 0.22) !important;
          box-shadow: none !important;
        }

        button:disabled,
        button[disabled] {
          background: rgba(94, 79, 65, 0.12) !important;
          color: rgba(35, 25, 18, 0.52) !important;
          border-color: rgba(111, 86, 60, 0.14) !important;
          opacity: 1 !important;
        }

        [data-testid="stTextArea"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stMultiSelect"] label,
        [data-testid="stSlider"] label,
        [data-testid="stToggle"] label,
        [data-testid="stRadio"] label {
          color: var(--ink) !important;
          font-weight: 600;
        }

        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {
          background: rgba(255, 250, 243, 0.96) !important;
          color: var(--ink) !important;
          border-color: rgba(140, 119, 101, 0.24) !important;
        }

        [data-testid="stTabs"] button {
          color: var(--ink) !important;
        }

        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, #262731 0%, #23242e 100%);
        }

        section[data-testid="stSidebar"] * {
          color: #f3ede8 !important;
        }

        .sidebar-panel {
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 18px;
          padding: 0.95rem 0.95rem 0.85rem;
          background: rgba(255, 255, 255, 0.04);
          margin: 0.8rem 0 1rem;
        }

        .sidebar-title {
          font-size: 0.88rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #d8b28d !important;
          font-weight: 700;
          margin-bottom: 0.55rem;
        }

        .sidebar-step {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          margin: 0.6rem 0;
          color: #f4eee9 !important;
        }

        .sidebar-step--done .sidebar-step-index {
          background: rgba(216, 178, 141, 0.92);
          color: #322319 !important;
        }

        .sidebar-step-index {
          width: 1.55rem;
          height: 1.55rem;
          border-radius: 999px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: rgba(216, 178, 141, 0.16);
          color: #f2d1b4 !important;
          font-size: 0.78rem;
          font-weight: 700;
        }

        .sidebar-step-copy {
          flex: 1;
        }

        .sidebar-step-note {
          display: block;
          color: #cfc4bb !important;
          font-size: 0.78rem;
          line-height: 1.45;
          margin-top: 0.12rem;
        }

        .sidebar-meta {
          color: #d8d0c9 !important;
          font-size: 0.92rem;
          line-height: 1.65;
        }

        .sidebar-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
          margin-top: 0.75rem;
        }

        .sidebar-pill {
          display: inline-flex;
          align-items: center;
          padding: 0.35rem 0.65rem;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid rgba(255, 255, 255, 0.08);
          color: #f6f0ea !important;
          font-size: 0.82rem;
        }

        section[data-testid="stSidebar"] .stProgress > div > div > div > div {
          background: linear-gradient(90deg, var(--accent), #e9ad76);
        }

        section[data-testid="stSidebar"] code {
          color: #ffd9b6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
