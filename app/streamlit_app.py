from __future__ import annotations

from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from asr_eval_system.service import list_saved_experiments
from ui.layout import render_hero, render_sidebar
from ui.sections.charts import render_chart_section
from ui.sections.dataset import render_dataset_section
from ui.sections.evaluation import render_evaluation_section
from ui.sections.models import render_model_section
from ui.sections.results import render_results_section
from ui.state import ensure_session_defaults, render_flash_notice
from ui.styles import inject_styles


st.set_page_config(page_title="ASR 多模型评测工作台", page_icon=":studio_microphone:", layout="wide")


def main() -> None:
    ensure_session_defaults()
    inject_styles()

    history_count = len(list_saved_experiments())
    render_sidebar(history_count)
    render_hero(history_count)
    render_flash_notice()

    render_dataset_section()
    st.divider()
    render_model_section()
    st.divider()
    render_evaluation_section()
    st.divider()
    render_results_section()
    st.divider()
    render_chart_section()


if __name__ == "__main__":
    main()
