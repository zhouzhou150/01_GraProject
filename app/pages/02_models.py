from __future__ import annotations

import streamlit as st


st.set_page_config(page_title="请使用首页工作台", page_icon=":house:")
st.info("模型选择与加载已经整合到首页工作台中。")
st.page_link("streamlit_app.py", label="返回首页工作台", icon=":material/home:")
