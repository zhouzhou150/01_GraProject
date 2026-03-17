from __future__ import annotations

from pathlib import Path

import streamlit as st

from asr_eval_system.data.audio_utils import (
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_TRANSCRIPT_EXTENSIONS,
    audio_player_format,
    expand_uploaded_archives,
    normalize_uploaded_name,
    resolve_transcript_text,
)
from ui.helpers import (
    build_sidecar_transcript_map,
    dataset_duration,
    dataset_preview_frame,
    load_demo_dataset_with_progress,
    merge_uploaded_entries,
    save_uploaded_dataset,
    section_header,
)
from ui.state import clear_upload_selection, reset_reports, set_flash_notice


def render_dataset_section() -> None:
    section_header(
        "01 / Dataset",
        "导入音频文件",
        "支持加载示例数据，也支持导入单个文件、多个文件或整个音频文件夹。音频会统一转换为 16k 单声道 WAV；参考文本支持 .txt、.lab、.trn。",
    )
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        quick_col, clear_col = st.columns([0.7, 0.3], gap="small")
        if quick_col.button("加载示例数据", type="primary", width="stretch"):
            items, issues = load_demo_dataset_with_progress()
            st.session_state["dataset_items"] = items if not issues else []
            st.session_state["dataset_name"] = "demo_manifest" if not issues else ""
            st.session_state["dataset_label"] = "示例数据集 / demo_manifest.json" if not issues else "示例数据集加载失败"
            st.session_state["dataset_issues"] = issues
            clear_upload_selection()
            reset_reports()
            if issues:
                st.error("示例数据校验未通过：" + "；".join(issues))
            else:
                set_flash_notice("success", "示例数据已加载，可以继续配置模型。")
                st.rerun()

        if clear_col.button("清空数据集", width="stretch"):
            st.session_state["dataset_items"] = []
            st.session_state["dataset_name"] = ""
            st.session_state["dataset_label"] = "尚未加载数据集"
            st.session_state["dataset_issues"] = []
            clear_upload_selection()
            reset_reports()
            set_flash_notice("info", "已清空当前数据集。")
            st.rerun()

        upload_nonce = int(st.session_state["upload_widget_nonce"])
        selected_audio_files = st.file_uploader(
            "上传音频文件",
            type=[suffix.lstrip(".") for suffix in SUPPORTED_AUDIO_EXTENSIONS],
            accept_multiple_files=True,
            key=f"audio-upload-files-{upload_nonce}",
        )
        selected_audio_directory = st.file_uploader(
            "或导入音频文件夹",
            type=[suffix.lstrip(".") for suffix in SUPPORTED_AUDIO_EXTENSIONS],
            accept_multiple_files="directory",
            key=f"audio-upload-directory-{upload_nonce}",
        )
        selected_audio_archives = st.file_uploader(
            "或上传音频 ZIP 压缩包",
            type=["zip"],
            accept_multiple_files=True,
            key=f"audio-upload-archives-{upload_nonce}",
        )
        archived_audio_files, audio_archive_issues = expand_uploaded_archives(
            selected_audio_archives,
            SUPPORTED_AUDIO_EXTENSIONS,
        )

        selected_text_files = st.file_uploader(
            "可选：上传参考文本文件",
            type=[suffix.lstrip(".") for suffix in SUPPORTED_TRANSCRIPT_EXTENSIONS],
            accept_multiple_files=True,
            key=f"text-upload-files-{upload_nonce}",
        )
        selected_text_directory = st.file_uploader(
            "或导入参考文本文件夹",
            type=[suffix.lstrip(".") for suffix in SUPPORTED_TRANSCRIPT_EXTENSIONS],
            accept_multiple_files="directory",
            key=f"text-upload-directory-{upload_nonce}",
        )
        selected_text_archives = st.file_uploader(
            "或上传参考文本 ZIP 压缩包",
            type=["zip"],
            accept_multiple_files=True,
            key=f"text-upload-archives-{upload_nonce}",
        )
        archived_text_files, text_archive_issues = expand_uploaded_archives(
            selected_text_archives,
            SUPPORTED_TRANSCRIPT_EXTENSIONS,
        )

        uploaded_audio_files = merge_uploaded_entries(
            selected_audio_files,
            selected_audio_directory,
            archived_audio_files,
        )
        uploaded_text_files = merge_uploaded_entries(
            selected_text_files,
            selected_text_directory,
            archived_text_files,
        )
        transcript_defaults = build_sidecar_transcript_map(uploaded_text_files)

        for issue in [*audio_archive_issues, *text_archive_issues]:
            st.warning(issue)

        if selected_audio_directory or selected_text_directory:
            st.caption("如果目录上传时出现大量 Network Error，建议改用 ZIP 压缩包批量导入，稳定性会更好。")

        if uploaded_audio_files:
            st.caption("支持批量导入。若上传了同名 .txt/.lab/.trn，系统会自动回填；你也可以继续手动修改。")
            for audio_file in uploaded_audio_files:
                transcript_key = f"transcript::{upload_nonce}::{audio_file.name}"
                default_text = resolve_transcript_text(
                    transcript_defaults,
                    normalize_uploaded_name(audio_file.name),
                )
                if transcript_key not in st.session_state:
                    st.session_state[transcript_key] = default_text
                elif default_text and not str(st.session_state.get(transcript_key, "")).strip():
                    st.session_state[transcript_key] = default_text
                st.text_area(
                    f"{audio_file.name} 的参考文本",
                    key=transcript_key,
                    height=78,
                )
        else:
            st.caption("先选择一个或多个音频文件，或直接导入整个音频文件夹，再执行导入。")

        upload_has_selection = bool(
            selected_audio_files
            or selected_audio_directory
            or selected_audio_archives
            or selected_text_files
            or selected_text_directory
            or selected_text_archives
        )
        upload_action_cols = st.columns([0.68, 0.32], gap="small")
        submit_uploaded = upload_action_cols[0].button(
            "导入上传音频",
            width="stretch",
            disabled=not bool(uploaded_audio_files),
        )
        clear_selected_uploads = upload_action_cols[1].button(
            "清空已选文件",
            width="stretch",
            disabled=not upload_has_selection,
        )

        if clear_selected_uploads:
            clear_upload_selection()
            set_flash_notice("info", "已清空当前上传选择。")
            st.rerun()

        if submit_uploaded:
            if not uploaded_audio_files:
                st.warning("请先选择至少一个音频文件。")
            else:
                transcript_lookup = {
                    normalize_uploaded_name(audio_file.name): st.session_state.get(
                        f"transcript::{upload_nonce}::{audio_file.name}",
                        "",
                    ).strip()
                    for audio_file in uploaded_audio_files
                }
                try:
                    items, issues, batch_name = save_uploaded_dataset(uploaded_audio_files, transcript_lookup)
                except Exception as exc:
                    st.session_state["dataset_items"] = []
                    st.session_state["dataset_name"] = ""
                    st.session_state["dataset_label"] = "上传数据集导入失败"
                    st.session_state["dataset_issues"] = [str(exc)]
                    reset_reports()
                    st.error(f"导入失败：{exc}")
                else:
                    st.session_state["dataset_items"] = items if not issues else []
                    st.session_state["dataset_name"] = batch_name if not issues else ""
                    st.session_state["dataset_label"] = f"上传数据集 / {batch_name}" if not issues else "上传数据集校验失败"
                    st.session_state["dataset_issues"] = issues
                    reset_reports()
                    if issues:
                        st.error("上传数据校验未通过：" + "；".join(issues))
                    else:
                        clear_upload_selection()
                        set_flash_notice("success", f"已导入 {len(items)} 个音频样本。")
                        st.rerun()

    with right:
        issues = st.session_state["dataset_issues"]
        st.markdown(
            f"""
            <div class="card">
              <h4>数据集概览</h4>
              <p><strong>来源：</strong>{st.session_state["dataset_label"]}</p>
              <p><strong>样本数量：</strong>{len(st.session_state["dataset_items"])}</p>
              <p><strong>总时长：</strong>{dataset_duration(st.session_state["dataset_items"]):.2f} 秒</p>
              <p><strong>校验状态：</strong>{'通过' if st.session_state["dataset_items"] and not issues else '待处理'}</p>
              <p class="muted">如果要跑完整实验，建议先确保所有参考文本都已填写完整。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state["dataset_items"]:
        st.dataframe(dataset_preview_frame(st.session_state["dataset_items"]), width="stretch", hide_index=True)
        with st.expander("试听与文本预览", expanded=False):
            for item in st.session_state["dataset_items"][:3]:
                audio_col, text_col = st.columns([0.7, 1.3], gap="large")
                with audio_col:
                    st.audio(Path(item.audio_path).read_bytes(), format=audio_player_format(item.audio_path))
                with text_col:
                    st.markdown(f"**文件名**：`{Path(item.audio_path).name}`")
                    st.markdown(f"**参考文本**：{item.transcript}")
                    st.caption(f"场景：{item.scene_tag} | 噪声：{item.noise_tag} | 时长：{item.duration_sec:.2f}s")
    else:
        st.info("当前还没有可用于评测的音频数据。可以先加载示例数据，或者上传自己的音频样本/文件夹。")
