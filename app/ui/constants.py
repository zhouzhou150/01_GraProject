from __future__ import annotations

MODEL_LIBRARY = {
    "cnn_ctc": {
        "label": "CNN-CTC",
        "desc": "轻量代理基线，用于快速对照，不代表真实 CNN-CTC 声学模型能力。",
        "accent": "#c96a2b",
    },
    "rnn_ctc": {
        "label": "RNN-CTC",
        "desc": "时序代理基线，用于快速对照，不代表真实 RNN-CTC 声学模型能力。",
        "accent": "#6a8f7a",
    },
    "faster_whisper": {
        "label": "Faster-Whisper",
        "desc": "精度与速度平衡，可切换 Whisper 规模。",
        "accent": "#2f6c8f",
    },
    "paddlespeech": {
        "label": "PaddleSpeech",
        "desc": "中文语音生态友好，适合工程对比。",
        "accent": "#8f5a44",
    },
}

OPTION_LABELS = {
    "decoder": "解码策略",
    "frontend": "前端配置",
    "hidden_size": "隐藏层规模",
    "dropout": "Dropout",
    "model_size": "Whisper 规模",
    "compute_type": "计算精度",
    "lang": "语言",
    "postprocess": "后处理",
}

SUMMARY_COLUMN_MAP = {
    "model_label": "模型",
    "runtime_mode": "运行模式",
    "backend": "后端",
    "sample_count": "样本数",
    "cer": "CER",
    "wer": "WER",
    "ser": "SER",
    "semdist": "SemDist",
    "avg_latency_ms": "平均延迟(ms)",
    "p95_latency_ms": "P95 延迟(ms)",
    "avg_upl_ms": "UPL(ms)",
    "avg_rtf": "RTF",
    "throughput": "吞吐量",
    "cpu_pct": "CPU(%)",
    "mem_mb": "内存(MB)",
    "load_time_ms": "加载时间(ms)",
    "robustness_score": "鲁棒性得分",
    "resource_score": "资源效率得分",
    "uss": "USS",
    "satisfaction_level": "满意度等级",
}

SAMPLE_COLUMN_MAP = {
    "model_label": "模型",
    "runtime_mode": "运行模式",
    "backend": "后端",
    "sample_id": "样本 ID",
    "pred_text": "识别文本",
    "ref_text": "参考文本",
    "latency_ms": "延迟(ms)",
    "upl_ms": "UPL(ms)",
    "rtf": "RTF",
    "throughput": "吞吐量",
    "cpu_pct": "CPU(%)",
    "mem_mb": "内存(MB)",
    "load_time_ms": "加载时间(ms)",
    "cer": "CER",
    "wer": "WER",
    "ser": "SER",
    "semdist": "SemDist",
    "scene_tag": "场景",
    "noise_tag": "噪声",
    "status": "状态",
    "error_message": "错误信息",
}

METRIC_SPECS = [
    {"key": "cer", "label": "CER", "hint": "字符错误率，越低越好", "fmt": ".4f", "color": "#c96a2b"},
    {"key": "wer", "label": "WER", "hint": "词错误率，越低越好", "fmt": ".4f", "color": "#bc7a45"},
    {"key": "ser", "label": "SER", "hint": "句错误率，越低越好", "fmt": ".4f", "color": "#b35a42"},
    {"key": "semdist", "label": "SemDist", "hint": "语义距离，越高越好", "fmt": ".2f", "color": "#648d7c"},
    {"key": "avg_latency_ms", "label": "平均延迟(ms)", "hint": "越低越好", "fmt": ".2f", "color": "#2f6c8f"},
    {"key": "p95_latency_ms", "label": "P95 延迟(ms)", "hint": "越低越好", "fmt": ".2f", "color": "#497f9d"},
    {"key": "avg_upl_ms", "label": "UPL(ms)", "hint": "越低越好", "fmt": ".2f", "color": "#7398ad"},
    {"key": "avg_rtf", "label": "RTF", "hint": "越低越好", "fmt": ".4f", "color": "#486d7d"},
    {"key": "throughput", "label": "吞吐量", "hint": "越高越好", "fmt": ".2f", "color": "#6279a0"},
    {"key": "load_time_ms", "label": "加载时间(ms)", "hint": "越低越好", "fmt": ".2f", "color": "#7d6748"},
    {"key": "cpu_pct", "label": "CPU(%)", "hint": "越低越好", "fmt": ".2f", "color": "#8e7a63"},
    {"key": "mem_mb", "label": "内存(MB)", "hint": "越低越好", "fmt": ".2f", "color": "#9a856d"},
    {"key": "robustness_score", "label": "鲁棒性得分", "hint": "越高越好", "fmt": ".2f", "color": "#6a8f7a"},
    {"key": "resource_score", "label": "资源效率得分", "hint": "越高越好", "fmt": ".2f", "color": "#758f68"},
    {"key": "uss", "label": "USS", "hint": "越高越好", "fmt": ".2f", "color": "#a35d32"},
]

LOWER_IS_BETTER = {
    "cer",
    "wer",
    "ser",
    "avg_latency_ms",
    "p95_latency_ms",
    "avg_upl_ms",
    "avg_rtf",
    "load_time_ms",
    "cpu_pct",
    "mem_mb",
}

SESSION_DEFAULTS = {
    "dataset_items": [],
    "dataset_name": "",
    "dataset_label": "尚未加载数据集",
    "dataset_issues": [],
    "loaded_models": {},
    "loaded_adapters": {},
    "performance_report": None,
    "overall_report": None,
    "overall_exports": None,
    "flash_notice": None,
    "upload_widget_nonce": 0,
}
