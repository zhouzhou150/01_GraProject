# 深度学习语音识别性能测度系统

这是一个面向本科毕业设计的语音识别性能测度系统样例工程，围绕中文语料下的多模型对比评测展开，提供统一的模型适配接口、指标体系、Web 端、桌面端、SQLite 存储和报告导出流程。

## 推荐环境

- Python `3.11`
- Windows 11
- CUDA 可选，非必需

## 快速开始

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .[app]
python scripts/generate_demo_dataset.py
python scripts/run_demo_experiment.py
streamlit run app/streamlit_app.py
```

## 说明

- 工程默认支持“真实模型后端 + 模拟后端”两种模式。
- 当 `faster-whisper`、`PaddleSpeech` 等依赖未安装时，系统会退回到模拟模式，保证流程仍可演示。
- 每个模块开发后都应立即补对应测试并执行，避免把问题堆到最后。

