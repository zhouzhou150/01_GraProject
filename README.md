# ASR 多模型评测工作台

这是一个面向毕业设计/课程项目的 ASR 多模型评测系统，提供：

- 示例数据集生成与导入
- 单文件、多文件、文件夹、ZIP 压缩包批量导入音频
- `.txt`、`.lab`、`.trn` 参考文本匹配
- 多模型加载、性能测试、总体测试
- JSON / CSV / Markdown 报告导出
- Streamlit 可视化工作台

## 推荐环境

- 操作系统：Windows 11
- Python：`3.11`
- 环境管理：`conda`

说明：

- `Python 3.13` 只建议用于界面联调，不建议用于真实模型测试。
- 真实加载 `faster-whisper` 和 `PaddleSpeech` 时，推荐使用 `Python 3.11`。
- `PaddleSpeech 1.5.x` 在当前项目里建议搭配 `paddlepaddle 2.6.x`，不要安装 `3.x`。

## 首次安装

### PowerShell

```powershell
conda create -n asr-eval311 python=3.11 -y
conda activate asr-eval311
Set-Location F:\01_GraProject

python -m pip install --upgrade pip
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .[app]
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple faster-whisper
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "paddlepaddle>=2.6,<3"
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-deps paddlespeech
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple psutil scipy soundfile

python scripts/generate_demo_dataset.py
streamlit run app/streamlit_app.py
```

注意：

- 在 `PowerShell` 里不要使用 `cd /d F:\01_GraProject`，那是 `cmd` 语法。
- 正确写法是 `Set-Location F:\01_GraProject` 或 `cd F:\01_GraProject`。
- Windows 下安装 `paddlespeech` 时，`webrtcvad` 可能因为本地 C++ 编译环境缺失而失败；当前项目的基础测试可先按上面的 `--no-deps paddlespeech` 方式安装。

## 后续再次运行

如果环境已经装好，后面每次启动只需要：

```powershell
conda activate asr-eval311
Set-Location F:\01_GraProject
streamlit run app/streamlit_app.py
```

如果需要重新生成示例数据：

```powershell
conda activate asr-eval311
Set-Location F:\01_GraProject
python scripts/generate_demo_dataset.py
```

如果需要下载一个更接近真实批量导入场景的小型公开数据集：

```powershell
conda activate asr-eval311
Set-Location F:\01_GraProject
python scripts/download_yesno_dataset.py
```

下载完成后，数据会放在：

- `data/external/yesno/waves_yesno/`
- 清单文件：`data/manifests/yesno_manifest.json`

## 常用检查命令

检查环境是否存在：

```powershell
conda env list
```

确认当前 Python 版本：

```powershell
conda activate asr-eval311
python -V
```

检查关键依赖是否装在当前环境里：

```powershell
conda activate asr-eval311
Set-Location F:\01_GraProject
python -m pip show streamlit
python -m pip show faster-whisper
python -m pip show paddlespeech
python -m pip show paddlepaddle
python -m pip show numpy
```

建议检查结果：

- `Python 3.11.x`
- `numpy 1.26.x`
- `paddlepaddle 2.6.x`

## 下载慢时如何加速

临时使用清华镜像：

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

永久设置 `pip` 镜像：

```powershell
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

如果也想给 `conda` 换源：

```powershell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/
conda config --set show_channel_urls yes
```

## 常见问题

### 1. `EnvironmentNameNotFound`

说明当前 `conda` 里没有这个环境。先执行：

```powershell
conda env list
```

如果没有 `asr-eval311`，重新创建即可。

### 2. README 看起来像“无法修改”或显示乱码

这通常不是文件损坏，而是编码显示问题：

- `README.md` 文件本身使用 `UTF-8`
- 某些终端会用错误编码显示中文
- IDE 如果不是按 `UTF-8` 打开，也会看起来异常

建议：

- 在 IDE 中把文件编码切到 `UTF-8`
- 优先使用 Windows Terminal / PowerShell

### 3. 真实模型是否真的加载了

当前系统已经区分了：

- 请求模式：你选择的是“真实”还是“模拟”
- 实际模式：模型最终到底是“真实”还是“模拟”
- 后端信息：例如 `faster-whisper`、`paddlespeech`

如果真实依赖缺失，真实模式下会直接报错，不会再悄悄回退到模拟模式。

## 项目入口

- Streamlit 应用：[app/streamlit_app.py](/F:/01_GraProject/app/streamlit_app.py)
- 示例数据生成脚本：[scripts/generate_demo_dataset.py](/F:/01_GraProject/scripts/generate_demo_dataset.py)
- 评测流程模型：[src/asr_eval_system/workflow.py](/F:/01_GraProject/src/asr_eval_system/workflow.py)
