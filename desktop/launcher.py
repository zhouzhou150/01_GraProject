from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_FILE = ROOT / "app" / "streamlit_app.py"


def wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.3)
    return False


def start_streamlit() -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP_FILE),
            "--server.headless=true",
            "--server.port=8501",
        ],
        cwd=str(ROOT),
    )


def main() -> None:
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QStatusBar, QToolBar
        from PySide6.QtGui import QAction
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError:
        print("缺少 PySide6，请执行: python -m pip install -e .[app]")
        return

    streamlit_process = start_streamlit()
    if not wait_for_port("127.0.0.1", 8501):
        streamlit_process.terminate()
        print("Streamlit 服务启动失败。")
        return

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("语音识别性能测度系统")
    window.resize(1280, 860)

    browser = QWebEngineView()
    browser.setUrl(QUrl("http://127.0.0.1:8501"))
    window.setCentralWidget(browser)

    toolbar = QToolBar("控制栏")
    window.addToolBar(toolbar)

    refresh_action = QAction("刷新", window)
    refresh_action.triggered.connect(browser.reload)
    toolbar.addAction(refresh_action)

    open_action = QAction("打开主页", window)
    open_action.triggered.connect(lambda: browser.setUrl(QUrl("http://127.0.0.1:8501")))
    toolbar.addAction(open_action)

    status_bar = QStatusBar()
    status_bar.showMessage("本地服务已启动：http://127.0.0.1:8501")
    window.setStatusBar(status_bar)

    def on_close() -> None:
        if streamlit_process.poll() is None:
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                streamlit_process.kill()

    app.aboutToQuit.connect(on_close)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
