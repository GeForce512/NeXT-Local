# -*- coding: utf-8 -*-
import sys, os, subprocess, json, time, shutil, requests, site, threading
from pathlib import Path
from PyQt5.QtCore import QUrl, QObject, pyqtSlot, pyqtSignal, QThread, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile
from PyQt5.QtWebChannel import QWebChannel


def get_base_dir():
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(
        os.path.abspath(__file__))


BASE_DIR = get_base_dir()


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(BASE_DIR, relative_path)


# ================= 智能环境路由 =================
def get_python_exe():
    if getattr(sys, 'frozen', False):
        py_path = os.path.join(BASE_DIR, 'python310', 'python.exe')
        if os.path.exists(py_path):
            return py_path
    return sys.executable


def get_site_packages():
    try:
        return site.getsitepackages()[0]
    except Exception:
        return os.path.join(os.path.dirname(get_python_exe()), 'Lib', 'site-packages')


# ================= 1. 环境检测线程 =================
class EnvCheckWorker(QThread):
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)

    def run(self):
        self.log.emit('🔍 开始进行环境深度体检...')
        base_dir = get_base_dir()
        python_exe = get_python_exe()
        site_packages = get_site_packages()
        env_script = os.path.join(base_dir, 'env_setup.py')

        if not os.path.exists(python_exe):
            self.log.emit(f'❌ 找不到 Python 解释器: {python_exe}')
            self.finished_signal.emit(False)
            return

        self.log.emit(f'✅ 核心解释器校验通过: {os.path.basename(python_exe)}')

        torch_path = os.path.join(site_packages, 'torch')
        if not os.path.exists(torch_path):
            self.log.emit('⚠️ 检测到 AI 核心库 (torch) 缺失！')
            if 'venv' in python_exe or python_exe == sys.executable:
                self.log.emit('💡 当前为开发环境，请在终端运行: pip install torch transformers')
                self.finished_signal.emit(False)
                return

            if not os.path.exists(env_script):
                self.log.emit(f'❌ 找不到修复脚本: {env_script}')
                self.finished_signal.emit(False)
                return

            try:
                env_vars = os.environ.copy()
                env_vars['PYTHONIOENCODING'] = 'utf-8'
                process = subprocess.Popen(
                    [python_exe, '-u', env_script],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    text=True, encoding='utf-8', errors='ignore', cwd=base_dir, env=env_vars
                )
                for line in process.stdout:
                    if line.strip(): self.log.emit(line.strip())
                process.wait()
                if process.returncode != 0:
                    self.log.emit(f'❌ 环境修复失败，退出码: {process.returncode}')
                    self.finished_signal.emit(False)
                    return
                self.log.emit('🎉 环境自动下载与修复完成！')
            except Exception as e:
                self.log.emit(f'❌ 调用异常: {str(e)}')
                self.finished_signal.emit(False)
                return
        else:
            self.log.emit('✅ AI 核心库校验通过 (torch 已存在)')

        self.log.emit('🚀 正在进行最终的 GPU 验证...')
        verify_script = """
import sys, os, importlib
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"🟢 GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    else: 
        print("🟡 CUDA 不可用")
    print("VERIFY_DONE")
except Exception as e: 
    print(f"❌ 验证崩溃: {str(e)[:50]}")
"""
        try:
            env_vars = os.environ.copy()
            env_vars['PYTHONIOENCODING'] = 'utf-8'
            process = subprocess.Popen([python_exe, '-u', '-c', verify_script], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore',
                                       cwd=base_dir, env=env_vars)
            for line in process.stdout:
                if line.strip(): self.log.emit(line.strip())
            self.log.emit('🎉 系统已完全就绪。')
        except Exception as e:
            self.log.emit(f'❌ 验证异常: {str(e)}')
            self.finished_signal.emit(False)
            return
        self.finished_signal.emit(True)


# ================= 2. 模型下载线程 =================
class ModelDownloadWorker(QThread):
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, alias, base_dir):
        super().__init__()
        self.alias = alias
        self.base_dir = base_dir

    def run(self):
        model_map = {
            'Qwen3.5-0.8B': 'Qwen/Qwen3.5-0.8B',
            'Qwen3.5-4B': 'Qwen/Qwen3.5-4B',
            'DeepSeek-1.5B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
        }
        repo_id = model_map.get(self.alias)
        if not repo_id:
            self.log.emit(f'❌ 未知的模型别名: {self.alias}')
            self.finished_signal.emit(False, self.alias)
            return

        python_exe = get_python_exe()
        models_dir = os.path.join(self.base_dir, 'models')
        cache_dir = os.path.join(models_dir, self.alias)

        dl_script = f"""
import sys, os, time
os.environ["HF_HOME"] = "D:/AI_Cache/huggingface"
os.environ["MODELSCOPE_CACHE"] = "D:/AI_Cache/modelscope"
from modelscope import snapshot_download
repo_id = '{repo_id}'
cache_dir = r'{cache_dir}'
alias = '{self.alias}'
os.makedirs(cache_dir, exist_ok=True)
print(f"🚀 [{{alias}}] 开始从 ModelScope 下载...", flush=True)
print(f"📂 存储目录: {{cache_dir}}", flush=True)
retries = 3
for attempt in range(1, retries + 1):
    try:
        start = time.time()
        model_dir = snapshot_download(repo_id, cache_dir=cache_dir, revision='master')
        elapsed = time.time() - start
        mins, secs = divmod(int(elapsed), 60)
        print(f"🎉 [{{alias}}] 下载完成，耗时 {{mins}}分{{secs}}秒", flush=True)
        print(f"📂 路径: {{model_dir}}", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"⚠️ [{{alias}}] 第{{attempt}}次失败: {{e}}", flush=True)
        if attempt < retries:
            wait = 5 * attempt
            print(f"⏳ 等待 {{wait}} 秒后重试...", flush=True)
            time.sleep(wait)
print(f"❌ [{{alias}}] 所有重试均失败！", flush=True)
sys.exit(1)
"""
        try:
            env_vars = os.environ.copy()
            env_vars['PYTHONIOENCODING'] = 'utf-8'
            process = subprocess.Popen(
                [python_exe, '-u', '-c', dl_script],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding='utf-8',
                errors='ignore', cwd=self.base_dir, env=env_vars
            )

            for line in process.stdout:
                line = line.strip()
                if line: self.log.emit(line)

            process.wait()
            if process.returncode == 0:
                self.finished_signal.emit(True, self.alias)
            else:
                self.finished_signal.emit(False, self.alias)
        except Exception as e:
            self.log.emit(f'❌ 异常: {str(e)}')
            self.finished_signal.emit(False, self.alias)


# ================= 3. 训练线程 =================
class TrainWorker(QThread):
    log = pyqtSignal(str)
    metrics = pyqtSignal(str)
    finished_signal = pyqtSignal(bool)

    def __init__(self, config, base_dir):
        super().__init__()
        self.config = config
        self.base_dir = base_dir

    def run(self):
        try:
            # 动态导入训练模块
            train_script = os.path.join(self.base_dir, '训练.py')
            if not os.path.exists(train_script):
                self.log.emit('❌ 找不到训练脚本')
                self.finished_signal.emit(False)
                return

            # 在子进程中运行训练以避免阻塞 UI
            python_exe = get_python_exe()
            # 构建训练启动脚本
            config_json = json.dumps(self.config, ensure_ascii=False)
            runner = f"""
import sys, os, json
sys.path.insert(0, r'{self.base_dir}')
from 训练 import run_training

config = json.loads(r'''{config_json}''')

def log_fn(msg):
    print(msg, flush=True)

def metrics_fn(msg):
    print(msg, flush=True)

result = run_training(log_fn=log_fn, config=config, metrics_fn=metrics_fn)
if result is True:
    print("__TRAIN_OK__", flush=True)
else:
    print("__TRAIN_FAIL__", flush=True)
"""
            env_vars = os.environ.copy()
            env_vars['PYTHONIOENCODING'] = 'utf-8'
            process = subprocess.Popen(
                [python_exe, '-u', '-c', runner],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding='utf-8',
                errors='ignore', cwd=self.base_dir, env=env_vars
            )

            success = False
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                if line == '__TRAIN_OK__':
                    success = True
                    continue
                if line == '__TRAIN_FAIL__':
                    success = False
                    continue
                # 尝试解析为 metrics JSON
                if line.startswith('{') and '"loss"' in line:
                    self.metrics.emit(line)
                else:
                    self.log.emit(line)

            process.wait()
            self.finished_signal.emit(success)
        except Exception as e:
            self.log.emit(f'❌ 训练异常: {str(e)}')
            self.finished_signal.emit(False)


# ================= 4. QWebChannel 桥接类 =================
class ChatHandler(QObject):
    # ★ 前端期望的信号名称
    envLog = pyqtSignal(str)
    downloadLog = pyqtSignal(str)
    trainLog = pyqtSignal(str)
    dataLoaded = pyqtSignal(str)        # JSON: {type:'datasets'|'loras', data:[...]}
    datasetContent = pyqtSignal(str)     # 数据集编辑器内容
    sessionsLoaded = pyqtSignal(str)     # JSON 会话列表
    gpuDetectResult = pyqtSignal(str)    # JSON GPU 信息
    envDone = pyqtSignal(bool)          # 环境检测完成
    downloadDone = pyqtSignal(bool)     # 模型下载完成
    trainDone = pyqtSignal(bool)        # 训练完成
    trainMetrics = pyqtSignal(str)       # 训练指标 JSON (loss, lr, step, epoch)
    firstLaunchResult = pyqtSignal(bool) # 首次启动检测结果
    languageLoaded = pyqtSignal(str)      # 语言加载完成 (zh-CN / en)
    fileDialogResult = pyqtSignal(str)     # 文件选择结果 JSON: {files: [...]}
    windowStateChanged = pyqtSignal(bool)  # 窗口最大化状态变化

    def __init__(self, parent=None):
        super().__init__(parent)
        self.inference_process = None
        self.env_worker = None
        self.dl_worker = None
        self.train_worker = None

    # ---------- 首次启动检测 ----------
    @pyqtSlot()
    def isFirstLaunch(self):
        settings_path = os.path.join(BASE_DIR, 'UI设置.json')
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                if settings.get('setup_complete', False):
                    self.firstLaunchResult.emit(False)
                    return
        except Exception:
            pass
        self.firstLaunchResult.emit(True)

    @pyqtSlot()
    def completeSetup(self):
        settings_path = os.path.join(BASE_DIR, 'UI设置.json')
        try:
            settings = {}
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            settings['setup_complete'] = True
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.envLog.emit(f'保存设置失败: {e}')

    # ---------- 语言持久化 ----------
    @pyqtSlot(str)
    def saveLanguage(self, lang):
        settings_path = os.path.join(BASE_DIR, 'UI设置.json')
        try:
            settings = {}
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            settings['language'] = lang
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.envLog.emit(f'保存语言设置失败: {e}')

    @pyqtSlot()
    def loadLanguage(self):
        settings_path = os.path.join(BASE_DIR, 'UI设置.json')
        lang = 'zh-CN'
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                lang = settings.get('language', 'zh-CN')
        except Exception:
            pass
        self.languageLoaded.emit(lang)

    # ---------- 窗口控制 ----------
    @pyqtSlot()
    def minimizeWindow(self):
        win = self.parent()
        if win and isinstance(win, QMainWindow):
            win.showMinimized()

    @pyqtSlot()
    def closeWindow(self):
        win = self.parent()
        if win and isinstance(win, QMainWindow):
            win.close()

    @pyqtSlot()
    def maximizeWindow(self):
        win = self.parent()
        if win and isinstance(win, QMainWindow):
            win.showMaximized()
            self.windowStateChanged.emit(True)

    @pyqtSlot()
    def restoreWindow(self):
        win = self.parent()
        if win and isinstance(win, QMainWindow):
            win.showNormal()
            self.windowStateChanged.emit(False)

    @pyqtSlot(int, int)
    def moveWindow(self, x, y):
        win = self.parent()
        if win and isinstance(win, QMainWindow):
            win.move(x, y)

    @pyqtSlot(str)
    def openFileDialog(self, accept_filter):
        """打开原生文件选择对话框，通过 fileDialogResult 信号返回结果"""
        win = self.parent()
        try:
            files, _ = QFileDialog.getOpenFileNames(
                win, '选择文件', BASE_DIR, accept_filter or '所有文件 (*.*)'
            )
            self.fileDialogResult.emit(json.dumps({'files': files}))
        except Exception as e:
            self.fileDialogResult.emit(json.dumps({'files': [], 'error': str(e)}))

    @pyqtSlot()
    def importDatasetFiles(self):
        """打开原生文件选择器，直接导入数据集文件到 datasets 目录"""
        win = self.parent()
        try:
            files, _ = QFileDialog.getOpenFileNames(
                win, '选择数据集文件', BASE_DIR,
                '数据集文件 (*.jsonl *.txt *.csv);;所有文件 (*.*)'
            )
            if not files:
                return
            data_dir = os.path.join(BASE_DIR, 'datasets')
            os.makedirs(data_dir, exist_ok=True)
            imported = 0
            for fpath in files:
                try:
                    name = os.path.basename(fpath)
                    # 读取文件内容
                    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    # TXT 文件转换为 JSONL 格式
                    if name.lower().endswith('.txt'):
                        import re as _re
                        chunks = _re.split(r'\n\s*\n|(?<=[。！？])\s*', content)
                        chunks = [c for c in chunks if len(c.strip()) > 50]
                        lines = []
                        for c in chunks:
                            split_idx = len(c) * 3 // 10
                            lines.append(json.dumps({
                                'messages': [
                                    {'role': 'user', 'content': 'Please continue: ' + c[:split_idx]},
                                    {'role': 'assistant', 'content': c[split_idx:]}
                                ]
                            }, ensure_ascii=False))
                        content = '\n'.join(lines)
                        name = name.rsplit('.', 1)[0] + '.jsonl'
                    # 保存
                    dest = os.path.join(data_dir, name)
                    with open(dest, 'w', encoding='utf-8') as f:
                        f.write(content)
                    imported += 1
                except Exception as e:
                    self.envLog.emit(f'⚠️ 导入失败 {os.path.basename(fpath)}: {e}')
            if imported > 0:
                self.envLog.emit(f'✅ 成功导入 {imported} 个数据集文件')
            self.getDatasetList()
        except Exception as e:
            self.envLog.emit(f'❌ 导入数据集失败: {e}')

    # ---------- 会话持久化 ----------
    @pyqtSlot()
    def loadSessions(self):
        sessions_path = os.path.join(BASE_DIR, 'chat_history', 'sessions.json')
        try:
            if os.path.exists(sessions_path):
                with open(sessions_path, 'r', encoding='utf-8') as f:
                    self.sessionsLoaded.emit(f.read())
            else:
                self.sessionsLoaded.emit('[]')
        except Exception:
            self.sessionsLoaded.emit('[]')

    @pyqtSlot(str)
    def saveSessions(self, json_str):
        sessions_dir = os.path.join(BASE_DIR, 'chat_history')
        os.makedirs(sessions_dir, exist_ok=True)
        try:
            with open(os.path.join(sessions_dir, 'sessions.json'), 'w', encoding='utf-8') as f:
                f.write(json_str)
        except Exception as e:
            self.envLog.emit(f'保存会话失败: {e}')

    # ---------- GPU 检测 ----------
    @pyqtSlot()
    def detectGpuInfo(self):
        """检测 GPU 信息并通过 gpuDetectResult 信号返回 JSON"""
        def _detect():
            python_exe = get_python_exe()
            script = """
import json, sys
result = {"gpu_name": None, "vram_gb": None, "torch_version": None, "cuda_version": None, "has_torch": False}
try:
    import torch
    result["has_torch"] = True
    result["torch_version"] = torch.__version__
    result["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        result["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        result["vram_gb"] = round(props.total_mem / 1024**3, 1)
except Exception:
    pass
print(json.dumps(result), flush=True)
"""
            try:
                env_vars = os.environ.copy()
                env_vars['PYTHONIOENCODING'] = 'utf-8'
                process = subprocess.Popen(
                    [python_exe, '-u', '-c', script],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, text=True, encoding='utf-8',
                    errors='ignore', cwd=BASE_DIR, env=env_vars
                )
                output = ''
                for line in process.stdout:
                    output += line.strip()
                process.wait()
                # 解析 JSON
                try:
                    self.gpuDetectResult.emit(output.strip())
                except Exception:
                    self.gpuDetectResult.emit(json.dumps({"gpu_name": None, "has_torch": False}))
            except Exception:
                self.gpuDetectResult.emit(json.dumps({"gpu_name": None, "has_torch": False}))

        threading.Thread(target=_detect, daemon=True).start()

    # ---------- 环境检测 ----------
    @pyqtSlot()
    def checkEnv(self):
        if self.env_worker and self.env_worker.isRunning(): return
        self.env_worker = EnvCheckWorker()
        self.env_worker.log.connect(self.envLog.emit)
        self.env_worker.finished_signal.connect(self._on_env_done)
        self.env_worker.start()

    def _on_env_done(self, success):
        self.env_worker = None
        self.envDone.emit(success)

    # ---------- 模型下载 ----------
    @pyqtSlot(str)
    def downloadModel(self, model_name):
        if self.dl_worker and self.dl_worker.isRunning():
            self.downloadLog.emit('⚠️ 已有下载任务正在进行中...')
            return

        self.downloadLog.emit(f'⏳ 准备下载模型: {model_name}...')
        self.dl_worker = ModelDownloadWorker(model_name, get_base_dir())
        self.dl_worker.log.connect(self.downloadLog.emit)
        self.dl_worker.finished_signal.connect(self._on_dl_done)
        self.dl_worker.start()

    def _on_dl_done(self, success, name):
        self.dl_worker = None
        if success:
            self.downloadLog.emit(f'✅ {name} 已就绪，可前往对话页使用！')
        else:
            self.downloadLog.emit(f'❌ {name} 下载失败，请检查网络后重试。')
        self.downloadDone.emit(success)

    # ---------- 推理后端 ----------
    @pyqtSlot()
    def startChat(self):
        if self.inference_process and self.inference_process.poll() is None:
            # 进程还在，检查是否已就绪
            def _check_health():
                for i in range(10):
                    try:
                        r = requests.get('http://127.0.0.1:5000/api/model/info', timeout=3)
                        if r.status_code == 200:
                            self.envLog.emit('✅ 推理后端已就绪')
                            return
                    except Exception:
                        pass
                    time.sleep(1)
                self.envLog.emit('⚠️ 推理后端启动中，请稍候...')
            threading.Thread(target=_check_health, daemon=True).start()
            return

        base_dir = get_base_dir()
        self.envLog.emit('⏳ 正在启动推理后端...')

        def _start_and_wait():
            try:
                python_exe = get_python_exe()
                inf_py = os.path.join(base_dir, '推理.py')

                if not os.path.exists(inf_py):
                    self.envLog.emit(f'❌ 找不到推理脚本: {inf_py}')
                    return
                if not os.path.exists(python_exe):
                    self.envLog.emit(f'❌ 找不到 Python 解释器: {python_exe}')
                    return

                self.envLog.emit(f'📌 推理后端: {os.path.basename(python_exe)} + 推理.py')
                self.inference_process = subprocess.Popen(
                    [python_exe, '-u', inf_py], cwd=base_dir,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='ignore'
                )

                # 后台线程读取子进程输出，转发到 UI 日志
                def _read_output():
                    try:
                        for line in self.inference_process.stdout:
                            if line.strip():
                                self.envLog.emit(f'[推理] {line.strip()}')
                    except Exception:
                        pass
                threading.Thread(target=_read_output, daemon=True).start()

                # 轮询等待后端就绪（最多 60 秒）
                for i in range(60):
                    time.sleep(1)
                    if self.inference_process.poll() is not None:
                        self.envLog.emit(f'❌ 推理后端异常退出，退出码: {self.inference_process.returncode}')
                        return
                    try:
                        r = requests.get('http://127.0.0.1:5000/api/model/info', timeout=3)
                        if r.status_code == 200:
                            info = r.json()
                            self.envLog.emit(f'✅ 推理后端已就绪 (状态: {info.get("status", "unknown")})')
                            return
                    except Exception:
                        pass
                self.envLog.emit('⚠️ 推理后端启动超时，请检查日志')
            except Exception as e:
                self.envLog.emit(f'❌ 启动推理后端失败: {e}')

        threading.Thread(target=_start_and_wait, daemon=True).start()

    # ---------- 数据集管理 ----------
    @pyqtSlot()
    def getDatasetList(self):
        data_dir = os.path.join(BASE_DIR, 'datasets')
        os.makedirs(data_dir, exist_ok=True)
        items = []
        for f in os.listdir(data_dir):
            if f.endswith('.jsonl'):
                try:
                    lines = sum(1 for _ in open(os.path.join(data_dir, f), encoding='utf-8'))
                except Exception:
                    lines = 0
                items.append({'name': f, 'path': os.path.join('datasets', f), 'lines': lines})
        self.dataLoaded.emit(json.dumps({'type': 'datasets', 'data': items}))

    @pyqtSlot(str, str)
    def saveDataset(self, name, content):
        path = os.path.join(BASE_DIR, 'datasets', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w', encoding='utf-8').write(content)
        self.getDatasetList()

    @pyqtSlot(str)
    def readDatasetContent(self, path):
        try:
            full_path = os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
            self.datasetContent.emit(open(full_path, 'r', encoding='utf-8').read())
        except Exception as e:
            self.datasetContent.emit(f'读取失败: {e}')

    @pyqtSlot(str, str)
    def updateDataset(self, path, content):
        full_path = os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
        open(full_path, 'w', encoding='utf-8').write(content)
        self.getDatasetList()

    @pyqtSlot(str)
    def deleteDataset(self, path):
        full_path = os.path.join(BASE_DIR, path) if not os.path.isabs(path) else path
        if os.path.exists(full_path):
            os.remove(full_path)
        self.getDatasetList()

    # ---------- 训练 ----------
    @pyqtSlot(str, str, str, str, str, str, str, str, str, str, str, str)
    def startTrain(self, lora_name, dataset_path, mode, base_lora, lr, epochs, seq_len,
                   batch_size, optimizer, weight_decay, lora_r, lora_alpha):
        if self.train_worker and self.train_worker.isRunning():
            self.trainLog.emit('⚠️ 训练正在进行中...')
            return

        config = {
            "mode": mode,
            "lora_name": lora_name,
            "dataset_path": dataset_path,
            "base_lora_path": base_lora,
            "learning_rate": float(lr),
            "epochs": int(epochs),
            "max_seq_length": int(seq_len),
            "batch_size": int(batch_size),
            "optimizer": optimizer,
            "weight_decay": float(weight_decay),
            "lora_r": int(lora_r),
            "lora_alpha": int(lora_alpha)
        }

        self.trainLog.emit(f'⏳ 准备训练: {lora_name} ({mode})')
        self.train_worker = TrainWorker(config, get_base_dir())
        self.train_worker.log.connect(self.trainLog.emit)
        self.train_worker.metrics.connect(self.trainMetrics.emit)
        self.train_worker.finished_signal.connect(self._on_train_done)
        self.train_worker.start()

    def _on_train_done(self, success):
        self.train_worker = None
        self.trainDone.emit(success)
        if success:
            self.trainLog.emit('✅ 训练完成！')
            self.getLoraWeights()  # 刷新 LoRA 列表

    # ---------- LoRA 管理 ----------
    @pyqtSlot()
    def getLoraWeights(self):
        lora_dir = os.path.join(BASE_DIR, 'lora_weights')
        os.makedirs(lora_dir, exist_ok=True)
        items = []
        for d in os.listdir(lora_dir):
            full = os.path.join(lora_dir, d)
            if os.path.isdir(full):
                # 读取 meta
                meta_path = os.path.join(full, 'lora_meta.json')
                created = ''
                name = d
                try:
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        name = meta.get('name', d)
                        created = meta.get('created_at', '')
                except Exception:
                    pass
                items.append({'name': name, 'path': f'lora_weights/{d}', 'created_at': created})
        self.dataLoaded.emit(json.dumps({'type': 'loras', 'data': items}))

    @pyqtSlot(str)
    def switchLora(self, path):
        try:
            requests.post('http://127.0.0.1:5000/api/lora/switch', json={'lora_path': path}, timeout=5)
            self.trainLog.emit('✅ LoRA 切换成功')
        except:
            self.trainLog.emit('❌ 无法连接后端')

    @pyqtSlot(str)
    def deleteLora(self, path):
        full_path = os.path.join(BASE_DIR, path)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        self.getLoraWeights()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEXT AI")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.resize(1200, 800)
        QWebEngineProfile.defaultProfile().setHttpCacheType(QWebEngineProfile.NoCache)
        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)
        self.handler = ChatHandler(self)
        self.channel = QWebChannel()
        self.channel.registerObject('chatObject', self.handler)
        self.browser.page().setWebChannel(self.channel)
        html_path = resource_path(os.path.join('前端', '主程序界面.html'))
        if os.path.exists(html_path):
            self.browser.setUrl(QUrl.fromLocalFile(html_path))
        # 页面加载完成后加载语言设置
        self.browser.loadFinished.connect(self._on_load_finished)

    def _on_load_finished(self, ok):
        if ok:
            QTimer.singleShot(300, self.handler.loadLanguage)

    def closeEvent(self, event):
        if self.handler.inference_process and self.handler.inference_process.poll() is None:
            self.handler.inference_process.terminate()
        if self.handler.train_worker and self.handler.train_worker.isRunning():
            self.handler.train_worker.terminate()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
