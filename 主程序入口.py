# -*- coding: utf-8 -*-
import sys, os, subprocess, json, time, shutil, requests
from pathlib import Path
from PyQt5.QtCore import QUrl, QObject, pyqtSlot, pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile
from PyQt5.QtWebChannel import QWebChannel


def get_base_dir():
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(
        os.path.abspath(__file__))


BASE_DIR = get_base_dir()


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'): return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(BASE_DIR, relative_path)


# ================= 1. 环境检测线程 =================
class EnvCheckWorker(QThread):
    log = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def run(self):
        self.log.emit('🔍 开始进行环境深度体检...')
        base_dir = get_base_dir()
        python_exe = os.path.join(base_dir, 'python310', 'python.exe')
        site_packages = os.path.join(base_dir, 'python310', 'Lib', 'site-packages')
        env_script = os.path.join(base_dir, 'env_setup.py')

        if not os.path.exists(python_exe):
            self.log.emit(f'❌ 找不到 Python 解释器: {python_exe}')
            self.finished_signal.emit();
            return

        self.log.emit('✅ 核心解释器校验通过')

        torch_path = os.path.join(site_packages, 'torch')
        if not os.path.exists(torch_path):
            self.log.emit('⚠️ 检测到 AI 核心库 (torch) 缺失！正在自动修复...')
            if not os.path.exists(env_script):
                self.log.emit(f'❌ 找不到修复脚本: {env_script}')
                self.finished_signal.emit();
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
                    self.finished_signal.emit();
                    return
                self.log.emit('🎉 环境自动下载与修复完成！')
            except Exception as e:
                self.log.emit(f'❌ 调用异常: {str(e)}')
                self.finished_signal.emit();
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
    else: print("🟡 CUDA 不可用")
    print("VERIFY_DONE")
except Exception as e: print(f"❌ 验证崩溃: {str(e)[:50]}")
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
        self.finished_signal.emit()


# ================= 2. 模型下载线程 (★ 复用 模型下载.py 核心逻辑) =================
class ModelDownloadWorker(QThread):
    log = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, alias, base_dir):
        super().__init__()
        self.alias = alias
        self.base_dir = base_dir

    def run(self):
        # ★ 完美复刻 模型下载.py 中的 MODEL_LIST 映射关系
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

        python_exe = os.path.join(self.base_dir, 'python310', 'python.exe')
        models_dir = os.path.join(self.base_dir, 'models')
        # ★ 复刻 模型下载.py 的目录结构：models/别名
        cache_dir = os.path.join(models_dir, self.alias)

        # ★ 动态生成下载脚本 (包含重试逻辑和 master 分支设定)
        dl_script = f"""
import sys, os, time
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
        # ★ 使用 snapshot_download，指定 master 分支
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


# ================= 3. QWebChannel 桥接类 =================
class ChatHandler(QObject):
    logUpdate = pyqtSignal(str, str)
    dataLoaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.inference_process = None
        self.env_worker = None
        self.dl_worker = None

    @pyqtSlot()
    def checkEnv(self):
        if self.env_worker and self.env_worker.isRunning(): return
        self.env_worker = EnvCheckWorker()
        self.env_worker.log.connect(lambda msg: self.logUpdate.emit('env', msg))
        self.env_worker.finished_signal.connect(self._clean_env)
        self.env_worker.start()

    def _clean_env(self):
        self.env_worker = None

    @pyqtSlot(str)
    def downloadModel(self, model_name):
        if self.dl_worker and self.dl_worker.isRunning():
            self.logUpdate.emit('download', '⚠️ 已有下载任务正在进行中...')
            return

        self.logUpdate.emit('download', f'⏳ 准备下载模型: {model_name}...')
        self.dl_worker = ModelDownloadWorker(model_name, get_base_dir())
        self.dl_worker.log.connect(lambda msg: self.logUpdate.emit('download', msg))
        self.dl_worker.finished_signal.connect(self._clean_dl)
        self.dl_worker.start()

    def _clean_dl(self, success, name):
        self.dl_worker = None
        if success:
            self.logUpdate.emit('download', f'✅ {name} 已就绪，可前往对话页使用！')
        else:
            self.logUpdate.emit('download', f'❌ {name} 下载失败，请检查网络后重试。')

    @pyqtSlot()
    def startChat(self):
        if self.inference_process and self.inference_process.poll() is None: return
        base_dir = get_base_dir()
        if getattr(sys, 'frozen', False):
            inf_exe = os.path.join(base_dir, '推理.exe')
            if os.path.exists(inf_exe):
                self.inference_process = subprocess.Popen([inf_exe], cwd=base_dir)
                self.logUpdate.emit('env', '✅ 推理后端已启动');
                return
        python_exe = os.path.join(base_dir, 'python310', 'python.exe')
        inf_py = os.path.join(base_dir, '推理.py')
        if os.path.exists(inf_py) and os.path.exists(python_exe):
            self.inference_process = subprocess.Popen([python_exe, '-u', inf_py], cwd=base_dir)

    @pyqtSlot()
    def getDatasetList(self):
        data_dir = os.path.join(BASE_DIR, 'datasets');
        os.makedirs(data_dir, exist_ok=True)
        items = [{'name': f, 'path': os.path.join(data_dir, f),
                  'lines': sum(1 for _ in open(os.path.join(data_dir, f), encoding='utf-8'))} for f in
                 os.listdir(data_dir) if f.endswith('.jsonl')]
        self.dataLoaded.emit(json.dumps({'type': 'dataset_list', 'items': items}))

    @pyqtSlot(str, str)
    def saveDataset(self, name, content):
        path = os.path.join(BASE_DIR, 'datasets', name);
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w', encoding='utf-8').write(content);
        self.getDatasetList()

    @pyqtSlot(str)
    def readDatasetContent(self, path):
        self.dataLoaded.emit(
            json.dumps({'type': 'dataset_content', 'content': open(path, 'r', encoding='utf-8').read()}))

    @pyqtSlot(str, str)
    def updateDataset(self, path, content):
        open(path, 'w', encoding='utf-8').write(content); self.getDatasetList()

    @pyqtSlot(str)
    def deleteDataset(self, path):
        os.remove(path); self.getDatasetList()

    @pyqtSlot(str)
    def startTrain(self, c):
        self.logUpdate.emit('train', '⏳ 训练已接收')

    @pyqtSlot()
    def getLoraWeights(self):
        lora_dir = os.path.join(BASE_DIR, 'lora_weights');
        os.makedirs(lora_dir, exist_ok=True)
        items = [{'name': d, 'path': f'lora_weights/{d}'} for d in os.listdir(lora_dir) if
                 os.path.isdir(os.path.join(lora_dir, d))]
        self.dataLoaded.emit(json.dumps({'type': 'lora_list', 'items': items}))

    @pyqtSlot(str)
    def switchLora(self, path):
        try:
            requests.post('http://127.0.0.1:5000/api/lora/switch', json={'lora_path': path},
                          timeout=5); self.logUpdate.emit('data-management', '✅ 切换成功')
        except:
            self.logUpdate.emit('data-management', '❌ 无法连接')

    @pyqtSlot(str)
    def deleteLora(self, path):
        shutil.rmtree(os.path.join(BASE_DIR, path)); self.getLoraWeights()

    @pyqtSlot()
    def goHome(self):
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEXT AI");
        self.resize(1200, 800)
        QWebEngineProfile.defaultProfile().setHttpCacheType(QWebEngineProfile.NoCache)
        self.browser = QWebEngineView();
        self.setCentralWidget(self.browser)
        self.handler = ChatHandler(self)
        self.channel = QWebChannel();
        self.channel.registerObject('chatObject', self.handler)
        self.browser.page().setWebChannel(self.channel)
        html_path = resource_path(os.path.join('前端', '主程序界面.html'))
        if os.path.exists(html_path): self.browser.setUrl(QUrl.fromLocalFile(html_path))

    def closeEvent(self, event):
        if self.handler.inference_process and self.handler.inference_process.poll() is None: self.handler.inference_process.terminate()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv);
    window = MainWindow();
    window.show();
    sys.exit(app.exec_())