#!/usr/bin/env python3
# build.py - 100MB 终极轻量版构建脚本
import shutil, subprocess, sys, os, urllib.request, zipfile, ssl
from pathlib import Path

# ★ 强制 UTF-8 输出，防止 Windows GBK 控制台崩溃
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ROOT = Path(__file__).parent.absolute()
DIST = ROOT / 'dist'
EXE_NAME = 'NeXT'
ICON_FILE = ROOT / 'app.ico'

import PyQt5

PYQT5_DIR = Path(PyQt5.__path__[0])
QT_DIR = PYQT5_DIR / 'Qt5' if (PYQT5_DIR / 'Qt5').exists() else PYQT5_DIR / 'Qt'
QTWEBENGINE_DIR = PYQT5_DIR / 'QtWebEngine'

MAIN_HIDDEN_IMPORTS = [
    'PyQt5', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets',
    'PyQt5.QtWebEngineWidgets', 'PyQt5.QtWebChannel',
    'PyQt5.QtWebEngine', 'PyQt5.QtWebEngineCore',
    'requests', 'yaml', 'jinja2'
]
EXCLUDE_MODULES = [
    'torch', 'torchvision', 'torchaudio', 'transformers', 'accelerate', 'peft',
    'sentencepiece', 'tokenizers', 'huggingface_hub', 'datasets', 'bitsandbytes',
    'gradio', 'numpy', 'scipy', 'sympy', 'pandas', 'pydub', 'scikit-learn',
    '训练', '环境', '模型下载'
]
EXTRA_SEARCH_PATHS = [
    r"C:\Program Files\Common Files\Adobe\Adobe Desktop Common\CEF",
    r"C:\Program Files\Norton\Suite",
    r"C:\Program Files (x86)\Common Files\Adobe\Adobe Desktop Common\CEF",
]


def recursive_find_and_copy(source_roots, dest_dir, patterns, is_dir=False):
    if isinstance(source_roots, (str, Path)): source_roots = [source_roots]
    results = {p: None for p in patterns}
    for root in source_roots:
        root = Path(root)
        if not root.exists(): continue
        for dirpath, dirnames, filenames in os.walk(root):
            for pattern in patterns:
                if results[pattern] is not None: continue
                if is_dir and pattern in dirnames:
                    results[pattern] = Path(dirpath) / pattern
                elif not is_dir and pattern in filenames:
                    results[pattern] = Path(dirpath) / pattern
    for pattern, src in results.items():
        dst = dest_dir / pattern
        if src is not None and not dst.exists():
            if is_dir:
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


def deploy_embedded_python(dest_dir):
    print("\n🐍 正在下载并部署 Python 3.10 (带 pip)...")
    py_dir = dest_dir / 'python310'
    if py_dir.exists(): shutil.rmtree(py_dir)
    py_dir.mkdir(parents=True)

    zip_url = "https://mirrors.huaweicloud.com/python/3.10.11/python-3.10.11-embed-amd64.zip"
    zip_path = ROOT / 'python-embed-temp.zip'

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False;
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as resp, open(zip_path, 'wb') as out:
            out.write(resp.read())

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(py_dir)
        zip_path.unlink()

        pth_file = py_dir / 'python310._pth'
        if pth_file.exists():
            content = pth_file.read_text(encoding='utf-8')
            content = content.replace("#import site", "import site")
            if "Lib\\site-packages" not in content: content += "\nLib\\site-packages\n"
            pth_file.write_text(content, encoding='utf-8')

        print("  ✔ 正在为空壳 Python 安装 pip...")
        pip_url = "https://bootstrap.pypa.io/get-pip.py"
        pip_path = py_dir / 'get-pip.py'
        urllib.request.urlretrieve(pip_url, pip_path)

        # 使用腾讯云镜像安装 pip
        pip_mirror = "https://mirrors.cloud.tencent.com/pypi/simple"
        subprocess.check_call(
            [str(py_dir / 'python.exe'), str(pip_path), '-i', pip_mirror, '--trusted-host',
             'mirrors.cloud.tencent.com'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        pip_path.unlink()
        print("  ✔ pip 安装完成！")

    except Exception as e:
        print(f"  ❌ 部署 Python 失败: {e}")
        sys.exit(1)


def build():
    if DIST.exists(): shutil.rmtree(DIST)
    exe_folder = DIST / EXE_NAME
    sep = ';' if os.name == 'nt' else ':'

    print("\n [1/3] 打包主程序 (NeXT.exe)...")
    # ★ PyInstaller --clean 会删除 build/ 但不会删除 dist/<name>/，需要手动确保为空
    if exe_folder.exists(): shutil.rmtree(exe_folder)
    cmd_main = [sys.executable, '-m', 'PyInstaller', '--windowed', f'--name={EXE_NAME}',
                f'--add-data=前端{sep}前端', '--clean', '--distpath', str(DIST), '主程序入口.py']
    if ICON_FILE.exists(): cmd_main.insert(3, f'--icon={ICON_FILE}')
    for mod in MAIN_HIDDEN_IMPORTS: cmd_main += ['--hidden-import', mod]
    for mod in EXCLUDE_MODULES: cmd_main += ['--exclude-module', mod]
    subprocess.check_call(cmd_main, cwd=ROOT)

    print("\n🧠 [2/3] 部署运行时环境...")
    deploy_embedded_python(exe_folder)

    # 复制环境配置脚本
    env_script = ROOT / 'env_setup.py'
    if env_script.exists():
        shutil.copy2(env_script, exe_folder / 'env_setup.py')
        print("  ✔ env_setup.py 已复制")
    else:
        # 兼容旧文件名
        old_env = ROOT / '环境.py'
        if old_env.exists():
            shutil.copy2(old_env, exe_folder / 'env_setup.py')
            print("  ✔ 环境.py 已重命名为 env_setup.py 并复制")

    # 复制批量下载脚本
    batch_dl_script = ROOT / '模型下载.py'
    if batch_dl_script.exists():
        shutil.copy2(batch_dl_script, exe_folder / 'download_all_models.py')
        print("  ✔ 模型下载.py 已重命名为 download_all_models.py 并复制")

    # ★ 复制推理和训练脚本（由嵌入式 Python 运行，不打包为 exe）
    for script_name in ['推理.py', '训练.py']:
        src = ROOT / script_name
        if src.exists():
            shutil.copy2(src, exe_folder / script_name)
            print(f"  ✔ {script_name} 已复制")
        else:
            print(f"  ⚠️ {script_name} 未找到，跳过")

    print("\n🛠️ [3/3] 补全 QtWebEngine 资源...")
    proc = QT_DIR / 'bin' / 'QtWebEngineProcess.exe'
    if proc.exists(): shutil.copy2(proc, exe_folder / 'QtWebEngineProcess.exe')
    qt_bin = QT_DIR / 'bin'
    if qt_bin.exists():
        for dll in qt_bin.glob('*.dll'):
            if not (exe_folder / dll.name).exists(): shutil.copy2(dll, exe_folder / dll.name)
    required_files = ['icudtl.dat', 'qtwebengine_resources.pak', 'qtwebengine_resources_100p.pak',
                      'qtwebengine_resources_200p.pak', 'snapshot_blob.bin', 'natives_blob.bin',
                      'v8_context_snapshot.bin']
    recursive_find_and_copy([PYQT5_DIR] + EXTRA_SEARCH_PATHS, exe_folder, required_files, is_dir=False)
    for sub in ['resources', 'translations', 'platforms']:
        src = QT_DIR / sub if sub != 'platforms' else QT_DIR / 'plugins' / 'platforms'
        if src.exists(): shutil.copytree(src, exe_folder / sub, dirs_exist_ok=True)
    if QTWEBENGINE_DIR.exists():
        for item in QTWEBENGINE_DIR.iterdir():
            if item.name == '__pycache__' or item.suffix in ('.py', '.pyc'): continue
            dst = exe_folder / item.name
            if item.is_file() and not dst.exists():
                shutil.copy2(item, dst)
            elif item.is_dir() and not dst.exists():
                shutil.copytree(item, dst)
    recursive_find_and_copy([PYQT5_DIR] + EXTRA_SEARCH_PATHS, exe_folder, ['swiftshader'], is_dir=True)

    build_dir = ROOT / 'build'
    if build_dir.exists(): shutil.rmtree(build_dir)
    for spec in ROOT.glob('*.spec'): spec.unlink()

    # Auto-deploy to F:\NeXT (full copy including _internal)
    deploy_target = Path('F:/NeXT')
    if deploy_target.exists():
        import time
        for attempt in range(3):
            try:
                shutil.rmtree(deploy_target)
                break
            except PermissionError as e:
                if attempt < 2:
                    print(f"  ⚠️  删除旧部署失败 ({e}), 等待 2 秒后重试...")
                    time.sleep(2)
                else:
                    # Fallback: use robocopy to overwrite instead of delete+copy
                    print(f"  ⚠️  无法删除旧部署，使用增量更新模式...")
                    # robocopy /E /Y /IS = copy all files, overwrite existing, include same files
                    result = subprocess.run(
                        ['robocopy', str(exe_folder), str(deploy_target), '/E', '/Y', '/IS'],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        print(f"\n  Deployed to {deploy_target} (incremental update)")
                        print(f"\nDone! Output: {exe_folder}")
                        print(f"\n🎉 打包完美完成！输出文件夹: {exe_folder}")
                        return
                    else:
                        print(f"  ❌ robocopy 失败: {result.stderr}")
                        raise
    shutil.copytree(exe_folder, deploy_target)
    print(f"\n  Deployed to {deploy_target} (with _internal)")

    print(f"\nDone! Output: {exe_folder}")

    print(f"\n🎉 打包完美完成！输出文件夹: {exe_folder}")


if __name__ == '__main__':
    build()