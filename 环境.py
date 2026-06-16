# -*- coding: utf-8 -*-
"""
NEXT AI 环境构建 (v3.0 最终修正版)
- 修复 PyTorch GPU 安装：使用 --index-url 指向 PyTorch wheel 源（独立安装，不与 -i 混用）
- 修复 Git LFS 源：移除 ghfast.top，替换为可信代理
- 修复 get-pip.py 下载：添加国内镜像备选
- PyTorch wheel 源三级降级：阿里云 → 南京大学 → 官方
- 更新 Git 版本至 v2.47.0
- 保留原生进度条与深度验证报告
"""
import os, sys, subprocess, urllib.request, zipfile, tempfile, time, shutil, ssl, json, importlib


def exe_dir():
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(
        os.path.abspath(__file__))


BASE_DIR = exe_dir()
INSTALL_DIR = os.path.join(BASE_DIR, "python310")
PYTHON_EXE = os.path.join(INSTALL_DIR, "python.exe")
STATUS_FILE = os.path.join(INSTALL_DIR, ".nextai_setup_ok")

# ================= ★ 四大顶级镜像源配置 =================
MIRRORS = [
    {"name": "阿里云", "url": "https://mirrors.aliyun.com/pypi/simple/", "host": "mirrors.aliyun.com"},
    {"name": "腾讯云", "url": "https://mirrors.cloud.tencent.com/pypi/simple", "host": "mirrors.cloud.tencent.com"},
    {"name": "华为云", "url": "https://repo.huaweicloud.com/repository/pypi/simple/", "host": "repo.huaweicloud.com"},
    {"name": "中科大", "url": "https://pypi.mirrors.ustc.edu.cn/simple/", "host": "pypi.mirrors.ustc.edu.cn"},
]

PYTHON_VERSION = "3.10.11"
PYTHON_URL = f"https://mirrors.huaweicloud.com/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"

# ================= ★ PyTorch 版本与镜像配置 =================
PYTORCH_VERSION = "2.5.1"
TORCHVISION_VERSION = "0.20.1"
TORCHAUDIO_VERSION = "2.5.1"
CUDA_TAG = "cu121"

# ★ PyTorch wheel 源三级降级（真正镜像，含完整依赖）
PYTORCH_INDEX_URLS = [
    "https://mirrors.aliyun.com/pytorch-wheels/cu121",      # 首选：阿里云真正镜像
    "https://mirrors.nju.edu.cn/pytorch/whl/cu121",          # 备选：南京大学
    f"https://download.pytorch.org/whl/{CUDA_TAG}",           # 兜底：官方
]

# get-pip.py 国内镜像备选
GET_PIP_URLS = [
    "http://mirrors.aliyun.com/pypi/get-pip.py",
    "https://bootstrap.pypa.io/get-pip.py",
]

# Git 安装源（v2.47.0）
GIT_VERSION = "2.47.0"
GIT_MIRROR_URLS = [
    f"https://registry.npmmirror.com/-/binary/git-for-windows/v{GIT_VERSION}.windows.1/Git-{GIT_VERSION}-64-bit.exe",
    f"https://github.com/git-for-windows/git/releases/download/v{GIT_VERSION}.windows.1/Git-{GIT_VERSION}-64-bit.exe",
]

# ★ Git LFS 源修复：移除 ghfast.top
GIT_LFS_VERSION = "3.5.2"
GIT_LFS_MIRROR_URLS = [
    f"https://github.com/git-lfs/git-lfs/releases/download/v{GIT_LFS_VERSION}/git-lfs-windows-amd64-v{GIT_LFS_VERSION}.zip",
    f"https://mirror.ghproxy.com/https://github.com/git-lfs/git-lfs/releases/download/v{GIT_LFS_VERSION}/git-lfs-windows-amd64-v{GIT_LFS_VERSION}.zip",
]

VC_REDIST_MIRROR_URLS = ["https://aka.ms/vs/17/release/vc_redist.x64.exe"]

# ★ 分离 PyTorch 包和普通包
PYTORCH_PACKAGES = {
    "torch": PYTORCH_VERSION,
    "torchvision": TORCHVISION_VERSION,
    "torchaudio": TORCHAUDIO_VERSION,
}

REQUIRED_PACKAGES = {
    "transformers": "4.48.0", "accelerate": "1.2.1", "datasets": "3.2.0",
    "tokenizers": "0.21.0", "peft": "0.14.0", "bitsandbytes": "0.45.0",
    "safetensors": "0.5.2", "PyQt5": "5.15.11", "PyQtWebEngine": "5.15.7",
    "flask": "3.1.0", "flask-cors": "5.0.0", "numpy": "1.26.4",
    "pillow": "10.4.0", "tqdm": None, "requests": None,
    "modelscope": "1.22.0", "duckduckgo-search": "6.2.1", "scipy": None
}


# ================= 工具函数 =================
def run_cmd(cmd, capture=True, timeout=1200):
    try:
        if capture:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', errors='ignore')
            output_lines = []
            for line in process.stdout:
                print(line, end='')
                output_lines.append(line)
            process.wait()
            return process.returncode == 0, "".join(output_lines), ""
        else:
            result = subprocess.run(cmd, shell=True, timeout=timeout)
            return result.returncode == 0, "", ""
    except Exception as e:
        return False, "", str(e)


def download_file(url, dest, log=None, retries=3):
    _log = log or print

    def fmt_size(n):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(n) < 1024.0:
                return f"{n:6.2f} {unit}"
            n /= 1024.0
        return f"{n:6.2f} PB"

    def fmt_time(s):
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{int(s)//60}m{int(s)%60}s"
        return f"{int(s)//3600}h{int(s)%3600//60}m"

    def draw_bar(downloaded, total, elapsed):
        if total > 0:
            pct = downloaded / total
            bar_w = 28
            filled = int(bar_w * pct)
            bar = "█" * filled + "░" * (bar_w - filled)
            speed = downloaded / (elapsed + 0.001)
            eta = (total - downloaded) / speed if speed > 0 else 0
            return (f"\r ⬇️ [{bar}] {pct*100:5.1f}%  "
                    f"{fmt_size(downloaded)}/{fmt_size(total)}  "
                    f"{fmt_size(speed)}/s  ETA {fmt_time(eta)}")
        else:
            return f"\r ⬇️ 已下载 {fmt_size(downloaded)} (大小未知)"

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp, open(dest, 'wb') as out:
                total = int(resp.headers.get('Content-Length', 0))
                downloaded = 0
                start = time.time()
                last_flush = 0

                _log(f" 📦 开始下载: {os.path.basename(dest)} "
                     f"({fmt_size(total) if total else '未知大小'})")

                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_flush > 0.25 or downloaded == total:
                        sys.stdout.write(draw_bar(downloaded, total, now - start))
                        sys.stdout.flush()
                        last_flush = now

                sys.stdout.write("\n")
                sys.stdout.flush()
                _log(f" ✅ 下载完成: {fmt_size(downloaded)}")

            return True

        except Exception as e:
            sys.stdout.write("\n")
            _log(f" ⚠️ 第 {attempt}/{retries} 次尝试失败: {str(e)[:80]}")
            if attempt < retries:
                time.sleep(3)
            else:
                return False
    return False


def download_file_with_fallback(urls, dest, log=None):
    for url in urls:
        if download_file(url, dest, log=log): return True
    return False


# ================= 安装函数 =================
def install_python(log=None):
    log("\n⬇️ 下载 Python 3.10...")
    os.makedirs(INSTALL_DIR, exist_ok=True)
    zip_path = os.path.join(tempfile.gettempdir(), "python-embed.zip")
    if not download_file(PYTHON_URL, zip_path, log=log): return False
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(INSTALL_DIR)
    os.remove(zip_path)
    pth_file = os.path.join(INSTALL_DIR, "python310._pth")
    if os.path.exists(pth_file):
        with open(pth_file, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace("#import site", "import site")
        if "Lib\\site-packages" not in content: content += "\nLib\\site-packages\n"
        with open(pth_file, 'w', encoding='utf-8') as f:
            f.write(content)
    return True


def install_pip(log=None):
    log("\n⬇️ 修复 pip...")
    get_pip_path = os.path.join(tempfile.gettempdir(), "get-pip.py")
    if not download_file_with_fallback(GET_PIP_URLS, get_pip_path, log=log):
        log("❌ get-pip.py 所有源均下载失败")
        return False
    success, _, _ = run_cmd(
        f'"{PYTHON_EXE}" "{get_pip_path}" -i {MIRRORS[0]["url"]} --trusted-host {MIRRORS[0]["host"]}')
    os.remove(get_pip_path)
    return success


def add_to_path():
    os.environ["PATH"] = f"{INSTALL_DIR};{os.path.join(INSTALL_DIR, 'Scripts')};" + os.environ.get("PATH", "")


def install_system_tools(log=None):
    log("\n" + "=" * 40 + "\n检查系统工具\n" + "=" * 40)
    if run_cmd("git --version", timeout=10)[0]:
        log("✅ Git 已安装")
    else:
        installer = os.path.join(tempfile.gettempdir(), "Git.exe")
        if download_file_with_fallback(GIT_MIRROR_URLS, installer):
            run_cmd(f'"{installer}" /VERYSILENT /NORESTART', timeout=600)
            os.remove(installer)
    if run_cmd("git lfs --version", timeout=10)[0]:
        log("✅ Git LFS 已安装")
    else:
        zip_path = os.path.join(tempfile.gettempdir(), "lfs.zip")
        if download_file_with_fallback(GIT_LFS_MIRROR_URLS, zip_path):
            ext_dir = os.path.join(tempfile.gettempdir(), "lfs")
            if os.path.exists(ext_dir): shutil.rmtree(ext_dir)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(ext_dir)
            for r, d, f in os.walk(ext_dir):
                if "git-lfs.exe" in f:
                    shutil.copy(os.path.join(r, "git-lfs.exe"), r"C:\Program Files\Git\cmd\git-lfs.exe")
                    run_cmd(r'"C:\Program Files\Git\cmd\git-lfs.exe" install')
                    break
            os.remove(zip_path)
            shutil.rmtree(ext_dir)
    if os.path.exists(os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "vcruntime140.dll")):
        log("✅ VC++ 已安装")
    else:
        installer = os.path.join(tempfile.gettempdir(), "vc.exe")
        if download_file_with_fallback(VC_REDIST_MIRROR_URLS, installer):
            run_cmd(f'"{installer}" /install /quiet /norestart', timeout=300)
            os.remove(installer)


# ================= ★ 智能依赖管理 =================
def smart_install_requirements(python_exe, log=None):
    log("\n" + "=" * 40 + "\n智能依赖管理\n" + "=" * 40)

    # 升级 pip
    run_cmd(f'"{python_exe}" -m pip install --upgrade pip -i {MIRRORS[0]["url"]} --trusted-host {MIRRORS[0]["host"]}',
            capture=False)

    success, stdout, _ = run_cmd(f'"{python_exe}" -m pip list --format=json', timeout=30)
    installed = {p['name'].lower(): p['version'] for p in json.loads(stdout)} if success else {}

    # ================= ★ 第一步：安装 PyTorch GPU 包（独立策略）====================
    log("\n" + "=" * 40 + "\n【阶段 1】安装 PyTorch GPU 包\n" + "=" * 40)

    pytorch_to_install = []
    for pkg, ver in PYTORCH_PACKAGES.items():
        inst_ver = installed.get(pkg.lower())
        if inst_ver and ver.split('+')[0] == inst_ver.split('+')[0]:
            log(f"✅ {pkg:<20} [{inst_ver}] (跳过)")
        else:
            pytorch_to_install.append(f"{pkg}=={ver}")
            log(f"⏳ {pkg:<20} [需安装: {ver}]")

    if pytorch_to_install:
        pkg_str = " ".join(pytorch_to_install)
        log(f"\n🚀 准备安装 PyTorch GPU: {pkg_str}")
        anti_stuck = "--progress-bar=on --timeout 1000 --retries 10"

        installed_ok = False
        for pt_url in PYTORCH_INDEX_URLS:
            mirror_name = "阿里云" if "aliyun" in pt_url else ("南京大学" if "nju" in pt_url else "官方")
            log(f"\n 🔄 尝试 PyTorch 源 [{mirror_name}]...")
            log(f"    URL: {pt_url}")

            # ★ 核心：使用 --index-url 单独指向 PyTorch wheel 源
            # 不混用 -i，确保 pip 只从 PyTorch 源搜索 torch 包
            # PyTorch wheel 源是完整的 PEP 503 仓库，包含 torch 及其所有依赖
            trusted_host = urllib.parse.urlparse(pt_url).netloc
            cmd = (f'"{python_exe}" -m pip install {pkg_str} '
                   f'--index-url {pt_url} '
                   f'--trusted-host {trusted_host} '
                   f'{anti_stuck}')

            ok = run_cmd(cmd, capture=False)[0]
            if ok:
                log(f"✅ [{mirror_name}] PyTorch GPU 安装成功")
                installed_ok = True
                break
            else:
                log(f"⚠️ [{mirror_name}] 失败，自动切换下一个 PyTorch 源...")

        if not installed_ok:
            log("❌ 所有 PyTorch 源均安装失败")
            return False
    else:
        log("✅ PyTorch GPU 包已全部安装")

    # ================= ★ 第二步：安装普通 Python 包（四大镜像降级）====================
    log("\n" + "=" * 40 + "\n【阶段 2】安装普通 Python 包\n" + "=" * 40)

    normal_to_install = []
    for pkg, ver in REQUIRED_PACKAGES.items():
        inst_ver = installed.get(pkg.lower())
        if inst_ver and (ver is None or ver == inst_ver):
            log(f"✅ {pkg:<20} [{inst_ver}] (跳过)")
        else:
            normal_to_install.append((pkg, ver))
            log(f"⏳ {pkg:<20} [需安装: {ver or '最新'}]")

    if not normal_to_install:
        log("✅ 所有普通包已安装")
        return True

    log(f"\n🚀 准备安装 {len(normal_to_install)} 个普通包...")
    anti_stuck = "--progress-bar=on --timeout 1000 --retries 10"

    for i, (pkg, ver) in enumerate(normal_to_install, 1):
        log(f"\n[{i}/{len(normal_to_install)}] {pkg}")
        pkg_str = f"{pkg}=={ver}" if ver else pkg

        installed_ok = False
        for mirror in MIRRORS:
            log(f" 🔄 尝试 [{mirror['name']}]...")
            trusted = f"--trusted-host {mirror['host']}"
            cmd = f'"{python_exe}" -m pip install {pkg_str} -i {mirror["url"]} {trusted} {anti_stuck}'
            ok = run_cmd(cmd, capture=False)[0]
            if ok:
                log(f"✅ [{mirror['name']}] 安装成功")
                installed_ok = True
                break
            else:
                log(f"⚠️ [{mirror['name']}] 失败或超时，自动切换下一个源...")

        if not installed_ok:
            log("❌ 所有镜像源均安装失败")
            return False

    return True


# ================= 深度验证 =================
def deep_verify_env(python_exe, log=None):
    verify_script = """
import sys, importlib, os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
passed = total = 0
def check(name, extra=None):
    global passed, total; total += 1
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "核心模块")
        if extra: extra(mod)
        print(f"✅ {name:<25} [版本: {ver}]"); passed += 1
    except Exception as e: print(f"❌ {name:<25} [导入失败: {str(e)[:40]}]")
print("【1. GPU】")
def chk(torch):
    if torch.cuda.is_available(): print(f"   🟢 {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)")
    else: print("   🟡 CPU (警告：未检测到 GPU)")
check("torch", chk)
print("\\n【2. 核心库】")
for lib in ["transformers", "peft", "bitsandbytes", "PyQt5", "flask", "duckduckgo_search", "scipy"]: check(lib)
print("\\n" + "="*60)
print(f"🎉 验证完成: {passed}/{total} 项通过。")
print("VERIFY_SCRIPT_DONE")
"""
    success, stdout, stderr = run_cmd(f'"{python_exe}" -c "{verify_script}"', timeout=60)
    log("\n" + "=" * 60 + "\n🚀 验证报告\n" + "=" * 60)
    if success and "VERIFY_SCRIPT_DONE" in stdout:
        for line in stdout.split('\n'):
            if any(k in line for k in ['✅', '❌', '🟢', '🟡', '【', '🎉', '=====']): log(line)
        return True
    return False


# ================= 核心调度 =================
def run_env_setup(log_fn=None):
    log = log_fn or print
    log("=" * 60 + "\nNEXT AI 环境构建 (v3.0 最终修正版)\n" + "=" * 60)

    need_python = not os.path.exists(PYTHON_EXE)
    need_pip = True

    if not need_python:
        add_to_path()
        if run_cmd(f'"{PYTHON_EXE}" -m pip --version', timeout=10)[0]:
            need_pip = False
            log("✅ Python/pip 健康")
        else:
            log("⚠️ pip 损坏，触发抢救...")

    if need_python:
        if not install_python(log=log): return False
        add_to_path()
    if need_pip:
        if not install_pip(log=log): return False

    install_system_tools(log=log)
    if not smart_install_requirements(PYTHON_EXE, log=log): return False

    if deep_verify_env(PYTHON_EXE, log=log):
        with open(STATUS_FILE, 'w') as f: f.write(str(time.time()))
        log("\n🎉 完美收官！")
        return True
    return False


if __name__ == "__main__":
    try:
        run_env_setup(print)
    except KeyboardInterrupt:
        print("\n中断")
    except Exception as e:
        print(f"\n异常: {e}")
    finally:
        input("\n按 Enter 退出...")
