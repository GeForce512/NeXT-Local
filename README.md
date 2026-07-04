# NeXT-Local

A desktop workstation for local LLM inference and fine-tuning. No command-line hassle, no bloated all-in-one packages — the main executable is only tens of MB, double-click and run.

## Why This Project

Running a large language model locally usually means wrestling with CLI configurations (Ollama, llama.cpp) or downloading sluggish "one-click bundles" that bundle everything under the sun. NeXT-Local is a clean, lightweight client that wraps environment setup, model downloading, LoRA fine-tuning, and inference into a single GUI — accessible even for users who don't write code.

## Architecture & Key Decisions

**Process Isolation** — The main app (`主程序入口.py`) handles only the PyQt5 UI and orchestration. The inference backend (`推理.py`) runs as an independent Flask process. They communicate via a local REST API, so the UI never freezes due to model loading or OOM errors.

**Shell-Bullet Separation (壳弹分离)** — PyInstaller packaging uses the "thin launcher + embedded Python" pattern: the main executable is ~12 MB, with an embedded `python310/` directory containing all ML dependencies. The inference (`推理.py`) and training (`训练.py`) scripts are deployed alongside and executed by the embedded Python at runtime — no heavy PyInstaller bundling needed.

**WebEngine Asset Recovery** — Packaging PyQtWebEngine with PyInstaller often misses `.pak` or `swiftshader` files, causing a blank screen. `build.py` includes a recursive search script that locates these missing CEF assets from system directories (Adobe, Norton, etc.) and patches them in.

**Weak-Network Downloader** — Model downloads wrap ModelScope's `snapshot_download` with 3-attempt exponential backoff retry. If the connection drops, it resumes. All models go into `models/`.

**Temperature-Reactive UI** — A frontend JS interpolation engine maps the model's `Temperature` parameter to the interface background hue in real-time. Cold blue for code questions, warm orange for creative writing. Purely aesthetic, but it feels great.

**Halftone Dot Background** — A full-screen `<canvas>` renders a grid of dots that enlarge near the mouse cursor, creating a subtle interactive texture. Uses squared-distance comparison (no `sqrt`) for performance.

**Pure Canvas Training Charts** — Training loss and learning rate are visualized as scrolling bezier curves drawn directly on `<canvas>`, with no external charting library. Keeps the bundle lightweight.

**File-Based Chat History** — Conversation sessions are persisted as JSON files via QWebChannel, not `localStorage`. This survives browser cache clears and makes backup/transfer trivial.

**First-Launch Wizard** — On first run, a 4-step modal guides the user: GPU detection → library installation → model download → ready. Users without a GPU are informed and can proceed with CPU-only torch.

## Project Structure

```
NeXT-Local/
├── 主程序入口.py          # Main entry: PyQt5 UI + QWebChannel bridge
├── 推理.py               # Flask inference backend (port 5000)
├── 训练.py               # QLoRA 4-bit fine-tuning with LoRA hot-swap
├── 环境.py               # Environment setup & dependency installer
├── 模型下载.py           # Model download via ModelScope
├── build.py              # PyInstaller build script
├── UI设置.json           # UI preferences (thinking mode toggle)
├── 前端/
│   ├── 主程序界面.html   # Main UI layout
│   ├── 交互逻辑.js       # Frontend logic (halftone, charts, chat, wizard)
│   └── 界面样式.css      # Styles (dark/light themes, responsive)
├── models/               # Downloaded base models
├── datasets/             # Training datasets (JSONL)
├── lora_weights/         # Trained LoRA adapters
└── 训练数据/             # Training temp output
```

## Getting Started

### Development

```bash
# 1. Install base dependencies
pip install -r requirements.txt

# 2. Set up local AI environment (PyTorch, etc.)
python 环境.py

# 3. Download a base model (supports resume)
python 模型下载.py

# 4. Start services (two terminals)
python 推理.py          # Terminal 1: inference backend
python 主程序入口.py     # Terminal 2: UI
```

### Packaged Build

```bash
python build.py
# Output: dist/NeXT-Local/ — copy anywhere and run
```

## Supported Models

| Model | Size | Notes |
|-------|------|-------|
| Qwen3.5-0.8B | ~1.6 GB | Lightweight, good for testing |
| Qwen3.5-4B | ~8 GB | Recommended base model |
| DeepSeek-R1-Distill-Qwen-1.5B | ~3 GB | Reasoning-focused |

## Tech Stack

Python 3.10 · PyQt5 + QWebEngineView · Flask · PyTorch + Transformers + PEFT (QLoRA 4-bit) · ModelScope · PyInstaller

---

# NeXT-Local（中文说明）

一个本地大模型推理和微调的桌面端工作台。没有黑乎乎的命令行，没有动辄几个 GB 的臃肿整合包。主程序只有几十 MB，双击就能跑。

## 为什么写这个项目

现在想在本机跑个大模型，要么面对 Ollama 复杂的命令行配置，要么下载解压慢得要死的"懒人包"。我想做一个干净、轻量的客户端，把环境配置、模型下载、LoRA 微调和推理全塞进一个 GUI 里，让不懂代码的人也能直接用。

## 核心实现

**主程序与推理进程物理隔离** — 主程序只负责 PyQt5 界面和调度；推理后端跑在独立的 Flask 进程里。两者通过本地 API 通信，UI 永远不会因为模型加载或 OOM 而卡死。

**壳弹分离打包** — PyInstaller 打包采用"薄启动器 + 内嵌 Python"模式：主 exe 约 12 MB，内嵌 `python310/` 目录包含全部 ML 依赖。推理 (`推理.py`) 和训练 (`训练.py`) 脚本随包部署，运行时由内嵌 Python 直接执行，无需再打包为独立 exe。

**WebEngine 资源修复** — PyInstaller 打包 PyQtWebEngine 经常缺 `.pak` 或 `swiftshader` 导致白屏。`build.py` 内置递归搜索脚本，自动从系统目录中找到并补齐缺失的 CEF 文件。

**带重试的弱网下载器** — 基于 ModelScope 的 `snapshot_download` 封装，3 次指数级退避重试。断网也能接着下，模型统一存放在 `models/` 目录。

**Temperature 驱动 UI 配色** — 前端 JS 插值算法将模型的 Temperature 参数实时映射到界面背景色温。问代码时冷蓝，写小说时暖橙。纯视觉体验，但很爽。

**半色调点阵背景** — 全屏 `<canvas>` 渲染点阵，鼠标附近的点会放大，形成细腻的交互纹理。使用平方距离比较（无 `sqrt`），性能友好。

**纯 Canvas 训练图表** — 训练 Loss 和学习率用 `<canvas>` 直接绘制滚动贝塞尔曲线，不依赖任何外部图表库，保持包体轻量。

**文件存储聊天记录** — 对话历史通过 QWebChannel 持久化为 JSON 文件，不依赖 `localStorage`。清除浏览器缓存不影响，备份和迁移也很方便。

**首次启动向导** — 首次运行时弹出 4 步引导：GPU 检测 → 安装运行库 → 下载模型 → 就绪。没有 GPU 的用户会收到提示，可以用 CPU 模式继续。

## 怎么跑起来

### 开发环境

```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 构建本地 AI 环境（自动处理 PyTorch 等）
python 环境.py

# 3. 下载基座模型（支持断点续传）
python 模型下载.py

# 4. 启动服务（需要两个终端）
python 推理.py          # 终端 1：启动后端
python 主程序入口.py     # 终端 2：启动 UI
```

### 打包构建

```bash
python build.py
# 输出: dist/NeXT-Local/ — 可复制到任意位置运行
```

## 支持模型

| 模型 | 大小 | 说明 |
|------|------|------|
| Qwen3.5-0.8B | ~1.6 GB | 轻量级，适合测试 |
| Qwen3.5-4B | ~8 GB | 推荐基座模型 |
| DeepSeek-R1-Distill-Qwen-1.5B | ~3 GB | 推理增强型 |

## 技术栈

Python 3.10 · PyQt5 + QWebEngineView · Flask · PyTorch + Transformers + PEFT (QLoRA 4-bit) · ModelScope · PyInstaller
