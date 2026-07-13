# NeXT-Local

**在 4GB 显存上运行你自己的 AI 智能体。**

当各大厂商的智能体平台接连下架服务、关闭 API 时，NeXT-Local 让你完全掌控自己的 AI 能力。无需依赖任何云端服务，所有模型推理和微调都在本地完成。

## 为什么需要这个项目

2025年以来，主流云服务商陆续关停或限制免费 AI 接口。企业级 API 价格飙升，个人开发者难以承担。更关键的是——**你无法控制别人的服务器何时关闭**。

NeXT-Local 的解决方案：
- **完全离线运行**：不依赖任何外部 API，断网也能用
- **超低配置要求**：4GB 显存即可运行，老旧计算卡（GTX 1060/1650）也能跑
- **一键定制专属智能体**：通过 LoRA 微调，让模型学会你的写作风格、业务知识或专业领域
- **零命令行门槛**：双击启动，图形化界面完成所有操作

## 核心特性

### 🖥️ 极简架构
主程序仅几十 MB，无臃肿整合包。PyQt5 + QWebEngineView 构建现代化 Liquid Glass UI，支持暗色/亮色主题切换。

### 🔧 进程隔离设计
- **GUI 外壳**（`NeXT.exe` / `主程序入口.py`）：只负责界面渲染和用户交互
- **推理后端**（`推理.py`）：独立 Flask 进程处理模型加载和推理
- **训练引擎**（`训练.py`）：独立进程执行 LoRA 微调任务

三者物理隔离，UI 永远不会因模型 OOM 或训练崩溃而卡死。

### 📦 智能打包与部署
`build.py` 自动完成：
1. PyInstaller 打包主程序为单文件 exe
2. 递归搜索并补齐 PyQtWebEngine 缺失的 CEF 资源（`.pak`、`swiftshader` 等）
3. 下载并部署独立 Python 3.10 运行时
4. 将推理/训练脚本以"壳弹分离"方式部署到目标目录

### 🌐 弱网环境优化
基于 ModelScope 的模型下载器内置指数级退避重试机制，学校机房、公司内网等不稳定网络环境下也能断点续传。

### 🎨 温度参数可视化
前端 JS 插值算法将 LLM 的 Temperature 参数映射为界面背景色：代码问答时呈现冷蓝色调，创意写作时转为暖橙色。虽无实际功能，但提供直观的参数反馈。

## 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 显存 | 4GB | 8GB+ |
| 显卡 | GTX 1060 / GTX 1650 / 任意支持 CUDA 的计算卡 | RTX 3060 / RTX 4060 |
| 内存 | 8GB | 16GB+ |
| 硬盘 | 20GB 可用空间 | 50GB SSD |

**重要说明**：本项目针对低配硬件优化，老旧游戏本、办公机搭载的入门级独显均可运行。无需购买高端显卡。

## 快速开始

### 方式一：从源码构建（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/GeForce512/NeXT-Local.git
cd NeXT-Local

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. 构建本地 AI 环境（自动处理 PyTorch 等重型依赖）
python 环境.py

# 4. 下载基座模型（支持断点续传，模型存入 models/ 目录）
python 模型下载.py

# 5. 一键打包并部署到 F:\NeXT
python build.py

# 6. 运行打包后的程序
F:\NeXT\NeXT.exe
```

### 方式二：开发模式运行（双终端）

```bash
# 终端 1：启动推理后端
python 推理.py

# 终端 2：启动 GUI 界面
python 主程序入口.py
```

## 项目结构

```
NeXT-Local/
├── 主程序入口.py          # PyQt5 GUI 入口
├── 推理.py                # Flask 推理后端（独立进程）
├── 训练.py                # LoRA 微调引擎（独立进程）
├── 环境.py                # 环境初始化脚本
├── 模型下载.py            # ModelScope 模型下载器
├── build.py               # PyInstaller 打包脚本
├── env_setup.py           # 运行时环境部署脚本
├── download_all_models.py # 全量模型下载脚本
├── 前端/
│   ├── 主程序界面.html    # Web UI 结构
│   ├── 界面样式.css       # Liquid Glass 样式系统
│   └── 交互逻辑.js        # 前端业务逻辑 + QWebChannel 通信
├── models/                # 模型存储目录（需自行下载）
└── dist/NeXT/            # 打包输出目录
```

## 技术栈

- **前端**：HTML5 + CSS3 (Liquid Glass Design) + Vanilla JS
- **桌面框架**：PyQt5 + QWebEngineView + QWebChannel
- **后端通信**：Flask REST API（本地 127.0.0.1）
- **AI 引擎**：PyTorch + Transformers + PEFT (LoRA)
- **模型源**：ModelScope（魔搭社区）
- **打包工具**：PyInstaller

## 常见问题

### Q: 真的只需要 4GB 显存吗？
A: 是的。我们使用量化后的 7B 以下小模型（如 Qwen2.5-1.5B/3B/7B-Int4），配合梯度检查点和 CPU offload 技术，4GB 显存足以运行推理和轻量级 LoRA 微调。

### Q: 没有 NVIDIA 显卡能用吗？
A: 目前仅支持 CUDA 设备。AMD 显卡可通过 ROCm 支持（需手动配置 PyTorch ROCm 版本），Intel 集显暂不支持。

### Q: 如何替换为自己的数据集进行微调？
A: 在"训练"页面上传 JSONL 格式的对话数据，选择基座模型后点击"开始训练"即可。训练完成后会自动生成 LoRA adapter，可在推理时加载。

### Q: 打包后的 exe 为什么只有几十 MB？
A: PyInstaller 仅打包 GUI 外壳，Python 运行时、PyTorch、模型文件均以独立目录形式存在。这是刻意设计的"壳弹分离"架构，便于单独更新各组件。

## License

MIT License

---

**记住：真正的智能不应该被任何人垄断。在你自己的机器上，运行你自己的 AI。**
