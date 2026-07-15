// ============================================================
// NeXT - Interaction Logic (v2)
// ============================================================

// ==================== DOM References ====================
const navBtns = document.querySelectorAll('.toolbar-circle-btn[data-page]');
const chatSidebar = document.getElementById('chat-sidebar');
const historyList = document.getElementById('history-list');
const newChatBtn = document.getElementById('new-chat-btn');
const sidebarToggle = document.getElementById('sidebar-toggle');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');
const chatStopBtn = document.getElementById('chat-stop-btn');
const loraCapsule = document.getElementById('lora-capsule');
const capsuleText = document.getElementById('capsule-text');
const capsuleDropdown = document.getElementById('capsule-dropdown');
const magicSwitches = document.getElementById('magic-switches');
const themeToggleBtn = document.getElementById('theme-toggle-btn');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const datasetGrid = document.getElementById('dataset-grid');
const datasetEmpty = document.getElementById('dataset-empty');
const datasetSelect = document.getElementById('dataset-select');
const editorModal = document.getElementById('editor-modal');
const modalTitle = document.getElementById('modal-title');
const modalEditor = document.getElementById('modal-editor');
const modalSaveBtn = document.getElementById('modal-save-btn');
const modalDeleteBtn = document.getElementById('modal-delete-btn');
const modalCancelBtn = document.getElementById('modal-cancel-btn');
const loraGrid = document.getElementById('lora-grid');
const loraEmpty = document.getElementById('lora-empty');
const loraNameInput = document.getElementById('lora-name-input');
const baseLoraGroup = document.getElementById('base-lora-group');
const baseLoraSelect = document.getElementById('base-lora-select');
const trainModeRadios = document.querySelectorAll('input[name="train-mode"]');
const languageSelect = document.getElementById('language-select');
const toastEl = document.getElementById('toast');
const toastText = document.getElementById('toast-text');

const logAreas = {
    env: document.getElementById('env-log'),
    download: document.getElementById('download-log'),
    train: document.getElementById('train-log'),
    'settings-lora': document.getElementById('settings-lora-log')
};

// ==================== State ====================
const API_BASE = 'http://127.0.0.1:5000';
let isGenerating = false, isWakingUp = false, currentAbortController = null;
let currentEditingDataset = null, selectedLoraPath = null;
let sessions = [], currentSessionId = null;
let magicStates = { thinking: false };
let currentPage = 'chat';
let currentLang = 'zh-CN';

// Temperature hue animation state
let currentHue = 220, targetHue = 220, animFrame = null;

// GPU monitoring state (from Python nvidia-smi)
let sidebarGpuStats = [];

// ==================== i18n ====================
const i18n = {
    'zh-CN': {
        'nav.train': '训练', 'nav.chat': '对话', 'nav.settings': '设置', 'nav.model': '模型',
        'chat.new_chat': '新建对话', 'chat.model_switch': '模型切换', 'chat.history_toggle': '历史记录',
        'train.config_title': '训练配置', 'train.lora_name': 'LoRA 名称',
        'train.mode': '训练模式', 'train.mode_new': '全新训练', 'train.mode_continue': '增量训练',
        'train.base_lora': '基础 LoRA', 'train.dataset': '训练数据集',
        'train.start_btn': '开始训练', 'train.monitor_title': '训练监控',
        'train.dataset_title': '数据集管理',
        'train.drop_text': '拖拽 JSONL / JSON / TXT / CSV 到此处，或', 'train.click_upload': '点击上传',
        'train.no_dataset': '暂无数据集，请上传文件',
        'chat.history': '历史记录', 'chat.thinking': '深度思考',
        'chat.thinking_tip': '深度思考 (CoT)', 'chat.input_placeholder': '输入消息...',
        'chat.send': '发送', 'chat.stop': '停止',
        'model.download_title': '模型下载', 'model.download_desc': '从 ModelScope 下载基础模型到本地',
        'settings.env_title': '环境与依赖', 'settings.env_desc': '检测 GPU 设备并安装所需运行库（PyTorch / Transformers 等）',
        'settings.env_btn': '开始检测', 'settings.model_title': '模型管理',
        'settings.download_btn': '开始下载', 'settings.lora_title': 'LoRA 权重',
        'settings.refresh': '刷新', 'settings.delete': '删除选中', 'settings.switch': '切换选中',
        'settings.no_lora': '暂无 LoRA 权重',
        'settings.params_title': '训练参数', 'settings.lr': '学习率',
        'settings.epochs': '训练轮数 (Epochs)', 'settings.seqlen': '最大序列长度',
        'settings.batch_size': '批次大小 (Batch Size)', 'settings.optimizer': '优化器',
        'settings.weight_decay': '正则化系数 (Weight Decay)',
        'settings.lora_r': 'LoRA Rank (r)', 'settings.lora_alpha': 'LoRA Alpha',
        'settings.appearance_title': '外观与语言', 'settings.theme_mode': '主题模式',
        'settings.toggle_theme': '切换明暗', 'settings.language': '语言',
        'log.train': '训练日志', 'log.env': '检测日志', 'log.download': '下载日志', 'log.lora': '操作日志',
        'modal.edit_dataset': '编辑数据集', 'modal.delete': '删除', 'modal.save': '保存',
        'toast.first_launch': '请在设置中检查设备配置',
    },
    'en': {
        'nav.train': 'Train', 'nav.chat': 'Chat', 'nav.settings': 'Settings', 'nav.model': 'Model',
        'chat.new_chat': 'New Chat', 'chat.model_switch': 'Model Switch', 'chat.history_toggle': 'History',
        'train.config_title': 'Training Config', 'train.lora_name': 'LoRA Name',
        'train.mode': 'Training Mode', 'train.mode_new': 'New Training', 'train.mode_continue': 'Continuation',
        'train.base_lora': 'Base LoRA', 'train.dataset': 'Dataset',
        'train.start_btn': 'Start Training', 'train.monitor_title': 'Training Monitor',
        'train.dataset_title': 'Dataset Management',
        'train.drop_text': 'Drag JSONL / JSON / TXT / CSV here, or', 'train.click_upload': 'click to upload',
        'train.no_dataset': 'No datasets yet, please upload files',
        'chat.history': 'History', 'chat.thinking': 'Deep Think',
        'chat.thinking_tip': 'Deep Thinking (CoT)', 'chat.input_placeholder': 'Type a message...',
        'chat.send': 'Send', 'chat.stop': 'Stop',
        'model.download_title': 'Model Download', 'model.download_desc': 'Download base models from ModelScope to local',
        'settings.env_title': 'Environment & Dependencies',
        'settings.env_desc': 'Detect GPU devices and install required libraries (PyTorch / Transformers, etc.)',
        'settings.env_btn': 'Detect', 'settings.model_title': 'Model Management',
        'settings.download_btn': 'Download', 'settings.lora_title': 'LoRA Weights',
        'settings.refresh': 'Refresh', 'settings.delete': 'Delete Selected', 'settings.switch': 'Switch to Selected',
        'settings.no_lora': 'No LoRA weights yet',
        'settings.params_title': 'Training Parameters', 'settings.lr': 'Learning Rate',
        'settings.epochs': 'Epochs', 'settings.seqlen': 'Max Sequence Length',
        'settings.batch_size': 'Batch Size', 'settings.optimizer': 'Optimizer',
        'settings.weight_decay': 'Weight Decay',
        'settings.lora_r': 'LoRA Rank (r)', 'settings.lora_alpha': 'LoRA Alpha',
        'settings.appearance_title': 'Appearance & Language', 'settings.theme_mode': 'Theme Mode',
        'settings.toggle_theme': 'Toggle Theme', 'settings.language': 'Language',
        'log.train': 'Training Log', 'log.env': 'Detection Log', 'log.download': 'Download Log', 'log.lora': 'Operation Log',
        'modal.edit_dataset': 'Edit Dataset', 'modal.delete': 'Delete', 'modal.save': 'Save',
        'toast.first_launch': 'Please check device configuration in Settings',
    }
};

function applyLanguage(lang) {
    currentLang = lang;
    const dict = i18n[lang] || i18n['zh-CN'];
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (dict[key]) el.textContent = dict[key];
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.getAttribute('data-i18n-placeholder');
        if (dict[key]) el.placeholder = dict[key];
    });
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.getAttribute('data-i18n-title');
        if (dict[key]) el.title = dict[key];
    });
}

function showToast(msg, duration) {
    if (!toastEl || !toastText) return;
    duration = duration || 3500;
    toastText.textContent = msg;
    toastEl.classList.add('show');
    setTimeout(() => toastEl.classList.remove('show'), duration);
}

// ==================== Utility ====================
function appendLog(tabId, msg) {
    const area = logAreas[tabId];
    if (area) {
        const p = document.createElement('p');
        p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        area.appendChild(p);
        area.scrollTop = area.scrollHeight;
        // Auto-expand log section when new message arrives
        if (area.classList.contains('collapsed')) {
            area.classList.remove('collapsed');
            const toggle = area.parentElement?.querySelector('.log-toggle');
            if (toggle) toggle.classList.add('open');
        }
    }
}

function simpleMarkdown(text) {
    if (!text) return '';
    return text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
               .replace(/`([^`]+)`/g, '<code>$1</code>')
               .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
               .replace(/\n/g, '<br>');
}

// ==================== Theme ====================
function applyTheme(pageId) {
    const container = document.getElementById('app-container');
    if (!container) return;
    // No per-page hue themes — only light/dark
    if (window._isLightTheme) container.classList.add('light-theme');
}

function toggleTheme() {
    const container = document.getElementById('app-container');
    if (!container) return;
    container.classList.toggle('light-theme');
    window._isLightTheme = container.classList.contains('light-theme');
    // Redraw halftone overlays
    if (window._halftoneInstances) {
        window._halftoneInstances.forEach(inst => inst.forceRedraw());
    }
}

// ==================== Navigation ====================
function switchPage(pageId) {
    currentPage = pageId;
    document.querySelectorAll('.page-container').forEach(p => p.classList.remove('active'));
    const contentEl = document.getElementById(pageId + '-content');
    if (contentEl) contentEl.classList.add('active');
    navBtns.forEach(d => d.classList.toggle('active', d.dataset.page === pageId));
    applyTheme(pageId);
}

// ==================== Window Controls ====================
function initWindowControls() {
    const minBtn = document.getElementById('win-minimize');
    const maxBtn = document.getElementById('win-maximize');
    const closeBtn = document.getElementById('win-close');
    let isMaximized = false;
    if (minBtn) {
        minBtn.addEventListener('click', () => {
            if (window.chatObject) window.chatObject.minimizeWindow();
        });
    }
    if (maxBtn) {
        const maximizeSVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>';
        const restoreSVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="10" height="10" rx="1.5"/><rect x="10" y="10" width="10" height="10" rx="1.5"/></svg>';
        maxBtn.addEventListener('click', () => {
            if (!window.chatObject) return;
            if (isMaximized) {
                window.chatObject.restoreWindow();
                maxBtn.innerHTML = maximizeSVG;
                maxBtn.title = '最大化';
            } else {
                window.chatObject.maximizeWindow();
                maxBtn.innerHTML = restoreSVG;
                maxBtn.title = '还原';
            }
            isMaximized = !isMaximized;
        });
    }
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            if (window.chatObject) window.chatObject.closeWindow();
        });
    }
    // Listen for window state changes from Python
    if (window.chatObject && window.chatObject.windowStateChanged) {
        window.chatObject.windowStateChanged.connect((maximized) => {
            isMaximized = maximized;
            if (maxBtn) {
                const maximizeSVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>';
                const restoreSVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="10" height="10" rx="1.5"/><rect x="10" y="10" width="10" height="10" rx="1.5"/></svg>';
                maxBtn.innerHTML = maximized ? restoreSVG : maximizeSVG;
                maxBtn.title = maximized ? '还原' : '最大化';
            }
        });
    }
}

// ==================== Window Drag (QWebEngineView) ====================
function initWindowDrag() {
    // Non-draggable elements (buttons, groups, etc.)
    function isNoDragTarget(el) {
        while (el && el !== document.body) {
            if (el.tagName === 'BUTTON' || el.tagName === 'SELECT' || el.tagName === 'INPUT') return true;
            if (el.classList && (el.classList.contains('toolbar-circle-group') || el.classList.contains('capsule'))) return true;
            if (el.id === 'window-controls') return true;
            el = el.parentElement;
        }
        return false;
    }

    function handleDragStart(e) {
        if (e.button !== 0) return;
        if (isNoDragTarget(e.target)) return;
        e.preventDefault();
        // Use native Windows drag — OS handles everything
        if (window.chatObject && window.chatObject.startSystemDrag) {
            window.chatObject.startSystemDrag();
        }
    }

    // Attach mousedown to drag regions + toolbar
    const dragTargets = [
        document.getElementById('sidebar-drag-region'),
        document.getElementById('main-drag-region'),
        document.getElementById('top-toolbar')
    ].filter(Boolean);

    dragTargets.forEach(el => {
        el.addEventListener('mousedown', handleDragStart);
    });
}

// ==================== Per-Card HalftoneEngine ====================
class CardHalftoneEngine {
    constructor(card) {
        this.card = card;
        this.canvas = card.querySelector('.halftone-overlay');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.spacing = 28;
        this.baseRadius = 1.0;
        this.maxRadius = 4.5;
        this.mouseRadius = 200;
        this.mouseX = -9999;
        this.mouseY = -9999;
        this.running = false;
        this._boundDraw = null;
        this._needsDraw = false;
        this._mouseRafPending = false;

        if (!this.canvas || !this.ctx) return;

        this._resize();

        // Throttled mouse tracking relative to card
        this.card.addEventListener('mousemove', (e) => {
            const rect = this.card.getBoundingClientRect();
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
            if (!this._mouseRafPending && this.running) {
                this._mouseRafPending = true;
                requestAnimationFrame(() => { this._mouseRafPending = false; });
            }
        });
        this.card.addEventListener('mouseleave', () => {
            this.mouseX = -9999;
            this.mouseY = -9999;
        });

        // Start on hover, stop on leave
        this.card.addEventListener('mouseenter', () => this.start());
        this.card.addEventListener('mouseleave', () => this.stop());

        // Resize observer (debounced)
        if (window.ResizeObserver) {
            this._resizeTimer = null;
            this._ro = new ResizeObserver(() => {
                clearTimeout(this._resizeTimer);
                this._resizeTimer = setTimeout(() => { this._resize(); if (!this.running) this._drawOnce(); }, 100);
            });
            this._ro.observe(this.card);
        }
    }

    _resize() {
        if (!this.canvas) return;
        const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
        const rect = this.card.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;
        this.canvas.width = w * dpr;
        this.canvas.height = h * dpr;
        this.canvas.style.width = w + 'px';
        this.canvas.style.height = h + 'px';
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.w = w;
        this.h = h;
    }

    start() {
        if (this.running || !this.ctx) return;
        this.running = true;
        this._draw();
    }

    stop() {
        this.running = false;
        if (this._boundDraw) {
            cancelAnimationFrame(this._boundDraw);
            this._boundDraw = null;
        }
    }

    forceRedraw() {
        this._resize();
        this._drawOnce();
    }

    _drawOnce() {
        if (!this.ctx) return;
        this.mouseX = -9999;
        this.mouseY = -9999;
        this._renderFrame();
    }

    _draw() {
        if (!this.running || !this.ctx) return;
        this._renderFrame();
        this._boundDraw = requestAnimationFrame(() => this._draw());
    }

    _renderFrame() {
        const { ctx, spacing, baseRadius, maxRadius, mouseRadius, mouseX, mouseY, w, h } = this;
        if (!ctx || w <= 0 || h <= 0) return;
        const isLight = document.getElementById('main-content-wrapper')?.classList.contains('light-theme') || false;
        const baseAlpha = isLight ? 0.08 : 0.12;
        const mr2 = mouseRadius * mouseRadius;

        ctx.clearRect(0, 0, w, h);

        // Base dots
        ctx.fillStyle = isLight ? `rgba(0,0,0,${baseAlpha})` : `rgba(255,255,255,${baseAlpha})`;
        ctx.beginPath();
        for (let x = spacing / 2; x < w; x += spacing) {
            for (let y = spacing / 2; y < h; y += spacing) {
                const dx = x - mouseX, dy = y - mouseY;
                if (dx * dx + dy * dy >= mr2) {
                    ctx.moveTo(x + baseRadius, y);
                    ctx.arc(x, y, baseRadius, 0, 6.2832);
                }
            }
        }
        ctx.fill();

        // Enhanced dots near mouse
        if (mouseX > -999) {
            ctx.fillStyle = isLight ? 'rgba(0,0,0,0.30)' : 'rgba(255,255,255,0.35)';
            ctx.beginPath();
            for (let x = spacing / 2; x < w; x += spacing) {
                for (let y = spacing / 2; y < h; y += spacing) {
                    const dx = x - mouseX, dy = y - mouseY;
                    const d2 = dx * dx + dy * dy;
                    if (d2 < mr2) {
                        const ratio = 1 - Math.sqrt(d2) / mouseRadius;
                        const r = baseRadius + (maxRadius - baseRadius) * ratio * ratio;
                        ctx.moveTo(x + r, y);
                        ctx.arc(x, y, r, 0, 6.2832);
                    }
                }
            }
            ctx.fill();
        }
    }
}

// ==================== Temperature Hue Animation ====================
function applyTemperatureHue(temp) {
    // Smooth gradient: blue(210) → cyan(180) → teal(150) → green(120) → yellow-green(90)
    // temp range 0.1~2.0 maps to hue 210~90, gentle slope
    let hue = 210 - ((temp - 0.1) / 1.9) * 120;
    if (hue < 90) hue = 90;
    if (hue > 210) hue = 210;
    targetHue = Math.round(hue);
    if (!animFrame) animateHue();
}

function animateHue() {
    let diff = targetHue - currentHue;
    if (diff > 180) diff -= 360;
    if (diff < -180) diff += 360;
    if (Math.abs(diff) < 0.5) {
        currentHue = targetHue;
        document.documentElement.style.setProperty('--temp-hue', currentHue);
        animFrame = null;
        return;
    }
    currentHue += diff * 0.08;
    if (currentHue < 0) currentHue += 360;
    if (currentHue >= 360) currentHue -= 360;
    document.documentElement.style.setProperty('--temp-hue', Math.round(currentHue));
    animFrame = requestAnimationFrame(animateHue);
}

// ==================== Organic Blob Animation (Apple Watch Mindfulness Style) ====================
let blobContainer = null;
let blobElements = [];
let blobAnimFrame = null;
let blobStartTime = 0;

// Blob configuration - each blob has unique movement pattern
const BLOB_CONFIGS = [
    { size: 280, x: 0.15, y: 0.6, speedX: 0.3, speedY: 0.2, ampX: 80, ampY: 60, phase: 0 },
    { size: 220, x: 0.55, y: 0.7, speedX: 0.25, speedY: 0.35, ampX: 100, ampY: 50, phase: 1.5 },
    { size: 320, x: 0.8, y: 0.5, speedX: 0.2, speedY: 0.25, ampX: 70, ampY: 80, phase: 3.0 },
    { size: 180, x: 0.35, y: 0.8, speedX: 0.4, speedY: 0.15, ampX: 60, ampY: 40, phase: 4.5 },
    { size: 260, x: 0.7, y: 0.3, speedX: 0.15, speedY: 0.3, ampX: 90, ampY: 70, phase: 2.0 },
    { size: 200, x: 0.25, y: 0.4, speedX: 0.35, speedY: 0.2, ampX: 50, ampY: 60, phase: 5.5 },
];

function createBlobs() {
    if (blobContainer) return; // Already created

    const chatBottomBar = document.querySelector('.chat-bottom-bar');
    if (!chatBottomBar) return;

    blobContainer = document.createElement('div');
    blobContainer.className = 'blob-container';

    const hue = currentHue || 200;

    BLOB_CONFIGS.forEach((config, i) => {
        const blob = document.createElement('div');
        blob.className = 'blob';

        // Size
        blob.style.width = config.size + 'px';
        blob.style.height = config.size + 'px';

        // Color - vary saturation and lightness per blob for depth
        const sat = 70 + (i % 3) * 10; // 70-90%
        const light = 55 + (i % 2) * 10; // 55-65%
        const alpha = 0.5 + (i % 3) * 0.1; // 0.5-0.7
        blob.style.background = `radial-gradient(circle, hsla(${hue}, ${sat}%, ${light}%, ${alpha}) 0%, hsla(${hue}, ${sat}%, ${light}%, 0) 70%)`;

        // Initial position
        blob.style.left = '0px';
        blob.style.top = '0px';

        blobContainer.appendChild(blob);
        blobElements.push({
            el: blob,
            config: config,
            baseX: config.x,
            baseY: config.y
        });
    });

    chatBottomBar.appendChild(blobContainer);

    // Create halftone dot overlay
    const halftoneOverlay = document.createElement('div');
    halftoneOverlay.className = 'halftone-overlay';
    chatBottomBar.appendChild(halftoneOverlay);

    blobStartTime = performance.now();

    // Start animation
    animateBlobs();
}

function animateBlobs() {
    const now = performance.now();
    const elapsed = (now - blobStartTime) / 1000; // seconds
    const hue = currentHue || 200;

    blobElements.forEach((blob, i) => {
        const config = blob.config;

        // Organic movement using multiple sine waves
        const x = blob.baseX +
            Math.sin(elapsed * config.speedX + config.phase) * 0.08 +
            Math.sin(elapsed * config.speedX * 0.7 + config.phase * 1.3) * 0.04;

        const y = blob.baseY +
            Math.sin(elapsed * config.speedY + config.phase * 0.8) * 0.06 +
            Math.cos(elapsed * config.speedY * 0.5 + config.phase * 1.5) * 0.05;

        // Breathing scale effect
        const scale = 1 +
            Math.sin(elapsed * 0.3 + config.phase) * 0.15 +
            Math.sin(elapsed * 0.2 + config.phase * 0.7) * 0.1;

        // Convert to pixels
        const containerWidth = blobContainer.offsetWidth || 800;
        const containerHeight = blobContainer.offsetHeight || 400;

        const px = x * containerWidth - config.size / 2;
        const py = y * containerHeight - config.size / 2;

        // Update color dynamically
        const sat = 70 + (i % 3) * 10;
        const light = 55 + (i % 2) * 10;
        const alpha = 0.5 + (i % 3) * 0.1;
        blob.el.style.background = `radial-gradient(circle, hsla(${hue}, ${sat}%, ${light}%, ${alpha}) 0%, hsla(${hue}, ${sat}%, ${light}%, 0) 70%)`;

        blob.el.style.transform = `translate(${px}px, ${py}px) scale(${scale})`;
    });

    blobAnimFrame = requestAnimationFrame(animateBlobs);
}

function stopBlobs() {
    if (blobAnimFrame) {
        cancelAnimationFrame(blobAnimFrame);
        blobAnimFrame = null;
    }

    if (blobContainer) {
        blobContainer.remove();
        blobContainer = null;
    }

    // Remove halftone overlay
    const halftoneOverlay = document.querySelector('.chat-bottom-bar .halftone-overlay');
    if (halftoneOverlay) halftoneOverlay.remove();

    blobElements = [];
}

// ==================== Chat: Model Status ====================
async function checkInferenceReady() {
    try {
        const res = await fetch(API_BASE + '/api/model/info');
        if (res.status === 200) {
            const info = await res.json();
            if (info.status === 'ready') {
                capsuleText.textContent = `🟢 ${info.current_lora ? info.current_lora.split('/').pop() : 'Base Model'}`;
                // ★ 标记推理后端已启动
                isInferenceRunning = true;
                return true;
            }
            // ★ 返回加载进度信息
            if (info.load_progress) {
                return { loading: true, progress: info.load_progress };
            }
        }
    } catch (e) {}
    // ★ 如果请求失败，标记推理后端未运行
    isInferenceRunning = false;
    return false;
}

async function ensureModelReady() {
    const initialCheck = await checkInferenceReady();
    if (initialCheck === true) return true;

    isWakingUp = true;
    const dict = i18n[currentLang] || i18n['zh-CN'];
    chatSendBtn.disabled = true;
    chatInput.disabled = true;
    window.chatObject?.startChat();

    // ★ 显示加载进度，根据模型状态提供更明确的反馈
    for (let i = 0; i < 300; i++) {
        const checkResult = await checkInferenceReady();
        if (checkResult === true) break;

        // 优先使用 API 返回的加载进度
        if (checkResult && checkResult.loading) {
            const prog = checkResult.progress;
            const pct = prog.progress || 0;
            const msg = prog.message || '加载中...';
            chatSendBtn.textContent = `${pct}% ${msg}`;
        } else {
            // 根据当前模型状态显示不同的提示
            if (currentModelState === 'starting') {
                chatSendBtn.textContent = currentLang === 'en' ? 'Starting backend...' : '启动后端...';
            } else if (currentModelState === 'loading') {
                chatSendBtn.textContent = currentLang === 'en' ? 'Loading model...' : '加载模型...';
            } else {
                chatSendBtn.textContent = currentLang === 'en' ? 'Connecting...' : '连接中...';
            }
        }

        await new Promise(r => setTimeout(r, 1000));
    }

    isWakingUp = false;
    chatInput.disabled = false;
    updateSendBtnState();
    if (!(await checkInferenceReady())) {
        showToast(currentLang === 'en' ? 'Model wake-up timeout' : '模型唤醒超时');
        return false;
    }
    return true;
}

function updateSendBtnState() {
    const dict = i18n[currentLang] || i18n['zh-CN'];
    if (isGenerating) {
        chatSendBtn.style.display = 'none';
        chatStopBtn.style.display = 'inline-block';
    } else {
        chatSendBtn.style.display = 'inline-flex';
        chatStopBtn.style.display = 'none';
        chatSendBtn.textContent = dict['chat.send'] || '发送';
        chatSendBtn.disabled = !chatInput.value.trim();
    }
}

// ==================== Chat: Message Bubbles ====================
function createAssistantBubble() {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble assistant';
    const thinkBlock = document.createElement('div');
    thinkBlock.className = 'thinking-block';
    const thinkTitleText = currentLang === 'en' ? 'Thinking...' : '思考中...';
    thinkBlock.innerHTML = `<div class="thinking-header"><span class="thinking-icon">💭</span><span class="thinking-title">${thinkTitleText}</span></div><div class="thinking-content"></div>`;
    thinkBlock.querySelector('.thinking-header').onclick = () => thinkBlock.classList.toggle('open');
    thinkBlock.style.display = 'none';
    const answerBlock = document.createElement('div');
    answerBlock.className = 'answer-block';
    answerBlock.textContent = '...';
    bubble.appendChild(thinkBlock);
    bubble.appendChild(answerBlock);
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return { bubble, thinkBlock, answerBlock };
}

function appendMessage(role, text) {
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble ' + role;
    bubble.innerHTML = role === 'user' ? text : simpleMarkdown(text);
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    const sess = getCurrentSession();
    if (sess) sess.messages.push({ role, content: text });
}

// ==================== Chat: Send Message ====================
async function sendMessage(text) {
    appendMessage('user', text);
    saveCurrentSession();
    const { thinkBlock, answerBlock } = createAssistantBubble();
    const thinkContent = thinkBlock.querySelector('.thinking-content');
    const thinkTitle = thinkBlock.querySelector('.thinking-title');
    isGenerating = true;
    document.getElementById('main-content-wrapper').classList.add('inferencing');
    createBlobs();
    updateSendBtnState();
    const controller = new AbortController();
    currentAbortController = controller;
    let isThinking = false, thinkingText = '', answerText = '', pendingRender = false;

    function scheduleRender() {
        if (!pendingRender) {
            pendingRender = true;
            requestAnimationFrame(() => {
                answerBlock.innerHTML = simpleMarkdown(answerText);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                pendingRender = false;
            });
        }
    }

    try {
        const res = await fetch(API_BASE + '/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                history: getCurrentSession().messages.slice(0, -1),
                enable_thinking: magicStates.thinking
            }),
            signal: controller.signal
        });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            for (let line of chunk.split('\n')) {
                if (line.startsWith('data:')) {
                    const data = line.slice(5).trim();
                    if (data === '[DONE]') break;
                    try {
                        const json = JSON.parse(data);
                        if (json.meta && json.meta.temperature !== undefined) {
                            applyTemperatureHue(json.meta.temperature);
                        } else if (json.choices) {
                            let delta = json.choices[0].delta.content;
                            // ★ 检查思考标签（支持 <think> 和 </think>）
                            if (delta.includes('<think>')) {
                                isThinking = true;
                                thinkBlock.style.display = 'block';
                                thinkBlock.classList.add('open');
                                delta = delta.replace('<think>', '');
                            }
                            if (delta.includes('</think>')) {
                                isThinking = false;
                                thinkTitle.textContent = currentLang === 'en' ? 'Thinking Process (click to collapse)' : '思考过程（点击折叠）';
                                thinkBlock.classList.remove('open');
                                delta = delta.replace('</think>', '');
                            }
                            if (isThinking) {
                                thinkingText += delta;
                                thinkContent.innerHTML = simpleMarkdown(thinkingText);
                            } else {
                                answerText += delta;
                                scheduleRender();
                            }
                        } else if (json.final) {
                            answerText = json.full_text || answerText;
                            answerBlock.innerHTML = simpleMarkdown(answerText);
                        } else if (json.error) {
                            answerBlock.textContent = 'Error: ' + json.error;
                        }
                    } catch (e) {}
                }
            }
        }
        const sess = getCurrentSession();
        if (sess) {
            sess.messages.push({ role: 'assistant', content: answerText });
            saveCurrentSession();
        }
    } catch (err) {
        if (err.name === 'AbortError') answerBlock.innerHTML += '<br><em>(stopped)</em>';
        else answerBlock.textContent = 'Request failed: ' + err.message;
    } finally {
        isGenerating = false;
        currentAbortController = null;
        document.getElementById('main-content-wrapper').classList.remove('inferencing');
        stopBlobs();
        updateSendBtnState();
    }
}

// ==================== Chat: Sessions (File-based) ====================
function getCurrentSession() {
    if (!currentSessionId) {
        const title = currentLang === 'en' ? 'New Chat' : '新对话';
        currentSessionId = 'sess_' + Date.now();
        sessions.unshift({ id: currentSessionId, title: title, messages: [], timestamp: Date.now() });
    }
    return sessions.find(s => s.id === currentSessionId);
}

function saveCurrentSession() {
    const sess = getCurrentSession();
    if (!sess) return;
    const newTitle = currentLang === 'en' ? 'New Chat' : '新对话';
    if (sess.messages.length > 0 && sess.title === newTitle) {
        sess.title = sess.messages[0].content.substring(0, 20) + '...';
    }
    sess.timestamp = Date.now();
    if (window.chatObject) {
        window.chatObject.saveSessions(JSON.stringify(sessions));
    }
    renderHistoryList();
}

function renderHistoryList() {
    if (!historyList) return;
    historyList.innerHTML = '';
    sessions.forEach(sess => {
        const item = document.createElement('div');
        item.className = 'history-item' + (sess.id === currentSessionId ? ' active' : '');
        const titleSpan = document.createElement('span');
        titleSpan.textContent = sess.title;
        titleSpan.style.flex = '1';
        titleSpan.style.overflow = 'hidden';
        titleSpan.style.textOverflow = 'ellipsis';
        titleSpan.style.whiteSpace = 'nowrap';
        titleSpan.style.cursor = 'pointer';
        titleSpan.onclick = () => loadSession(sess.id);
        const delBtn = document.createElement('button');
        delBtn.className = 'history-delete-btn';
        delBtn.innerHTML = '&times;';
        delBtn.title = currentLang === 'en' ? 'Delete' : '删除';
        delBtn.onclick = (e) => {
            e.stopPropagation();
            deleteSession(sess.id);
        };
        item.style.display = 'flex';
        item.style.alignItems = 'center';
        item.style.gap = '4px';
        item.appendChild(titleSpan);
        item.appendChild(delBtn);
        historyList.appendChild(item);
    });
}

function deleteSession(id) {
    sessions = sessions.filter(s => s.id !== id);
    if (currentSessionId === id) {
        currentSessionId = sessions.length > 0 ? sessions[sessions.length - 1].id : null;
        chatMessages.innerHTML = '';
        if (currentSessionId) {
            const sess = sessions.find(s => s.id === currentSessionId);
            if (sess) sess.messages.forEach(m => appendMessage(m.role, m.content));
        }
    }
    if (window.chatObject) {
        window.chatObject.saveSessions(JSON.stringify(sessions));
    }
    renderHistoryList();
}

function loadSession(id) {
    currentSessionId = id;
    const sess = sessions.find(s => s.id === id);
    if (!sess) return;
    chatMessages.innerHTML = '';
    sess.messages.forEach(m => appendMessage(m.role, m.content));
    renderHistoryList();
}

function loadSessionsFromPython(jsonStr) {
    try {
        sessions = JSON.parse(jsonStr || '[]');
    } catch (e) {
        sessions = [];
    }
    renderHistoryList();
}

// ==================== Chat: Sidebar & Input ====================
if (newChatBtn) {
    newChatBtn.onclick = () => {
        currentSessionId = null;
        chatMessages.innerHTML = '';
        renderHistoryList();
    };
}

if (sidebarToggle && chatSidebar) {
    const hamburgerSVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h18M3 6h18M3 18h18"/></svg>';
    const closeSVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
    sidebarToggle.onclick = () => {
        chatSidebar.classList.toggle('collapsed');
        sidebarToggle.innerHTML = chatSidebar.classList.contains('collapsed') ? hamburgerSVG : closeSVG;
    };
}

if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !chatSendBtn.disabled && !isGenerating) {
            e.preventDefault();
            const text = chatInput.value.trim();
            if (text) {
                chatInput.value = '';
                sendMessage(text);
            }
        }
    });
    chatInput.addEventListener('input', () => {
        if (!isGenerating) updateSendBtnState();
    });
}

if (chatSendBtn) {
    chatSendBtn.onclick = () => {
        const text = chatInput.value.trim();
        if (text && !isGenerating) {
            chatInput.value = '';
            sendMessage(text);
        }
    };
}

if (chatStopBtn) {
    chatStopBtn.onclick = () => {
        if (currentAbortController) currentAbortController.abort();
        fetch(API_BASE + '/api/stop', { method: 'POST' }).catch(() => {});
    };
}

// ==================== Magic Switches ====================
if (magicSwitches) {
    magicSwitches.querySelectorAll('.switch-pill').forEach(pill => {
        pill.onclick = () => {
            pill.classList.toggle('active');
            magicStates[pill.dataset.feature] = pill.classList.contains('active');
        };
    });
}

// ==================== LoRA Capsule ====================
if (loraCapsule) {
    loraCapsule.addEventListener('click', (e) => {
        e.stopPropagation();
        loraCapsule.classList.toggle('open');
    });
    if (capsuleDropdown) {
        capsuleDropdown.addEventListener('click', (e) => e.stopPropagation());
    }
    document.addEventListener('click', (e) => {
        if (loraCapsule && !loraCapsule.contains(e.target)) {
            loraCapsule.classList.remove('open');
        }
    });
}

async function switchLoraViaCapsule(path) {
    // ★ 并发保护：如果模型正在关闭或 LoRA 正在切换中，忽略操作
    if (isModelClosing) {
        showToast(`⚠️ ${currentLang === 'en' ? 'Please wait for model to close' : '请等待模型关闭完成'}`, 3000);
        return;
    }
    if (isLoraSwitching) {
        showToast(`⚠️ ${currentLang === 'en' ? 'LoRA switching in progress' : 'LoRA 切换进行中'}`, 3000);
        return;
    }
    
    loraCapsule.classList.remove('open');
    if (!(await ensureModelReady())) return;
    
    isLoraSwitching = true;
    capsuleText.textContent = currentLang === 'en' ? ' Switching...' : '⏳ 切换中...';
    
    try {
        const res = await fetch(API_BASE + '/api/lora/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lora_path: path })
        });
        if (res.status === 200) {
            const data = await res.json();
            const loraName = path ? path.split('/').pop() : (currentLang === 'en' ? 'Base Model' : '基础模型');
            capsuleText.textContent = ` ${loraName}`;
            // ★ 显示更明显的成功提示（绿色背景，更长显示时间）
            showToast(`✅ ${currentLang === 'en' ? `LoRA switched to: ${loraName}` : `已切换到 LoRA: ${loraName}`}`, 5000);
        } else {
            const errorData = await res.json().catch(() => ({}));
            const errorMsg = errorData.error || (currentLang === 'en' ? 'Switch failed' : '切换失败');
            capsuleText.textContent = `🔴 ${currentLang === 'en' ? 'Failed' : '失败'}`;
            // ★ 显示更明显的失败提示（红色背景，更长显示时间）
            showToast(`❌ ${errorMsg}`, 6000);
            checkInferenceReady();
        }
    } catch (e) {
        capsuleText.textContent = `🔴 ${currentLang === 'en' ? 'Error' : '错误'}`;
        showToast(`❌ ${currentLang === 'en' ? 'Connection failed' : '连接失败'}: ${e.message}`, 6000);
    } finally {
        isLoraSwitching = false;
    }
}

function updateCapsuleDropdown(loras) {
    if (!capsuleDropdown) return;
    capsuleDropdown.innerHTML = '';
    const baseItem = document.createElement('div');
    baseItem.className = 'capsule-item';
    baseItem.textContent = currentLang === 'en' ? 'Base Model' : '基础模型';
    baseItem.onclick = () => switchLoraViaCapsule(null);
    capsuleDropdown.appendChild(baseItem);
    loras.forEach(lora => {
        const item = document.createElement('div');
        item.className = 'capsule-item';
        item.textContent = lora.name;
        item.onclick = () => switchLoraViaCapsule(lora.path);
        capsuleDropdown.appendChild(item);
    });
}

// ==================== Dataset Management ====================
function initDropZone() {
    if (!dropZone) return;
    dropZone.addEventListener('click', () => {
        if (window.chatObject && window.chatObject.importDatasetFiles) {
            window.chatObject.importDatasetFiles();
        } else if (fileInput) {
            fileInput.click();
        }
    });
    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
    });
    dropZone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files));
    fileInput.addEventListener('change', (e) => { handleFiles(e.target.files); fileInput.value = ''; });
}

function handleFiles(files) {
    Array.from(files).forEach(file => {
        const reader = new FileReader();
        reader.onload = (e) => {
            let content = e.target.result;
            let outputName = file.name;
            
            if (file.name.endsWith('.txt')) {
                // ★ TXT 文件：先尝试 JSON 解析，失败再按纯文本分块
                let isJson = false;
                try {
                    const dataObj = JSON.parse(content);
                    let items = [];
                    if (typeof dataObj === 'object' && dataObj !== null && 'data' in dataObj && Array.isArray(dataObj.data)) {
                        items = dataObj.data;
                    } else if (Array.isArray(dataObj)) {
                        items = dataObj;
                    }
                    if (items.length > 0) {
                        content = items.map(item => {
                            if (item.messages) return JSON.stringify(item);
                            else if (item.instruction && item.output !== undefined) {
                                const userContent = item.input ? `${item.instruction}\n\n${item.input}` : item.instruction;
                                return JSON.stringify({ messages: [{ role: 'user', content: userContent }, { role: 'assistant', content: item.output }] });
                            }
                            return JSON.stringify(item);
                        }).join('\n');
                        isJson = true;
                    }
                } catch (_) {}
                if (!isJson) {
                    // 纯文本分块逻辑
                    const chunks = content.split(/\n\s*\n|(?<=[。！？])\s*/).filter(c => c.trim().length > 50);
                    content = chunks.map(c => {
                        const splitIdx = Math.floor(c.length * 0.3);
                        return JSON.stringify({
                            messages: [
                                { role: 'user', content: "Please continue: " + c.substring(0, splitIdx) },
                                { role: 'assistant', content: c.substring(splitIdx) }
                            ]
                        });
                    }).join('\n');
                }
                outputName = file.name.replace('.txt', '.jsonl');
            } else if (file.name.endsWith('.json')) {
                // ★ JSON 文件：检测格式并转换为 JSONL
                try {
                    const dataObj = JSON.parse(content);
                    let items = [];
                    
                    if (typeof dataObj === 'object' && dataObj !== null && 'data' in dataObj && Array.isArray(dataObj.data)) {
                        // 格式：{"system": "...", "data": [...]}
                        items = dataObj.data;
                    } else if (Array.isArray(dataObj)) {
                        // 格式：[{...}, {...}]
                        items = dataObj;
                    }
                    
                    // 将每个 item 转换为标准的 messages 格式
                    content = items.map(item => {
                        if (item.messages) {
                            // 已经是标准格式
                            return JSON.stringify(item);
                        } else if (item.instruction && item.output !== undefined) {
                            // 指令格式：{instruction, input, output}
                            const userContent = item.input ? `${item.instruction}\n\n${item.input}` : item.instruction;
                            return JSON.stringify({
                                messages: [
                                    { role: 'user', content: userContent },
                                    { role: 'assistant', content: item.output }
                                ]
                            });
                        }
                        return JSON.stringify(item);
                    }).join('\n');
                    
                    outputName = file.name.replace('.json', '.jsonl');
                } catch (err) {
                    // JSON 解析失败，保持原样
                    console.warn('JSON parse failed, keeping as-is:', err);
                }
            }
            
            window.chatObject?.saveDataset(outputName, content);
        };
        reader.readAsText(file, 'UTF-8');
    });
}

function renderDatasetGrid(datasets) {
    if (!datasetGrid) return;
    datasetGrid.innerHTML = '';
    if (!datasets || datasets.length === 0) {
        if (datasetEmpty) datasetEmpty.style.display = 'block';
        return;
    }
    if (datasetEmpty) datasetEmpty.style.display = 'none';
    datasets.forEach(ds => {
        const card = document.createElement('div');
        card.className = 'dataset-card';
        const editLabel = currentLang === 'en' ? 'Edit' : '编辑';
        const deleteLabel = currentLang === 'en' ? 'Delete' : '删除';
        card.innerHTML = `<div class="card-info"><div class="card-title-text">${ds.name}</div><div class="card-meta">${ds.lines || 0} ${currentLang === 'en' ? 'items' : '条'}</div></div><div class="card-actions"><button class="card-btn edit-btn">${editLabel}</button><button class="card-btn danger delete-btn">${deleteLabel}</button></div>`;
        card.querySelector('.edit-btn').onclick = (e) => { e.stopPropagation(); openEditorModal(ds); };
        card.querySelector('.delete-btn').onclick = (e) => {
            e.stopPropagation();
            const msg = currentLang === 'en' ? 'Delete this dataset?' : '确认删除此数据集？';
            if (confirm(msg)) window.chatObject?.deleteDataset(ds.path);
        };
        datasetGrid.appendChild(card);
    });
}

function updateDatasetSelect(datasets) {
    if (!datasetSelect) return;
    const label = currentLang === 'en' ? 'Select dataset' : '选择数据集';
    datasetSelect.innerHTML = `<option value="">${label}</option>`;
    datasets.forEach(ds => {
        const opt = document.createElement('option');
        opt.value = ds.path;
        opt.textContent = ds.name;
        datasetSelect.appendChild(opt);
    });
}

// ==================== Dataset Editor Modal ====================
function openEditorModal(ds) {
    currentEditingDataset = ds;
    if (modalTitle) modalTitle.textContent = `${currentLang === 'en' ? 'Edit' : '编辑'}: ${ds.name}`;
    if (editorModal) editorModal.classList.add('active');
    window.chatObject?.readDatasetContent(ds.path);
}

function closeModal() {
    if (editorModal) editorModal.classList.remove('active');
}

if (modalCancelBtn) modalCancelBtn.onclick = closeModal;
if (modalSaveBtn) modalSaveBtn.onclick = () => {
    if (currentEditingDataset) {
        window.chatObject.updateDataset(currentEditingDataset.path, modalEditor.value);
        closeModal();
    }
};
if (modalDeleteBtn) modalDeleteBtn.onclick = () => {
    if (currentEditingDataset) {
        const msg = currentLang === 'en' ? 'Delete this dataset?' : '确认删除此数据集？';
        if (confirm(msg)) {
            window.chatObject.deleteDataset(currentEditingDataset.path);
            closeModal();
        }
    }
};

// ==================== LoRA Grid (Settings Page) ====================
function renderLoraGrid(loras) {
    if (!loraGrid) return;
    loraGrid.innerHTML = '';
    if (!loras || loras.length === 0) {
        if (loraEmpty) loraEmpty.style.display = 'block';
        return;
    }
    if (loraEmpty) loraEmpty.style.display = 'none';
    loras.forEach(lora => {
        const card = document.createElement('div');
        card.className = 'lora-card';
        card.innerHTML = `<div class="card-info"><div class="card-title-text">${lora.name}</div><div class="card-meta">${lora.created_at ? new Date(lora.created_at).toLocaleDateString() : ''}</div></div>`;
        card.onclick = () => {
            document.querySelectorAll('.lora-card').forEach(c => c.style.borderColor = '');
            card.style.borderColor = 'var(--accent-blue)';
            selectedLoraPath = lora.path;
        };
        loraGrid.appendChild(card);
    });
}

function updateBaseLoraSelect(loras) {
    if (!baseLoraSelect) return;
    const label = currentLang === 'en' ? 'Select base LoRA' : '选择基础 LoRA';
    baseLoraSelect.innerHTML = `<option value="">${label}</option>`;
    loras.forEach(lora => {
        const opt = document.createElement('option');
        opt.value = lora.path;
        opt.textContent = lora.name;
        baseLoraSelect.appendChild(opt);
    });
}

// ==================== Collapsible Log Toggles ====================
function initLogToggles() {
    document.querySelectorAll('.log-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-target');
            const logBox = document.getElementById(targetId);
            if (!logBox) return;
            logBox.classList.toggle('collapsed');
            btn.classList.toggle('open');
        });
    });
}

// ==================== Training ====================
trainModeRadios.forEach(radio => {
    radio.addEventListener('change', () => {
        // ★ 更新所有 radio-card 的 active 类
        document.querySelectorAll('.radio-card').forEach(card => {
            const input = card.querySelector('input[type="radio"]');
            card.classList.toggle('active', input && input.checked);
        });
        // 增量训练时显示基础 LoRA 选择
        const checked = document.querySelector('input[name="train-mode"]:checked');
        if (baseLoraGroup) baseLoraGroup.style.display = checked && checked.value === 'continue' ? 'block' : 'none';
    });
});

// ==================== Settings Page Bindings ====================
document.getElementById('env-btn')?.addEventListener('click', () => {
    appendLog('env', currentLang === 'en' ? 'Starting environment check...' : '开始环境检测...');
    if (window.chatObject) window.chatObject.checkEnv();
});

document.getElementById('download-btn')?.addEventListener('click', () => {
    const modelSelect = document.getElementById('model-select');
    const modelName = modelSelect ? modelSelect.value : 'Qwen3.5-4B';
    appendLog('download', (currentLang === 'en' ? 'Starting download: ' : '开始下载: ') + modelName);
    if (window.chatObject) window.chatObject.downloadModel(modelName);
});

document.getElementById('refresh-lora-btn')?.addEventListener('click', () => {
    if (window.chatObject) window.chatObject.getLoraWeights();
});

document.getElementById('delete-lora-btn')?.addEventListener('click', () => {
    if (selectedLoraPath) {
        const msg = currentLang === 'en' ? 'Delete selected LoRA?' : '确认删除选中的 LoRA？';
        if (confirm(msg)) {
            if (window.chatObject) window.chatObject.deleteLora(selectedLoraPath);
            selectedLoraPath = null;
        }
    } else {
        appendLog('settings-lora', currentLang === 'en' ? 'Please select a LoRA first' : '请先选择一个 LoRA');
    }
});

document.getElementById('switch-lora-btn')?.addEventListener('click', async () => {
    if (!selectedLoraPath) return appendLog('settings-lora', currentLang === 'en' ? 'Please select a LoRA first' : '请先选择一个 LoRA');
    if (!(await ensureModelReady())) return;
    appendLog('settings-lora', currentLang === 'en' ? 'Switching LoRA...' : '切换 LoRA 中...');
    try {
        const res = await fetch(API_BASE + '/api/lora/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lora_path: selectedLoraPath })
        });
        if (res.status === 200) {
            appendLog('settings-lora', currentLang === 'en' ? 'Switched successfully' : '切换成功');
            checkInferenceReady();
        } else {
            appendLog('settings-lora', currentLang === 'en' ? 'Switch failed' : '切换失败');
        }
    } catch (e) {
        appendLog('settings-lora', currentLang === 'en' ? 'Connection failed' : '连接失败');
    }
});

document.getElementById('train-btn')?.addEventListener('click', () => {
    const loraName = loraNameInput?.value.trim();
    if (!loraName) return appendLog('train', currentLang === 'en' ? 'Please enter a LoRA name' : '请输入 LoRA 名称');
    const datasetPath = datasetSelect?.value;
    if (!datasetPath) return appendLog('train', currentLang === 'en' ? 'Please select a dataset' : '请选择数据集');
    const mode = document.querySelector('input[name="train-mode"]:checked')?.value || 'new';
    const baseLora = baseLoraSelect?.value || '';
    // Read all training params
    const lr = document.getElementById('lr-input')?.value || '0.001';
    const epochs = document.getElementById('epochs-input')?.value || '4';
    const seqLen = document.getElementById('seqlen-input')?.value || '512';
    const batchSize = document.getElementById('batchsize-input')?.value || '4';
    const optimizer = document.getElementById('optimizer-select')?.value || 'adamw';
    const weightDecay = document.getElementById('weightdecay-input')?.value || '0.01';
    const loraR = document.getElementById('lorar-input')?.value || '16';
    const loraAlpha = document.getElementById('loraalpha-input')?.value || '32';

    appendLog('train', `${currentLang === 'en' ? 'Starting training' : '开始训练'}: ${loraName} (${mode})`);

    // Initialize sidebar training state
    sidebarTrainState.status = 'running';
    sidebarTrainState.loraName = loraName;
    sidebarTrainState.mode = mode === 'new' ? (currentLang === 'en' ? 'New' : '全新') : (currentLang === 'en' ? 'Continue' : '增量');
    sidebarTrainState.epoch = 0;
    sidebarTrainState.totalEpochs = parseInt(epochs) || 0;
    sidebarTrainState.step = 0;
    sidebarTrainState.totalSteps = 0;
    sidebarTrainState.loss = null;
    sidebarTrainState.lr = null;
    sidebarTrainState.lossHistory = [];  // ★ 清空历史数据
    sidebarTrainState.lrHistory = [];    // ★ 清空历史数据
    sidebarTrainState.startTime = Date.now();
    sidebarTrainState.logs = [];
    renderSidebarBody();
    startSidebarTimer();

    if (window.chatObject) {
        if (typeof window.chatObject.startTrain === 'function') {
            appendLog('train', `[JS] 调用 startTrain (JSON模式)...`);
            const trainConfig = JSON.stringify({
                lora_name: loraName,
                dataset_path: datasetPath,
                mode: mode,
                base_lora: baseLora,
                lr: lr,
                epochs: epochs,
                seq_len: seqLen,
                batch_size: batchSize,
                optimizer: optimizer,
                weight_decay: weightDecay,
                lora_r: loraR,
                lora_alpha: loraAlpha
            });
            window.chatObject.startTrain(trainConfig);
            appendLog('train', `[JS] startTrain 调用完成`);
        } else {
            appendLog('train', `[JS] startTrain 不是函数!`);
        }
    } else {
        appendLog('train', `[JS] window.chatObject 不存在!`);
    }
});

if (themeToggleBtn) {
    themeToggleBtn.onclick = toggleTheme;
}

// ==================== Language Select ====================
if (languageSelect) {
    languageSelect.addEventListener('change', () => {
        const lang = languageSelect.value;
        applyLanguage(lang);
        // Persist via Python
        if (window.chatObject) window.chatObject.saveLanguage(lang);
    });
}

// ==================== Python Signal Handlers ====================
function connectSignals() {
    const obj = window.chatObject;
    if (!obj) return;

    obj.envLog.connect((msg) => appendLog('env', msg));
    obj.modelLog.connect((msg) => addSidebarLog(msg));  // Model logs go to sidebar, NOT env log
    obj.downloadLog.connect((msg) => appendLog('download', msg));
    obj.trainLog.connect((msg) => {
        appendLog('train', msg);
        addSidebarLog(msg);
    });

    obj.dataLoaded.connect((data) => {
        try {
            const parsed = JSON.parse(data);
            if (parsed.type === 'datasets') {
                renderDatasetGrid(parsed.data);
                updateDatasetSelect(parsed.data);
            } else if (parsed.type === 'loras') {
                renderLoraGrid(parsed.data);
                updateBaseLoraSelect(parsed.data);
                updateCapsuleDropdown(parsed.data);
            }
        } catch (e) {}
    });

    obj.sessionsLoaded.connect((jsonStr) => {
        loadSessionsFromPython(jsonStr);
    });

    obj.gpuDetectResult.connect((jsonStr) => {
        // No longer used for wizard; GPU info is shown in settings if needed
    });

    obj.gpuStats.connect((jsonStr) => {
        try {
            sidebarGpuStats = JSON.parse(jsonStr);
            renderSidebarBody();
        } catch (e) {}
    });

    obj.modelState.connect((state) => {
        // Update model state in sidebar: starting|loading|running|closed
        currentModelState = state;
        if (state === 'starting' || state === 'loading') {
            isInferenceRunning = true;
            renderSidebarBody();
        } else if (state === 'running') {
            isInferenceRunning = true;
            isModelClosing = false; // Reset closing flag when backend is running
            renderSidebarBody();
        } else if (state === 'closed') {
            isInferenceRunning = false;
            isModelClosing = false; // Reset closing flag
            renderSidebarBody();
        }
    });

    obj.envDone.connect((success) => {
        appendLog('env', success ? (currentLang === 'en' ? 'Environment check complete' : '环境检测完成') : (currentLang === 'en' ? 'Environment check failed' : '环境检测失败'));
    });

    obj.downloadDone.connect((success) => {
        appendLog('download', success ? (currentLang === 'en' ? 'Download complete' : '下载完成') : (currentLang === 'en' ? 'Download failed' : '下载失败'));
    });

    obj.trainMetrics.connect((jsonStr) => {
        try {
            const m = JSON.parse(jsonStr);
            // Update sidebar training state
            if (m.loss !== undefined) {
                sidebarTrainState.loss = m.loss;
                sidebarTrainState.lossHistory.push(m.loss);
            }
            if (m.lr !== undefined) {
                sidebarTrainState.lr = m.lr;
                sidebarTrainState.lrHistory.push(m.lr);
            }
            if (m.epoch !== undefined) sidebarTrainState.epoch = m.epoch;
            if (m.step !== undefined) sidebarTrainState.step = m.step;
            if (m.total_steps !== undefined) sidebarTrainState.totalSteps = m.total_steps;
            renderSidebarBody();
        } catch (e) {}
    });

    obj.trainDone.connect((success) => {
        appendLog('train', success ? (currentLang === 'en' ? 'Training complete!' : '训练完成！') : (currentLang === 'en' ? 'Training failed' : '训练失败'));
        // Update sidebar state
        sidebarTrainState.status = success ? 'complete' : 'error';
        if (sidebarTimerInterval) { clearInterval(sidebarTimerInterval); sidebarTimerInterval = null; }
        addSidebarLog(success ? (currentLang === 'en' ? 'Training complete!' : '训练完成！') : (currentLang === 'en' ? 'Training failed' : '训练失败'));
        // ★ Clear startTime to stop elapsed time counter
        sidebarTrainState.startTime = null;
        renderSidebarBody();
    });

    obj.datasetContent.connect((content) => {
        if (modalEditor) modalEditor.value = content;
    });

    obj.fileDialogResult.connect((jsonStr) => {
        try {
            const result = JSON.parse(jsonStr);
            if (result.error) {
                showToast(result.error, 3000);
            }
        } catch (e) {}
    });

    // First launch: show toast instead of wizard
    obj.firstLaunchResult.connect((isFirst) => {
        if (isFirst) {
            const dict = i18n[currentLang] || i18n['zh-CN'];
            showToast(dict['toast.first_launch'] || '请在设置中检查设备配置', 4000);
            if (window.chatObject) window.chatObject.completeSetup();
        }
    });

    // Language loaded from Python
    obj.languageLoaded.connect((lang) => {
        if (lang && i18n[lang]) {
            currentLang = lang;
            if (languageSelect) languageSelect.value = lang;
            applyLanguage(lang);
        }
    });
}

// ==================== Sidebar Training Progress Panel ====================
let sidebarTrainState = {
    status: 'idle', // idle | running | complete | error
    loraName: '',
    mode: '',
    epoch: 0,
    totalEpochs: 0,
    step: 0,
    totalSteps: 0,
    loss: null,
    lr: null,
    lossHistory: [],  // ★ 历史loss数据用于绘图
    lrHistory: [],    // ★ 历史LR数据用于绘图
    startTime: null,
    logs: []
};

// ★ 推理后端状态（用于决定是否显示"关闭模型"按钮）
let isInferenceRunning = false;
let currentModelState = 'closed'; // closed|starting|loading|running

// ★ 并发操作保护：防止在模型关闭过程中进行其他操作
let isModelClosing = false; // 模型正在关闭中
let isLoraSwitching = false; // LoRA 正在切换中

// ==================== Simple Line Chart Drawing ====================
function drawLineChart(canvasId, data, color, label) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !data || data.length === 0) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const padding = 20;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Find min/max for scaling
    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);
    const range = maxVal - minVal || 1;
    
    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padding + (height - 2 * padding) * i / 4;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
    }
    
    // Draw the line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();
    
    data.forEach((val, i) => {
        const x = padding + (width - 2 * padding) * i / (data.length - 1 || 1);
        const y = padding + (height - 2 * padding) * (1 - (val - minVal) / range);
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
    
    // Fill area under line
    ctx.globalAlpha = 0.1;
    ctx.fillStyle = color;
    ctx.lineTo(padding + (width - 2 * padding), height - padding);
    ctx.lineTo(padding, height - padding);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1.0;
}

function renderSidebarBody() {
    const body = document.querySelector('.sidebar-body');
    if (!body) return;

    const isTraining = sidebarTrainState.status === 'running';
    const isComplete = sidebarTrainState.status === 'complete';
    const isError = sidebarTrainState.status === 'error';
    const isIdle = sidebarTrainState.status === 'idle';

    const statusClass = isTraining ? 'running' : isError ? 'error' : 'idle';
    const statusText = isTraining ? (currentLang === 'en' ? 'Running' : '训练中')
        : isComplete ? (currentLang === 'en' ? 'Complete' : '已完成')
        : isError ? (currentLang === 'en' ? 'Error' : '错误')
        : (currentLang === 'en' ? 'Idle' : '空闲');

    const progressPct = sidebarTrainState.totalSteps > 0
        ? Math.round((sidebarTrainState.step / sidebarTrainState.totalSteps) * 100)
        : (isComplete ? 100 : 0);

    const elapsed = sidebarTrainState.startTime
        ? Math.floor((Date.now() - sidebarTrainState.startTime) / 1000)
        : 0;
    const elapsedStr = formatTime(elapsed);
    const etaStr = (isTraining && progressPct > 0)
        ? formatTime(Math.round(elapsed * (100 / progressPct - 1)))
        : '--:--:--';

    // ★ Model state display
    const modelStatusText = isInferenceRunning 
        ? (currentLang === 'en' ? '🟢 Running' : '🟢 运行中')
        : isModelClosing
        ? (currentLang === 'en' ? '⏳ Closing...' : '⏳ 关闭中...')
        : (currentLang === 'en' ? '🔴 Closed' : '🔴 已关闭');
    const modelStatusClass = isInferenceRunning ? 'running' : isModelClosing ? '' : 'idle';

    body.innerHTML = `
        <div class="sidebar-section">
            <div class="sidebar-section-title">${currentLang === 'en' ? 'Model Status' : '模型状态'}</div>
            <div class="sidebar-status-card">
                <div class="sidebar-status-row">
                    <span class="sidebar-status-label">${currentLang === 'en' ? 'Status' : '状态'}</span>
                    <span class="sidebar-status-value ${modelStatusClass}">${modelStatusText}</span>
                </div>
                ${(isInferenceRunning || isModelClosing) ? `
                <button id="close-model-btn" class="sidebar-close-btn ${isModelClosing ? 'disabled' : ''}" style="margin-top:12px;width:100%;" ${isModelClosing ? 'disabled' : ''}>
                    ${isModelClosing ? (currentLang === 'en' ? '⏳ Closing...' : '⏳ 关闭中...') : (currentLang === 'en' ? ' Close Model' : '🔴 关闭模型')}
                </button>
                ` : ''}
            </div>
        </div>
        ${(isTraining || isComplete) ? `
        <div class="sidebar-section">
            <div class="sidebar-section-title">${currentLang === 'en' ? 'Training Status' : '训练状态'}</div>
            <div class="sidebar-status-card">
                <div class="sidebar-status-row">
                    <span class="sidebar-status-label">${currentLang === 'en' ? 'Status' : '状态'}</span>
                    <span class="sidebar-status-value ${statusClass}">${statusText}</span>
                </div>
                ${!isIdle ? `
                <div class="sidebar-status-row">
                    <span class="sidebar-status-label">${currentLang === 'en' ? 'LoRA' : '名称'}</span>
                    <span class="sidebar-status-value">${sidebarTrainState.loraName || '--'}</span>
                </div>
                <div class="sidebar-status-row">
                    <span class="sidebar-status-label">${currentLang === 'en' ? 'Mode' : '模式'}</span>
                    <span class="sidebar-status-value">${sidebarTrainState.mode || '--'}</span>
                </div>
                <div class="sidebar-status-row">
                    <span class="sidebar-status-label">${currentLang === 'en' ? 'Epoch' : '轮次'}</span>
                    <span class="sidebar-status-value">${sidebarTrainState.epoch}/${sidebarTrainState.totalEpochs || '?'}</span>
                </div>
                ` : ''}
                ${isTraining || isComplete ? `
                <div class="sidebar-progress-bar">
                    <div class="sidebar-progress-fill ${isComplete ? 'complete' : ''}" style="width:${progressPct}%"></div>
                </div>
                ` : ''}
            </div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-section-title">${currentLang === 'en' ? 'Metrics' : '实时指标'}</div>
            <div class="sidebar-metrics-grid" style="grid-template-columns: 1fr 1fr; gap: 12px;">
                <div class="sidebar-metric-item" style="display: flex; flex-direction: column; align-items: center;">
                    <canvas id="loss-chart" width="180" height="80" style="width: 100%; height: 80px; background: rgba(0,0,0,0.2); border-radius: 6px;"></canvas>
                    <div class="sidebar-metric-label" style="margin-top: 4px;">Loss ${sidebarTrainState.loss !== null ? sidebarTrainState.loss.toFixed(4) : '--'}</div>
                </div>
                <div class="sidebar-metric-item" style="display: flex; flex-direction: column; align-items: center;">
                    <canvas id="lr-chart" width="180" height="80" style="width: 100%; height: 80px; background: rgba(0,0,0,0.2); border-radius: 6px;"></canvas>
                    <div class="sidebar-metric-label" style="margin-top: 4px;">LR ${sidebarTrainState.lr !== null ? sidebarTrainState.lr.toExponential(2) : '--'}</div>
                </div>
                <div class="sidebar-metric-item">
                    <div class="sidebar-metric-value">${elapsedStr}</div>
                    <div class="sidebar-metric-label">${currentLang === 'en' ? 'Elapsed' : '已用时间'}</div>
                </div>
                <div class="sidebar-metric-item">
                    <div class="sidebar-metric-value">${etaStr}</div>
                    <div class="sidebar-metric-label">ETA</div>
                </div>
            </div>
        </div>
        ` : ''}
        <div class="sidebar-section">
            <div class="sidebar-section-title">${currentLang === 'en' ? 'Live Feed' : '实时反馈'}</div>
            <div class="sidebar-live-log" id="sidebar-log">
                ${sidebarTrainState.logs.length === 0 && !isInferenceRunning
                    ? `<div class="sidebar-empty-state">${currentLang === 'en' ? 'No activity yet' : '暂无活动'}</div>`
                    : sidebarTrainState.logs.slice(-20).map(l => `<p>${l}</p>`).join('')}
            </div>
        </div>
        <div class="sidebar-section">
            <div class="sidebar-section-title">${currentLang === 'en' ? 'GPU Monitor' : '显卡监控'}</div>
            ${sidebarGpuStats.length === 0
                ? `<div class="sidebar-status-card"><div class="sidebar-empty-state">${currentLang === 'en' ? 'No GPU detected' : '未检测到显卡'}</div></div>`
                : sidebarGpuStats.map((gpu, i) => `
                    <div class="sidebar-status-card" style="margin-bottom:8px;">
                        <div class="sidebar-status-row">
                            <span class="sidebar-status-label">GPU ${i}</span>
                            <span class="sidebar-status-value ${gpu.util > 90 ? 'running' : gpu.util > 70 ? '' : 'idle'}">${gpu.util}%</span>
                        </div>
                        <div class="sidebar-progress-bar">
                            <div class="sidebar-progress-fill" style="width:${gpu.util}%"></div>
                        </div>
                        <div class="sidebar-metrics-grid" style="margin-top:8px;">
                            <div class="sidebar-metric-item">
                                <div class="sidebar-metric-value">${gpu.mem_used}MB</div>
                                <div class="sidebar-metric-label">${currentLang === 'en' ? 'VRAM Used' : '显存已用'}</div>
                            </div>
                            <div class="sidebar-metric-item">
                                <div class="sidebar-metric-value">${gpu.mem_total}MB</div>
                                <div class="sidebar-metric-label">${currentLang === 'en' ? 'VRAM Total' : '显存总量'}</div>
                            </div>
                        </div>
                        <div class="sidebar-status-row" style="margin-top:4px;">
                            <span class="sidebar-status-label">${currentLang === 'en' ? 'Temperature' : '温度'}</span>
                            <span class="sidebar-status-value ${gpu.temp > 85 ? 'error' : gpu.temp > 70 ? '' : 'idle'}">${gpu.temp}°C</span>
                        </div>
                    </div>
                `).join('')}
        </div>
    `;

    // ★ 绘制 Loss 和 LR 曲线图
    setTimeout(() => {
        drawLineChart('loss-chart', sidebarTrainState.lossHistory, '#ff6b6b', 'Loss');
        drawLineChart('lr-chart', sidebarTrainState.lrHistory, '#4ecdc4', 'LR');
    }, 0);

    // Auto-scroll live log
    const logEl = document.getElementById('sidebar-log');
    if (logEl) logEl.scrollTop = logEl.scrollHeight;
}

function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
}

function addSidebarLog(msg) {
    const time = new Date().toLocaleTimeString();
    sidebarTrainState.logs.push(`[${time}] ${msg}`);
    if (sidebarTrainState.logs.length > 100) sidebarTrainState.logs.shift();
    // Update just the log section if possible (avoid full re-render)
    const logEl = document.getElementById('sidebar-log');
    if (logEl) {
        const emptyState = logEl.querySelector('.sidebar-empty-state');
        if (emptyState) emptyState.remove();
        const p = document.createElement('p');
        p.textContent = `[${time}] ${msg}`;
        logEl.appendChild(p);
        logEl.scrollTop = logEl.scrollHeight;
    }
}

// Update sidebar timer every second during training
let sidebarTimerInterval = null;
function startSidebarTimer() {
    if (sidebarTimerInterval) clearInterval(sidebarTimerInterval);
    sidebarTimerInterval = setInterval(() => {
        if (sidebarTrainState.status === 'running') {
            // Update elapsed/ETA without full re-render
            const metricValues = document.querySelectorAll('.sidebar-metric-value');
            if (metricValues.length >= 4) {
                const elapsed = Math.floor((Date.now() - sidebarTrainState.startTime) / 1000);
                metricValues[2].textContent = formatTime(elapsed);
                const progressPct = sidebarTrainState.totalSteps > 0
                    ? Math.round((sidebarTrainState.step / sidebarTrainState.totalSteps) * 100) : 0;
                metricValues[3].textContent = progressPct > 0
                    ? formatTime(Math.round(elapsed * (100 / progressPct - 1)))
                    : '--:--:--';
            }
        } else {
            clearInterval(sidebarTimerInterval);
            sidebarTimerInterval = null;
        }
    }, 1000);
}

// ==================== Initialization ====================
function init() {
    // Per-card halftone engines
    window._halftoneInstances = [];
    document.querySelectorAll('.halftone-target').forEach(card => {
        const engine = new CardHalftoneEngine(card);
        window._halftoneInstances.push(engine);
    });

    // Navigation
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => switchPage(btn.dataset.page));
    });

    // Tutorial button — show tutorial page
    const actionTutorial = document.getElementById('action-tutorial');
    if (actionTutorial) actionTutorial.addEventListener('click', () => {
        document.querySelectorAll('.page-container').forEach(p => p.classList.remove('active'));
        const tutEl = document.getElementById('tutorial-content');
        if (tutEl) tutEl.classList.add('active');
        navBtns.forEach(d => d.classList.remove('active'));
    });
    // Tutorial back button — return to chat
    const tutorialBackBtn = document.getElementById('tutorial-back-btn');
    if (tutorialBackBtn) tutorialBackBtn.addEventListener('click', () => {
        switchPage('chat');
    });

    // Sidebar collapse/expand toggle — controlled by toolbar button only
    function toggleGlassSidebar() {
        const sidebar = document.getElementById('glass-sidebar');
        if (!sidebar) return;
        sidebar.classList.toggle('collapsed');
    }
    const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
    if (sidebarToggleBtn) sidebarToggleBtn.addEventListener('click', toggleGlassSidebar);

    // Window controls
    initWindowControls();
    initWindowDrag();

    // Collapsible log toggles
    initLogToggles();

    // Apply initial theme
    applyTheme('chat');

    // Drop zone
    initDropZone();

    // Render sidebar training progress (initial idle state)
    renderSidebarBody();

    // Attach close model button handler (called after renderSidebarBody)
    function attachCloseModelHandler() {
        const closeModelBtn = document.getElementById('close-model-btn');
        if (closeModelBtn) {
            closeModelBtn.onclick = () => {
                // ★ 并发保护：如果已经在关闭中，忽略点击
                if (isModelClosing) return;
                
                isModelClosing = true;
                renderSidebarBody(); // 重新渲染以显示"关闭中..."状态
                
                if (window.chatObject) {
                    window.chatObject.stopInference();
                    showToast(currentLang === 'en' ? 'Closing model...' : '正在关闭模型...');
                    
                    // Note: Don't set isInferenceRunning = false here
                    // Wait for modelState signal to emit 'closed'
                } else {
                    isModelClosing = false;
                    currentModelState = 'closed';
                    renderSidebarBody();
                    attachCloseModelHandler(); // Re-attach for next time
                }
            };
        }
    }
    attachCloseModelHandler();

    // Connect Python signals
    connectSignals();

    // Request initial data
    if (window.chatObject) {
        window.chatObject.isFirstLaunch();
        window.chatObject.loadSessions();
        window.chatObject.getDatasetList();
        window.chatObject.getLoraWeights();
        window.chatObject.loadLanguage();
        window.chatObject.startGpuMonitor();
    }
}

// Wait for QWebChannel to be ready
function waitForChatObject(cb) {
    if (window.chatObject) {
        cb();
    } else {
        let attempts = 0;
        const timer = setInterval(() => {
            attempts++;
            if (window.chatObject || attempts > 50) {
                clearInterval(timer);
                cb();
            }
        }, 100);
    }
}

waitForChatObject(init);
