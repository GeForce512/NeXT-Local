// ============================================================
// NeXT - Interaction Logic (v2)
// ============================================================

// ==================== DOM References ====================
const navBtns = document.querySelectorAll('.nav-btn');
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
let currentPage = 'train';
let currentLang = 'zh-CN';

// Temperature hue animation state
let currentHue = 220, targetHue = 220, animFrame = null;

// ==================== i18n ====================
const i18n = {
    'zh-CN': {
        'nav.train': '训练', 'nav.chat': '推理', 'nav.settings': '设置',
        'train.config_title': '训练配置', 'train.lora_name': 'LoRA 名称',
        'train.mode': '训练模式', 'train.mode_new': '全新训练', 'train.mode_continue': '增量训练',
        'train.base_lora': '基础 LoRA', 'train.dataset': '训练数据集',
        'train.start_btn': '开始训练', 'train.monitor_title': '训练监控',
        'train.dataset_title': '数据集管理',
        'train.drop_text': '拖拽 JSONL / TXT / CSV 到此处，或', 'train.click_upload': '点击上传',
        'train.no_dataset': '暂无数据集，请上传文件',
        'chat.history': '历史记录', 'chat.thinking': '深度思考',
        'chat.thinking_tip': '深度思考 (CoT)', 'chat.input_placeholder': '输入消息...',
        'chat.send': '发送', 'chat.stop': '停止',
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
        'nav.train': 'Train', 'nav.chat': 'Chat', 'nav.settings': 'Settings',
        'train.config_title': 'Training Config', 'train.lora_name': 'LoRA Name',
        'train.mode': 'Training Mode', 'train.mode_new': 'New Training', 'train.mode_continue': 'Continuation',
        'train.base_lora': 'Base LoRA', 'train.dataset': 'Dataset',
        'train.start_btn': 'Start Training', 'train.monitor_title': 'Training Monitor',
        'train.dataset_title': 'Dataset Management',
        'train.drop_text': 'Drag JSONL / TXT / CSV here, or', 'train.click_upload': 'click to upload',
        'train.no_dataset': 'No datasets yet, please upload files',
        'chat.history': 'History', 'chat.thinking': 'Deep Think',
        'chat.thinking_tip': 'Deep Thinking (CoT)', 'chat.input_placeholder': 'Type a message...',
        'chat.send': 'Send', 'chat.stop': 'Stop',
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
const themeMap = { train: 'theme-train', chat: 'theme-chat', settings: 'theme-settings' };

function applyTheme(pageId) {
    document.body.className = document.body.className.replace(/theme-\S+/g, '').trim();
    if (themeMap[pageId]) document.body.classList.add(themeMap[pageId]);
    // Restore light/dark from memory variable (no localStorage)
    if (window._isLightTheme) document.body.classList.add('light-theme');
}

function toggleTheme() {
    document.body.classList.toggle('light-theme');
    window._isLightTheme = document.body.classList.contains('light-theme');
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
        maxBtn.addEventListener('click', () => {
            if (!window.chatObject) return;
            if (isMaximized) {
                window.chatObject.restoreWindow();
                maxBtn.innerHTML = '&#x25A1;';
                maxBtn.title = '最大化';
            } else {
                window.chatObject.maximizeWindow();
                maxBtn.innerHTML = '&#x29C9;';
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
                maxBtn.innerHTML = maximized ? '&#x29C9;' : '&#x25A1;';
                maxBtn.title = maximized ? '还原' : '最大化';
            }
        });
    }
}

// ==================== Window Drag (QWebEngineView) ====================
function initWindowDrag() {
    const dragRegion = document.getElementById('drag-region');
    if (!dragRegion) return;

    let dragging = false;
    let winX = 0, winY = 0;
    let lastScreenX = 0, lastScreenY = 0;

    dragRegion.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        dragging = true;
        lastScreenX = e.screenX;
        lastScreenY = e.screenY;
        const rect = dragRegion.getBoundingClientRect();
        winX = Math.round(e.screenX - e.clientX + rect.left);
        winY = Math.round(e.screenY - e.clientY + rect.top);
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!dragging || !window.chatObject) return;
        const dx = e.screenX - lastScreenX;
        const dy = e.screenY - lastScreenY;
        if (dx === 0 && dy === 0) return;
        lastScreenX = e.screenX;
        lastScreenY = e.screenY;
        winX += dx;
        winY += dy;
        window.chatObject.moveWindow(winX, winY);
    });

    document.addEventListener('mouseup', () => {
        dragging = false;
    });
}

// ==================== Per-Card HalftoneEngine ====================
class CardHalftoneEngine {
    constructor(card) {
        this.card = card;
        this.canvas = card.querySelector('.halftone-overlay');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.spacing = 24;
        this.baseRadius = 1.0;
        this.maxRadius = 4.0;
        this.mouseRadius = 180;
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
        const isLight = document.body.classList.contains('light-theme');
        const baseAlpha = isLight ? 0.06 : 0.04;
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
            ctx.fillStyle = isLight ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.1)';
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

// ==================== TrainingChart ====================
class TrainingChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.activeTab = 'loss';
        this.lossData = [];
        this.lrData = [];
        this.maxPoints = 100;
        this._resize();
        window.addEventListener('resize', () => { this._resize(); this.draw(); });
    }

    _resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.w = rect.width;
        this.h = rect.height;
    }

    addPoint(loss, lr) {
        this.lossData.push(loss);
        this.lrData.push(lr);
        if (this.lossData.length > this.maxPoints) {
            this.lossData.shift();
            this.lrData.shift();
        }
        this.draw();
    }

    setTab(tab) {
        this.activeTab = tab;
        this.draw();
    }

    clear() {
        this.lossData = [];
        this.lrData = [];
        this.draw();
    }

    draw() {
        const { ctx, w, h } = this;
        const pad = { top: 24, right: 20, bottom: 32, left: 56 };
        const cw = w - pad.left - pad.right;
        const ch = h - pad.top - pad.bottom;

        ctx.clearRect(0, 0, w, h);

        const data = this.activeTab === 'loss' ? this.lossData : this.lrData;
        const label = this.activeTab === 'loss' ? 'Loss' : 'Learning Rate';
        const color = this.activeTab === 'loss' ? '#ff5c5c' : '#3dd68c';

        // Grid lines
        const isLight = document.body.classList.contains('light-theme');
        ctx.strokeStyle = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = pad.top + (ch / 4) * i;
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(pad.left + cw, y);
            ctx.stroke();
        }

        if (data.length < 2) {
            ctx.fillStyle = isLight ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.3)';
            ctx.font = '13px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for data...', w / 2, h / 2);
            return;
        }

        // Y-axis range
        let minV = Infinity, maxV = -Infinity;
        for (let i = 0; i < data.length; i++) {
            if (data[i] < minV) minV = data[i];
            if (data[i] > maxV) maxV = data[i];
        }
        const range = maxV - minV || 1;
        minV -= range * 0.1;
        maxV += range * 0.1;

        // Y labels
        ctx.fillStyle = isLight ? '#64748b' : '#888899';
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
            const val = maxV - ((maxV - minV) / 4) * i;
            const y = pad.top + (ch / 4) * i;
            ctx.fillText(val.toPrecision(4), pad.left - 8, y + 4);
        }

        // X labels
        ctx.textAlign = 'center';
        const startIdx = Math.max(0, data.length - this.maxPoints);
        const step = Math.max(1, Math.floor(data.length / 5));
        for (let i = 0; i < data.length; i += step) {
            const x = pad.left + (i / (data.length - 1)) * cw;
            ctx.fillText(String(startIdx + i + 1), x, h - 8);
        }

        // Area fill gradient
        const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch);
        if (this.activeTab === 'loss') {
            grad.addColorStop(0, 'rgba(255,92,92,0.15)');
            grad.addColorStop(1, 'rgba(255,92,92,0)');
        } else {
            grad.addColorStop(0, 'rgba(61,214,140,0.15)');
            grad.addColorStop(1, 'rgba(61,214,140,0)');
        }

        // Build points
        const pts = [];
        for (let i = 0; i < data.length; i++) {
            const x = pad.left + (i / (data.length - 1)) * cw;
            const y = pad.top + ch - ((data[i] - minV) / (maxV - minV)) * ch;
            pts.push({ x, y });
        }

        // Area fill
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pad.top + ch);
        ctx.lineTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
            const prev = pts[i - 1], curr = pts[i];
            const cpx = (prev.x + curr.x) / 2;
            ctx.bezierCurveTo(cpx, prev.y, cpx, curr.y, curr.x, curr.y);
        }
        ctx.lineTo(pts[pts.length - 1].x, pad.top + ch);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        // Line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) {
            const prev = pts[i - 1], curr = pts[i];
            const cpx = (prev.x + curr.x) / 2;
            ctx.bezierCurveTo(cpx, prev.y, cpx, curr.y, curr.x, curr.y);
        }
        ctx.stroke();

        // Latest value label
        const lastVal = data[data.length - 1];
        ctx.fillStyle = color;
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`${label}: ${lastVal.toPrecision(4)}`, pad.left + 8, pad.top + 14);
    }
}

// ==================== Temperature Hue Animation ====================
function applyTemperatureHue(temp) {
    let hue = temp <= 0.7 ? 200 - ((temp - 0.1) / 0.6) * 80 : 120 - ((temp - 0.7) / 0.8) * 120;
    targetHue = Math.round(hue);
    if (targetHue < 0) targetHue += 360;
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

// ==================== Chat: Model Status ====================
async function checkInferenceReady() {
    try {
        const res = await fetch(API_BASE + '/api/model/info');
        if (res.status === 200) {
            const info = await res.json();
            if (info.status === 'ready') {
                capsuleText.textContent = `🟢 ${info.current_lora ? info.current_lora.split('/').pop() : 'Base Model'}`;
                return true;
            }
        }
    } catch (e) {}
    return false;
}

async function ensureModelReady() {
    if (await checkInferenceReady()) return true;
    isWakingUp = true;
    const dict = i18n[currentLang] || i18n['zh-CN'];
    chatSendBtn.textContent = currentLang === 'en' ? 'Waking up...' : '唤醒中...';
    chatSendBtn.disabled = true;
    chatInput.disabled = true;
    window.chatObject?.startChat();
    for (let i = 0; i < 60; i++) {
        await new Promise(r => setTimeout(r, 1000));
        if (await checkInferenceReady()) break;
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
                            if (delta.includes('')) {
                                isThinking = true;
                                thinkBlock.style.display = 'block';
                                thinkBlock.classList.add('open');
                                delta = delta.replace('', '');
                            }
                            if (delta.includes('')) {
                                isThinking = false;
                                thinkTitle.textContent = currentLang === 'en' ? 'Thinking Process (click to collapse)' : '思考过程（点击折叠）';
                                thinkBlock.classList.remove('open');
                                delta = delta.replace('', '');
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
        item.textContent = sess.title;
        item.onclick = () => loadSession(sess.id);
        historyList.appendChild(item);
    });
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
    sidebarToggle.onclick = () => {
        chatSidebar.classList.toggle('collapsed');
        sidebarToggle.textContent = chatSidebar.classList.contains('collapsed') ? '☰' : '✕';
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
    loraCapsule.classList.remove('open');
    if (!(await ensureModelReady())) return;
    capsuleText.textContent = currentLang === 'en' ? '⏳ Switching...' : '⏳ 切换中...';
    try {
        const res = await fetch(API_BASE + '/api/lora/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lora_path: path })
        });
        if (res.status === 200) {
            capsuleText.textContent = `🟢 ${path ? path.split('/').pop() : 'Base Model'}`;
        } else {
            showToast(currentLang === 'en' ? 'Switch failed' : '切换失败');
            checkInferenceReady();
        }
    } catch (e) {
        showToast(currentLang === 'en' ? 'Connection failed' : '连接失败');
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
            if (file.name.endsWith('.txt')) {
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
            window.chatObject?.saveDataset(file.name.replace('.txt', '.jsonl'), content);
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
    radio.addEventListener('change', (e) => {
        if (baseLoraGroup) baseLoraGroup.style.display = e.target.value === 'continue' ? 'block' : 'none';
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
    const lr = document.getElementById('lr-input')?.value || '0.0003';
    const epochs = document.getElementById('epochs-input')?.value || '4';
    const seqLen = document.getElementById('seqlen-input')?.value || '512';
    const batchSize = document.getElementById('batchsize-input')?.value || '4';
    const optimizer = document.getElementById('optimizer-select')?.value || 'adamw';
    const weightDecay = document.getElementById('weightdecay-input')?.value || '0.01';
    const loraR = document.getElementById('lorar-input')?.value || '16';
    const loraAlpha = document.getElementById('loraalpha-input')?.value || '32';

    appendLog('train', `${currentLang === 'en' ? 'Starting training' : '开始训练'}: ${loraName} (${mode})`);
    if (window._trainChart) window._trainChart.clear();

    if (window.chatObject) {
        window.chatObject.startTrain(loraName, datasetPath, mode, baseLora, lr, epochs, seqLen, batchSize, optimizer, weightDecay, loraR, loraAlpha);
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

// ==================== Chart Tab Switching ====================
document.querySelectorAll('.chart-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        if (window._trainChart) {
            window._trainChart.setTab(tab.dataset.chart);
        }
    });
});

// ==================== Python Signal Handlers ====================
function connectSignals() {
    const obj = window.chatObject;
    if (!obj) return;

    obj.envLog.connect((msg) => appendLog('env', msg));
    obj.downloadLog.connect((msg) => appendLog('download', msg));
    obj.trainLog.connect((msg) => appendLog('train', msg));

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

    obj.envDone.connect((success) => {
        appendLog('env', success ? (currentLang === 'en' ? 'Environment check complete' : '环境检测完成') : (currentLang === 'en' ? 'Environment check failed' : '环境检测失败'));
    });

    obj.downloadDone.connect((success) => {
        appendLog('download', success ? (currentLang === 'en' ? 'Download complete' : '下载完成') : (currentLang === 'en' ? 'Download failed' : '下载失败'));
    });

    obj.trainMetrics.connect((jsonStr) => {
        try {
            const m = JSON.parse(jsonStr);
            if (window._trainChart && m.loss !== undefined) {
                window._trainChart.addPoint(m.loss, m.lr || 0);
            }
        } catch (e) {}
    });

    obj.trainDone.connect((success) => {
        appendLog('train', success ? (currentLang === 'en' ? 'Training complete!' : '训练完成！') : (currentLang === 'en' ? 'Training failed' : '训练失败'));
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

// ==================== Initialization ====================
function init() {
    // Per-card halftone engines
    window._halftoneInstances = [];
    document.querySelectorAll('.halftone-target').forEach(card => {
        const engine = new CardHalftoneEngine(card);
        window._halftoneInstances.push(engine);
    });

    // Training chart
    const chartCanvas = document.getElementById('train-chart');
    if (chartCanvas) {
        window._trainChart = new TrainingChart(chartCanvas);
        window._trainChart.draw();
    }

    // Navigation
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => switchPage(btn.dataset.page));
    });

    // Window controls
    initWindowControls();
    initWindowDrag();

    // Collapsible log toggles
    initLogToggles();

    // Apply initial theme
    applyTheme('train');

    // Drop zone
    initDropZone();

    // Connect Python signals
    connectSignals();

    // Request initial data
    if (window.chatObject) {
        window.chatObject.isFirstLaunch();
        window.chatObject.loadSessions();
        window.chatObject.getDatasetList();
        window.chatObject.getLoraWeights();
        window.chatObject.loadLanguage();
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
