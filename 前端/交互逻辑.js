// ============================================================
// NEXT AI 交互逻辑 - 60fps 丝滑色温插值版
// ============================================================

const dockItems = document.querySelectorAll('.dock-item');
const pageContents = document.querySelectorAll('.page-container');
const pageTitle = document.getElementById('page-title');
const homeBtn = document.getElementById('home-btn');
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

const logAreas = {
    env: document.getElementById('env-log'), download: document.getElementById('download-log'),
    data: document.getElementById('data-log'), train: document.getElementById('train-log'),
    'data-management': document.getElementById('data-management-log')
};

const API_BASE = 'http://127.0.0.1:5000';
let isGenerating = false, isWakingUp = false, currentAbortController = null;
let currentEditingDataset = null, selectedLoraPath = null;
let sessions = JSON.parse(localStorage.getItem('next_ai_sessions') || '[]');
let currentSessionId = null;
let magicStates = { thinking: false };

// ★ 核心：60fps 色温插值引擎状态
let currentHue = 220;
let targetHue = 220;
let animFrame = null;

function appendLog(tabId, msg) {
    const area = logAreas[tabId];
    if (area) {
        const p = document.createElement('p');
        p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        area.appendChild(p); area.scrollTop = area.scrollHeight;
    }
}

function simpleMarkdown(text) {
    if (!text) return '';
    return text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>').replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
}

const themeMap = { env: 'theme-env', download: 'theme-download', data: 'theme-data', train: 'theme-train', chat: 'theme-chat', 'data-management': 'theme-data-management' };

function applyTheme(pageId) {
    document.body.className = document.body.className.replace(/theme-\S+/g, '').trim();
    if (themeMap[pageId]) document.body.classList.add(themeMap[pageId]);
    if (localStorage.getItem('next_ai_theme') === 'light') document.body.classList.add('light-theme');
}

// ★ 核心：触发 60fps 颜色渐变
function applyTemperatureHue(temp) {
    let hue = temp <= 0.7 ? 200 - ((temp - 0.1) / 0.6) * 80 : 120 - ((temp - 0.7) / 0.8) * 120;
    targetHue = Math.round(hue);
    if (targetHue < 0) targetHue += 360;

    // 启动动画循环
    if (!animFrame) animateHue();
}

// ★ 核心：JS 缓动插值算法 (自动计算色相环最短路径)
function animateHue() {
    let diff = targetHue - currentHue;

    // 寻找色相环上的最短路径 (防止从 350度 到 10度 时绕远路)
    if (diff > 180) diff -= 360;
    if (diff < -180) diff += 360;

    // 如果已经到达目标，停止动画
    if (Math.abs(diff) < 0.5) {
        currentHue = targetHue;
        document.documentElement.style.setProperty('--temp-hue', currentHue);
        animFrame = null;
        return;
    }

    // 缓动插值 (Ease-out 效果，0.08 控制速度)
    currentHue += diff * 0.08;

    // 限制在 0-360 之间
    if (currentHue < 0) currentHue += 360;
    if (currentHue >= 360) currentHue -= 360;

    // 更新 CSS 变量
    document.documentElement.style.setProperty('--temp-hue', Math.round(currentHue));

    // 请求下一帧
    animFrame = requestAnimationFrame(animateHue);
}

async function checkInferenceReady() {
    try {
        const res = await fetch(API_BASE + '/api/model/info');
        if (res.status === 200) {
            const info = await res.json();
            if (info.status === 'ready') {
                capsuleText.textContent = `🟢 ${info.current_lora ? info.current_lora.split('/').pop() : '基座模型'}`;
                return true;
            }
        }
    } catch (e) {} return false;
}

async function ensureModelReady() {
    if (await checkInferenceReady()) return true;
    isWakingUp = true; chatSendBtn.textContent = '⏳ 唤醒中...'; chatSendBtn.disabled = true; chatInput.disabled = true;
    window.chatObject?.startChat();
    for (let i = 0; i < 60; i++) { await new Promise(r => setTimeout(r, 1000)); if (await checkInferenceReady()) break; }
    isWakingUp = false; chatInput.disabled = false; updateSendBtnState();
    if (!(await checkInferenceReady())) { alert('模型唤醒超时'); return false; } return true;
}

function updateSendBtnState() {
    if (isGenerating) { chatSendBtn.style.display = 'none'; chatStopBtn.style.display = 'inline-block'; }
    else { chatSendBtn.style.display = 'inline-flex'; chatStopBtn.style.display = 'none'; chatSendBtn.textContent = '⚡ 发送'; chatSendBtn.disabled = !chatInput.value.trim(); }
}

function createAssistantBubble() {
    const bubble = document.createElement('div'); bubble.className = 'chat-bubble assistant';
    const thinkBlock = document.createElement('div'); thinkBlock.className = 'thinking-block';
    thinkBlock.innerHTML = `<div class="thinking-header"><span class="thinking-icon">💭</span><span class="thinking-title">思考中...</span></div><div class="thinking-content"></div>`;
    thinkBlock.querySelector('.thinking-header').onclick = () => thinkBlock.classList.toggle('open');
    thinkBlock.style.display = 'none';
    const answerBlock = document.createElement('div'); answerBlock.className = 'answer-block'; answerBlock.textContent = '...';
    bubble.appendChild(thinkBlock); bubble.appendChild(answerBlock);
    chatMessages.appendChild(bubble); chatMessages.scrollTop = chatMessages.scrollHeight;
    return { bubble, thinkBlock, answerBlock };
}

async function sendMessage(text) {
    appendMessage('user', text); saveCurrentSession();
    const { thinkBlock, answerBlock } = createAssistantBubble();
    const thinkContent = thinkBlock.querySelector('.thinking-content');
    const thinkTitle = thinkBlock.querySelector('.thinking-title');
    isGenerating = true; updateSendBtnState();
    const controller = new AbortController(); currentAbortController = controller;
    let isThinking = false, thinkingText = '', answerText = '', pendingRender = false;

    function scheduleRender() {
        if (!pendingRender) {
            pendingRender = true;
            requestAnimationFrame(() => {
                answerBlock.innerHTML = simpleMarkdown(answerText);
                chatMessages.scrollTop = chatMessages.scrollHeight; pendingRender = false;
            });
        }
    }

    try {
        const res = await fetch(API_BASE + '/api/chat', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, history: getCurrentSession().messages.slice(0, -1), enable_thinking: magicStates.thinking }),
            signal: controller.signal
        });
        const reader = res.body.getReader(); const decoder = new TextDecoder();
        while (true) {
            const { value, done } = await reader.read(); if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            for (let line of chunk.split('\n')) {
                if (line.startsWith('data:')) {
                    const data = line.slice(5).trim(); if (data === '[DONE]') break;
                    try {
                        const json = JSON.parse(data);
                        // ★ 拦截 Meta 帧，触发 60fps 色温渐变
                        if (json.meta && json.meta.temperature !== undefined) applyTemperatureHue(json.meta.temperature);
                        else if (json.choices) {
                            let delta = json.choices[0].delta.content;
                            if (delta.includes('<think>')) { isThinking = true; thinkBlock.style.display = 'block'; thinkBlock.classList.add('open'); delta = delta.replace('<think>', ''); }
                            if (delta.includes('</think>')) { isThinking = false; thinkTitle.textContent = '思考过程 (点击折叠)'; thinkBlock.classList.remove('open'); delta = delta.replace('</think>', ''); }
                            if (isThinking) { thinkingText += delta; thinkContent.innerHTML = simpleMarkdown(thinkingText); }
                            else { answerText += delta; scheduleRender(); }
                        } else if (json.final) { answerText = json.full_text || answerText; answerBlock.innerHTML = simpleMarkdown(answerText); }
                        else if (json.error) { answerBlock.textContent = '错误: ' + json.error; }
                    } catch (e) {}
                }
            }
        }
        getCurrentSession().messages.push({ role: 'assistant', content: answerText }); saveCurrentSession();
    } catch (err) {
        if (err.name === 'AbortError') answerBlock.innerHTML += '<br><em>(已停止)</em>';
        else answerBlock.textContent = '请求失败: ' + err.message;
    } finally { isGenerating = false; currentAbortController = null; updateSendBtnState(); }
}

function appendMessage(role, text) {
    const bubble = document.createElement('div'); bubble.className = 'chat-bubble ' + role;
    bubble.innerHTML = role === 'user' ? text : simpleMarkdown(text);
    chatMessages.appendChild(bubble); chatMessages.scrollTop = chatMessages.scrollHeight;
    getCurrentSession().messages.push({ role, content: text });
}

function getCurrentSession() {
    if (!currentSessionId) { currentSessionId = 'sess_' + Date.now(); sessions.unshift({ id: currentSessionId, title: '新对话', messages: [], timestamp: Date.now() }); }
    return sessions.find(s => s.id === currentSessionId);
}
function saveCurrentSession() {
    const sess = getCurrentSession();
    if (sess.messages.length > 0 && sess.title === '新对话') sess.title = sess.messages[0].content.substring(0, 20) + '...';
    localStorage.setItem('next_ai_sessions', JSON.stringify(sessions)); renderHistoryList();
}
function renderHistoryList() {
    if (!historyList) return; historyList.innerHTML = '';
    sessions.forEach(sess => {
        const item = document.createElement('div');
        item.className = 'history-item' + (sess.id === currentSessionId ? ' active' : '');
        item.textContent = sess.title; item.onclick = () => loadSession(sess.id);
        historyList.appendChild(item);
    });
}
function loadSession(id) {
    currentSessionId = id; const sess = getCurrentSession(); chatMessages.innerHTML = '';
    sess.messages.forEach(m => appendMessage(m.role, m.content)); renderHistoryList();
}
if (newChatBtn) newChatBtn.onclick = () => { currentSessionId = null; chatMessages.innerHTML = ''; renderHistoryList(); };
if (sidebarToggle && chatSidebar) {
    sidebarToggle.onclick = () => {
        chatSidebar.classList.toggle('collapsed');
        sidebarToggle.textContent = chatSidebar.classList.contains('collapsed') ? '☰' : '✕';
    };
}

if (magicSwitches) {
    magicSwitches.querySelectorAll('.switch-pill').forEach(pill => {
        pill.onclick = () => { pill.classList.toggle('active'); magicStates[pill.dataset.feature] = pill.classList.contains('active'); };
    });
}
if (loraCapsule) {
    loraCapsule.onclick = (e) => { e.stopPropagation(); loraCapsule.classList.toggle('open'); };
    document.addEventListener('click', () => loraCapsule.classList.remove('open'));
}
async function switchLoraViaCapsule(path) {
    loraCapsule.classList.remove('open'); if (!(await ensureModelReady())) return;
    capsuleText.textContent = '⏳ 切换中...';
    try {
        const res = await fetch(API_BASE + '/api/lora/switch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ lora_path: path }) });
        if (res.status === 200) capsuleText.textContent = `🟢 ${path ? path.split('/').pop() : '基座模型'}`;
        else { alert('切换失败'); checkInferenceReady(); }
    } catch (e) { alert('无法连接'); }
}

function initDropZone() {
    if (!dropZone) return;
    dropZone.addEventListener('click', () => fileInput.click());
    ['dragenter', 'dragover'].forEach(evt => dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach(evt => dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); }));
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
                    return JSON.stringify({messages: [{role:'user', content:"请续写："+c.substring(0, splitIdx)}, {role:'assistant', content:c.substring(splitIdx)}]});
                }).join('\n');
            }
            window.chatObject?.saveDataset(file.name.replace('.txt', '.jsonl'), content);
        };
        reader.readAsText(file, 'UTF-8');
    });
}
function renderDatasetGrid(datasets) {
    if (!datasetGrid) return; datasetGrid.innerHTML = '';
    if (!datasets || datasets.length === 0) { if (datasetEmpty) datasetEmpty.style.display = 'block'; return; }
    if (datasetEmpty) datasetEmpty.style.display = 'none';
    datasets.forEach(ds => {
        const card = document.createElement('div'); card.className = 'dataset-card';
        card.innerHTML = `<div class="card-info"><div class="card-title-text">${ds.name}</div><div class="card-meta">${ds.lines || 0} 条</div></div><div class="card-actions"><button class="card-btn edit-btn">编辑</button><button class="card-btn danger delete-btn">删除</button></div>`;
        card.querySelector('.edit-btn').onclick = (e) => { e.stopPropagation(); openEditorModal(ds); };
        card.querySelector('.delete-btn').onclick = (e) => { e.stopPropagation(); if(confirm('删除？')) window.chatObject?.deleteDataset(ds.path); };
        datasetGrid.appendChild(card);
    });
}
function updateDatasetSelect(datasets) {
    if (!datasetSelect) return; datasetSelect.innerHTML = '<option value="">请选择数据集</option>';
    datasets.forEach(ds => { const opt = document.createElement('option'); opt.value = ds.path; opt.textContent = ds.name; datasetSelect.appendChild(opt); });
}
function openEditorModal(ds) {
    currentEditingDataset = ds; if (modalTitle) modalTitle.textContent = `编辑: ${ds.name}`;
    if (editorModal) editorModal.classList.add('active'); window.chatObject?.readDatasetContent(ds.path);
}
function closeModal() { if (editorModal) editorModal.classList.remove('active'); }
if (modalCancelBtn) modalCancelBtn.onclick = closeModal;
if (modalSaveBtn) modalSaveBtn.onclick = () => { if (currentEditingDataset) { window.chatObject.updateDataset(currentEditingDataset.path, modalEditor.value); closeModal(); } };
if (modalDeleteBtn) modalDeleteBtn.onclick = () => { if (currentEditingDataset && confirm('删除？')) { window.chatObject.deleteDataset(currentEditingDataset.path); closeModal(); } };

function renderLoraGrid(loras) {
    if (!loraGrid) return; loraGrid.innerHTML = '';
    if (!loras || loras.length === 0) { if (loraEmpty) loraEmpty.style.display = 'block'; return; }
    if (loraEmpty) loraEmpty.style.display = 'none';
    loras.forEach(lora => {
        const card = document.createElement('div'); card.className = 'lora-card';
        card.innerHTML = `<div class="card-info"><div class="card-title-text">${lora.name}</div><div class="card-meta">${lora.created_at ? new Date(lora.created_at).toLocaleDateString() : ''}</div></div>`;
        card.onclick = () => { document.querySelectorAll('.lora-card').forEach(c => c.style.borderColor = ''); card.style.borderColor = 'var(--accent-blue)'; selectedLoraPath = lora.path; };
        loraGrid.appendChild(card);
    });
}
function updateBaseLoraSelect(loras) {
    if (!baseLoraSelect) return; baseLoraSelect.innerHTML = '<option value="">请选择基础 LoRA</option>';
    loras.forEach(lora => { const opt = document.createElement('option'); opt.value = lora.path; opt.textContent = lora.name; baseLoraSelect.appendChild(opt); });
}
function updateCapsuleDropdown(loras) {
    if (!capsuleDropdown) return; capsuleDropdown.innerHTML = '';
    const baseItem = document.createElement('div'); baseItem.className = 'capsule-item'; baseItem.textContent = '基座模型'; baseItem.onclick = () => switchLoraViaCapsule(null); capsuleDropdown.appendChild(baseItem);
    loras.forEach(lora => { const item = document.createElement('div'); item.className = 'capsule-item'; item.textContent = lora.name; item.onclick = () => switchLoraViaCapsule(lora.path); capsuleDropdown.appendChild(item); });
}

trainModeRadios.forEach(radio => radio.addEventListener('change', (e) => { if (baseLoraGroup) baseLoraGroup.style.display = e.target.value === 'continue' ? 'block' : 'none'; }));
if (document.getElementById('train-btn')) {
    document.getElementById('train-btn').onclick = () => {
        const loraName = loraNameInput?.value.trim(); if (!loraName) return appendLog('train', '⚠️ 请输入 LoRA 名称');
        const mode = document.querySelector('input[name="train-mode"]:checked')?.value || 'new';
        const datasetPath = datasetSelect?.value; if (!datasetPath) return appendLog('train', '⚠️ 请选择数据集');
        const baseLoraPath = mode === 'continue' ? baseLoraSelect?.value : '';
        window.chatObject?.startTrain(JSON.stringify({ mode, lora_name: loraName, dataset_path: datasetPath, base_lora_path: baseLoraPath }));
    };
}
if (document.getElementById('refresh-lora-btn')) document.getElementById('refresh-lora-btn').onclick = () => window.chatObject?.getLoraWeights();
if (document.getElementById('delete-lora-btn')) document.getElementById('delete-lora-btn').onclick = () => { if (selectedLoraPath && confirm('删除？')) window.chatObject?.deleteLora(selectedLoraPath); };
if (document.getElementById('switch-lora-btn')) document.getElementById('switch-lora-btn').onclick = async () => { if (!selectedLoraPath) return; if (await ensureModelReady()) window.chatObject?.switchLora(selectedLoraPath); };
if (document.getElementById('env-btn')) document.getElementById('env-btn').onclick = () => window.chatObject?.checkEnv();
if (document.getElementById('download-btn')) document.getElementById('download-btn').onclick = () => window.chatObject?.downloadModel(document.getElementById('model-select')?.value);
if (homeBtn) homeBtn.onclick = () => window.chatObject?.goHome();

if (chatInput) {
    chatInput.addEventListener('input', updateSendBtnState);
    chatInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); wakeUpAndSend(); } });
}
if (chatSendBtn) chatSendBtn.onclick = wakeUpAndSend;
if (chatStopBtn) chatStopBtn.onclick = () => { currentAbortController?.abort(); fetch(API_BASE + '/api/stop', {method:'POST'}).catch(()=>{}); };

async function wakeUpAndSend() {
    const text = chatInput.value.trim(); if (!text || isGenerating || isWakingUp) return;
    if (await ensureModelReady()) { chatInput.value = ''; updateSendBtnState(); sendMessage(text); }
}

dockItems.forEach(item => {
    item.addEventListener('click', () => {
        const pageId = item.getAttribute('data-page'); applyTheme(pageId);
        dockItems.forEach(d => d.classList.remove('active')); pageContents.forEach(p => p.classList.remove('active'));
        item.classList.add('active'); document.getElementById(pageId + '-content')?.classList.add('active');
        const names = { env: '环境监测', download: '模型获取', data: '数据准备', train: '模型微调训练', chat: 'NeXT 对话', 'data-management': 'LoRA 权重管理' };
        if (pageTitle) pageTitle.textContent = names[pageId] || pageId;
        if (pageId === 'data') window.chatObject?.getDatasetList();
        if (pageId === 'data-management' || pageId === 'train') { window.chatObject?.getLoraWeights(); window.chatObject?.getDatasetList(); }
    });
});

if (themeToggleBtn) {
    const savedTheme = localStorage.getItem('next_ai_theme');
    if (savedTheme === 'light') document.body.classList.add('light-theme');
    else if (!savedTheme && window.matchMedia('(prefers-color-scheme: light)').matches) document.body.classList.add('light-theme');
    themeToggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('light-theme');
        localStorage.setItem('next_ai_theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
    });
}

let _signalsBound = false;
function setupSignalListeners() {
    if (_signalsBound || !window.chatObject) { if (!window.chatObject) setTimeout(setupSignalListeners, 500); return; }
    window.chatObject.logUpdate.connect(appendLog);
    window.chatObject.dataLoaded.connect(function(rawData) {
        if (!rawData) return;
        try {
            const parsed = JSON.parse(rawData);
            if (parsed.type === 'lora_list') { renderLoraGrid(parsed.items); updateBaseLoraSelect(parsed.items); updateCapsuleDropdown(parsed.items); }
            else if (parsed.type === 'dataset_list') { renderDatasetGrid(parsed.items); updateDatasetSelect(parsed.items); }
            else if (parsed.type === 'dataset_content' && modalEditor) { modalEditor.value = parsed.content; }
        } catch (e) {}
    });
    _signalsBound = true; window.chatObject.getDatasetList(); window.chatObject.getLoraWeights(); checkInferenceReady();
}
setupSignalListeners(); initDropZone(); renderHistoryList(); updateSendBtnState(); applyTheme('env');