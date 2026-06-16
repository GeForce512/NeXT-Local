# -*- coding: utf-8 -*-
import os, sys, json, time, threading, argparse, random
from flask import Flask, request, jsonify, Response
from flask_cors import CORS


def exe_dir():
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(
        os.path.abspath(__file__))


BASE_DIR = os.getcwd() if getattr(sys, 'frozen', False) else exe_dir()
MODEL_DIR = os.path.join(BASE_DIR, "models")

_model = None
tokenizer = None
device = "cpu"
is_generating = False
is_loading = False
stop_event = threading.Event()
_lock = threading.Lock()
current_lora_path = None
_base_model = None


def log(msg): print(f"[推理] {msg}")


def find_model():
    if not os.path.exists(MODEL_DIR): return None
    for root, dirs, files in os.walk(MODEL_DIR):
        if "config.json" in files: return root
    return None


def find_lora():
    lora_dir = os.path.join(BASE_DIR, "lora_weights")
    if os.path.exists(lora_dir):
        candidates = []
        for name in os.listdir(lora_dir):
            path = os.path.join(lora_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                candidates.append((path, os.path.getmtime(path)))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
    return None


# ★ 核心：语义感知 + 深度累积 的混合温度引擎
def decide_temperature(history_length, user_message):
    # 1. 基础温度：随对话深度累积 (0.3 起步，每轮 +0.08，最高累积到 1.0)
    base_temp = 0.3 + min(history_length * 0.08, 0.7)

    # 2. 语义感知：根据提示词瞬间加热或冷却
    msg = user_message.lower()
    hot_words = ['想象', '故事', '诗', '创意', '脑洞', '小说', '唯美', '爱情', '发散', '角色扮演', '写一篇文章', '灵感',
                 '浪漫']
    cold_words = ['代码', 'code', 'python', 'c++', 'java', '数学', '计算', '逻辑', '翻译', '总结', '严格', '修复',
                  'bug', 'sql', 'excel']

    if any(w in msg for w in hot_words):
        # 瞬间加热到至少 1.2 (浪漫紫红/狂热橙)
        base_temp = max(base_temp, 1.2)
    elif any(w in msg for w in cold_words):
        # 瞬间冷却到最多 0.2 (极寒冰蓝)
        base_temp = min(base_temp, 0.2)

    return round(min(1.4, max(0.2, base_temp)), 2)


def load_model():
    global _model, _base_model, tokenizer, device, current_lora_path, is_loading
    is_loading = True
    path = find_model()
    if not path: log("❌ 模型未找到"); sys.exit(1)
    try:
        import torch
    except ImportError:
        log("❌ 请安装 torch"); sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = None
    dtype = torch.float32
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        if vram_gb >= 8.0:
            log(f"🚀 显存 {vram_gb:.1f}GB，启用 BF16 满血！")
            dtype = torch.bfloat16
        elif vram_gb >= 5.0:
            log(f"🌟 显存 {vram_gb:.1f}GB，启用 INT8 高精度！")
            quant = BitsAndBytesConfig(load_in_8bit=True)
            dtype = None
        else:
            log(f"⚠️ 显存 {vram_gb:.1f}GB，启用 INT4 极限压缩！")
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            dtype = None

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        path, quantization_config=quant, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True, attn_implementation="sdpa"
    )
    if device == "cpu": _model = _model.to("cpu")
    _model.eval()
    _base_model = _model

    lora_path = find_lora()
    if lora_path:
        try:
            from peft import PeftModel
            _model = PeftModel.from_pretrained(_model, lora_path)
            current_lora_path = lora_path[len(BASE_DIR):].lstrip(os.sep).replace(os.sep, "/") if lora_path.startswith(
                BASE_DIR) else lora_path.replace(os.sep, "/")
            log(f"✅ LoRA 已加载: {lora_path}")
        except Exception as e:
            log(f"❌ LoRA 加载失败: {e}");
            _model = _base_model;
            current_lora_path = None
    else:
        log("ℹ️ 未找到 LoRA，使用基座模型")

    is_loading = False
    log("✅ 模型就绪")


def load_lora(lora_path):
    global _model, current_lora_path, is_loading, _base_model
    with _lock:
        if is_generating: return False, "生成中"
        if is_loading: return False, "加载中"
        is_loading = True
        try:
            if current_lora_path:
                try:
                    from peft import PeftModel
                    if isinstance(_model, PeftModel): _model = _model.unload()
                    if hasattr(_model, 'peft_config'): _model = _base_model
                except:
                    _model = _base_model
                current_lora_path = None

            if lora_path:
                from peft import PeftModel
                full_path = lora_path if os.path.isabs(lora_path) else os.path.join(BASE_DIR,
                                                                                    lora_path.replace("/", os.sep))
                if not os.path.exists(os.path.join(full_path, "adapter_config.json")):
                    is_loading = False;
                    return False, "配置不存在"
                _model = PeftModel.from_pretrained(_model, full_path)
                current_lora_path = lora_path.replace(os.sep, "/")
                log(f"✅ LoRA loaded: {lora_path}")

            if device == "cpu": _model = _model.to("cpu")
            _model.eval()
            is_loading = False
            return True, "OK"
        except Exception as e:
            log(f"❌ LoRA 异常: {e}");
            _model = _base_model;
            current_lora_path = None;
            is_loading = False
            return False, str(e)


def generate(messages, thinking=False):
    global is_generating
    if _model is None: yield '{"error": "未加载"}'; return
    acquired = _lock.acquire(blocking=False)
    if not acquired:
        time.sleep(0.5)
        acquired = _lock.acquire(blocking=True, timeout=10)
        if not acquired: yield '{"error": "繁忙"}'; return

    try:
        is_generating = True;
        stop_event.clear()

        # ★ 提取用户最新消息，用于语义感知
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        history_length = len([m for m in messages if m.get("role") == "user"]) - 1
        history_length = max(0, history_length)

        # ★ 传入 user_msg，让引擎“听懂”提示词
        current_temp = decide_temperature(history_length, user_msg)
        log(f"🌡️ 轮数: {history_length}, 提示词: '{user_msg[:15]}...', 生成温度: {current_temp}")

        # 下发 Meta 帧 (触发前端 60fps 色温渐变)
        yield f"data: {json.dumps({'meta': {'temperature': current_temp}}, ensure_ascii=False)}\n\n"

        chat = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages if
                m.get("role") in ("user", "assistant", "system")]
        if not chat or chat[-1]["role"] != "user": chat.append({"role": "user", "content": ""})

        try:
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda": inputs = {k: v.to("cuda") for k, v in inputs.items()}

        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            **inputs, "max_new_tokens": 2048, "temperature": current_temp, "top_p": 0.9,
            "do_sample": True, "repetition_penalty": 1.15,
            "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id, "streamer": streamer
        }

        gen_thread = threading.Thread(target=lambda: _model.generate(**gen_kwargs))
        gen_thread.start()

        full = ""
        for t in streamer:
            if stop_event.is_set(): break
            if t:
                full += t
                yield f"data: {json.dumps({'choices': [{'delta': {'content': t}}]}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'final': True, 'full_text': full}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        gen_thread.join(timeout=5)
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
    finally:
        is_generating = False;
        _lock.release()


app = Flask(__name__)
CORS(app)


@app.route("/api/chat", methods=["POST"])
def chat():
    if _model is None: return jsonify({"error": "未加载"}), 503
    if is_generating: return jsonify({"error": "生成中"}), 429
    if is_loading: return jsonify({"error": "加载中"}), 503
    data = request.get_json(force=True)
    msg = data.get("message", "")
    hist = data.get("history", [])
    think = data.get("enable_thinking", False)
    msgs = [m for m in hist if m.get("role") in ("user", "assistant", "system")]
    if msg: msgs.append({"role": "user", "content": msg})
    return Response(generate(msgs, think), mimetype="text/event-stream")


@app.route("/api/stop", methods=["POST"])
def stop():
    stop_event.set()
    return jsonify({"status": "ok"})


@app.route("/api/model/info")
def info():
    return jsonify({
        "model": "NeXT", "device": device,
        "status": "loading" if is_loading else ("ready" if _model else "offline"),
        "current_lora": current_lora_path,
        "is_generating": is_generating, "is_loading": is_loading
    })


@app.route("/api/lora/list", methods=["GET"])
def lora_list():
    lora_dir = os.path.join(BASE_DIR, "lora_weights")
    if not os.path.exists(lora_dir): return jsonify([])
    loras = []
    for name in sorted(os.listdir(lora_dir), reverse=True):
        path = os.path.join(lora_dir, name)
        if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "adapter_config.json")): continue
        meta = {}
        meta_path = os.path.join(path, "lora_meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except:
                pass
        loras.append({"name": meta.get("name", name), "folder": name, "created_at": meta.get("created_at", ""),
                      "path": os.path.join("lora_weights", name).replace("\\", "/")})
    return jsonify(loras)


@app.route("/api/lora/switch", methods=["POST"])
def lora_switch():
    if is_loading: return jsonify({"error": "加载中"}), 503
    if is_generating: return jsonify({"error": "生成中"}), 429
    data = request.get_json(force=True)
    ok, msg = load_lora(data.get("lora_path"))
    return jsonify({"status": "ok", "current_lora": current_lora_path}) if ok else jsonify({"error": msg}), 500


@app.route("/api/lora/unload", methods=["POST"])
def lora_unload():
    if is_loading or is_generating: return jsonify({"error": "忙碌中"}), 429
    ok, msg = load_lora(None)
    return jsonify({"status": "ok", "current_lora": None}) if ok else jsonify({"error": msg}), 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    load_model()
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()