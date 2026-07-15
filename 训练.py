# -*- coding: utf-8 -*-
import os, sys, torch, json, hashlib, re, math

# ★ 强制 stdout 使用 UTF-8，防止 Windows GBK 控制台编码崩溃
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, \
    DataCollatorForSeq2Seq, TrainerCallback
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


def exe_dir():
    return os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(
        os.path.abspath(__file__))


BASE_DIR = exe_dir()
MODEL_DIR = os.path.join(BASE_DIR, "models")
MAX_SEQ_LENGTH, LEARNING_RATE, EPOCHS, BATCH_SIZE, GRAD_ACC = 512, 3e-4, 4, 4, 8


def find_model():
    for r, d, f in os.walk(MODEL_DIR):
        if "config.json" in f: return r
    return None


BASE_MODEL = find_model()


def log(msg):
    """GBK 安全日志函数，防止 emoji 在 Windows 控制台崩溃"""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(str(msg).encode('utf-8', errors='replace').decode('utf-8', errors='replace'), flush=True)


class Monitor(TrainerCallback):
    def __init__(s, log, metrics_fn=None):
        s._log, s._metrics, s.steps, s.losses = log or print, metrics_fn, [], []

    def on_log(s, a, state, c, logs=None, **k):
        if logs and 'loss' in logs:
            loss = float(logs['loss']) if isinstance(logs['loss'], str) else logs['loss']
            if math.isnan(loss): raise RuntimeError("NaN")
            lr = float(logs.get('learning_rate', 0))
            epoch = float(logs.get('epoch', 0))
            s.steps.append(state.global_step);
            s.losses.append(loss)
            # 推送文本日志到前端
            s._log(
                f"📊 Step {state.global_step} | Loss: {loss:.4f} | LR: {lr:.2e} | Epoch: {epoch:.3f}")
            # ★ 推送结构化 JSON 给前端图表
            if s._metrics:
                try:
                    s._metrics(json.dumps({"loss": round(loss, 6), "lr": round(lr, 8), "step": state.global_step, "epoch": round(epoch, 3)}))
                except Exception:
                    pass


def fp(item):
    u = item['messages'][0]['content'] if len(item['messages']) > 0 else ""
    a = item['messages'][1]['content'] if len(item['messages']) > 1 else ""
    return hashlib.md5(f"{u}||{a}".encode()).hexdigest()


def load_fps(p):
    f = os.path.join(p, "fps.json")
    return set(json.load(open(f, encoding='utf-8'))) if os.path.exists(f) else set()


def save_fps(p, fps):
    os.makedirs(p, exist_ok=True)
    json.dump(list(fps), open(os.path.join(p, "fps.json"), 'w', encoding='utf-8'))


def load_data(p, exist):
    new, all_fp, empty = [], set(), 0
    for line in open(p, 'r', encoding='utf-8'):
        try:
            item = json.loads(line.strip())
            if len(item.get('messages', [])) >= 2:
                # ★ 核心：过滤空答案
                if not item['messages'][1].get('content', '').strip(): empty += 1; continue
                h = fp(item);
                all_fp.add(h)
                if h not in exist: new.append(item)
        except:
            pass
    return new, all_fp, empty


def run_training(log_fn=None, config=None, metrics_fn=None):
    log = log_fn or print
    
    # ★ DEBUG: 诊断 exe 运行时问题
    log("=" * 50)
    log("🔍 [DEBUG] 训练脚本启动")
    log(f"  sys.executable: {sys.executable}")
    log(f"  sys.frozen: {getattr(sys, 'frozen', False)}")
    log(f"  __file__: {__file__}")
    log(f"  BASE_DIR: {BASE_DIR}")
    log(f"  MODEL_DIR: {MODEL_DIR}")
    log(f"  cwd: {os.getcwd()}")
    log(f"  config: {json.dumps(config, ensure_ascii=False) if config else 'None'}")
    log(f"  BASE_MODEL (find_model): {find_model()}")
    log("=" * 50)
    
    if not config: return log("❌ 无配置")
    mode, name, ds_rel, base_rel = config.get("mode", "new"), config.get("lora_name", "un"), config.get("dataset_path",
                                                                                                        ""), config.get(
        "base_lora_path", "")
    # ★ 动态训练参数（从前端设置页读取）
    lr = float(config.get("learning_rate", LEARNING_RATE))
    epochs = int(config.get("epochs", EPOCHS))
    max_seq = int(config.get("max_seq_length", MAX_SEQ_LENGTH))
    batch_size = int(config.get("batch_size", BATCH_SIZE))
    optimizer = config.get("optimizer", "adamw_torch")
    weight_decay = float(config.get("weight_decay", 0.01))
    lora_r = int(config.get("lora_r", 16))
    lora_alpha = int(config.get("lora_alpha", 32))
    ds = os.path.join(BASE_DIR, ds_rel) if ds_rel else None
    base = os.path.join(BASE_DIR, base_rel) if base_rel else None
    
    log(f"🔍 [DEBUG] 数据集路径: {ds}")
    log(f"🔍 [DEBUG] 数据集存在: {os.path.exists(ds) if ds else False}")
    log(f"🔍 [DEBUG] BASE_MODEL: {BASE_MODEL}")
    
    if not ds or not os.path.exists(ds): return log(f"❌ 找不到数据集: {ds_rel}")
    if not BASE_MODEL: return log("❌ 找不到基座模型")

    log(f"📂 数据集: {os.path.basename(ds)}")
    exist_fps = load_fps(base) if mode == 'continue' and base and os.path.exists(base) else set()
    log("🔄 增量模式" if mode == 'continue' and base else "🚀 全新模式")

    new_items, cur_fps, empty = load_data(ds, exist_fps)
    if empty > 0: log(f"⚠️ 过滤 {empty} 条空答案数据")
    if not new_items: return log("⚠️ 无新数据")
    log(f"📊 准备训练 {len(new_items)} 条数据...")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if not tok.pad_token: tok.pad_token = tok.eos_token

    def fmt(ex):
        ids, masks, lbs = [], [], []
        for m in ex['messages']:
            u, a = m[0]['content'], m[1]['content']
            u_ids = tok(f"<|im_start|>user\n{u}\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)[
                'input_ids']
            a_ids = tok(a, add_special_tokens=False)['input_ids']
            e_ids = tok("<|im_end|>", add_special_tokens=False)['input_ids']
            f_ids = u_ids + a_ids + e_ids
            f_lbs = [-100] * len(u_ids) + a_ids + [-100] * len(e_ids)
            ids.append(f_ids[:max_seq]);
            masks.append([1] * len(f_ids[:max_seq]));
            lbs.append(f_lbs[:max_seq])
        return {"input_ids": ids, "attention_mask": masks, "labels": lbs}

    ds_obj = Dataset.from_list(new_items).map(fmt, batched=True, remove_columns=['messages'])

    log(f"🔍 [DEBUG] 数据集准备完成，开始加载模型...")
    log(f"🔍 [DEBUG] 模型路径: {BASE_MODEL}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)

    # ★ transformers 5.x 用 dtype，4.x 用 torch_dtype（inspect检测不到，用版本号判断）
    import transformers as _tf
    _fpkw = "dtype" if int(_tf.__version__.split('.')[0]) >= 5 else "torch_dtype"
    _load_kwargs = dict(quantization_config=bnb, device_map="auto", trust_remote_code=True,
                        attn_implementation="sdpa")
    _load_kwargs[_fpkw] = torch.bfloat16

    try:
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **_load_kwargs)
    except (ValueError, RuntimeError, OSError) as e:
        log(f"⚠️ 模型加载异常 ({e})，尝试不带 sdpa 加载...")
        _load_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **_load_kwargs)

    model = prepare_model_for_kbit_training(model)

    if mode == 'continue' and base and os.path.exists(base):
        model = PeftModel.from_pretrained(model, base)
    else:
        model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                                                 target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj",
                                                                 "up_proj", "down_proj"], lora_dropout=0.05,
                                                 bias="none", task_type="CAUSAL_LM"))

    model.gradient_checkpointing_enable();
    model.train()
    out_dir = os.path.join(BASE_DIR, "训练数据", "temp")
    os.makedirs(out_dir, exist_ok=True)

    # ★ 核心：disable_tqdm=True 防止控制台乱码，logging_steps=1 保证每步推送前端
    args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=batch_size,
                             gradient_accumulation_steps=GRAD_ACC, num_train_epochs=epochs, learning_rate=lr,
                             warmup_ratio=0.1, lr_scheduler_type="cosine", bf16=True, logging_steps=1,
                             save_strategy="no", optim=optimizer, weight_decay=weight_decay, disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=ds_obj,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True), callbacks=[Monitor(log, metrics_fn)])

    try:
        trainer.train()
    except Exception as e:
        return log(f"❌ 训练出错: {e}")

    # ★ 保存 LoRA 权重
    try:
        safe = re.sub(r'[\\/:*?"<>|]', '_', name).strip()[:50] or "un"
        save_dir = os.path.join(BASE_DIR, "lora_weights", safe)
        log(f"💾 正在保存 LoRA 到: {save_dir}")
        
        # 如果已存在旧版本，先备份再覆盖
        if os.path.exists(save_dir):
            backup_dir = save_dir + "_backup"
            if os.path.exists(backup_dir):
                import shutil
                shutil.rmtree(backup_dir)
            os.rename(save_dir, backup_dir)
            log(f"📦 旧版本已备份到：{os.path.basename(backup_dir)}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存元数据
        meta = {"name": name, "created_at": datetime.now().isoformat(), "mode": mode}
        meta_path = os.path.join(save_dir, "lora_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        log(f"✓ 元数据已保存: {meta_path}")
        
        # 保存模型和分词器
        log("📦 正在保存模型权重...")
        model.save_pretrained(save_dir)
        log("✓ 模型权重已保存")
        
        log("📦 正在保存分词器...")
        tok.save_pretrained(save_dir)
        log("✓ 分词器已保存")
        
        # 保存指纹
        save_fps(save_dir, exist_fps.union(cur_fps))
        log("✓ 指纹已保存")
        
        log(f"✅ 训练完成，LoRA 已保存至: {os.path.basename(save_dir)}")
        return True
    except Exception as e:
        import traceback
        log(f"❌ 保存 LoRA 失败: {e}")
        log(f"❌ 错误详情: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    run_training(config={"mode": "new", "lora_name": "test", "dataset_path": "datasets/test.jsonl"})