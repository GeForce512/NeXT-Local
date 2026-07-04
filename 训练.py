# -*- coding: utf-8 -*-
import os, sys, torch, json, hashlib, re, math
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
MAX_SEQ_LENGTH, LEARNING_RATE, EPOCHS, BATCH_SIZE, GRAD_ACC = 512, 3e-4, 4, 1, 8


def find_model():
    for r, d, f in os.walk(MODEL_DIR):
        if "config.json" in f: return r
    return None


BASE_MODEL = find_model()


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
    if not config: return log("❌ 无配置")
    mode, name, ds_rel, base_rel = config.get("mode", "new"), config.get("lora_name", "un"), config.get("dataset_path",
                                                                                                        ""), config.get(
        "base_lora_path", "")
    # ★ 动态训练参数（从前端设置页读取）
    lr = float(config.get("learning_rate", LEARNING_RATE))
    epochs = int(config.get("epochs", EPOCHS))
    max_seq = int(config.get("max_seq_length", MAX_SEQ_LENGTH))
    batch_size = int(config.get("batch_size", BATCH_SIZE))
    optimizer = config.get("optimizer", "paged_adamw_8bit")
    weight_decay = float(config.get("weight_decay", 0.0))
    lora_r = int(config.get("lora_r", 32))
    lora_alpha = int(config.get("lora_alpha", 64))
    ds = os.path.join(BASE_DIR, ds_rel) if ds_rel else None
    base = os.path.join(BASE_DIR, base_rel) if base_rel else None
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

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto",
                                                 trust_remote_code=True)
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

    safe = re.sub(r'[\\/:*?"<>|]', '_', name).strip()[:50] or "un"
    save_dir = os.path.join(BASE_DIR, "lora_weights", f"{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)
    json.dump({"name": name, "created_at": datetime.now().isoformat(), "mode": mode},
              open(os.path.join(save_dir, "lora_meta.json"), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    model.save_pretrained(save_dir);
    tok.save_pretrained(save_dir)
    save_fps(save_dir, exist_fps.union(cur_fps))
    log(f"✅ 训练完成，LoRA 已保存至: {os.path.basename(save_dir)}")
    return True


if __name__ == "__main__":
    run_training(config={"mode": "new", "lora_name": "test", "dataset_path": "datasets/test.jsonl"})