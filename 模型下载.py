# -*- coding: utf-8 -*-
"""
模型批量下载脚本
通过 ModelScope SDK（snapshot_download）下载 Qwen3.5-0.8B / Qwen3.5-4B / DeepSeek-1.5B
所有模型统一存储到 exe 同级目录的 models/ 文件夹下。
"""

import os
import sys
import time
from modelscope import snapshot_download

# ==================== 可移植路径配置 ====================
def exe_dir():
    """获取可执行文件所在目录（兼容 PyInstaller 打包和直接运行脚本）"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = exe_dir()
MODEL_DIR = os.path.join(BASE_DIR, "models")          # 统一模型根目录

# ==================== 模型配置 ====================
MODEL_LIST = [
    {
        "id": "Qwen/Qwen3.5-0.8B",
        "cache_dir": os.path.join(MODEL_DIR, "Qwen3.5-0.8B"),
        "revision": "master",
        "alias": "Qwen3.5-0.8B",
    },
    {
        "id": "Qwen/Qwen3.5-4B",
        "cache_dir": os.path.join(MODEL_DIR, "Qwen3.5-4B"),
        "revision": "master",
        "alias": "Qwen3.5-4B",
    },
    {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "cache_dir": os.path.join(MODEL_DIR, "DeepSeek-1.5B"),
        "revision": "master",
        "alias": "DeepSeek-1.5B",
    },
]

# ==================== 下载函数 ====================

def download_model(model_cfg, retries=3):
    """下载单个模型，支持重试"""
    model_id = model_cfg["id"]
    cache_dir = model_cfg["cache_dir"]
    revision = model_cfg["revision"]
    alias = model_cfg["alias"]

    os.makedirs(cache_dir, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            print(f"[{alias}] 开始下载 (第{attempt}/{retries}次)...")
            start = time.time()

            model_dir = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                revision=revision,
            )

            elapsed = time.time() - start
            mins, secs = divmod(int(elapsed), 60)
            print(f"[{alias}] 下载完成，耗时 {mins}分{secs}秒")
            print(f"[{alias}] 路径: {model_dir}\n")
            return True, model_dir

        except Exception as e:
            print(f"[{alias}] 第{attempt}次失败: {e}")
            if attempt < retries:
                wait = 5 * attempt
                print(f"[{alias}] 等待 {wait} 秒后重试...\n")
                time.sleep(wait)

    print(f"[{alias}] 所有重试均失败，跳过\n")
    return False, None


# ==================== 主流程 ====================

if __name__ == "__main__":
    total = len(MODEL_LIST)
    success_list = []
    fail_list = []

    # 确保模型根目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print(f"模型批量下载 · 共 {total} 个")
    print(f"程序根目录: {BASE_DIR}")
    print(f"模型统一存储目录: {MODEL_DIR}")
    print("=" * 60)
    print()

    overall_start = time.time()

    for i, cfg in enumerate(MODEL_LIST, 1):
        print(f"{'─' * 60}")
        print(f"[{i}/{total}] {cfg['alias']}")
        print(f"  ModelScope ID: {cfg['id']}")
        print(f"  存储目录:      {cfg['cache_dir']}")
        print(f"{'─' * 60}")

        ok, path = download_model(cfg)
        if ok:
            success_list.append(cfg["alias"])
        else:
            fail_list.append(cfg["alias"])

    # 汇总
    overall_elapsed = time.time() - overall_start
    mins, secs = divmod(int(overall_elapsed), 60)

    print()
    print("=" * 60)
    print(f"下载完毕 · 总耗时 {mins}分{secs}秒")
    print(f"  成功: {len(success_list)} 个  {', '.join(success_list) if success_list else '无'}")
    if fail_list:
        print(f"  失败: {len(fail_list)} 个  {', '.join(fail_list)}")
    print("=" * 60)

    if fail_list:
        input("\n部分模型下载失败，按 Enter 键退出...")