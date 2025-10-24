#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, random
from typing import Dict, Any, Tuple

import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)
OK, WARN, ERR, INFO = Fore.GREEN+"[OK]"+Style.RESET_ALL, Fore.YELLOW+"[WARN]"+Style.RESET_ALL, Fore.RED+"[ERR]"+Style.RESET_ALL, Fore.CYAN+"[INFO]"+Style.RESET_ALL

DEFAULT_MODEL = "/mnt/data1/models/Qwen3-VL-30B-A3B-Instruct/"

PROMPT_TEMPLATE = (
    "以下段落为我的数据集的其中一个文本样本，我想透过(在不影响整体格式和逻辑的前提下)"
    "修改用词和句子去提升样本的多样性(variation)。以下是文本样本: \"\"\"\n{analysis_text}\n\"\"\"\n"
    "修改从 \"**图像分析**\" 开始，\"**图像分析**\" 之前的部份都不用修改。\n"
    "请只输出修改后的文本样本，输出文本样本的所有部份(包括**图像分析**之前和之后)。\n"
    "请根据原本的文本样本保持整体格式不变，保留原有的formatting syntax(如 \\n)。\n"
    "以下为特别文字部份，不用修改这些部份: 有 ** ** 包含着的所有object title、\"**图像分析**\" 之前的部份、\"**结论**\"的内容部份。\n"
    "以下为特别用词，如出现不用修改这些用词: \"缺陷区域\", \"缺陷\", \"激光切割路径\", \"红色\", \"黄绿色\", \"Drain\", \"Gate\", \"Source\", \"TFT\", "
    "\"Data\", \"Com\", \"Mesh\", \"Mesh_Hole\", \"ITO\", \"VIA_Hole\", \"ITO remove\"。\n"
    "请确保文本中的推理逻辑不变。请使用简体中文字，并确保新文本中没有错字或漏字情况。可以在不歪曲文句意思的前提，自由调整每部分的文句次序。修改可以不用过于保守，修改程度大约为中至高等。"
)

def build_prompt(analysis_text: str) -> str:
    return PROMPT_TEMPLATE.format(analysis_text=analysis_text.strip())

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def find_samples(root: str):
    for dirpath, _, filenames in os.walk(root):
        if "analysis_gt.json" in filenames:
            yield dirpath

def init_model(model_path: str, verbose: bool=False):
    if verbose:
        print(f"{INFO} Loading model from: {model_path}")
    # Use the VL MoE classes
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",          # avoids torch_dtype deprecation warning
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    if verbose:
        print(f"{OK} Model loaded. device_map=auto")
    return processor, model

def generate_text(
    processor: AutoProcessor,
    model: Qwen3VLMoeForConditionalGeneration,
    prompt: str,
    temperature: float = 0.65,
    top_p: float = 0.9,
    top_k: int = 50,
    max_new_tokens: int = 2048,
) -> str:
    # Text-only chat; still use the chat template expected by the VL model.
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k) if top_k is not None else 50,
            repetition_penalty=1.1,
            max_new_tokens=int(max_new_tokens),
        )
    # trim prompt tokens
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)]
    text_list = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text_list[0].strip() if isinstance(text_list, list) and text_list else ""

def remix_sample(sample_dir: str, processor, model, args) -> Tuple[bool, str]:
    orig_path = os.path.join(sample_dir, "analysis_gt.json")
    remix_path = os.path.join(sample_dir, "remix_analysis_gt.json")

    if args.skip_existing and os.path.isfile(remix_path):
        return False, f"{WARN} Skip (exists): {os.path.relpath(remix_path, start=args.root)}"

    try:
        data = load_json(orig_path)
    except Exception as e:
        return False, f"{WARN} Failed to load analysis_gt.json: {sample_dir} ({e})"

    analysis_text = data.get("analysis", "")
    if not isinstance(analysis_text, str) or not analysis_text.strip():
        return False, f"{WARN} Empty or invalid 'analysis' → skip: {sample_dir}"

    prompt = build_prompt(analysis_text)
    remix = generate_text(
        processor, model, prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )

    new_obj = dict(data)
    new_obj["analysis"] = remix or analysis_text  # fallback if model returns empty

    try:
        save_json(new_obj, remix_path)
        return True, f"{OK} Wrote {os.path.relpath(remix_path, start=args.root)}"
    except Exception as e:
        return False, f"{ERR} Failed to write remix: {sample_dir} ({e})"

def main():
    ap = argparse.ArgumentParser(description="Remix analysis_gt.json 'analysis' via local Qwen3 VL MoE model.")
    ap.add_argument("--root", type=str, default=".", help="Search root (run from gt_datasets_20250915).")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Local HF model path.")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    ap.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    ap.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables).")
    ap.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens to generate.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip samples with existing remix_analysis_gt.json.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs.")
    args = ap.parse_args()

    args.root = os.path.abspath(args.root)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    try:
        processor, model = init_model(args.model, verbose=args.verbose)
    except Exception as e:
        print(f"{ERR} Failed to load model: {e}")
        return

    total = changed = skipped = 0
    for sdir in find_samples(args.root):
        total += 1
        ok, msg = remix_sample(sdir, processor, model, args)
        print(msg)
        if ok:
            changed += 1
        else:
            if msg.startswith(WARN) and "Skip" in msg:
                skipped += 1

    print(f"\n{OK} Done. total={total}, remixed={changed}, skipped={skipped}")

if __name__ == "__main__":
    main()
