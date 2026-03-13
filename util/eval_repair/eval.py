#!/usr/bin/env python3
import os, json, argparse
from typing import List, Dict, Any
import numpy as np
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

def load_json_file(filename: str) -> Any:
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def infer(model, processor, messages, max_new_tokens: int = 2048):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def combine_repair_rules(json_data: List[Dict[str, Any]]) -> str:
    combined_text = []
    for item in json_data:
        damaged_component = item["damaged_component"]
        rules = [rule["repair_rule"] for rule in item["rules"]]
        rules_text = " ".join(rules)
        combined_text.append(f"对于{damaged_component}缺陷，{rules_text}")
    return "；".join(combined_text) + "。"

def default_save_name(model_path: str) -> str:
    mp = model_path.rstrip("/") or "model"
    tail = os.path.basename(mp) or "ckpt"
    parent = os.path.basename(os.path.dirname(mp)) or "model"
    return f"pred_{parent}_{tail}.json"

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Qwen2.5/3-VL models on Repair Test Set.")
    p.add_argument("--model", required=True, help="Path to model directory")
    p.add_argument("--save-dir", required=True, help="Directory to write the output JSON file")
    p.add_argument("--save-name", default=None, help="Output filename (defaults to pred_<parent>_<basename>.json)")
    p.add_argument("--json-path", default="/mnt/workspace/kennethliu/data/20260121_dual/repair_test_dataset_dual_20260121.json",
                   help="Path to the evaluation JSON data")
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                   help="Torch dtype for model weights")
    p.add_argument("--device-map", default="auto", help='Device map for accelerate/transformers (e.g. "auto")')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = args.save_name or default_save_name(args.model)
    out_path = os.path.join(args.save_dir, save_name)

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # System & prompt templates
    system_msg = "你是一个专业的半导体显示面板业务专家和图像算法专家。"
    request_template = (
        "如图所示是一张半导体显示面板图像，蓝色曲线内区域为缺陷，\
        算法根据输入的缺陷位置及对应的修补信息输出了需要修补的路径，\
        红色直线代表模拟的激光切割的路径，黄绿色直线（较淡较宽）代表模拟的ITO remove的路径，\
        两种路径有可能有重合。你需要检查图上的可视化路径是否符合以下修补规则：一般规则，\
        1. 对当前组件进行激光修补时，激光切割路径需要避开其他组件，不可以与其他组件区域有重合；\
        2. 若由于修补手法产生的ITO remove路径进而与TFT或VIA_Hole相交，则需进一步修补，具体修补手法为：\
        a. 在靠近Data线位置激光切断Source组件；b. 在靠近ITO位置激光切断Drain组件；c. 切断Drain部件的相同路径上覆盖ITO remove；\
        3. 除非规则中特别指明修补路径位置，修补路径应尽可能靠近缺陷区域。；{repair_rule}\
        请输出两个内容：1. 判断图中可视化切割线是否符合所述修补规则，回答符合或者不符合；\
        2. 如果不符合，请说出不符合的地方。输出格式请参考以下示例(需包含规则确认、图像分析、应用规则、结论、理由)："
    )

    # request_template = (
    #     "<image>如图所示是一张半导体显示面板的组件掩膜合成图像，其中各组件掩膜以不同颜色区分，并且可能有重叠。\
    #     各组件对应的配色如下：Source(青色)；Drain(绿色)；TFT(橙色)；VIA_Hole(粉色)；Mesh(靛色)；Mesh_Hole(橄榄色)；Com(灰色)；Data(棕色)；Gate(紫色)；ITO(蓝色)。\
    #     背景为黑色。图中有可能出现各种可视化模拟路径，奶油色曲线内区域为缺陷，算法根据输入的缺陷位置及对应的修补信息输出了需要修补的路径，\
    #     红色直线代表模拟的激光切割的路径，萤光黄色线（较宽）代表模拟的ITO remove的路径，\
    #     路径有可能有重合。你需要检查图上的可视化路径是否符合以下修补规则：一般规则，\
    #     1. 对当前组件进行激光修补时，激光切割路径需要避开其他组件，不可以与其他组件区域有重合；\
    #     2. 若由于修补手法产生的ITO remove路径进而与TFT或VIA_Hole相交，则需进一步修补，具体修补手法为：\
    #     a. 在靠近Data线位置激光切断Source组件；b. 在靠近ITO位置激光切断Drain组件；c. 切断Drain部件的相同路径上覆盖ITO remove；\
    #     3. 除非规则中特别指明修补路径位置，修补路径应尽可能靠近缺陷区域。；{repair_rule}\
    #     请输出两个内容：1. 判断图中可视化切割线是否符合所述修补规则，回答符合或者不符合；\
    #     2. 如果不符合，请说出不符合的地方。输出格式请参考以下示例(包含颜色确认、规则确认、图像分析、应用规则、结论、理由)："
    # )

    # Load model & processor
    # --- Qwen2.5-VL ---
    # BASE_MODEL = "/mnt/data1/models/Qwen2.5-VL-7B-Instruct"
    # cfg = AutoConfig.from_pretrained(args.model)
    # base_cfg = AutoConfig.from_pretrained(BASE_MODEL)
    # if getattr(cfg, "rope_scaling", None) is None:
    #     cfg.rope_scaling = getattr(base_cfg, "rope_scaling", None)
    # if getattr(cfg, "text_config", None) is not None and getattr(cfg.text_config, "rope_scaling", None) is None:
    #     if getattr(base_cfg, "text_config", None) is not None:
    #         cfg.text_config.rope_scaling = base_cfg.text_config.rope_scaling
    #     else:
    #         cfg.text_config.rope_scaling = base_cfg.rope_scaling
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     args.model, config=cfg, torch_dtype=torch_dtype, device_map=args.device_map
    # )
    # processor = AutoProcessor.from_pretrained(BASE_MODEL)

    # --- Qwen3-VL MoE  ---
    # BASE_MODEL = "/mnt/data1/models/Qwen3-VL-30B-A3B-Instruct"
    # cfg = AutoConfig.from_pretrained(args.model)
    # if getattr(cfg, "text_config", None) is not None and getattr(cfg.text_config, "rope_scaling", None) is None:
    #     base_cfg = AutoConfig.from_pretrained(BASE_MODEL)
    #     cfg.text_config.rope_scaling = base_cfg.text_config.rope_scaling
    # model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    #     args.model,
    #     config=cfg,
    #     dtype=torch.bfloat16,
    #     device_map=args.device_map or "auto",
    # )
    # model.set_attn_implementation("sdpa")
    # model.eval()
    # processor = AutoProcessor.from_pretrained(BASE_MODEL)

    # --- Qwen3-VL DENSE ---
    BASE_MODEL = "/mnt/data1/models/Qwen3-VL-8B-Instruct"
    cfg = AutoConfig.from_pretrained(args.model)
    if getattr(cfg, "text_config", None) is not None and getattr(cfg.text_config, "rope_scaling", None) is None:
        base_cfg = AutoConfig.from_pretrained(BASE_MODEL)
        cfg.text_config.rope_scaling = base_cfg.text_config.rope_scaling
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        config=cfg,
        dtype=torch_dtype,
        device_map=args.device_map or "auto",
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    # ------------------------------------------------------

    # Load eval data
    data_list = load_json_file(args.json_path)

    results = []
    for idx, data in enumerate(tqdm(data_list), start=1):
        img_path = data["images"][0]

        user_msg = ""
        gt_msg = ""
        for it in data['messages']:
            if it['role'] == "user":
                user_msg = it['content']
            elif it['role'] == "assistant":
                gt_msg = it['content']

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_msg}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_msg},
                ],
            },
        ]

        output_text = infer(model, processor, messages, max_new_tokens=args.max_new_tokens)

        meta = {k: v for k, v in data.items() if k != "messages"}
        pred = output_text[0] if isinstance(output_text, list) and len(output_text) == 1 else output_text

        results.append({
            **meta,
            "eval_index": idx,
            "user_msg": user_msg,
            "gt": gt_msg,
            "qwen_pred": pred
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("saved to", out_path)
