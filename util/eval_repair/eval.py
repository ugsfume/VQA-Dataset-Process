#!/usr/bin/env python3
import os, json, argparse
from typing import List, Dict, Any
import numpy as np
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
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
    p = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL models on TCL auto-repair test JSON.")
    p.add_argument("--model", required=True, help="Path to model directory or HF model ID")
    p.add_argument("--save-dir", required=True, help="Directory to write the output JSON file")
    p.add_argument("--save-name", default=None, help="Output filename (defaults to pred_<parent>_<basename>.json)")
    p.add_argument("--json-path", default="/mnt/workspace/yangsidi/LLaMA-Factory/data/TCL_auto_repair_test_20250524.json",
                   help="Path to the evaluation JSON data")
    p.add_argument("--max-new-tokens", type=int, default=2048)
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
        "如图所示是一张半导体显示面板图像,\
        蓝色曲线内区域为缺陷，算法根据输入的缺陷位置及对应的修补信息输出了需要修补的路径，红色直线代表模拟的激光切割的路径，\
        黄绿色直线（较淡较宽）代表模拟的ITO remove的路径，两种路径有可能有重合。\
        你需要检查图上的可视化路径是否符合以下修补规则：{repair_rule}。\
        请输出两个内容：1.判断图中可视化切割线是否符合所述修补规则，回答符合或者不符合；\
        2.如果不符合，请说出不符合的地方。\
        输出格式请参考以下示例(需包含规则确认、图像分析、应用规则、结论、理由)："
    )
    right_answer_template = (
        "1.规则确认：规则1：如果缺陷区域是直线型，ITO remove路径应为缺陷外围的矩形。\
        规则2：如果缺陷区域是“U”型，ITO remove路径应沿着缺陷区域外部轮廓做一个U型。\
        2.图像分析：缺陷形状：图中蓝色曲线勾勒出的缺陷区域，其形状明显是一个“U”型。\
        ITO Remove 路径：图中黄绿色较宽的线条代表模拟的ITO remove路径。观察这个路径的形状，它也是一个“U”型。\
        路径与缺陷的关系：黄绿色的U型路径是沿着蓝色U型缺陷的外部轮廓绘制的。\
        3.应用规则：由于缺陷是“U”型的，应适用规则2;规则2要求ITO remove路径是沿着缺陷外部轮廓的U型;图中的黄绿色路径确实是沿着蓝色U型缺陷外部轮廓绘制的U型。\
        4.结论：图中可视化的ITO remove路径（黄绿色线）符合您所述的修补规则。\
        符合理由：缺陷区域呈现“U”型，而黄绿色的ITO remove路径也相应地沿着缺陷区域的外部轮廓呈现为一个U型，这完全符合规则2的要求。"
    )
    wrong_answer_template = (
        "规则确认：条件：缺陷区域（蓝色圆圈）与Data线（纵向组件）有相交。\
        操作：将与缺陷相交的Data线在缺陷区域的两端切断。约束：切割路径（红色直线）必须仅在Data线上，不能与其他组件或区域（如像素区、Gate线等）重合。\
        图像分析：缺陷与Data线相交：蓝色圆圈代表的缺陷区域，确实与左侧那条垂直的Data线有接触或重叠。因此，满足规则的条件。\
        切割路径位置：图中显示了两条红色的水平直线，位于缺陷区域的上方和下方。\
        切割目标：这两条红线并没有切割与缺陷相交的Data线本身。它们横跨了左侧Data线和中间Data线之间的区域（这通常是像素电极区域）。\
        路径重合：根据规则约束，切割路径应仅在Data线上。但图中的红色路径是水平的，并且位于Data线之间的像素区域上，明显与像素区域或其他非Data线组件重合了。\
        结论：图中可视化的切割线不符合所述的修补规则。 不符合的地方：\
        切割位置错误：规则要求切断与缺陷相交的Data线本身（应为在该Data线上的短切割），但图中的红线是水平切割，横跨在Data线之间的像素区域，并未切割Data线。\
        违反路径约束：规则要求切割路径仅在Data线上，但图中的红色路径位于像素区域上，与其他组件/区域发生了重合，违反了此约束。\
        总结来说，正确的做法（根据您描述的规则）应该是在与缺陷相交的那条左侧Data线上，缺陷范围的上方和下方各画一条非常短的、\
        垂直于Data线（即水平）或者沿着Data线（垂直）的切割线（具体方向取决于实际工艺，但目标是断开Data线），\
        并且这条短切割线应该仅限于Data线宽度范围内。而图中显示的横跨整个像素区域的水平长切割线是不符合规则的。"
    )

    # Load model & processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=args.device_map
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # Load eval data
    data_list = load_json_file(args.json_path)

    results = []
    for data in tqdm(data_list):
        # Each item contains 'images' and 'messages' with user/assistant content
        img_path = data['images'][0]

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

        results.append({
            "gt": gt_msg,
            "qwen_pred": output_text
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("saved to", out_path)
