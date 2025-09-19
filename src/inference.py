'''
python inference.py \
  --model /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage2/checkpoint-250 \
  --image /mnt/workspace/kennethliu/bin/color_test_512_2.png \
  --prompt "若存在，请返回该图中全部TFT的bbox" \
  --min_pixels $((324*28*28)) \
  --max_pixels $((324*28*28))

  圖中TFT的顏色是甚麼
  请列出图中全部组件及其颜色

python inference.py \
  --model /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage2/checkpoint-250 \
  --image /mnt/workspace/autorepair_vlm/gt_datasets/positive/TSMRN/TSMRN_TSRAS_AS_0916211526/repair_image.jpg \
  --prompt "如图所示是一张半导体显示面板图像，蓝色曲线内区域为缺陷，算法根据输入的缺陷位置及对应的修补信息输出了需要修补的路径，红色直线代表模拟的激光切割的路径，黄绿色直线（较淡较宽）代表模拟的ITO remove的路径，两种路径有可能有重合。你需要检查图上的可视化路径是否符合以下修补规则：对于Drain缺陷，当缺陷发生在此组件上时，需做修补。具体手法为：a.在靠近Data线位置激光切断Source组件；b.在靠近ITO位置激光切断Drain组件；c.切断Drain部件的相同路径上覆盖ITO remove。；对于TFT缺陷，当缺陷发生在此组件上时，需做修补。具体手法为：a.在靠近Data线位置激光切断Source组件；b.在靠近ITO位置激光切断Drain组件；c.切断Drain部件的相同路径上覆盖ITO remove。。请输出两个内容：1.判断图中可视化切割线是否符合所述修补规则，回答符合或者不符合；2.如果不符合，请说出不符合的地方。输出格式请参考以下示例(需包含规则确认、图像分析、应用规则、结论、理由)：" \
  --min_pixels $((324*28*28)) \
  --max_pixels $((324*28*28))

python inference.py \
  --model /mnt/workspace/kennethliu/ckpt/qwen2_5vl-7b_dual_rehearsal_3/stage2/checkpoint-250 \
  --image /mnt/workspace/autorepair_vlm/gt_datasets/negative/ito_rm_scaling/TSCVD_TSCOK_0919162906/repair_image.jpg \
  --prompt "如图所示是一张半导体显示面板图像，蓝色曲线内区域为缺陷，算法根据输入的缺陷位置及对应的修补信息输出了需要修补的路径，红色直线代表模拟的激光切割的路径，黄绿色直线（较淡较宽）代表模拟的ITO remove的路径，两种路径有可能有重合。你需要检查图上的可视化路径是否符合以下修补规则：对于TSCOK缺陷，此种类型缺陷需做修补，首先需要判断缺陷的形状是“U”型还是直线型，当缺陷形状是“U”型时，具体手法为用ITO remove围绕“U”型缺陷轮廓外部一圈，ITO remove路径呈现为一个空心的“U”型；当缺陷形状为直线型时，具体修补手法为用ITO remove沿着缺陷的外接矩形一圈，此时ITO remove路径呈现为一个矩形。然后需要检查上述ITO remove的区域内是否包含TFT组件，或ITO remove路径与TFT组件有相交，如有包含TFT或相交，则需要进行进一步修补，具体手法为a.在靠近Data线位置激光切断Source组件；b.在靠近ITO位置激光切断Drain组件；c.切断Drain部件的相同路径上覆盖ITO remove。。请输出两个内容：1.判断图中可视化切割线是否符合所述修补规则，回答符合或者不符合；2.如果不符合，请说出不符合的地方。输出格式请参考以下示例(需包含规则确认、图像分析、应用规则、结论、理由)：" \
  --min_pixels $((324*28*28)) \
  --max_pixels $((324*28*28))

    
  
'''

import argparse
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to HF checkpoint dir")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--prompt", required=True, help="User prompt/question")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    # keep min/max pixels consistent with training (snap-to-28 happens inside Qwen utils)
    ap.add_argument("--min_pixels", type=int, default=324*28*28)
    ap.add_argument("--max_pixels", type=int, default=324*28*28)
    args = ap.parse_args()

    device_map = "auto"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Processor: pass pixel policy so resize matches training
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )

    # Model
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Load image
    img = Image.open(args.image).convert("RGB")

    # Chat-style input (image + text)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    # Build template and tensors
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat_text], images=[img], return_tensors="pt").to(model.device)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0.0,
            use_cache=True,
        )

    # Decode; keep only the assistant's reply (after the chat prompt)
    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    # Try to split on assistant marker if present
    reply = decoded.split("<|assistant|>")[-1].strip()
    print(reply)

if __name__ == "__main__":
    main()
