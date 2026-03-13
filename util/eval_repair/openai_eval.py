import http.client
import json


def load_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
    

prompt = "你是一个专业的半导体显示面板业务专家和图像算法专家，你有着能对评估结果进行判决的能力。现在我需要你帮助我评估两个评估结果的相似程度，其中第一个是参考答案，是绝对正确的，另一个是预测答案，有可能会存在不正确的地方。参考答案为{gt_text},预测答案为{pred_text}。现在我需要你在评估后给出评估结果。如果预测结果正确性判断正确，为正样本，请你：1.输出它的正确性得分为1，不加任何其余的文本；2.给出它与参考答案的相似程度，范围在0-1之间，保留两位小数；3.给出混淆矩阵的类别: True Positive (TP), True Negative (TN)。例如参考答案和预测答案同样判断为「不符合」，应归类为TN；4.给出为什么你给出这个相似程度得分的原因。比如一个正样本的输出格式为：Acc: 1. Match: 0.75. Class: TP. Reason: 给出预测结果正确的原因。如果预测结果正确性判断错误，为负样本，请你：1.输出它的正确性得分为0，不加任何其余的文本；2.给出它与参考答案的相似程度，范围在0-1之间，保留两位小数；3.给出混淆矩阵的类别: False Positive (FP), False Negative (FN)。例如参考答案的判断是「不符合」，但预测答案却判断为「符合」，应归类为FP；4.给出为什么你给出这个相似程度得分的原因。比如一个负样本的输出格式为：Acc: 0. Match: 0.12. Class: FN. Reason: 给出预测结果错误的原因。"

eval_file = "/mnt/workspace/kennethliu/eval/grpo_20260220_qwen3vl-8b_dual_white_fixed_repair_20260217/step_100/pred_grpo_20260220_qwen3vl-8b_dual_white_fixed_repair_20260217_global_step_100_merged.json"
    
save_file = "gpt5_eval_grpo_20260220_qwen3vl-8b_dual_white_fixed_repair_20260217_global_step_100_merged.json"
results = []
data = load_json_file(eval_file)

conn = http.client.HTTPSConnection("test-turing.cn.llm.tcljd.com")

for it in data:
    gt = it["gt"]
    pred = it["qwen_pred"]
    
    user_msg = prompt.format(gt_text=gt, pred_text=pred)
    
    payload = json.dumps({
       "model": "turing/gpt-5",
       "messages": [
          {
             "role": "user",
             "content": user_msg
          }
       ],
       "stream": False
    })
    headers = {
       'Authorization': 'Bearer sk-ZISnDqR3Yi5FHVScO5hZCWLO0xx4luKRgT29r9X4sbh',
       'Content-Type': 'application/json'
    }
    conn.request("POST", "/api/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    out = res.read()
    print(out.decode("utf-8"))
    results.append(out.decode("utf-8"))
    
with open(save_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print('saved to', save_file)
