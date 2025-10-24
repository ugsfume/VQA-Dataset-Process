# legacy

'''
This script preps dataset using color-coded mask images for SFT 
'''

    # TODO: 
        # change directory path    O
        # change user content      O
        # change assistant content (blue -> cyan, laser_cut)

import os, json
from collections import defaultdict
from random import shuffle

def load_json_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def safe_listdir(path):
    """ Skip hidden files """
    return [f for f in os.listdir(path) if not f.startswith('.')]

def combine_repair_rules(json_data):
    """
    Combine all repair rules into a single string
    """
    combined_text = []
    for item in json_data:
        damaged_component = item["damaged_component"]
        rules = [rule["repair_rule"] for rule in item["rules"]]
        rules_text = " ".join(rules)
        combined_text.append(f"对于{damaged_component}缺陷，{rules_text}")
    return "；".join(combined_text)

def count_image_placeholders(messages):
    return sum(msg["content"].count("<image>") for msg in messages)

def stratified_sample_split(root_path):
    """ 按照每个类别下的样本进行 8:2 分割 """
    train_set, test_set = [], []
    stats = {}

    for node in safe_listdir(root_path): 
        node_path = os.path.join(root_path, node)
        for cl in safe_listdir(node_path):  # Classes (TOPEN, TSMRN etc.)
            cls_path = os.path.join(node_path, cl)

            samples = []
            for sample in safe_listdir(cls_path):  # Sample folders
                data_folder = os.path.join(cls_path, sample)

                rule_file = os.path.join(data_folder, 'repair_rule.json')
                gt_file = os.path.join(data_folder, 'analysis_gt.json')
                img_file = os.path.join(data_folder, 'components_image.bmp')

                # Only include samples with all required files
                if not all(os.path.exists(f) for f in [rule_file, gt_file, img_file]):
                    continue

                # Combine repair rules into a single string for the prompt
                repair_rules = combine_repair_rules(load_json_file(rule_file))
                gt_data = load_json_file(gt_file)

                # Construct the user prompt, describing the image and repair rules
                user_content = (
                    "<image>如图所示是一张半导体显示面板的映射图像。"
                    "图像中不同的部件会用不同的颜色表示(部件名称: 颜色名字), Com: 深绿色, Data: 浅蓝色, Gate: 橙色, ITO: 灰色, "
                    "Source: 棕色, Drain: 蓝色, TFT: 紫色, Mesh: 深蓝色, Mesh_Hole: 绿色, VIA_Hole: 白色, defect: 青色, "
                    "激光切断/Cut: 红色, ITO_remove: 黄色, 背景: 黑色。 青色曲线内区域为缺陷，"
                    "算法根据输入的缺陷位置及对应的修补信息输出了需要修补的路径，"
                    "红色直线代表模拟的激光切割的路径Cut，黄色直线（较宽）代表模拟的ITO_remove的路径。"
                    f"Cut和ITO_remove两种路径有可能有重合。你需要检查图上的可视化路径是否符合以下修补规则：{repair_rules}。"
                    "请输出两个内容：1.判断图中可视化切割线是否符合所述修补规则，回答符合或者不符合；"
                    "2.如果不符合，请说出不符合的地方。输出格式请参考以下示例(需包含规则确认、图像分析、应用规则、结论、理由)："
                )

                # Construct the assistant's expected output (ground truth)
                assistant_content = f"结果: {gt_data['result']}\n\n分析报告:\n{gt_data['analysis']}"

                messages = [
                    {"content": user_content, "role": "user"},
                    {"content": assistant_content, "role": "assistant"},
                ]

                # Determine how many images to attach based on <image> placeholders
                img_count = count_image_placeholders(messages)
                images = [img_file] * img_count if img_count > 0 else [img_file]

                samples.append({
                    "messages": messages,
                    "images": images,
                    "class": cl  # 可选，方便后续统计
                })

            # Shuffle and split samples for this class into train/test sets (80/20)
            shuffle(samples)
            total = len(samples)
            train_count = int(total * 0.8)
            test_count = total - train_count

            train_samples = samples[:train_count]
            test_samples = samples[train_count:]

            train_set.extend(train_samples)
            test_set.extend(test_samples)

            stats[cl] = {
                "total": total,
                "train": train_count,
                "test": test_count
            }

    return train_set, test_set, stats

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    root_path = "/mnt/workspace/autorepair_vlm/gt_datasets20250722"
    train_file = "/mnt/workspace/yangsidi/LLaMA-Factory/data/TCL_auto_repair_train_20250812_mask.json"
    test_file = "/mnt/workspace/yangsidi/LLaMA-Factory/data/TCL_auto_repair_test_20250812_mask.json"

    train_data, test_data, stats = stratified_sample_split(root_path)

    save_json(train_data, train_file)
    save_json(test_data, test_file)

    # Print statistics by class
    print("按类别统计：")
    for cl, s in stats.items():
        print(f"类别: {cl} - 总数: {s['total']}, 训练集: {s['train']}, 测试集: {s['test']}")

    print("\n总样本数统计：")
    print(f"训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")
    print(f"已保存至:\n训练集 -> {train_file}\n测试集 -> {test_file}")
