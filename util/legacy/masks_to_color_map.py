# legacy

import cv2
import json
import numpy as np
import os, glob
import math
from tqdm import tqdm


def load_contours_from_json(json_path):
    """
    Load contours and labels from JSON file as parallel lists.

    Args:
        json_path (str): Path to the JSON file

    Returns:
        labels (list): List of labels
        contours (list): List of contours, each contour is a numpy array of points
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    labels = []
    contours = []

    # Process each item in the JSON contours list
    for item in data['contours']:
        labels.append(item['label'])
        contours.append(np.array(item['points'], dtype=np.int32))

    return labels, contours


def contour_to_mask(contour, image_size):
    """
    Convert a single contour to binary mask.

    Args:
        contour (numpy.ndarray): Contour points of shape (N, 2)
        image_size (tuple): Target image size (height, width)

    Returns:
        numpy.ndarray: Binary mask of shape (height, width)
    """
    # Create empty mask
    mask = np.zeros(image_size, dtype=np.uint8)

    # Convert contour to integer coordinates
    contour = contour.astype(np.int32)

    # Draw filled contour on mask
    cv2.fillPoly(mask, [contour], 1)

    return mask


def contours_to_mask(contours, image_size):
    """
    Convert a list of contours to a single binary mask.

    Args:
        contours (list): List of contours, each contour is a numpy array of shape (N, 2)
        image_size (tuple): Target image size (height, width)

    Returns:
        numpy.ndarray: Binary mask of shape (height, width) with 0s and 1s
    """
    # Create empty mask
    mask = np.zeros(image_size, dtype=np.uint8)

    # Draw all contours
    for contour in contours:
        # Convert contour to integer coordinates
        contour = contour.astype(np.int32)
        # Fill the contour
        cv2.fillPoly(mask, [contour], 1)

    return mask


def read_image(file):
    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def contour_to_mask(points, height, width):
    """
    将OpenCV轮廓转换为二值掩膜
    参数：
        contour: OpenCV轮廓（通过findContours获取）
        height: 原图高度
        width: 原图宽度
    返回：
        mask: 二值化掩膜（uint8类型，0为背景，255为前景）
    """
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    # 创建全黑画布
    mask = np.zeros((height, width), dtype=np.uint8)

    # 绘制填充轮廓（厚度-1表示填充内部）
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)

    # 或者使用fillPoly方法（两种方法二选一）
    # cv2.fillPoly(mask, [contour], color=255)

    return mask


def get_components(mask, component_threshold=1):
    from scipy.ndimage.measurements import label as scipy_label
    from scipy.ndimage.measurements import center_of_mass
    import numpy as np

    struct = np.ones((3, 3), dtype=np.int)

    components_labels, ncomponents = scipy_label(mask, struct)
    component_masks = []

    for l in range(1, ncomponents + 1):
        component = (components_labels == l)
        count = np.count_nonzero(component)
        # print('count: ', count)
        if count < component_threshold:
            continue

        component_masks.append(component)

    return component_masks


def split_masks(mask, masks, min_area=10):
    contours, _ = cv2.findContours(
        image=mask,
        mode=cv2.RETR_EXTERNAL,  # 只检测外层轮廓
        method=cv2.CHAIN_APPROX_SIMPLE
    )
    mask_count = 0
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        mask_count = mask_count + 1
        mask_tmp = np.zeros_like(mask)
        cv2.drawContours(mask_tmp, contours, idx, (255, 255, 255), thickness=-1)

        masks.append(mask_tmp)
    return mask_count


class ProductTemplate:
    def __init__(self, temp_dir: str, prod_name: str = "50UD"):

        self.temp_dir = temp_dir
        self.image_dir = os.path.join(self.temp_dir, prod_name, 'image')
        self.temp_image = read_image(os.path.join(self.image_dir, 'copy.png'))
        self.mask_dir = os.path.join(self.temp_dir, prod_name, 'mask')
        com_image = cv2.imread(os.path.join(self.mask_dir, 'A-Com.png'), 0)
        data_image = cv2.imread(os.path.join(self.mask_dir, 'Data.png'), 0)
        gate_image = cv2.imread(os.path.join(self.mask_dir, 'Gate.png'), 0)
        # ITO_image = np.logical_or(cv2.imread(os.path.join(self.mask_dir, 'COM_ITO.png'), 0),
        #                           cv2.imread(os.path.join(self.mask_dir, 'P_ITO.png'), 0))
        ITO_image = cv2.imread(os.path.join(self.mask_dir, 'COM_ITO.png'), 0)
        source_image = cv2.imread(os.path.join(self.mask_dir, 'Source.png'), 0)
        drain_image = cv2.imread(os.path.join(self.mask_dir, 'Drain.png'), 0)
        tft_image = cv2.imread(os.path.join(self.mask_dir, 'TFT_CH.png'), 0)
        mesh_image = cv2.imread(os.path.join(self.mask_dir, 'Mesh.png'), 0)
        mesh_hole_image = cv2.imread(os.path.join(self.mask_dir, 'Mesh_Hole.png'), 0)
        via_image = cv2.imread(os.path.join(self.mask_dir, 'VIA_Hole.png'), 0)
        self.components = {"Com": com_image, "Data": data_image, "Gate": gate_image, "ITO": ITO_image,
                           "Source": source_image, "Drain": drain_image, "TFT": tft_image, "Mesh": mesh_image,
                           "Mesh_Hole": mesh_hole_image, "VIA_Hole": via_image}
        #https://blog.csdn.net/qq_51985653/article/details/113392665
        self.component_colors = {"Com":  [0, 128, 0],#13 BGR 深绿色
                                 "Data":  [255, 128, 128], #9 BGR 浅蓝色
                                 "Gate":  [0, 128, 255],#12 BGR 橙色
                                 "ITO":  [128, 128, 128], #15 BGR 灰色
                                 "Source": [42, 42, 128],#14 BGR 棕色
                                 "Drain":  [255, 0, 0],#5 BGR 蓝色
                                 "TFT":  [255, 0, 255],#8 BGR 紫色
                                 "Mesh":  [128, 0, 0],#11 BGR 深蓝色:
                                 "Mesh_Hole":  [0, 255, 0], #4 BGR 绿色
                                 "VIA_Hole": [255, 255, 255],#2 BGR 白色
                                 "defect": [255, 255, 0]#7 BGR 青色
                                 }

        #3  "Cut":  [0, 0, 255], # BGR 红色
        #6"ITO_remove":  [0, 255, 255],# BGR 黄色
        #1 背景[0,0,0]



    def template_matching(self, input_image, method=cv2.TM_CCOEFF_NORMED):

        template_gray = cv2.cvtColor(self.temp_image, cv2.COLOR_BGR2GRAY)
        input_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        h, w = input_gray.shape
        res = cv2.matchTemplate(template_gray, input_gray, method)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        # bottom_right = (top_left[0] + w, top_left[1] + h)
        matched_img = self.temp_image[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

        matched_components = {}
        for name, image in self.components.items():
            matched_components[name] = image[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]

        return matched_img, matched_components

    def get_damaged_components(self, defect_mask, components):
        masks = []
        split_masks(defect_mask, masks, 10)
        damaged_components = []
        for d_mask in masks:
            d_mask = (d_mask > 0).astype(np.uint8)
            damaged_components_tmp = []
            for label, component_mask in components.items():

                component_mask = (component_mask > 0).astype(np.uint8)
                components_masks = get_components(component_mask)
                for c_m in components_masks:
                    intersection = np.logical_and(d_mask, c_m)
                    if np.any(intersection):
                        damaged_components_tmp.append(label)

            damaged_components.append(damaged_components_tmp)

        return damaged_components

import repair_rules_lookup_table

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    lookup = repair_rules_lookup_table.RepairRules()

    # json_file = "dataset/agent_data_samples/TOPEN_TOITO_repair/review_results_TOPEN_TOITO_ITO_0917151816.json"
    # labels, contours = load_contours_from_json(json_file)
    # # print(labels, "\n", contours)

    # for contour in contours:
    #     mask = contour_to_mask(contour, (968, 968))
    #     plt.imshow(mask)
    #     plt.show()
    # TOPEN  TSMRN TSCVD TSHRT
    data_dir = "/mnt/workspace/autorepair_vlm/original_data/TSCVD"
    cls = os.path.basename(data_dir)
    image_files = glob.glob(os.path.join(data_dir, 'original_*.jpg'))
    save_folder_name = "tmp"
    if "TSCVD" in cls:
        files = glob.glob(os.path.join("/mnt/workspace/autorepair_vlm/original_data/AArepair_testset/TSCVD", 'TSCOK', '*.jpg'))
        TSCOK = [os.path.basename(f).split('.')[0] for f in files]
        files = glob.glob(os.path.join("/mnt/workspace/autorepair_vlm/original_data/AArepair_testset/TSCVD", 'TSCNG', '*.jpg'))
        TSCNG = [os.path.basename(f).split('.')[0] for f in files]

    for files in tqdm(image_files):
        # if "TSHRT_TSBLK_0917113818" not in files:
        #     continue
        save_folder_name = os.path.basename(files).split('.')[0]
        id = save_folder_name.split('original_')[-1]
        id_files = glob.glob(os.path.join(data_dir, '*_' + id + '.*'))

        for file in id_files:
            if len(id_files) < 3:
                continue
            if 'repair' in os.path.basename(file):
                rep_img = read_image(file)
            elif 'original' in os.path.basename(file):
                save_folder_name = os.path.basename(file).split('.')[0]
                ori_img = read_image(file)
            elif 'review' in os.path.basename(file):
                json_parser = json.load(open(file))
            elif 'defect_mask' in os.path.basename(file):
                defect_mask = read_image(file)

        if rep_img is None:
            continue


        # plt.imshow(rep_img)
        # plt.show()
        def bounding_box(points):
            x_coordinates, y_coordinates = zip(*points)

            return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]


        brepair = False
        json_new = json_parser.copy()
        if True:
            # TODO: get cropping position depending on defect and repair locations
            points = []

            for ps in json_parser["contours"]:
                if "rps_points" in ps['label'] and len(ps['points']) > 0:
                    brepair = True
                points += ps["points"]

            start_pt = np.array([340, 410])
            crop_size = 400

            # crop_repair_img = rep_img[start_pt[1]:start_pt[1]+crop_size, start_pt[0]:start_pt[0]+crop_size]
            W, H, C = rep_img.shape
            bbox = bounding_box(points)
            size = max(abs(bbox[0][1] - bbox[1][1]), abs(bbox[0][0] - bbox[1][0]))
            size = size * 1.2  # dilate 1.2
            if size < 512:
                size = 512
            elif size > W:
                size = W
            cy = (bbox[0][1] + bbox[1][1]) / 2
            cx = (bbox[0][0] + bbox[1][0]) / 2
            x_start = math.ceil(max(0, cx - size / 2))
            x_start = math.ceil(min(x_start, W - size))
            y_start = math.ceil(max(0, cy - size / 2))
            y_start = math.ceil(min(y_start, H - size))
            size = math.ceil(size)
            crop_repair_img = rep_img[y_start:y_start + size, x_start:x_start + size]
            # cv2.imshow("result", crop_repair_img)
            # cv2.waitKey(0)
            # crop_repair_img = cv2.cvtColor(crop_repair_img, cv2.COLOR_BGR2RGB)
            # plt.imshow(crop_repair_img)
            # plt.show()
            for ps in json_new["contours"]:

                for i in range(0, len(ps['points']), 2):
                    p1 = [int(f) for f in ps['points'][i]]
                    ps['points'][i][0] = ps['points'][i][0] - x_start
                    ps['points'][i][1] = ps['points'][i][1] - y_start
                    if i + 1 >= len(ps['points']):
                        break
                    p2 = [int(f) for f in ps['points'][i + 1]]
                    ps['points'][i + 1][0] = ps['points'][i + 1][0] - x_start
                    ps['points'][i + 1][1] = ps['points'][i + 1][1] - y_start

            temp_dir = "/mnt/workspace/autorepair_vlm/original_data/template"
            prod_temp = ProductTemplate(temp_dir)
            matched_img, matched_components = prod_temp.template_matching(ori_img)

            damaged_components_list = prod_temp.get_damaged_components(defect_mask, matched_components)
            # compose gt_analysis
            compose_list = ["Gate", "Data", "Gate&Data", "Data&Data", "Data&Com", "Gate&Gate", "Gate&Com", "Gate&TFT", \
                            "Gate&Drain", "Gate&Mesh", "TFT", "Data&Drain", "Data&Mesh", "Data&ITO", "Gate&ITO", "ITO", \
                            "Mesh_Hole", "VIA_Hole", "Com", "Drain"]
            # self.components = {"Com": com_image, "Data": data_image, "Gate": gate_image, "ITO": ITO_image,
            #                    "Source": source_image, "Drain": drain_image, "TFT": tft_image, "Mesh": mesh_image,
            #                    "VIA_Hole": via_image}
            compose_repair = []
            for d_comps in damaged_components_list:
                for compose in compose_list:
                    for d_comp in d_comps:
                        if d_comp in compose:
                            if '&' in compose:
                                split_comp = compose.split('&')
                                matched = True
                                import copy

                                d_comps_tmp = copy.deepcopy(d_comps)
                                d_comps_tmp.remove(d_comp)
                                split_comp.remove(d_comp)
                                for split in split_comp:

                                    if split not in d_comps_tmp:
                                        matched = False
                                if matched:
                                    compose_repair.append(compose)
                            else:
                                compose_repair.append(compose)
            if "TOITO" in cls:
                if "ITO" in compose_repair:
                    compose_repair = ["ITO"]
                else:
                    compose_repair = []
            cls_key = cls
            if "TSCVD" in cls:
                if id in TSCNG:
                    cls_key = "TSCNG"
                    compose_repair = ["TSCNG"]
                if id in TSCOK:
                    cls_key = "TSCOK"
                    compose_repair = ["TSCOK"]
            repair_rules = []
            compose_repair = list(set(compose_repair))
            for i, compose in enumerate(compose_repair):
                gt_str = {}
                gt_str["damaged_component"] = compose
                rule = lookup.get_value(cls_key, compose)
                rules = []
                for r in rule:
                    repair_rule = {}
                    repair_rule["repair_rule"] = r["repair_rule"]
                    repair_rule["operations"] = r["operations"]
                    repair_rule["repair_components"] = r["repair_components"]
                    rules.append(repair_rule)
                gt_str["rules"] = rules
                repair_rules.append(gt_str)

            # save_folder = os.path.join(os.path.dirname(data_dir), "results", os.path.basename(data_dir), id)
            save_folder = os.path.join("autorepair_vlm/original_data/results/positive",
                                       os.path.basename(data_dir), id)
            os.makedirs(save_folder, exist_ok=True)
            crop_original_img = ori_img[y_start:y_start + size, x_start:x_start + size]
            cv2.imwrite(os.path.join(save_folder, 'repair_image.jpg'), crop_repair_img)
            cv2.imwrite(os.path.join(save_folder, 'original_image.jpg'), crop_original_img)
            # Create empty RGB channels
            height, width, _ = crop_original_img.shape
            components_image = np.zeros((height, width, 3), dtype=np.uint8)
            for label, component_mask in matched_components.items():
                save_filename = os.path.join(save_folder, 'mask_{}.jpg'.format(label))

                crop_mask = component_mask[y_start:y_start + size, x_start:x_start + size]
                crop_mask = (crop_mask > 0).astype(np.uint8) * 255
                # cv2.imwrite(save_filename, crop_mask)
                target_color = prod_temp.component_colors[label] #
                components_image[crop_mask == 255] = target_color
                # save_filename = os.path.join(save_folder, 'components_image.bmp')
                # cv2.imwrite(save_filename, components_image)
            # components_image[defect_mask[y_start:y_start + size, x_start:x_start + size] == 255] =[255, 255, 255]


            thickness = 5
            canvas_ITO = np.zeros_like(components_image)
            for ps in json_parser["contours"]:
                if "rps_points" in ps['label'] and len(ps['points']) > 0:
                    brepair = True
                    if "rps_points:10"==ps['label']: #cut:source v, drain h:
                        tp=10
                        for i in range(0, len(ps['points']), 2):
                            p1 = [int(f) for f in ps['points'][i]]
                            p2 = [int(f) for f in ps['points'][i + 1]]
                            if p1[0]==p2[0] : #source v
                                cv2.line(canvas_ITO, p1, p2, (0, 0, 255), thickness, cv2.LINE_8)
                                # cv2.line(ori_img_copy, p1, p2, (0,0,255), thickness, cv2.LINE_8)
                                bhit = True
                            if  p1[1]==p2[1]: #drain h
                                cv2.line(canvas_ITO, p1, p2, (0,0,255), thickness, cv2.LINE_8)
                                # bhit = True
                    elif "rps_points:11" == ps['label']:  # ITO remove
                        tp = 11
                        for i in range(0, len(ps['points']), 2):
                            p1 = [int(f) for f in ps['points'][i]]
                            p2 = [int(f) for f in ps['points'][i + 1]]
                            cv2.line(components_image, p1, p2, (0, 255, 255), 4 * thickness, cv2.LINE_8)


                    elif "rps_points:110" == ps['label']:  # U left
                        tp = 110
                        for i in range(0, len(ps['points']), 2):
                            p1 = [int(f) for f in ps['points'][i]]
                            p3 = [int(f) for f in ps['points'][i + 1]]
                            p2 = [p3[0], p1[1]]
                            p4 = [p1[0], p3[1]]
                            cv2.line(components_image, p1, p2, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            cv2.line(components_image, p2, p3, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            cv2.line(components_image, p3, p4, (0, 255, 255), 4 * thickness, cv2.LINE_8)

                    elif "rps_points:111" == ps['label']:  # U upper
                        tp = 111
                        for i in range(0, len(ps['points']), 2):
                            p1 = [int(f) for f in ps['points'][i]]
                            p3 = [int(f) for f in ps['points'][i + 1]]
                            p2 = [p3[0], p1[1]]
                            p4 = [p1[0], p3[1]]
                            # cv2.line(canvas_ITO, p1, p2, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            cv2.line(components_image, p2, p3, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            cv2.line(components_image, p3, p4, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            cv2.line(components_image, p4, p1, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                    elif "rps_points:112" == ps['label']:  # U right
                        tp = 112
                        for i in range(0, len(ps['points']), 2):
                            p1 = [int(f) for f in ps['points'][i]]
                            p3 = [int(f) for f in ps['points'][i + 1]]
                            p2 = [p3[0], p1[1]]
                            p4 = [p1[0], p3[1]]
                            cv2.line(components_image, p1, p2, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            # cv2.line(canvas_ITO, p2, p3, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            cv2.line(components_image, p3, p4, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            cv2.line(components_image, p4, p1, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                    elif "rps_points:113" == ps['label']:  # U down
                        tp = 113
                        for i in range(0, len(ps['points']), 2):
                            p1 = [int(f) for f in ps['points'][i]]
                            p3 = [int(f) for f in ps['points'][i + 1]]
                            p2 = [p3[0], p1[1]]
                            p4 = [p1[0], p3[1]]
                            cv2.line(components_image, p1, p2, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            cv2.line(components_image, p2, p3, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            # cv2.line(canvas_ITO, p3, p4, (0, 255, 255), 4 * thickness, cv2.LINE_8)
                            cv2.line(components_image, p4, p1, (0, 255, 255), 4 * thickness, cv2.LINE_8)

                else:
                    contours = contour = np.array(ps["points"], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.drawContours(components_image, [contours], -1, prod_temp.component_colors["defect"], 2)
                    # save_filename = os.path.join(save_folder, 'components_image.bmp')
                    # cv2.imwrite(save_filename, components_image)
            ''''''
            alpha_ITO = 1.0
            beta = 1.0
            components_image[canvas_ITO[:,:,2]>0]=[0,0,0]
            cv2.addWeighted(canvas_ITO, alpha_ITO, components_image, beta, 0.0, components_image)
            save_filename = os.path.join(save_folder, 'components_image.bmp')
            cv2.imwrite(save_filename, components_image)
            # json.dump(repair_rules, open(os.path.join(save_folder, 'repair_rule.json'), "w", encoding="utf8"),
            #           sort_keys=False,
            #           indent=4, separators=(',', ':'),
            #           ensure_ascii=False)

            json.dump(json_new, open(os.path.join(save_folder, 'output.json'), "w", encoding="utf8"),
                      sort_keys=False,
                      indent=4, separators=(',', ':'),
                      ensure_ascii=False)


