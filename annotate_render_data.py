import os
import csv
import json
import glob
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict


ann_dir="../synthetic_data/scenes/specimen/output/exp_TG_5f1818dd"
save_dir="multimodal-data/welding_dataset"
target_class_names = ["horizontal_plate", "vertical_plate"]


def read_dataset(ann_dir: str, target_class_names: list[str]):
    """
    ann_dir 예: /workspace/welding_line_detection/synthetic_data/scenes/specimen/output/exp_TG_5f1818dd
      - ann_dir/rgb/XXXX_CG_YYYYYYYY_camZ.png
      - ann_dir/semantic_info/XXXX_CG_YYYYYYYY_camZ.json  (bbox_annotations 형식)

    반환 형식은 기존과 동일:
      { image_path: { 'boxes': [[x1,y1,x2,y2], ...],
                      'captions': [label, ...] } }
    """
    sem_dir = os.path.join(ann_dir, "semantic_info")
    rgb_dir = os.path.join(ann_dir, "rgb")

    ann_Dict = defaultdict(lambda: defaultdict(list))

    for jp in sorted(glob.glob(os.path.join(sem_dir, "*.json"))):
        with open(jp, "r") as f:
            data = json.load(f)

        b = data.get("bbox_annotations", {})
        anns = b.get("annotations", [])
        box_format = (b.get("box_format") or "xyxy").lower()
        img_width = b.get("img_width")
        img_height = b.get("img_height")

        # JSON 파일명과 동일한 베이스 이름의 PNG가 rgb/에 있다고 가정
        base = os.path.splitext(os.path.basename(jp))[0]
        img_path = os.path.join(rgb_dir, base + ".png")

        # 만약 실제 파일이 없으면 JSON 안의 image_path로 폴백
        if not os.path.exists(img_path):
            img_path = b.get("image_path", img_path)

        for ann in anns:
            label = ann.get("class_name", "")
            if label not in target_class_names:
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            # box_format에 따라 통일: xyxy
            if box_format in ("xyxy", "x1y1x2y2"):
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
            elif box_format in ("xywh", "x1y1wh"):
                x1, y1, w, h = bbox
            else:
                # 알 수 없는 형식이면 xywh로 간주
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

            # 기존 코드 호환을 위해 정수로 반올림
            ann_Dict[img_path]["boxes"].append([
                int(round(x1)), int(round(y1)), int(round(w)), int(round(h))
            ])
            ann_Dict[img_path]["captions"].append(label.replace('_', ' '))
            ann_Dict[img_path]["img_widths"].append(img_width)
            ann_Dict[img_path]["img_heights"].append(img_height)

    return ann_Dict


def save_annotation(save_dir, ann_dict, train_ratio=0.8):
    save_dir = Path(save_dir)
    train_ann_path = save_dir / "train_annotations.csv"
    val_ann_path = save_dir / "val_annotations.csv"
    train_image_dir = save_dir / "images" / "train"; train_image_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir = save_dir / "images" / "val"; val_image_dir.mkdir(parents=True, exist_ok=True)

    num_img = len(ann_dict)
    num_train = int(np.round(train_ratio*num_img))
    num_val = int(np.round((1-train_ratio)*num_img))
    train_ids, val_ids, _ = np.split(np.random.permutation(len(ann_dict)), [num_train, num_train+num_val])

    train_ann_dict_list = []
    val_ann_dict_list = []

    for i, (image_path, ann) in enumerate(ann_dict.items()):
        if i in train_ids:
            image_dir = train_image_dir
            ann_dict_list = train_ann_dict_list
        else:
            image_dir = val_image_dir
            ann_dict_list = val_ann_dict_list

        image_name = os.path.basename(image_path)
        shutil.copy2(
            image_path,
            image_dir / image_name,
        )

        for box, label, img_width, img_height in zip(ann["boxes"], ann["captions"], ann["img_widths"], ann["img_heights"]):
            # label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,width,height
            ann_dict_list.append({
                "label_name": label,
                "bbox_x": box[0],
                "bbox_y": box[1],
                "bbox_width": box[2],
                "bbox_height": box[3],
                "image_name": image_name,
                "width": img_width,
                "height": img_height,
            })

    with open(train_ann_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(train_ann_dict_list[0].keys()))
        writer.writeheader()
        writer.writerows(train_ann_dict_list)
    with open(val_ann_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(val_ann_dict_list[0].keys()))
        writer.writeheader()
        writer.writerows(val_ann_dict_list)


if __name__ == "__main__":
    np.random.seed(42)
    ann_dict = read_dataset(ann_dir, target_class_names)
    save_annotation(save_dir, ann_dict)
