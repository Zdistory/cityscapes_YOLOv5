import os
import json
from tqdm import tqdm
from PIL import Image
from glob import glob

# 类别映射
TARGET_CLASSES = ['person', 'rider', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']
label2id = {label: idx for idx, label in enumerate(TARGET_CLASSES)}

def get_bbox(polygon):
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max

def convert_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

def convert_cityscapes_to_yolo(gt_dir, img_dir, output_img_dir, output_label_dir):
    sets = ['train', 'val', 'test']

    for split in sets:
        gt_split_dir = os.path.join(gt_dir, split)
        img_split_dir = os.path.join(img_dir, split)

        output_img_split_dir = os.path.join(output_img_dir, split)
        output_label_split_dir = os.path.join(output_label_dir, split)

        os.makedirs(output_img_split_dir, exist_ok=True)
        os.makedirs(output_label_split_dir, exist_ok=True)

        cities = os.listdir(gt_split_dir)
        for city in tqdm(cities, desc=f'Processing {split}'):
            json_files = glob(os.path.join(gt_split_dir, city, '*_gtFine_polygons.json'))

            for json_path in json_files:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                img_file = json_path.replace('gtFine', 'leftImg8bit').replace('_leftImg8bit_polygons.json', '_leftImg8bit.png')
                img_name = os.path.basename(img_file)
                img = Image.open(img_file)
                img_w, img_h = img.size

                # 拷贝图片
                img.save(os.path.join(output_img_split_dir, img_name))

                # 生成YOLO格式标注
                label_file = os.path.join(output_label_split_dir, img_name.replace('.png', '.txt'))
                with open(label_file, 'w') as out_f:
                    for obj in data['objects']:
                        label = obj['label']
                        if label in TARGET_CLASSES:
                            bbox = get_bbox(obj['polygon'])
                            x_center, y_center, w, h = convert_to_yolo(bbox, img_w, img_h)
                            class_id = label2id[label]
                            out_f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n')

if __name__ == '__main__':
    convert_cityscapes_to_yolo(
        gt_dir='D:/learning/SUMMER/cityscapes/gtFine',
        img_dir='D:/learning/SUMMER/cityscapes/leftImg8bit',
        output_img_dir='D:/learning/SUMMER/cityscapes_yolo/data/images',
        output_label_dir='D:/learning/SUMMER/cityscapes_yolo/data/labels'
    )
