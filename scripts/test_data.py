import matplotlib.pyplot as plt
import os
from PIL import Image

# 参数
img_path = '../data/images/train/aachen_000000_000019_leftImg8bit.png'
label_path = '../data/labels/train/aachen_000000_000019_leftImg8bit.txt'

# 加载图片
img = Image.open(img_path)
w, h = img.size

# 画图
plt.figure(figsize=(10, 6))
plt.imshow(img)

# 加载并画 bbox
with open(label_path, 'r') as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.strip().split())
        # 还原为像素坐标
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        x2 = (x + bw / 2) * w
        y2 = (y + bh / 2) * h
        # 画矩形
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(x1, y1 - 5, f"{int(cls)}", color='white', bbox=dict(facecolor='red', alpha=0.5))
plt.axis('off')
plt.show()
