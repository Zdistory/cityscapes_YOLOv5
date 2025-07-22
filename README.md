# 斑马线区域行人与车辆检测系统

## 1. 项目概述

本项目基于**Cityscapes数据集**开发了一套面向斑马线区域的行人与车辆检测系统，通过改进YOLOv5模型并融合Transformer架构，实现了复杂交通场景下的高精度目标检测。系统支持实时视频分析，可为自动驾驶和智能交通系统提供核心感知能力。



## 2. 项目目标

1. **基础目标检测**：实现行人/车辆/斑马线的精准检测
2. **模型改进验证**：对比原始YOLOv5与Transformer增强版性能
3. **预警系统原型**：开发区域入侵检测功能（待实现）

## 3. 数据集 - Cityscapes

|   属性   |          描述          |
| :------: | :--------------------: |
|   类型   | 像素级标注城市街景图像 |
|  样本量  |    5,000帧精细标注     |
| 关键类别 |    行人、车辆等7类     |
|  预处理  |        转换脚本        |

## 4. 模型架构演进

### 4.1 基准模型：YOLOv5s

```
graph LR
A[640×640输入] --> B[CSPDarknet骨干]
B --> C[FPN+PAN特征融合]
C --> D[三尺度检测头]
```

### 4.2 **Transformer增强变体**

|   **模型**    |            **创新点**            |        **配置文件**        |         **训练指令**          |
| :-----------: | :------------------------------: | :------------------------: | :---------------------------: |
| **原始YOLO**  |          CSPDarkNet骨干          |       `yolov5s.yaml`       |       `--batch-size 64`       |
|   **C3TR**    |    深层嵌入Transformer Block     | `yolov5s-transformer.yaml` |       `--batch-size 64`       |
| **SWINStage** |       骨干末端嵌入Swin模块       |    `yolov5s-swin.yaml`     |       `--batch-size 64`       |
| **SWIN-Tiny** | 完整Swin-Tiny骨干（24×24大窗口） |     `yolov5s-ST.yaml`      | `--batch-size 16` ⚠️显存需求高 |

> ⚠️ **注**：SWIN-Tiny因计算量较大，需降低`batch-size`至16

## 5. 训练与推理

#### 5.1 环境配置

```bash
# 安装依赖
pip install -r requirements.txt  # 包含torch>=1.12, torchvision>=0.13
```

#### 5.2 **训练命令**

```bash
# 所有模型通用参数
python train.py \
  --data data.yaml \    # Cityscapes数据集配置
  --epochs 100 \        # 训练轮次
  --weights '' \        # 从零开始训练
  --batch-size <SIZE>   # 根据模型选择64或16

# 示例：训练SWIN-Tiny
python train.py --cfg yolov5s-ST.yaml --batch-size 16
```

#### 5.3 **推理演示**

```bash
python detect.py \
  --weights best.pt \          # 训练生成的权重
  --source video.mp4 \         # 输入视频/图像
  --conf-thres 0.5 \           # 置信度阈值
  --device 0                   # 使用GPU加速
```

## 6. 未来方向

1. **小目标优化**：在浅层（160×160分辨率）添加坐标注意力层

   ```yaml
   backbone:
     [-1, 1, CoordAttention, [192]]  # 添加在80×80特征层后
   ```

2. **3D融合感知**：结合LiDAR点云数据提升空间定位精度

3. **行为预警系统**：基于检测结果的越线分析与轨迹预测
