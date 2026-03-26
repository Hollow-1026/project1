#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_yolo.py
# @Time      :2025/6/30 14:11:40
# @Author    :雨霓同学
# @Project   :SafeYolo
# @Function  :
if __name__ == "__main__":
    from ultralytics import YOLO
    # 第一阶段模型训练
    # model = YOLO("yolov8n.pt")  # 实例化一个模型，模型不存在时会自动下载
    # results = model.train(data="coco8.yaml", epochs=2) # 训练模型

    # 第二阶段模型推理
    model = YOLO("runs/detect/train/weights/best.pt")
    results = model.predict(source="0", show=True)
