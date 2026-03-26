#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_infer_v0.py
# @Time      :2025/7/7 10:39:54
# @Author    :雨霓同学
# @Project   :SafeYolo
# @Function  :模型推理

import argparse
from ultralytics import YOLO

def parser_args():
    parser = argparse.ArgumentParser(description="YOLO11n Inference")
    parser.add_argument("--model", type=str,
        default=r"E:\PythonProject\SafeYolo\yolo_server\models\checkpoints\train14-20250705-145820-yolo11n-last.pt",
        help="模型权重文件")
    parser.add_argument("--source", type=str, default="0", help="推理图片")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU阈值")
    return parser.parse_args()

def main():
    args = parser_args()
    model = YOLO(args.model)
    model.predict(args.source, conf=args.conf, iou=args.iou,show=True)


if __name__ == "__main__":
    main()
