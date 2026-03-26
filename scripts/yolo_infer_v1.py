#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_infer_v1.py
# @Time      :2025/7/7 11:20:20
# @Author    :雨霓同学
# @Project   :SafeYolo
# @Function  :优化退出机制，支持显示大小设定
import argparse

import cv2
from ultralytics import YOLO
from pathlib import Path

from utils.paths import CHECKPOINTS_DIR

def parser_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference")
    parser.add_argument("--model", type=str,
        default=r"E:\PythonProject\SafeYolo\yolo_server\models\checkpoints\train14-20250705-145820-yolo11n-last.pt",
        help="模型权重文件")
    parser.add_argument("--source", type=str, default="1", help="推理图片")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU阈值")
    parser.add_argument("--save", type=bool, default=True, help="是否保存推理结果")
    parser.add_argument("--show", type=bool, default=True, help="是否显示推理结果")
    parser.add_argument("--save_txt", type=bool, default=False, help="是否保存推理结果txt文件")
    parser.add_argument("--save_conf", type=bool, default=False, help="是否保存推理结果置信度")
    parser.add_argument("--save_crop", type=bool, default=False, help="是否保存推理结果裁剪图片")
    parser.add_argument("--save_frames", type=bool, default=False, help="是否保存推理结果帧")
    parser.add_argument("--display_size", type=str,default="720",
                    choices=["360","480","720","1080","1440"], help="显示图片大小")

    return parser.parse_args()

def main():
    args = parser_args()

    resolution_map = {
        "360": (640, 360),
        "480": (640, 480),
        "720": (1280, 720),
        "1080": (1920, 1080),
        "1440": (2560, 1440),
    }
    display_width, display_height = resolution_map[args.display_size]

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = CHECKPOINTS_DIR / model_path
    source = args.source


    model = YOLO(args.model)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频源 {source}")

        window_name = "YOLOv8 Inference"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                save=args.save,
                show=False,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                save_crop=args.save_crop,
                save_frames=args.save_frames,
                project = "runs/predict",
                name = "exp"
            )
            annotated_frame = results[0].plot()

            annotated_frame = cv2.resize(annotated_frame, (display_width, display_height))

            cv2.imshow(window_name, annotated_frame)

            # 新的退出机制
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            frame_idx += 1
        cap.release()
        cv2.destroyAllWindows()
        print(f"摄像头推理结束，已处理帧数: {frame_idx}")
    else:
        results = model.predict(
            source=source,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            show=args.show,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            save_frames=args.save_frames,
        )

if __name__ == "__main__":
    main()
