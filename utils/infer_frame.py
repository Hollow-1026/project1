#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @FileName :infer_frame.py
# @Time     :2025/7/8 09:28:09
# @Author   :南京同学
# @Project  :SafeYolo
# @Function :单帧推理

import cv2
from utils.beautify import custom_plot


def process_frame(frame, result, project_args, beautify_params, current_fps=None):
    """
    处理单帧图像，完成基本颜色和标注排序
    :param frame: 输入图像帧
    :param result: YOLO检测结果
    :param project_args: 项目参数
    :param beautify_params: 美化参数
    :param current_fps: 当前帧率（可选）
    :return: 标注后的图像帧
    """
    annotated_frame = frame.copy()
    original_height, original_width = frame.shape[:2]  # 修正原代码中的错误索引

    # 提取检测结果
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    labels_idx = result.boxes.cls.cpu().numpy().astype(int)
    labels = [result.names[int(cls_idx)] for cls_idx in labels_idx]

    # 选择美化或原始标注方式
    if project_args.beautify:
        annotated_frame = custom_plot(
            annotated_frame,
            boxes,
            confs,
            labels,
            **beautify_params  # 修正参数传递方式
        )
    else:
        annotated_frame = result.plot()

    # 添加FPS显示（如果提供了帧率）
    if current_fps is not None and current_fps > 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_color = (0, 255, 0)  # 绿色文字
        text_background_color = (0, 0, 0)  # 黑色背景

        # 计算文本尺寸
        text = f"FPS: {current_fps:.1f}"
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, font_thickness)

        # 添加半透明背景
        padding = 10
        box_x1 = original_width - text_width - padding * 2
        box_y1 = original_height - text_height - padding * 2
        box_x2 = original_width
        box_y2 = original_height

        cv2.rectangle(annotated_frame, (box_x1, box_y1), (box_x2, box_y2),
                      text_background_color, -1)

        # 添加文本
        text_x = original_width - text_width - padding
        text_y = original_height - padding
        cv2.putText(annotated_frame, text, (text_x, text_y),
                    font, font_scale, text_color, font_thickness)

    return annotated_frame


if __name__ == "__main__":
    # 运行时获取环境信息（调试用）
    import sys
    import os
    import platform

    print("\n环境信息")
    print(f"解释器路径: {sys.executable}")
    print(f"脚本路径: {os.path.abspath(__file__)}")
    print(f"操作系统: {platform.system()} {platform.release()}")