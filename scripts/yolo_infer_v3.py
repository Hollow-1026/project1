#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: yolo_infer_v3.py
# @Time    : 2025/7/6 09:37:36
# @Author  : 雨痕同学
# @Project : SafeVolo
# @Function: 集成之前所有的功能-添加美化功能
# @Desc    : 支持语音触发间隔,冷却间隔, 以及语音触发

import argparse
import cv2
from pathlib import Path
import sys

from scipy.fft import idstn

from beautify import calculate_beautify_params

yolo_server_root_path = Path(__file__).resolve().parent.parent
utils_path = yolo_server_root_path / "utils"
if str(yolo_server_root_path) not in sys.path:
    sys.path.insert(0,str(yolo_server_root_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1,str(utils_path))
from utils.paths import CHECKPOINTS_DIR
from utils.config_utils import load_yaml_config, merger_configs
from utils.paths import LOGS_DIR, CHECKPOINTS_DIR, CONFIGS_DIR, YOLOSERVER_ROOT
from utils.infer_frame import process_frame
def parser_args():  # 1 个用法
    parser = argparse.ArgumentParser(description="工地安全生产检测系统推理脚本")
    parser.add_argument(name_or_flags="--model", type=str,
                        default="train2-20250784-16165-45.pth", help="模型权重文件")
    parser.add_argument(name_or_flags="--source", type=str, default="./demo.mp4", help="推理图片")
    parser.add_argument(name_or_flags="--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument(name_or_flags="--save", type=bool, default=True, help="是否保存推理结果")
    parser.add_argument(name_or_flags="--show", type=bool, default=True, help="是否显示推理结果")
    parser.add_argument(name_or_flags="--save_txt", type=bool, default=True, help="是否保存推理结果txt文件")
    parser.add_argument(name_or_flags="--use_conf", type=bool, default=True, help="是否保存推理结果置信度")
    parser.add_argument(name_or_flags="--save_json", type=bool, default=False, help="是否保存推理结果json文件")
    parser.add_argument(name_or_flags="--save_frames", type=bool, default=False, help="是否保存推理结果帧")
    # 美化参数
    parser.add_argument(name_or_flags="--display_size", nargs=2, type=int, choices=[(480, 480), (720, 1280), (1440, 2560)], help="显示图片大小")
    parser.add_argument(name_or_flags="--beautify", type=bool, default=True, help="是否美化推理结果")
    parser.add_argument(name_or_flags="--font_size", type=int, default=32, help="字体大小")
    parser.add_argument(name_or_flags="--font_flags", type=int, default=4, help="字体样式")
    parser.add_argument(name_or_flags="--line_width", type=int, default=4, help="边框宽度")
    parser.add_argument(name_or_flags="--label_width", type=int, default=54, help="标签的宽度")
    parser.add_argument(name_or_flags="--label_padding_y", type=int, default=5, help="标签内边距Y")
    parser.add_argument(name_or_flags="--label_flags", type=int, default=4, help="标签的样式")
    parser.add_argument(name_or_flags="--radius", type=int, default=3, help="边框圆角半径")
    parser.add_argument(name_or_flags="--use_yaml", type=bool, default=True, help="是否使用yaml配置文件")
    return parser.parse_args()

def main():  # 1 个用法
    args = parser_args()
    # 1. 设置日志
    logger = setup_logging()
    base_path = LOGS_DIR,
    log_type="infer",
    model_name=args.model,
    log=False

    # 2. 打印设备信息
    yaml_config = {}
    if args.use_yaml:
        yaml_config = load_yaml_config(config_type="infer")
    yolo_args,project_args =merger_configs(args,yaml_config,mode="infer")
    # 3. 合并配置文件
    # 4. 计算分辨率
    resolution_map = {
        "480": (640, 360),
        "720": (640, 480),
        "1080": (1280, 720),
        "1440": (1920, 1080),
        "2160": (2560, 1440),
    }
    display_width, display_height = args.display_size

    # 5. 计算美化参数
    beautiful_params =calculate_beautify_params(
        current_image_height=display_height,
        current_image_width=display_width,
        base_font_size=args.font_size,
        base_line_width=args.line_width,
        base_label_padding_y=args.label_padding_y,
        base_radius=args.radius,
        ref_dim_for_scaling=720,
        font_path="utils/chinese/SimHei-Bold.ttf",
        text_color_bgr=(0, 0, 0),
        use_chinese_mapping=args.use_chinese_mapping,
        label_mapping=yaml_config["beautify_settings"]['label_mapping'],
        color_mapping=yaml_config["beautify_settings']['color_mapping"]
    )

    # 6. 打印一下参数来源信息
    # 7. 加载模型
    model = YOLO(CHECKPOINTS_DIR/args.model)
    logger.info(f"模型加载成功模型：{CHECKPOINTS_DIR/args.model}")
    source = args.source

    # 模型推理
    if source.isdigit() or source.endswith((".mp4", ".avi", ".mov", ".mkv")):
        # 初始化视频捕获
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            logger.error(f"无法打开视频源: {source}")
            raise RuntimeError(f"无法打开视频源: {source}")
        window_name = "YOLOv8 Inference"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 流式推理
        video_writer = None
        frames_dir = None
        save_dir = None
        yolo_args.stream = True
        yolo_args.show = False
        yolo_args.save = False

        print('YOLO参数:', yolo_args)
        for idx, result in enumerate(model.predict(**vars(yolo_args))):
            # 第一帧初始化保存路径
            if idx == 0:
                save_dir = YOLOSERVER_ROOT / Path(result.save_dir)
                logger.info(f"此次推理结果保存路径: {save_dir}")
                if args.save_frames:
                    frames_dir = save_dir / "0_frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"保存帧图像路径: {frames_dir}")
                if args.save:
                    video_path = save_dir / "output.mp4"
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (display_width, display_height),
                    )
                    logger.info(f"保存视频路径: {video_path}")
                    if video_writer:
                        logger.info(f"视频写入器创建成功")

            # 获取每一帧
            frame = result.orig_img

            # 针对每一帧进行美化
            annotated_frame = process_frame(frame, result, project_args, beautify_params)

            # 保存视频
            if video_writer:
                annotated_frame = cv2.resize(annotated_frame, (display_width, display_height))
                video_writer.write(annotated_frame)

            # 保存帧图像
            if frames_dir:
                cv2.imwrite(str(frames_dir / f"{idx}.png"), annotated_frame)

            # 显示
            if args.show:
                cv2.imshow(window_name, annotated_frame)

            # 退出机制
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        # 释放资源
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        logger.info(f"视频推理结束，已处理帧数: {idx}")
    else:
        logger.info(f"推理结束".center(50, '-'))
        yolo_args.stream = False
        results = model.predict(**vars(yolo_args))
        save_dir = Path(results[0].save_dir)
        base_save_dir = save_dir / "beautify"
        base_save_dir.mkdir(parents=True, exist_ok=True)

        for idx, result in enumerate(results):
            annotated_frame = process_frame(result.orig_img, result, project_args, beautify_params)
            if args.save:
                save_path = base_save_dir / f"{ids}.png"
                cv2.imwrite(str(save_path), annotated_frame)

        logger.info(f"图片推理结束，已处理图片数: {ids}")

    if __name__ == "__main__":
        main()