import argparse
import logging
import sys
import pathlib
from pathlib import Path
from ultralytics import YOLO

# Add utils directory to sys.path
yolo_server_root_path = Path(__file__).resolve().parent.parent
utils_path = yolo_server_root_path / "utils"
if str(yolo_server_root_path) not in sys.path:
    sys.path.insert(0, str(yolo_server_root_path))
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

from utils.paths import CHECKPOINTS_DIR, LOGS_DIR, CONFIGS_DIR
from logging_utils import setup_logging
from performance_utils import time_it
from result_utils import log_results
from config_utils import load_yaml_config, log_parameters, merger_configs
from system_utils import log_device_info
from datainfo_utils import log_dataset_info

def name_or_flags(name):
    if "_" in name:
        return ["--" + name]
    else:
        return ["--" + name]

def parser_args():
    parser = argparse.ArgumentParser(description="YOLOv8 validation")
    parser.add_argument(*name_or_flags("data"), type=str, default="data.yaml", help="yaml配置文件")
    parser.add_argument(*name_or_flags("weights"), type=str, default="yolo11n.pt", help="模型权重文件")
    parser.add_argument(*name_or_flags("batch"), type=int, default=16, help="训练批次大小")
    parser.add_argument(*name_or_flags("batch_size"), type=int, default=16, help="训练批次大小")
    parser.add_argument(*name_or_flags("device_id"), type=int, default=8, help="训练设备ID")
    parser.add_argument(*name_or_flags("workers"), type=int, default=8, help="训练数据加载线程数")
    parser.add_argument(*name_or_flags("conf"), type=float, default=0.25, help="置信度阈值")
    parser.add_argument(*name_or_flags("iou"), type=float, default=0.45, help="IOU阈值")
    parser.add_argument(*name_or_flags("split"), type=str, default="test", choices=["val", "test"], help="数据集划分")
    parser.add_argument(*name_or_flags("use_yaml"), type=bool, default=True, help="是否使用yaml配置文件")
    return parser.parse_args()

def validate_model(model, yolo_args):
    results = model.val(**vars(yolo_args))
    return results

def main():
    args = parser_args()
    model_name = Path(args.weights).stem
    # 移除 log 参数
    logger = setup_logging(base_path=LOGS_DIR, log_type="val", model_name=model_name)
    logger.info("YOLO 工地安全生产检测模型验证程序启动")
    try:
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_yaml_config(config_type="val")
        # 合并参数
        yolo_args, project_args = merger_configs(args, yaml_config)

        # 记录设备信息
        log_device_info()
        # 记录参数信息
        log_parameters(project_args)
        # 记录数据集信息
        log_dataset_info(args.data, mode="test")

        # 检查数据集配置
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / args.data
        if not data_path.exists():
            logger.error(f"数据集配置文件不存在: {data_path}")
            raise FileNotFoundError(f"数据集配置文件不存在: {data_path}")
        logger.info(f"加载数据集配置文件: {data_path}")
        model = YOLO(args.weights)

        # 执行模型验证
        decorated_run_validation = time_it(iterations=1, name="模型验证", logger_instance=logger)(validate_model)
        results = decorated_run_validation(model, yolo_args)

    except Exception as e:
        logger.error(f"模型验证程序异常: {e}")

if __name__ == "__main__":
    # 运行时获取实际路径信息
    import sys, os, platform

    print("\n===== 环境信息 =====")
    print(f"解释器路径: {sys.executable}")
    print(f"脚本路径: {os.path.abspath(__file__)}")
    print(f"操作系统: {platform.system()} {platform.release()}")

    run_code = 0
    main()