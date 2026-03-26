#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2025/7/2 09:05:52
# @Author    :雨霓同学
# @Project   :SafeYolo
# @Function  :
if __name__ == "__main__":
    # 运行时获取实际路径信息
    import sys, os, platform

    print("\n===== 环境信息 =====")
    print(f"解释器路径: {sys.executable}")
    print(f"脚本路径: {os.path.abspath(__file__)}")
    print(f"操作系统: {platform.system()} {platform.release()}")

    run_code = 0
