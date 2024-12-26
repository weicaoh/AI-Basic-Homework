"""
Function：打印文件目录结构，与实现功能无关
"""
import os


def print_directory_structure(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 打印目录路径
        print(f"目录：{dirpath}")

        # 打印子目录
        if dirnames:
            print(f"  子目录: {', '.join(dirnames)}")

        # 打印文件
        if filenames:
            print(f"  文件: {', '.join(filenames)}")
        print("-" * 40)


# 调用函数并打印某个目录的结构
print_directory_structure('D:\\HuaweiMoveData\\Users\\10534\\Desktop\\AIF-homework')
