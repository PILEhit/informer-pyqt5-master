import argparse
import datetime
from pathlib import Path
import os
import shutil
import compileall

def main():
    parser = argparse.ArgumentParser()  # 获取项目的parameters
    parser.add_argument("src", type=Path)
    args = parser.parse_args()
    root = args.src  # 根目录
    for src_file in root.rglob("*.pyc"):  # 先删除根目录下的pyc文件
        os.remove(src_file)
    compileall.compile_dir(root, force=True)  # 将项目下的py都编译成pyc文件
    current_day = datetime.date.today()  # 当前日期
    edition = "1.1.2"  # 设置版本号
    dest = root.parent / f"{root.name}_{edition}.{'001'}_beta_{current_day}"  # 目标文件夹名称
    for src_file in root.glob("**/*.pyc"):  # 遍历所有pyc文件
        relative_path = src_file.relative_to(root)  # pyc文件对应模块文件夹名称
        dest_folder = dest / str(relative_path.parent.parent)  # 在目标文件夹下创建同名模块文件夹
        os.makedirs(dest_folder, exist_ok=True)
        dest_file = dest_folder / (src_file.stem.rsplit(".", 1)[0] + src_file.suffix)  # 创建同名文件
        print(f"install {relative_path}")
        shutil.copyfile(src_file, dest_file)  # 将pyc文件复制到同名文件

if __name__ == '__main__':
    main()