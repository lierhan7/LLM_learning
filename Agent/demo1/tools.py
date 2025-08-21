import os
import shutil
from typing import List, Optional


def list_files(directory: str) -> List[str]:
    """
    列出指定目录中的所有文件
    
    Args:
        directory (str): 目录路径
        
    Returns:
        List[str]: 文件列表
        
    Raises:
        FileNotFoundError: 如果目录不存在
        PermissionError: 如果没有访问权限
    """
    print(f"列出目录: {directory}")

    try:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        if not os.path.isdir(directory):
            raise ValueError(f"路径不是一个目录: {directory}")
        
        # 获取目录中的所有项目
        items = os.listdir(directory)
        
        # 过滤出文件（不包括子目录）
        files = []
        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                files.append(item)
        
        return sorted(files)  # 返回排序后的文件列表
        
    except PermissionError as e:
        raise PermissionError(f"没有访问权限: {directory}") from e
    except Exception as e:
        raise Exception(f"列出文件时发生错误: {str(e)}") from e


def read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    读取指定文件的内容
    
    Args:
        file_path (str): 文件路径
        encoding (str): 文件编码，默认为 utf-8
        
    Returns:
        str: 文件内容
        
    Raises:
        FileNotFoundError: 如果文件不存在
        PermissionError: 如果没有读取权限
        UnicodeDecodeError: 如果编码错误
    """
    print(f"读取文件: {file_path}，编码: {encoding}")

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if not os.path.isfile(file_path):
            raise ValueError(f"路径不是一个文件: {file_path}")
        
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
        
        return content
        
    except PermissionError as e:
        raise PermissionError(f"没有读取权限: {file_path}") from e
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end,
            f"无法使用 {encoding} 编码解码文件: {file_path}"
        ) from e
    except Exception as e:
        raise Exception(f"读取文件时发生错误: {str(e)}") from e


def rename_file(old_path: str, new_path: str, overwrite: bool = False) -> bool:
    """
    重命名指定文件
    
    Args:
        old_path (str): 原文件路径
        new_path (str): 新文件路径
        overwrite (bool): 是否覆盖已存在的文件，默认为 False
        
    Returns:
        bool: 重命名是否成功
        
    Raises:
        FileNotFoundError: 如果原文件不存在
        FileExistsError: 如果目标文件已存在且不允许覆盖
        PermissionError: 如果没有操作权限
    """
    print(f"重命名文件: {old_path} -> {new_path}, 允许覆盖: {overwrite}")

    try:
        if not os.path.exists(old_path):
            raise FileNotFoundError(f"原文件不存在: {old_path}")
        
        if not os.path.isfile(old_path):
            raise ValueError(f"原路径不是一个文件: {old_path}")
        
        # 检查目标文件是否已存在
        if os.path.exists(new_path):
            if not overwrite:
                raise FileExistsError(f"目标文件已存在: {new_path}")
            else:
                # 如果允许覆盖，先删除目标文件
                os.remove(new_path)
        
        # 确保目标目录存在
        new_dir = os.path.dirname(new_path)
        if new_dir and not os.path.exists(new_dir):
            os.makedirs(new_dir, exist_ok=True)
        
        # 重命名文件
        shutil.move(old_path, new_path)
        
        return True
        
    except PermissionError as e:
        raise PermissionError(f"没有操作权限: {str(e)}") from e
    except Exception as e:
        raise Exception(f"重命名文件时发生错误: {str(e)}") from e


# 使用示例
if __name__ == "__main__":
    # 示例1: 列出当前目录的文件
    try:
        current_dir = "."
        files = list_files(current_dir)
        print(f"当前目录的文件: {files}")
    except Exception as e:
        print(f"列出文件失败: {e}")
    
    # 示例2: 读取文件内容（如果存在）
    try:
        if "tools.py" in os.listdir("."):
            content = read_file("tools.py")
            print(f"文件内容长度: {len(content)} 字符")
    except Exception as e:
        print(f"读取文件失败: {e}")
    
    # 示例3: 重命名文件（示例，实际使用时请谨慎）
    # try:
    #     success = rename_file("old_file.txt", "new_file.txt")
    #     print(f"重命名成功: {success}")
    # except Exception as e:
    #     print(f"重命名失败: {e}")

