#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径管理工具模块
用于统一管理项目中的路径问题，避免在每个文件中重复设置 sys.path.append
"""

import os
import sys
from pathlib import Path


def setup_project_path():
    """
    设置项目根路径到 sys.path 中
    只需要在项目入口点调用一次即可
    """
    # 获取项目根目录（当前文件所在目录）
    project_root = Path(__file__).parent.absolute()
    
    # 将项目根目录添加到 Python 路径中
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root


def get_project_root():
    """
    获取项目根目录路径
    """
    return Path(__file__).parent.absolute()


def get_module_path(module_name):
    """
    获取指定模块的路径
    
    Args:
        module_name: 模块名称，如 'CasRel_RE', 'LSTM_CRF' 等
    
    Returns:
        模块的绝对路径
    """
    project_root = get_project_root()
    return project_root / module_name


def get_data_path(module_name, data_file=None):
    """
    获取数据文件路径
    
    Args:
        module_name: 模块名称
        data_file: 数据文件名（可选）
    
    Returns:
        数据路径
    """
    module_path = get_module_path(module_name)
    data_path = module_path / 'data'
    
    if data_file:
        return data_path / data_file
    return data_path


def get_model_save_path(module_name, model_file=None):
    """
    获取模型保存路径
    
    Args:
        module_name: 模块名称
        model_file: 模型文件名（可选）
    
    Returns:
        模型保存路径
    """
    module_path = get_module_path(module_name)
    
    # 尝试不同的保存目录名称
    for save_dir in ['save_model', 'save', 'models']:
        save_path = module_path / save_dir
        if save_path.exists() or save_dir == 'save_model':  # 默认使用 save_model
            if model_file:
                return save_path / model_file
            return save_path
    
    # 如果都不存在，返回默认的 save_model
    save_path = module_path / 'save_model'
    if model_file:
        return save_path / model_file
    return save_path


# 自动设置项目路径（当模块被导入时）
setup_project_path()