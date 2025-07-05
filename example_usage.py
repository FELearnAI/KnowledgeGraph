#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径管理工具使用示例
演示如何在项目中使用新的路径管理方式
"""

# 方式1：只导入路径管理工具（推荐用于大多数情况）
from path_utils import setup_project_path

# 现在可以正常导入项目模块，无需手动设置 sys.path
try:
    from CasRel_RE.config import Config
    print("✅ 成功导入 CasRel_RE.config")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

# 方式2：使用特定的路径获取功能
from path_utils import (
    get_project_root,
    get_module_path,
    get_model_save_path,
    get_data_path
)

def demonstrate_path_utils():
    """
    演示路径管理工具的各种功能
    """
    print("\n=== 路径管理工具功能演示 ===")
    
    # 获取项目根目录
    project_root = get_project_root()
    print(f"项目根目录: {project_root}")
    
    # 获取各个模块的路径
    modules = ['CasRel_RE', 'LSTM_CRF', 'Bilstm_Attention_RE']
    for module in modules:
        module_path = get_module_path(module)
        print(f"{module} 模块路径: {module_path}")
        
        # 检查模块是否存在
        if module_path.exists():
            print(f"  ✅ {module} 模块存在")
        else:
            print(f"  ❌ {module} 模块不存在")
    
    # 获取模型保存路径
    print("\n--- 模型保存路径 ---")
    casrel_model_dir = get_model_save_path('CasRel_RE')
    casrel_model_file = get_model_save_path('CasRel_RE', 'best_f1.pth')
    print(f"CasRel_RE 模型目录: {casrel_model_dir}")
    print(f"CasRel_RE 模型文件: {casrel_model_file}")
    
    # 获取数据路径
    print("\n--- 数据路径 ---")
    casrel_data_dir = get_data_path('CasRel_RE')
    casrel_data_file = get_data_path('CasRel_RE', 'train.json')
    print(f"CasRel_RE 数据目录: {casrel_data_dir}")
    print(f"CasRel_RE 数据文件: {casrel_data_file}")

def old_vs_new_comparison():
    """
    对比旧方式和新方式的代码
    """
    print("\n=== 旧方式 vs 新方式对比 ===")
    
    print("\n【旧方式】每个文件都需要:")
    print("```python")
    print("import sys")
    print("import os")
    print("sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))")
    print("")
    print("# 获取模型路径")
    print("model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_model', 'best_f1.pth')")
    print("```")
    
    print("\n【新方式】简洁明了:")
    print("```python")
    print("from path_utils import get_model_save_path")
    print("")
    print("# 获取模型路径")
    print("model_path = get_model_save_path('CasRel_RE', 'best_f1.pth')")
    print("```")
    
    print("\n✅ 新方式的优势:")
    print("  - 代码更简洁")
    print("  - 易于理解和维护")
    print("  - 避免重复代码")
    print("  - 减少出错可能")
    print("  - 跨平台兼容")

if __name__ == '__main__':
    print("🚀 路径管理工具使用示例")
    
    # 演示路径管理功能
    demonstrate_path_utils()
    
    # 对比旧方式和新方式
    old_vs_new_comparison()
    
    print("\n✨ 示例运行完成！")
    print("\n💡 提示: 现在你可以在项目的任何地方使用这种方式来管理路径了！")