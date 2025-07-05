# 项目路径管理指南

## 问题描述

之前项目中多个文件都使用了相同的代码来设置 Python 路径：
```python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

这种方式存在以下问题：
1. **代码重复**：每个文件都需要重复相同的路径设置代码
2. **维护困难**：如果项目结构发生变化，需要修改多个文件
3. **可读性差**：路径计算逻辑复杂，不易理解
4. **容易出错**：手动计算相对路径容易出现错误

## 解决方案

### 1. 创建统一的路径管理模块

我们创建了 `path_utils.py` 模块来统一管理项目路径：

```python
from path_utils import get_project_root, get_model_save_path, get_data_path
```

### 2. 主要功能

- **自动路径设置**：导入 `path_utils` 时自动将项目根目录添加到 `sys.path`
- **路径获取工具**：提供便捷的函数获取各种路径
- **跨平台兼容**：使用 `pathlib.Path` 确保跨平台兼容性

### 3. 使用方法

#### 基本导入（推荐）
```python
# 只需要导入路径管理工具，自动设置项目路径
from path_utils import setup_project_path

# 然后正常导入项目模块
from CasRel_RE.model.CasrelModel import CasRel
from CasRel_RE.config import Config
```

#### 获取特定路径
```python
from path_utils import get_model_save_path, get_data_path, get_project_root

# 获取模型保存路径
model_path = get_model_save_path('CasRel_RE', 'best_f1.pth')

# 获取数据路径
data_path = get_data_path('CasRel_RE', 'train.json')

# 获取项目根目录
project_root = get_project_root()
```

## 迁移步骤

### 对于现有文件：

1. **删除旧的路径设置代码**：
   ```python
   # 删除这些行
   import sys
   import os
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   ```

2. **添加新的导入**：
   ```python
   # 添加这行（自动设置路径）
   from path_utils import setup_project_path
   
   # 或者如果需要特定路径功能
   from path_utils import get_model_save_path, get_data_path
   ```

3. **更新路径使用**：
   ```python
   # 旧方式
   model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_model', 'best_f1.pth')
   
   # 新方式
   model_path = get_model_save_path('CasRel_RE', 'best_f1.pth')
   ```

## 最佳实践

### 1. 项目入口点
在主要的入口文件（如 `train.py`, `predict.py`）中导入路径管理工具：
```python
from path_utils import setup_project_path  # 自动设置项目路径
```

### 2. 模块内部文件
在模块内部的文件中，只需要导入路径管理工具即可：
```python
from path_utils import get_model_save_path  # 根据需要导入特定功能
```

### 3. 配置文件
在配置文件中使用路径管理工具：
```python
from path_utils import get_data_path, get_model_save_path

class Config:
    def __init__(self):
        self.data_path = get_data_path('CasRel_RE')
        self.model_save_path = get_model_save_path('CasRel_RE')
```

## 优势

1. **代码简洁**：消除重复的路径设置代码
2. **易于维护**：路径逻辑集中管理
3. **可读性强**：函数名清晰表达意图
4. **灵活性高**：支持不同模块的路径需求
5. **错误减少**：避免手动计算路径错误
6. **跨平台**：使用 `pathlib` 确保兼容性

## 注意事项

1. **导入顺序**：确保在导入项目模块之前导入 `path_utils`
2. **项目结构**：如果项目结构发生重大变化，只需要更新 `path_utils.py`
3. **向后兼容**：现有代码可以逐步迁移，新旧方式可以共存

## 扩展功能

`path_utils.py` 模块可以根据需要扩展更多功能：
- 配置文件路径管理
- 日志文件路径管理
- 临时文件路径管理
- 资源文件路径管理

通过这种方式，项目的路径管理变得更加规范和易于维护。