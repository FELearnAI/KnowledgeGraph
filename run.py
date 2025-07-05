import argparse
import sys

# 动态导入并执行指定模型的训练函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model training for various models.")
    parser.add_argument(
        'model_name',
        type=str,
        help="The name of the model to train (e.g., 'CasRel_RE', 'Bilstm_Attention_RE')."
    )
    args = parser.parse_args()

    try:
        # 构建模块路径并导入
        # 例如，对于 'CasRel_RE'，将导入 CasRel_RE.train
        train_module = __import__(f"{args.model_name}.train", fromlist=['model2train'])
        
        # 假设训练函数名为 model2train 或 train
        if hasattr(train_module, 'model2train'):
            train_func = train_module.model2train
        elif hasattr(train_module, 'train'):
            train_func = train_module.train
        else:
            print(f"Error: Could not find a 'model2train' or 'train' function in {args.model_name}.train.")
            sys.exit(1)

        print(f"--- Starting training for {args.model_name} ---")
        train_func()
        print(f"--- Finished training for {args.model_name} ---")

    except ImportError:
        print(f"Error: Could not import training module for '{args.model_name}'.")
        print("Please ensure the model name is correct and the corresponding directory and train.py exist.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)