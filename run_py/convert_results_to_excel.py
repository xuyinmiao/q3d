import os
import re
import argparse
import pandas as pd

# ===================== 配置区域 =====================
# 默认文件路径配置（可根据需要修改）
# 支持的数据集: model_history_mnist, model_history_fmnist, model_history_cifar10, model_history_cifar100
# 脚本会自动检测数据集类型并使用相应的列名 (MNIST: CNN/QCNN, CIFAR: WideResNet/QWideResNet)
DEFAULT_PATHS = {
    'pbs': {
        'input': 'model_history_cifar100/robustness_vs_pbs_results.txt',
        'output': 'model_history_cifar100/robustness_vs_pbs_results.xlsx',
    },
    'it': {
        'input': 'model_history_cifar100/robustness_vs_iterations_results.txt',
        'output': 'model_history_cifar100/robustness_vs_iterations_results.xlsx',
    }
}

# 是否过滤零值（True: 只保留非零数据，False: 保留所有数据）
FILTER_ZERO = True
# ================================================


def parse_results_txt(path, mode='pbs'):
    """
    解析结果文本文件
    
    Args:
        path: 文件路径
        mode: 'pbs' (Perturbation Budget) 或 'it' (Iterations)
    
    Returns:
        list of dict: 解析后的数据行
    """
    rows = []
    with open(path, 'r') as f:
        lines = f.readlines()
    
    attack = None
    in_section = False
    
    # 根据模式确定列名
    if mode == 'pbs':
        param_key = 'Epsilon'
        header_keyword = 'Epsilon'
    elif mode == 'it':
        param_key = 'Iterations'
        header_keyword = 'Iterations'
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'pbs' or 'it'.")
    
    for line in lines:
        s = line.rstrip('\n')
        if s.endswith('Attack Results:'):
            attack = s.replace(' Attack Results:', '').strip()
            in_section = True
            continue
        
        if in_section:
            if not s.strip():
                in_section = False
                continue
            if set(s.strip()) == set('-'):
                continue
            if s.strip().startswith(header_keyword):
                continue
            
            parts = s.split()
            if len(parts) >= 3:
                try:
                    param_value = float(parts[0])
                    hybrid = float(parts[1])
                    cnn = float(parts[2])
                    
                    # 根据配置过滤零值
                    if FILTER_ZERO and param_value == 0:
                        continue
                    
                    rows.append({
                        'Attack': attack,
                        param_key: param_value,
                        'Classical': cnn,
                        'Hybrid': hybrid,
                    })
                except ValueError:
                    continue
    
    return rows, param_key


def detect_dataset_type(path):
    """
    从路径中检测数据集类型
    
    Args:
        path: 文件路径
    
    Returns:
        str: 'mnist' 或 'cifar'
    """
    path_lower = path.lower()
    if 'mnist' in path_lower or 'fmnist' in path_lower:
        return 'mnist'
    elif 'cifar' in path_lower:
        return 'cifar'
    else:
        # 默认使用 cifar 格式
        return 'cifar'


def main():
    parser = argparse.ArgumentParser(
        description='Convert robustness evaluation results to Excel format.'
    )
    parser.add_argument('-i', '--input', default=None, help='Input txt file path')
    parser.add_argument('-o', '--output', default=None, help='Output xlsx file path')
    parser.add_argument('-m', '--mode', default='pbs', choices=['pbs', 'it'],
                        help="Mode: 'pbs' for Perturbation Budget, 'it' for Iterations")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 根据模式确定默认路径
    if args.input:
        in_path = args.input
    else:
        default_input = DEFAULT_PATHS[args.mode]['input']
        in_path = os.path.join(project_root, default_input)
    
    if args.output:
        out_path = args.output
    else:
        default_output = DEFAULT_PATHS[args.mode]['output']
        out_path = os.path.join(project_root, default_output)
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

    # 解析文件
    rows, param_key = parse_results_txt(in_path, mode=args.mode)
    
    if not rows:
        print(f'⚠️  Warning: No valid data rows parsed from {in_path}')
        if FILTER_ZERO:
            print(f'   Note: FILTER_ZERO is enabled, only non-zero {param_key} values are included.')
        raise SystemExit('No rows parsed from input file.')

    # 检测数据集类型
    dataset_type = detect_dataset_type(in_path)
    
    # 构建 DataFrame
    df = pd.DataFrame(rows)
    
    # 根据数据集类型重命名列
    if dataset_type == 'mnist':
        classical_col = 'CNN'
        hybrid_col = 'QCNN'
    else:  # cifar
        classical_col = 'WideResNet'
        hybrid_col = 'QWideResNet'
    
    df = df.rename(columns={'Classical': classical_col, 'Hybrid': hybrid_col})
    
    # 根据模式创建标签
    if args.mode == 'pbs':
        df['Label'] = df.apply(lambda r: f"{r['Attack']} (ε={float(r['Epsilon']):.2f})", axis=1)
        sort_col = 'Epsilon'
    else:  # mode == 'it'
        df['Label'] = df.apply(lambda r: f"{r['Attack']} (iter={int(r['Iterations'])})", axis=1)
        sort_col = 'Iterations'
    
    df['__order'] = range(len(df))
    
    # 保留原始攻击顺序
    attack_order = df['Attack'].drop_duplicates()
    df['AttackCat'] = pd.Categorical(df['Attack'], categories=list(attack_order), ordered=True)
    df = df.sort_values(['AttackCat', sort_col, '__order']).drop(columns=['AttackCat'])
    
    # 生成最终结果
    result = df[['Label', classical_col, hybrid_col]].set_index('Label')

    # 写入 Excel
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        result.to_excel(writer, sheet_name='results', index_label='Attack')

    print(f'✅ Successfully converted {args.mode.upper()} results to Excel')
    print(f'   Dataset: {dataset_type.upper()} (Columns: {classical_col}, {hybrid_col})')
    print(f'   Input:  {in_path}')
    print(f'   Output: {out_path}')
    print(f'   Rows:   {len(rows)} (FILTER_ZERO={FILTER_ZERO})')


if __name__ == '__main__':
    main()
