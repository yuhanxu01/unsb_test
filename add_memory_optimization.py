#!/usr/bin/env python3
"""
批量为 OT_input 实验脚本添加内存优化参数
"""

import os
import re

# 需要优化的实验列表
EXPERIMENTS_TO_OPTIMIZE = [
    'exp1_fully_pair_OT_input.sh',
    'exp2_fully_pair_OT_input_E.sh',
    'exp5_twostage_10p_OT_input.sh',
    'exp6_twostage_10p_OT_input_E.sh',
    'exp9_twostage_100p_OT_input.sh',
    'exp10_twostage_100p_OT_input_E.sh'
]

# 内存优化配置
MEMORY_OPTIMIZATION_BLOCK = """
# Memory optimization (极致优化方案 - 推荐)
# 使用梯度累积 + 选择性梯度 + 混合精度
# 预期内存: 14+ GB → 1-2 GB (90-95% 减少)
export USE_GRADIENT_ACCUMULATION="--use_gradient_accumulation"
export SELECTIVE_GRADIENT_STEPS="--selective_gradient_steps 3"
export USE_MIXED_PRECISION="--use_mixed_precision"
# export USE_GRADIENT_CHECKPOINTING="--use_gradient_checkpointing"  # 可选：如果需要进一步优化
"""

MEMORY_INFO_BLOCK = """
echo "Memory Optimizations:"
echo "  Gradient Accumulation: ENABLED (lowest memory peak)"
echo "  Selective Gradient: Last 3 steps only"
echo "  Mixed Precision: FP16 enabled"
echo "  Expected Memory: 1-2 GB (vs 14+ GB without optimization)"
echo ""
"""


def update_experiment_script(filepath):
    """更新单个实验脚本"""
    if not os.path.exists(filepath):
        print(f"⚠️  文件不存在: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # 检查是否已经添加了优化
    if 'USE_GRADIENT_ACCUMULATION' in content:
        print(f"✓  已优化: {os.path.basename(filepath)}")
        return True

    # 在 Loss configuration 后添加内存优化
    pattern = r'(export DISABLE_NCE="--disable_nce"\n)'
    replacement = r'\1' + MEMORY_OPTIMIZATION_BLOCK

    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
    else:
        print(f"⚠️  无法找到插入点: {filepath}")
        return False

    # 在配置输出后添加内存优化信息
    pattern = r'(echo "  Epochs: 1-.*\n)(echo ""\n)'
    replacement = r'\1' + MEMORY_INFO_BLOCK + r'\2'

    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)

    # 保存更新后的文件
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"✓  已更新: {os.path.basename(filepath)}")
    return True


def main():
    base_dir = '/home/user/unsb_test/experiments/ablation_studies'

    print("=" * 60)
    print("批量添加内存优化参数")
    print("=" * 60)
    print()

    success_count = 0
    total_count = len(EXPERIMENTS_TO_OPTIMIZE)

    for exp_file in EXPERIMENTS_TO_OPTIMIZE:
        filepath = os.path.join(base_dir, exp_file)
        if update_experiment_script(filepath):
            success_count += 1

    print()
    print("=" * 60)
    print(f"完成: {success_count}/{total_count} 个实验已优化")
    print("=" * 60)
    print()
    print("优化参数:")
    print("  ✓ 梯度累积 (--use_gradient_accumulation)")
    print("  ✓ 选择性梯度 (--selective_gradient_steps 3)")
    print("  ✓ 混合精度 (--use_mixed_precision)")
    print()
    print("预期效果:")
    print("  内存: 14+ GB → 1-2 GB (90-95% 减少)")
    print("  速度: 轻微影响 (~5-15% 变慢)")
    print()


if __name__ == '__main__':
    main()
