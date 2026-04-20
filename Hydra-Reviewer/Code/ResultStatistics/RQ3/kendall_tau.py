import numpy as np
import json_and_jsonl_handler as jh
from scipy.stats import kendalltau, ttest_rel

# 路径按你自己实际改
gpt_comments_path = r'D:\Data\RQ2\hydra_single_comments-dimension-judge.jsonl'
variant3_comments_path = r'D:\Data\RQ3\variant3_single_comments-dimension-judge.jsonl'
patches_path = r'D:\Data\RQ3\ablation_dataset.jsonl'

gpt_comments = jh.read_jsonl_file(gpt_comments_path)
variant3_comments = jh.read_jsonl_file(variant3_comments_path)
patches = jh.read_jsonl_file(patches_path)

priority = {
    'Fault Tolerance': 1,
    'Code Semantic Correctness': 2,
    'Compatibility': 3,
    'Performance': 4,
    'Security Compliance': 5,
    'Comment Quality': 6,
    'Runtime Observability': 7,
    'Identifier Naming Style': 8,
    'Code Formatting Style': 9,
}

# 可选：有问题的 patch id，跳过
inconformity_list = []

def get_kendall_tau(actual_list):
    expected_list = sorted(actual_list)
    actual_ranks = np.argsort(actual_list).argsort() + 1
    expected_ranks = np.argsort(expected_list).argsort() + 1
    tau, p = kendalltau(actual_ranks, expected_ranks)
    return tau, p

def get_patch_tau_list(comments):
    tau_list = []
    p_list = []
    for i, patch in enumerate(patches):
        if i == 384:
            break
        patch_id = patch['id']
        if patch_id in inconformity_list:
            continue
        dimension_list = []
        for comment in comments:
            if comment['id'] == patch_id:
                dimensions = comment['dimension'].split('、')
                max_dimension = 10
                for dimension in dimensions:
                    if dimension in priority:
                        value = priority[dimension]
                        if value < max_dimension:
                            max_dimension = value
                dimension_list.append(max_dimension)
        if len(dimension_list) > 0:
            tau, p = get_kendall_tau(dimension_list)
            tau_list.append(tau)
            p_list.append(p)
    return tau_list, p_list

hydra_tau_list, hydra_p_list = get_patch_tau_list(gpt_comments)
variant3_tau_list, variant3_p_list = get_patch_tau_list(variant3_comments)

# 平均 Kendall Tau
print(f"Hydra-Reviewer average Kendall Tau: {np.mean(hydra_tau_list):.4f}")
print(f"Variant3 average Kendall Tau: {np.mean(variant3_tau_list):.4f}")

# 统计有多少 patch 显著 (p < 0.05)
hydra_significant = sum(p < 0.05 for p in hydra_p_list)
variant3_significant = sum(p < 0.05 for p in variant3_p_list)
total = len(hydra_p_list)

print(f"Hydra-Reviewer significant patches (p < 0.05): {hydra_significant}/{total} ({hydra_significant/total:.1%})")
print(f"Variant3 significant patches (p < 0.05): {variant3_significant}/{total} ({variant3_significant/total:.1%})")

# 配对 t 检验，判断两个方法排序一致性差异是否显著
t_stat, p_val = ttest_rel(hydra_tau_list, variant3_tau_list)
print(f"Paired t-test p-value for Kendall Tau difference: {p_val:.4f}")

# 统计多少 patch 是 hydra 的 tau 更高
count_hydra_better = sum(h > v for h, v in zip(hydra_tau_list, variant3_tau_list))
print(f"Number of patches where Hydra-Reviewer has higher Kendall Tau than Variant3: {count_hydra_better}/{total}")
