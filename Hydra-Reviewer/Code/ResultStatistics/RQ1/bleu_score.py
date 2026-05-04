from smooth_bleu import bleu_fromstr

import re
import json_and_jsonl_handler as jh


def split_suggestions(comment):
    # Remove the leading numbers and dots from each suggestion
    pattern = re.compile(r'(\d+\..*?)(?=\d+\.\s|$)', re.DOTALL)
    matches = pattern.findall(comment)

    comments = [re.sub(r'^\d+\.\s*', '', match).strip() for match in matches]
    return comments


def split_patch(patch_text):
    # Regular expression pattern to match hunk, ensuring it follows the format "@@ -number,number +number,number @@"
    pattern = r'(@@ -\d+,\d+ \+\d+,\d+ @@.*?(?=^@@ -\d+,\d+ \+\d+,\d+ @@|\Z))'
    hunks = re.findall(pattern, patch_text, re.DOTALL | re.MULTILINE)
    return hunks


def get_diff_num(old_hunk, tmp_patch):
    diffs = split_patch(tmp_patch)
    if len(diffs) == 0:
        diffs = [tmp_patch]
    old_hunk_lines = old_hunk.split('\n')
    line_number = len(old_hunk_lines)
    while line_number >= 1:
        last_lines = old_hunk_lines[-line_number:]
        splice_last_lines = ''
        for i, last_line in enumerate(last_lines):
            if i == 0:
                splice_last_lines += last_line
            else:
                splice_last_lines += '\n' + last_line

        if line_number > 1:
            for i, diff in enumerate(diffs):
                if splice_last_lines in diff:
                    return i
        elif line_number == 1:
            for i, diff in enumerate(diffs):
                diff_lines = diff.split('\n')
                if splice_last_lines in diff_lines:
                    return i
        line_number -= 1
    return 0


def get_acr_bleu(tmp_patch, gr_comment, preds):
    # Determine which diff block the comment is on and extract the corresponding acr_comment
    diff_num = get_diff_num(gr_comment['old_hunk'], tmp_patch)
    acr_comment = preds[diff_num]
    bleu = bleu_fromstr([acr_comment], [gr_comment['comment']], rmstop=False)
    return bleu


def get_gpt_bleu(comment, preds):
    bleus = []
    for pred in preds:
        bleu = bleu_fromstr([pred], [comment], rmstop=False)
        bleus.append(bleu)
    return max(bleus)


import os as _os
result_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../../Dataset/FinalDatasetPython.jsonl')
result_datas = jh.read_jsonl_file(result_path)

count = 0
patch_cr_bleu = 0
patch_lr_bleu = 0
patch_chatgpt_bleu = 0
patch_hydra_bleu = 0
patch_reproduction_bleu = 0
patch_readme_bleu = 0
patch_compat_only_bleu = 0
patch_naming_only_bleu = 0
for data in result_datas:
    cr_bleu = 0
    lr_bleu = 0
    chatgpt_bleu = 0
    hydra_bleu = 0
    reproduction_bleu = 0
    readme_bleu = 0
    compat_only_bleu = 0
    naming_only_bleu = 0

    print("id: ", data['id'])
    patch = data['patch']
    ground_truth_comments = data['ground_truth_comments']
    cr_comments = data['cr_comments']
    lr_comments = data['lr_comments']

    chatgpt_comment = data['chatgpt_comment']
    chatgpt_comments = split_suggestions(chatgpt_comment)

    hydra_comment = data['hydra_comment']
    hydra_comments = split_suggestions(hydra_comment)

    reproduction_comment = data.get('hydra_comment_reproduction', '')
    reproduction_comments = split_suggestions(reproduction_comment) if reproduction_comment else []

    readme_comment = data.get('hydra_comment_with_readme', '')
    readme_comments = split_suggestions(readme_comment) if readme_comment else []

    compat_only_comment = data.get('hydra_readme_compat_only', '')
    compat_only_comments = split_suggestions(compat_only_comment) if compat_only_comment else []

    naming_only_comment = data.get('hydra_readme_naming_only', '')
    naming_only_comments = split_suggestions(naming_only_comment) if naming_only_comment else []

    for ground_truth_comment in ground_truth_comments:
        cr_max_bleu = get_acr_bleu(patch, ground_truth_comment, cr_comments)
        lr_max_bleu = get_acr_bleu(patch, ground_truth_comment, lr_comments)
        chatgpt_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], chatgpt_comments)
        hydra_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], hydra_comments)
        reproduction_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], reproduction_comments) if reproduction_comments else 0
        readme_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], readme_comments) if readme_comments else 0
        compat_only_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], compat_only_comments) if compat_only_comments else 0
        naming_only_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], naming_only_comments) if naming_only_comments else 0

        cr_bleu += cr_max_bleu
        lr_bleu += lr_max_bleu
        chatgpt_bleu += chatgpt_max_bleu
        hydra_bleu += hydra_max_bleu
        reproduction_bleu += reproduction_max_bleu
        readme_bleu += readme_max_bleu
        compat_only_bleu += compat_only_max_bleu
        naming_only_bleu += naming_only_max_bleu
        count += 1

    patch_cr_bleu += (cr_bleu / len(ground_truth_comments))
    patch_lr_bleu += (lr_bleu / len(ground_truth_comments))
    patch_chatgpt_bleu += (chatgpt_bleu / len(ground_truth_comments))
    patch_hydra_bleu += (hydra_bleu / len(ground_truth_comments))
    patch_reproduction_bleu += (reproduction_bleu / len(ground_truth_comments))
    patch_readme_bleu += (readme_bleu / len(ground_truth_comments))
    patch_compat_only_bleu += (compat_only_bleu / len(ground_truth_comments))
    patch_naming_only_bleu += (naming_only_bleu / len(ground_truth_comments))

print("count: ", count)
print("bleu score patch:")
print("CodeReviewer: ", patch_cr_bleu / len(result_datas))
print("LLaMA-Reviewer: ", patch_lr_bleu / len(result_datas))
print("ChatGPT: ", patch_chatgpt_bleu / len(result_datas))
print("Hydra-Reviewer: ", patch_hydra_bleu / len(result_datas))
print("Hydra-Reproduction: ", patch_reproduction_bleu / len(result_datas))
print("Hydra-with-README: ", patch_readme_bleu / len(result_datas))
print("Hydra-README-compat-only: ", patch_compat_only_bleu / len(result_datas))
print("Hydra-README-naming-only: ", patch_naming_only_bleu / len(result_datas))
