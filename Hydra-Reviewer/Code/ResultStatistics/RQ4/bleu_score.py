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


result_path = r'D:\Data\RQ4\unseen_dataset.jsonl'
result_datas = jh.read_jsonl_file(result_path)

count = 0
patch_cr_bleu = 0
patch_lr_bleu = 0
patch_end2end_gpt_bleu = 0
patch_comprehensive_chatgpt_bleu = 0
patch_gpt_bleu = 0
patch_deepseek_bleu = 0
for data in result_datas:
    cr_bleu = 0
    lr_bleu = 0
    end2end_gpt_bleu = 0
    comprehensive_chatgpt_bleu = 0
    gpt_bleu = 0
    deepseek_bleu = 0

    print("id: ", data['id'])
    patch = data['patch']
    ground_truth_comments = data['ground_truth_comments']
    cr_comments = data['cr_comments']
    lr_comments = data['lr_comments']
    end2end_gpt_comment = data['chatgpt_comment']
    end2end_gpt_comments = split_suggestions(end2end_gpt_comment)
    comprehensive_chatgpt_comment = data['comprehensive_gpt_comment']
    comprehensive_chatgpt_comments = split_suggestions(comprehensive_chatgpt_comment)
    gpt_comment = data['hydra_comment']
    gpt_comments = split_suggestions(gpt_comment)
    deepseek_comment = data['deepseek_comment']
    deepseek_comments = split_suggestions(deepseek_comment)

    for ground_truth_comment in ground_truth_comments:
        cr_max_bleu = get_acr_bleu(patch, ground_truth_comment, cr_comments)
        lr_max_bleu = get_acr_bleu(patch, ground_truth_comment, lr_comments)
        end2end_gpt_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], end2end_gpt_comments)
        comprehensive_chatgpt_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], comprehensive_chatgpt_comments)
        gpt_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], gpt_comments)
        deepseek_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], deepseek_comments)

        cr_bleu += cr_max_bleu
        lr_bleu += lr_max_bleu
        end2end_gpt_bleu += end2end_gpt_max_bleu
        comprehensive_chatgpt_bleu += comprehensive_chatgpt_max_bleu
        gpt_bleu += gpt_max_bleu
        deepseek_bleu += deepseek_max_bleu
        count += 1

    patch_cr_bleu += (cr_bleu / len(ground_truth_comments))
    patch_lr_bleu += (lr_bleu / len(ground_truth_comments))
    patch_end2end_gpt_bleu += (end2end_gpt_bleu / len(ground_truth_comments))
    patch_comprehensive_chatgpt_bleu += (comprehensive_chatgpt_bleu / len(ground_truth_comments))
    patch_gpt_bleu += (gpt_bleu / len(ground_truth_comments))
    patch_deepseek_bleu += (deepseek_bleu / len(ground_truth_comments))

print("Comment Count: ", count)
print("bleu score patch:")
print("CodeReviewer: ", patch_cr_bleu / len(result_datas))
print("LLaMA-Reviewer: ", patch_lr_bleu / len(result_datas))
print("ChatGPT: ", patch_end2end_gpt_bleu / len(result_datas))
print("Comprehensive ChatGPT: ", patch_comprehensive_chatgpt_bleu / len(result_datas))
print("Hydra-Reviewer: ", patch_gpt_bleu / len(result_datas))
print("DeepSeek: ", patch_deepseek_bleu / len(result_datas))
