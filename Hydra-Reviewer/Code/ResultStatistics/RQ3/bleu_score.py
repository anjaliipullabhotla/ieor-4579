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


result_path = r'D:\Data\RQ3\ablation_dataset.jsonl'
result_datas = jh.read_jsonl_file(result_path)
print(len(result_datas))

count = 0
patch_gpt_bleu = 0
patch_variant1_bleu = 0
patch_variant2_bleu = 0
patch_variant3_bleu = 0
patch_variant4_bleu = 0
patch_chatgpt_bleu = 0
patch_deepseek_bleu = 0

for number, data in enumerate(result_datas):
    gpt_bleu = 0
    variant1_bleu = 0
    variant2_bleu = 0
    variant3_bleu = 0
    variant4_bleu = 0
    chatgpt_bleu = 0
    deepseek_bleu = 0

    print('number: ', number)
    # print("id: ", data['id'])
    ground_truth_comments = data['ground_truth_comments']
    gpt_comment = data['gpt_comment']
    gpt_comments = split_suggestions(gpt_comment)
    variant1_comment = data['variant1_comment']
    variant1_comments = split_suggestions(variant1_comment)
    variant2_comment = data['variant2_comment']
    variant2_comments = split_suggestions(variant2_comment)
    variant3_comment = data['variant3_comment']
    variant3_comments = split_suggestions(variant3_comment)
    variant4_comment = data['variant4_comment']
    variant4_comments = split_suggestions(variant4_comment)
    chatgpt_comment = data['end2end_comment']
    chatgpt_comments = split_suggestions(chatgpt_comment)
    deepseek_comment = data['deepseek_comment']
    deepseek_comments = split_suggestions(deepseek_comment)

    for ground_truth_comment in ground_truth_comments:
        gpt_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], gpt_comments)
        variant1_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], variant1_comments)
        variant2_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], variant2_comments)
        variant3_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], variant3_comments)
        variant4_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], variant4_comments)
        chatgpt_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], chatgpt_comments)
        deepseek_max_bleu = get_gpt_bleu(ground_truth_comment['comment'], deepseek_comments)

        gpt_bleu += gpt_max_bleu
        variant1_bleu += variant1_max_bleu
        variant2_bleu += variant2_max_bleu
        variant3_bleu += variant3_max_bleu
        variant4_bleu += variant4_max_bleu
        chatgpt_bleu += chatgpt_max_bleu
        deepseek_bleu += deepseek_max_bleu
        count += 1

    patch_gpt_bleu += (gpt_bleu / len(ground_truth_comments))
    patch_variant1_bleu += (variant1_bleu / len(ground_truth_comments))
    patch_variant2_bleu += (variant2_bleu / len(ground_truth_comments))
    patch_variant3_bleu += (variant3_bleu / len(ground_truth_comments))
    patch_variant4_bleu += (variant4_bleu / len(ground_truth_comments))
    patch_chatgpt_bleu += (chatgpt_bleu / len(ground_truth_comments))
    patch_deepseek_bleu += (deepseek_bleu / len(ground_truth_comments))

print("count: ", count)
print("bleu score:")
print("Hydra-Reviewer: ", patch_gpt_bleu / len(result_datas))
print('Variant1: ', patch_variant1_bleu / len(result_datas))
print('Variant2: ', patch_variant2_bleu / len(result_datas))
print('Variant3: ', patch_variant3_bleu / len(result_datas))
print('Variant4: ', patch_variant4_bleu / len(result_datas))
print('ChatGPT: ', patch_chatgpt_bleu / len(result_datas))
print('Deepseek-V3: ', patch_deepseek_bleu / len(result_datas))
