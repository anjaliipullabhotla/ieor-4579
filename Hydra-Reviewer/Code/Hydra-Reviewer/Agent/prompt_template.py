def get_reflect(requirements):
    reflect_prompt = (f"You are a reviewer, and a user has commented on code changes based on the following "
                      f"requirements: '{requirements}' The user's comment is your input. Please evaluate whether the "
                      f"user's comment meets the requirements. If you believe the user's comment meets the "
                      f"requirements, only output the identifier '<Good>'. If you think the user's comment does not "
                      f"meet the requirements or the content is too vague, provide your suggestions for "
                      f"improvement.")
    reflect_prompt_example = '''
Example 1:
User Comment: *A user comment that meets the requirements*
Response: <Good>

Example 2:
User Comment: *A user comment that does not meet the requirements or is too superficial*
Response: *Your suggestions for improvement*'''
    reflect_prompt = reflect_prompt + reflect_prompt_example
    return reflect_prompt


def get_clean_up(requirements):
    clean_up_prompt = (f"You are a clean-up expert, and a user has commented on code changes based on the following "
                       f"requirements: '{requirements}' The user's comment is your input. Please clean up any content "
                       f"in the comments that is irrelevant to the requirements, approval of the code changes, "
                       f"statements about the implementation of the code, or suggestions that lack practical "
                       f"significance. Retain any suggestions you find meaningful along with their corresponding code "
                       f"snippets. Only output the cleaned comment and refrain from adding meaningless words.")
    return clean_up_prompt


def get_input_introduction(flag):
    if flag:
        input_introduction_prompt = ("The input is a patch and, optionally, additional information. The patch consists "
                                     "of all the diffs for a single file within a single code change. The additional "
                                     "information provides code snippets of certain functions or classes in the patch "
                                     "to help enhance your understanding of the patch.")
    else:
        input_introduction_prompt = ("The input is a patch, which consists of all the diffs for a single file within a "
                                     "single code change. ")
    return input_introduction_prompt


def get_function_calling_prompt(flag):
    if flag:
        function_calling_prompt = ("Please comment on the patch after fully understanding any additional information "
                                   "and the patch itself, and avoid commenting on the additional information. ")
    else:
        function_calling_prompt = ""
    return function_calling_prompt


def get_summarize_prompt(flag):
    if flag:
        summarize_prompt = ("You are a code reviewer. There are 18 other reviewers who have commented on the same "
                            "patch, which includes all the diffs for a single file within a single code change. Your "
                            "input is a patch, along with possible additional information, and comments or "
                            "suggestions regarding the patch. The additional information provides code snippets of "
                            "certain functions or classes in the patch to help enhance your understanding of the "
                            "patch. Your task is to critically review the patch and then output every specific "
                            "suggestion from these comments that you confirm to be correct and can genuinely improve "
                            "the quality of the code. Exclude any incorrect or redundant suggestions. Before "
                            "outputting, ensure to cross-check each suggestion against others to guarantee that no "
                            "duplicate suggestions with similar meanings are included. Check if the suggestions are "
                            "incorrect for code that is already correct, and ignore those suggestions if they are "
                            "erroneous. Only output correct specific suggestions and refrain from adding superfluous "
                            "words.")
    else:
        summarize_prompt = ("You are a code reviewer. There are 18 other reviewers who have commented on the same "
                            "patch, which includes all the diffs for a single file within a single code change. Your "
                            "input is the patch and these comments. Your task is to critically review the patch and "
                            "then output every specific suggestion from these comments that you confirm to be correct "
                            "and can genuinely improve the quality of the code. Exclude any incorrect or redundant "
                            "suggestions. Before outputting, ensure to cross-check each suggestion against others to "
                            "guarantee that no duplicate suggestions with similar meanings are included. Check if the "
                            "suggestions are incorrect for code that is already correct, and ignore those suggestions "
                            "if they are erroneous. Only output correct specific suggestions and refrain from adding "
                            "superfluous words.")
    return summarize_prompt


def get_clean_up_prompt(flag):
    if flag:
        clean_up_head = ("Your input consists of a patch, optionally additional information, and comments. The patch "
                         "is a collection of diffs for a file within a code change. The additional information "
                         "provides code snippets of certain functions or classes in the patch to help enhance your "
                         "understanding of the patch. The comments provide critiques and suggestions for the patch.")
    else:
        clean_up_head = ("Your input consists of a patch and comments. The patch is a collection of diffs for a file "
                         "within a code change, and the comments provide critiques and suggestions for the patch.")
    clean_up_body = (" Please first analyze and understand the patch code. Then strictly examine each suggestion based "
                     "on the patch and clean them up according to these rules:\n1.Remove a preventative suggestion if "
                     "you find it pointless. However, do not remove preventative suggestions related to "
                     "security.\n2.If a suggestion involves renaming or adding comments, and you find it unnecessary "
                     "or pointless—for example, if the readability of the identifier suggested for renaming is "
                     "already good, or adding or changing comments is unnecessary for the code snippet "
                     "referenced—then remove this suggestion.\n3.If two suggestions have a similar effect (i.e., "
                     "they are duplicate), keep only one suggestion and remove the other.\n4.If a suggestion is a "
                     "statement about the implementation of the code, remove this suggestion.\n5.If a suggestion is "
                     "too vague and not specific enough (e.g., does not provide clear, actionable steps, "
                     "lacks examples or details on how to implement the improvement, or does not specify which parts "
                     "of the code need changes), remove that suggestion.\n6.If a suggestion is incorrect (e.g., "
                     "leads to logical errors, misunderstands the code's purpose, or contains factual inaccuracies), "
                     "remove that suggestion.\nDo not change the content of the suggestions. Your output should be a "
                     "list of suggestions, and refrain from adding any words. Do not output which suggestions you "
                     "deleted, only output the suggestions you kept, and renumber the suggestions in order.")
    clean_up_example = ("\nCorrect Output Example:\n1.suggestion1\n2.suggestion2\n3.suggestion3\n4.suggestion4\n"
                        "5.suggestion5\n...")
    clean_up_prompt = clean_up_head + clean_up_body + clean_up_example
    return clean_up_prompt
