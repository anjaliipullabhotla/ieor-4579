from Agent import *
from generate_data import get_pr_rows
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import re
import json_and_jsonl_handler as jh
import time
import os
from dotenv import load_dotenv
from github import Github

load_dotenv()

GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")


def post_to_github(repo_name, pr_number, comment_body, token):
    """Posts the final_comment as a general comment on the Pull Request."""
    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        pr = repo.get_pull(int(pr_number))
        pr.create_issue_comment(comment_body)
        print(f"Successfully posted comment to {repo_name} PR #{pr_number}")
    except Exception as e:
        print(f"Failed to post to GitHub: {e}")


def run_agent_with_retry(agent, patch, progress=None, max_retries=100, function_calling_flag=True):
    retries = 0
    while retries <= max_retries:
        try:
            if hasattr(agent, 'using_function_calling'):
                agent.using_function_calling(function_calling_flag)
            result = agent.run_graph(patch)
            if progress:
                progress.update(1)
            return result
        except Exception as e:
            error_message = str(e)
            if "context_length_exceeded" in error_message:
                print(f"Critical error in {agent.__name__}: {error_message}")
                raise

            print(f"\nAgent {agent.__name__} failed with error: {e}")
            retries += 1
            if retries > max_retries:
                raise
            print(f"\nRetrying {agent.__name__}... Attempt {retries}")


def renumber_suggestions(text):
    lines = text.split('\n')
    suggestion_count = 1

    for i in range(len(lines)):
        # Check if the line starts with a number followed by a dot and a space
        if re.match(r'^\d+\.', lines[i]):
            # Replace the starting number with the suggestion_count
            lines[i] = re.sub(r'^\d+\.', f'{suggestion_count}.', lines[i], 1)
            suggestion_count += 1

    return '\n'.join(lines)


def get_review_comment(patch_with_additional_information):
    with tqdm(total=18, desc="Processing Agents") as pbar:
        with ThreadPoolExecutor(max_workers=18) as executor:
            futures = [
                executor.submit(run_agent_with_retry, code_semantic_correctness_agent,
                                patch_with_additional_information, progress=pbar),
                executor.submit(run_agent_with_retry, code_syntax_correctness_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, security_compliance_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, programming_handling_conventions_agent,
                                patch_with_additional_information, progress=pbar),
                executor.submit(run_agent_with_retry, identifier_naming_style_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, code_formatting_style_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, comment_style_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, identifier_naming_readability_agent,
                                patch_with_additional_information, progress=pbar),
                executor.submit(run_agent_with_retry, code_logic_readability_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, comment_quality_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, redundancy_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, compatibility_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, name_and_logic_consistency_agent,
                                patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, runtime_observability_agent,
                                patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, fault_tolerance_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, code_testing_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, extensibility_agent, patch_with_additional_information,
                                progress=pbar),
                executor.submit(run_agent_with_retry, performance_agent,
                                patch_with_additional_information, progress=pbar)
            ]

            results = [future.result() for future in futures]
            input_comment = patch_with_additional_information
            count = 1
            for result in results:
                input_comment += "{comment" + str(count) + "}:\n" + result + "\n"
                count += 1
            print(input_comment)
            print("-" * 80)
            summary_comment = run_agent_with_retry(summarizer_agent, input_comment, None)
            summarizer_clean_up_input = patch_with_additional_information + '\n{comments}:\n' + summary_comment
            clean_up_comment = run_agent_with_retry(summarizer_clean_up_agent, summarizer_clean_up_input, None)
            rerank_comment = run_agent_with_retry(suggestions_rerank_agent, clean_up_comment, None)
            final_comment = renumber_suggestions(rerank_comment)
            return input_comment, summary_comment, clean_up_comment, final_comment
        

def run_jsonl(filename):
    datas = jh.read_jsonl_file(filename)
    print(f'Data loaded successfully. {len(datas)} rows total.')
    times = []
    for i, data in enumerate(datas):
        if data.get('hydra_comment_reproduction'):
            print(f'Row {i}: already processed, skipping.')
            continue

        print(f'Row {i}: processing...')
        patch_with_additional_information = get_additional_information_agent.get_additional_information(data)
        start_time = time.time()
        _, _, _, final_comment = get_review_comment(
            patch_with_additional_information)
        end_time = time.time()
        print("-" * 80)
        duration = end_time - start_time
        times.append(duration)
        print(f"Generating comments took {duration:.2f} seconds.", flush=True)
        print(final_comment)

        datas[i]['hydra_comment_reproduction'] = final_comment
        with open(args.path, 'w', encoding='utf-8') as f:
            for item in datas:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f'Row {i}: saved.')
    print(f"Average duration to compute review comment: {sum(times) / len(times)}")


def run_pr(repo, pr_num):
    rows = get_pr_rows(repo, pr_num)
    if not rows:
        print("No Python files found in this PR.")
    else:
        all_final_comments = []
        for i, row in enumerate(rows):
            print(f"\n--- Processing file {i+1}/{len(rows)}: {row['path']} ---")
            patch_with_info = get_additional_information_agent.get_additional_information(row)
            _, _, _, final_comment = get_review_comment(patch_with_info)
            all_final_comments.append(f"### `{row['path']}`\n{final_comment}")
        combined_comment = "\n\n---\n\n".join(all_final_comments)
        print("\n" + "=" * 80)
        print(combined_comment)
        post_to_github(args.repo, args.number, combined_comment, GITHUB_API_TOKEN)




if __name__ == "__main__":
    # 1. Use`pip install -r requirements.txt` to install required libraries.
    # 2. Copy .env.example to .env and fill in OPENAI_API_KEY, OPENAI_API_BASE, and OPENAI_GPT_MODEL
    # 3. Specify the datas_path (jsonl)
    # 4. Run the function
    parser = argparse.ArgumentParser(description="Hydra Reviewer: Process JSONL or a GitHub PR.")
    
    # Mode Selection
    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")

    # JSONL Mode
    jsonl_parser = subparsers.add_parser("jsonl", help="Process a local JSONL file")
    jsonl_parser.add_argument("path", type=str, help="Path to .jsonl file")

    # PR Mode
    pr_parser = subparsers.add_parser("pr", help="Review a specific GitHub PR")
    pr_parser.add_argument("--repo", required=True, help="Repo name (e.g., 'owner/repo')")
    pr_parser.add_argument("--number", required=True, type=int, help="PR number")

    args = parser.parse_args()
    if args.mode == "jsonl":
        run_jsonl(args.path)
    elif args.mode == "pr":
        run_pr(args.repo, args.number)
    else:
        parser.print_help()

