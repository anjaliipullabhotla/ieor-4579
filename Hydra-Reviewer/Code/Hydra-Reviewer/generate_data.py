import requests
import base64
import json
from dotenv import load_dotenv
import os
import argparse
from collections import defaultdict

load_dotenv()

GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")
REPO = "dmlc/dgl"
PR_NUMBER = 1217
HEADERS = {"Authorization": f"token {GITHUB_API_TOKEN}"}
FILENAME = "GeneratedDatasetPython.jsonl"

PYTHON_EXTENSION = "py"
LANGUAGE = "python"

def get_json(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def get_file_at_sha(repo, path, sha, headers):
    try:
        url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={sha}"
        data = get_json(url, headers)
        return base64.b64decode(data['content']).decode('utf-8')
    except: return ""


def get_pr_rows(repo, pr_num):
    """Fetch PR data from GitHub and return rows as a list of dicts (in-memory)."""
    headers = {"Authorization": f"token {GITHUB_API_TOKEN}"}

    files_data = get_json(f"https://api.github.com/repos/{repo}/pulls/{pr_num}/files", headers)
    comments_data = get_json(f"https://api.github.com/repos/{repo}/pulls/{pr_num}/comments", headers)

    grouped_comments = defaultdict(list)
    for c in comments_data:
        grouped_comments[c['path']].append(c)

    pr_data = get_json(f"https://api.github.com/repos/{repo}/pulls/{pr_num}", headers)
    base_sha = pr_data['base']['sha']
    head_sha = pr_data['head']['sha']

    rows = []
    for file_info in files_data:
        filename = file_info['filename']
        ext = filename.split('.')[-1] if '.' in filename else ''
        if ext != PYTHON_EXTENSION:
            print(f"Skipping {filename}, could not process non-Python file!")
            continue

        patch = file_info.get('patch', '')
        prev_file = get_file_at_sha(repo, filename, base_sha, headers)
        curr_file = get_file_at_sha(repo, filename, head_sha, headers)

        comments_for_file = grouped_comments[filename]
        comments_for_file.sort(key=lambda x: x['created_at'])
        first_comment = comments_for_file[0] if comments_for_file else None

        rows.append({
            "id": f"{pr_num}_{filename}",
            "repo": repo,
            "pr_number": int(pr_num),
            "comment_id": first_comment['id'] if first_comment else None,
            "created_at": first_comment['created_at'] if first_comment else None,
            "path": filename,
            "lang": LANGUAGE,
            "status": file_info['status'],
            "patch": patch,
            "old_hunk": first_comment.get('diff_hunk', '') if first_comment else '',
            "previous_file": prev_file,
            "current_file": curr_file,
            "input_comment": f"{{Patch}}:\n{patch}",
            "ground_truth_comments": [
                {"comment_id": str(c['id']), "body": c['body']}
                for c in comments_for_file
            ],
        })

    print(f"Fetched {len(rows)} patch-centric rows.")
    return rows


def generate_patch_centric_dataset(repo, pr_num, output_file):
    rows = get_pr_rows(repo, pr_num)
    with open(output_file, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Generated {len(rows)} patch-centric rows.")


def main():
    parser = argparse.ArgumentParser(description="Extract GitHub PR comments into a JSONL dataset.")
    
    parser.add_argument("--repo", required=True, help="The GitHub repository (e.g., dmlc/dgl)")
    parser.add_argument("--pr", required=True, type=int, help="The Pull Request number")
    parser.add_argument("--output", default=FILENAME, help="Output file name")
    
    args = parser.parse_args()
    if not GITHUB_API_TOKEN:
        print("Failed to find GitHub token, exiting.")
        return
    generate_patch_centric_dataset(args.repo, args.pr, args.output)

if __name__ == "__main__":
    main()