# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sentence-transformers",
#   "torch",
# ]
# ///

import sys
from sentence_transformers import SentenceTransformer, util
import json


FILENAME = '../Dataset/FinalDatasetPython.jsonl'


def avg_similarity_against_ground_truth(model, pred, ground_truth_comments):
    """Return the average cosine similarity of pred against all ground truth comments."""
    if not pred or not ground_truth_comments:
        return 0.0
    gt_texts = [c['comment'] for c in ground_truth_comments if c.get('comment')]
    if not gt_texts:
        return 0.0
    pred_emb = model.encode(pred, convert_to_tensor=True)
    scores = [util.cos_sim(pred_emb, model.encode(gt, convert_to_tensor=True)).item() for gt in gt_texts]
    return sum(scores) / len(scores)


def semantic_compare(filename):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    baseline_scores, readme_scores = [], []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            ground_truth = data.get('ground_truth_comments', [])
            baseline = data.get('hydra_comment_reproduction', '')
            readme = data.get('hydra_comment_with_readme', '')

            if not baseline or not readme or not ground_truth:
                continue

            b_score = avg_similarity_against_ground_truth(model, baseline, ground_truth)
            r_score = avg_similarity_against_ground_truth(model, readme, ground_truth)
            baseline_scores.append(b_score)
            readme_scores.append(r_score)

            print(f"--- id: {data['id']} ---")
            print(f"  Baseline (no README) vs ground truth:    {b_score:.4f}")
            print(f"  With README vs ground truth:             {r_score:.4f}")
            print(f"  Delta:                                   {r_score - b_score:+.4f}")

    if baseline_scores:
        avg_b = sum(baseline_scores) / len(baseline_scores)
        avg_r = sum(readme_scores) / len(readme_scores)
        print(f"\n=== RESULTS ({len(baseline_scores)} rows) ===")
        print(f"  Avg baseline (no README) vs ground truth:  {avg_b:.4f}")
        print(f"  Avg with README vs ground truth:           {avg_r:.4f}")
        print(f"  Delta:                                     {avg_r - avg_b:+.4f}")


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else FILENAME
    semantic_compare(filename)