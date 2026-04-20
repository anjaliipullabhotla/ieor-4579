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

def semantic_compare(filename):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    scores = []

    # 2. Read file contents
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            embeddings1 = model.encode(data['hydra_comment'], convert_to_tensor=True)
            embeddings2 = model.encode(data['hydra_comment_reproduction'], convert_to_tensor=True)
            cosine_score = util.cos_sim(embeddings1, embeddings2)
            score = cosine_score.item()
            scores.append(score)
            print(f"--- Semantic Analysis ---")
            print(f"Hydra Comment: {data['hydra_comment']}")
            print(f"Hydra Comment Reproduction: {data['hydra_comment_reproduction']}")
            print(f"Similarity Score: {score:.4f}")

            if score > 0.9:
                print("Verdict: These files are semantically nearly identical.")
            elif score > 0.7:
                print("Verdict: These files are highly similar in meaning.")
            else:
                print("Verdict: These files cover different topics or meanings.")
        print(f"Average Score: {sum(scores) / len(scores)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare.py <file>")
    else:
        semantic_compare(sys.argv[1])