# Extension Plan: Incorporating README Context into Hydra-Reviewer

## Motivation

Currently each agent sees only the patch + extracted code snippets from the same file. A repo's README often encodes project-level conventions (naming, architecture, security policies, supported platforms) that agents like `compatibility_agent`, `extensibility_agent`, `identifier_naming_style_agent`, and `security_compliance_agent` could use to give more grounded suggestions.

---

## Part 1: Implementation

### Step 1 — Fetch README in `generate_data.py`

Add a `get_readme(repo, sha, headers)` helper that fetches `README.md` from the repo root at the base SHA (the state before the PR). Append `readme` as a new field on each row.

### Step 2 — Inject into context in `get_additional_information_agent.py`

After building the existing `additional_informational` string, append the README under a new section tag, e.g. `{repository context}:\n<readme text>`. Truncate to a reasonable token budget (e.g. first 2000 chars) to avoid blowing up context windows.

### Step 3 — Update `prompt_template.py`

Add a `readme_flag` parameter (alongside the existing `flag`) to `get_input_introduction` and `get_clean_up_prompt`. When true, mention that a repository README is provided and agents should use it to assess project conventions and intent.

### Step 4 — Surface the flag through agents

The agents all call `get_input_introduction(flag)` and `get_clean_up(requirements)`. Adding the README field to the structured input is enough — no per-agent prompt changes needed unless you want to steer specific agents (e.g. tell `compatibility_agent` to cross-check against stated platform support in the README).

---

## Part 2: Evaluation

### Baseline

Run the full pipeline on `FinalDataset.jsonl` (which has `ground_truth_comments`) and record outputs as the no-README baseline. This is already partially done via `hydra_comment_reproduction`.

### Treatment

Re-run the same rows with README context injected. Store results in a new field, e.g. `hydra_comment_with_readme`.

### Automatic metrics

Compare both conditions against `ground_truth_comments` using:
- BLEU / ROUGE-L (lexical overlap)
- BERTScore (semantic similarity)

These are already in `Code/ResultStatistics/`, so just point them at the new field.

### Targeted analysis

Beyond aggregate scores, isolate the delta on suggestions that are likely README-sensitive:
- Filter suggestions that mention class/module names, platform names, or style conventions — these are where README context should help most.
- Manually check a random sample (~30 PRs) for: (a) new correct suggestions README enabled, (b) hallucinated suggestions README caused.

### Ablation dimension

Since the paper already ablates individual agents, you can also ablate README injection per agent type — inject it only into `compatibility_agent` or only into naming agents — to identify which agents actually benefit from it vs. which get confused by the extra context.

---

---

## Part 3: Ablation Experiment

The goal is to isolate *which agents* actually benefit from README context, and whether injecting it everywhere helps or hurts.

### Design

Run four conditions on the same 101-row dataset, each writing to a separate field:

| Condition | Field | README injected into |
|---|---|---|
| Baseline | `hydra_comment_reproduction` | None (existing) |
| All agents | `hydra_comment_with_readme` | All 18 agents (existing) |
| Compatibility only | `hydra_readme_compat_only` | `compatibility_agent` only |
| Naming only | `hydra_readme_naming_only` | `identifier_naming_style_agent`, `identifier_naming_readability_agent` |

### Implementation

Add a `readme_agents` set parameter to `get_additional_information_agent.get_additional_information()` that controls which agent names receive the README section. Pass this through `run_jsonl` via a `--readme-agents` flag (comma-separated agent names, or `all`/`none`).

### Evaluation

Run `bleu_score.py` on each condition's field and compare the four BLEU scores. Agents where README helps should show improvement in their condition vs. baseline; agents where it hurts will drag down the all-agents score.

### Expected finding

Compatibility and naming agents are most likely to benefit (they need project conventions). Correctness and syntax agents may not benefit at all — they operate on universal rules, not project-specific context.

---

## Implementation Order

1. `generate_data.py` — add `get_readme`, add `readme` field to rows ✓
2. `get_additional_information_agent.py` — inject README into the context string ✓
3. `prompt_template.py` — add README mention to input introduction ✓
4. Run on dataset, compare BLEU baseline vs. treatment ✓ (9.01 vs 8.40)
5. Run ablation: inject README into subsets of agents, compare per-condition BLEU
6. Manual review of ~30 sampled suggestions for qualitative delta
