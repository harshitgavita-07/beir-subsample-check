"""Stress-test: how stable are nDCG@10 rankings under small query subsamples?

Context: wondersearch.ai/benchmarks (fetched 2026-11-10) compares its internal
"300 questions from the BEIR test set" scores against published MTEB full-test
numbers, citing e.g. Small 74.68 vs bge-large-en-v1.5 74.64 (a 0.04 gap) with
95% bootstrap intervals ~4 points wide. This script measures how much rankings
between genuinely close retrieval systems move under subsamples of that size.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from beir_data import load
from retrievers import build_systems
from ndcg import ndcg_cut
from subsample import subsample_queries, bootstrap_ci

DATASETS = ["scifact", "nfcorpus"]
SEEDS = 20
SUBSAMPLE_N_PER_DATASET = 37  # illustrative uniform allocation; their per-dataset allocation is unpublished
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def per_query_scores(retriever, queries, qrels, k=10):
    out = {}
    for qid, qtext in queries.items():
        if qid not in qrels:
            continue
        ranked = retriever.rank(qtext, top_k=k)
        out[qid] = ndcg_cut(ranked, qrels[qid], k)
    return out


def main():
    results = {"datasets": {}, "subsample": {"n_per_dataset": SUBSAMPLE_N_PER_DATASET, "seeds": SEEDS, "runs": []}}
    for name in DATASETS:
        corpus, queries, qrels = load(name, DATA_DIR)
        systems = build_systems(corpus)
        results["datasets"][name] = {"n_queries": len(qrels), "n_docs": len(corpus), "full": {}}
        per_system = {}
        for sys_name, ret in systems.items():
            scores = per_query_scores(ret, queries, qrels)
            full = sum(scores.values()) / len(scores)
            results["datasets"][name]["full"][sys_name] = round(full * 100, 2)
            per_system[sys_name] = scores
        # subsample runs: for each seed, subsample this dataset's queries,
        # average across datasets (their shape), record both systems' scores
        results["datasets"][name]["_scores"] = per_system
    sys_names = list(results["datasets"][DATASETS[0]]["_scores"].keys())
    for seed in range(SEEDS):
        run = {"seed": seed}
        for s in sys_names:
            pooled = []
            for name in DATASETS:
                qids = list(results["datasets"][name]["_scores"][s].keys())
                sub = subsample_queries(qids, SUBSAMPLE_N_PER_DATASET, seed=seed * 100 + hash(name) % 100)
                pooled.extend(results["datasets"][name]["_scores"][s][q] for q in sub)
            run[s] = round(100 * sum(pooled) / len(pooled), 2)
        run["winner"] = max(sys_names, key=lambda s: run[s])
        run["margin"] = round(abs(run[sys_names[0]] - run[sys_names[1]]), 2)
        results["subsample"]["runs"].append(run)
    # full-set pooled reference + CI on a 300-question pooled sample
    pooled_full = {}
    for s in sys_names:
        all_scores = []
        for name in DATASETS:
            all_scores.extend(results["datasets"][name]["_scores"][s].values())
        pooled_full[s] = round(100 * sum(all_scores) / len(all_scores), 2)
        lo, hi = bootstrap_ci(all_scores, iters=2000, seed=42)
        results["datasets"].setdefault("pooled", {})
        results["datasets"]["pooled"][s] = {
            "full_mean": pooled_full[s],
            "bootstrap95_full": [round(100 * lo, 2), round(100 * hi, 2)],
        }
    for name in DATASETS:
        results["datasets"][name].pop("_scores")
    flips = sum(1 for r in results["subsample"]["runs"] if r["winner"] != results["subsample"]["runs"][0]["winner"])
    results["subsample"]["winner_flips"] = flips
    margins = [r["margin"] for r in results["subsample"]["runs"]]
    results["subsample"]["margin_min_max"] = [min(margins), max(margins)]
    # their cited pairs (wondersearch.ai/benchmarks, 2026-11-10)
    results["their_cited_pairs"] = {
        "Small 74.68 vs bge-large-en-v1.5 74.64": 0.04,
        "Medium 77.79 vs text-embedding-3-large 77.77": 0.02,
        "Large 80.06 vs Qwen3-Embedding-8B 78.46": 1.60,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != "datasets"}, indent=1)[:1500])
    print(json.dumps(results["datasets"], indent=1))


if __name__ == "__main__":
    main()
