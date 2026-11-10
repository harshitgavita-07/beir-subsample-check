import math
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytrec_eval

from ndcg import ndcg_cut, mean_ndcg, dcg


def test_dcg_first_rank_undiscounted():
    assert dcg([3, 2, 1], 3) == 3 + 2 / math.log2(3) + 1 / math.log2(4)


def test_ndcg_perfect_ranking_is_one():
    qrels = {"a": 2, "b": 1, "c": 0}
    assert ndcg_cut(["a", "b", "c"], qrels, 3) == 1.0


def test_ndcg_unjudged_counts_zero():
    qrels = {"a": 1}
    # unjudged doc ranked first wastes rank 1
    score = ndcg_cut(["x", "a"], qrels, 2)
    ideal = 1.0
    assert abs(score - (1 / math.log2(3)) / ideal) < 1e-9


def test_ndcg_empty_qrels_zero():
    assert ndcg_cut(["a"], {}, 10) == 0.0


def test_matches_pytrec_eval_random_fixture():
    rng = random.Random(7)
    qrels = {}
    run = {}
    for q in range(15):
        qid = f"q{q}"
        docs = [f"d{q}_{i}" for i in range(30)]
        qrels[qid] = {d: rng.randint(0, 2) for d in docs}
        qrels[qid] = {d: s for d, s in qrels[qid].items() if s > 0}
        ranked = docs[:]
        rng.shuffle(ranked)
        run[qid] = {d: float(len(ranked) - i) for i, d in enumerate(ranked)}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    ref = evaluator.evaluate(run)
    for qid in qrels:
        ranked = sorted(run[qid], key=lambda d: -run[qid][d])
        ours = ndcg_cut(ranked, qrels[qid], 10)
        assert abs(ours - ref[qid]["ndcg_cut_10"]) < 1e-6, qid


def test_mean_ndcg_matches_pytrec_eval():
    rng = random.Random(11)
    qrels, run = {}, {}
    for q in range(8):
        qid = f"q{q}"
        docs = [f"d{q}_{i}" for i in range(12)]
        qrels[qid] = {d: 1 for d in rng.sample(docs, 3)}
        ranked = docs[:]
        rng.shuffle(ranked)
        run[qid] = ranked
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut.10"})
    ref = evaluator.evaluate({q: {d: float(len(r) - i) for i, d in enumerate(r)}
                              for q, r in run.items()})
    ref_mean = sum(v["ndcg_cut_10"] for v in ref.values()) / len(ref)
    assert abs(mean_ndcg(run, qrels, 10) - ref_mean) < 1e-6
