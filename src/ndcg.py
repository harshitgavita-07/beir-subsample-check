"""nDCG@k, matching trec_eval's ndcg_cut definition.

DCG uses the trec_eval discount: gain / log2(rank + 1). Graded relevance from qrels; unjudged documents count as 0.
"""
import math


def dcg(gains, k):
    total = 0.0
    for i, g in enumerate(gains[:k]):
        rank = i + 1
        discount = math.log2(rank + 1)
        total += g / discount
    return total


def ndcg_cut(ranked_doc_ids, qrels, k):
    """qrels: dict doc_id -> graded relevance. ranked_doc_ids: ranked list."""
    gains = [qrels.get(d, 0) for d in ranked_doc_ids]
    ideal = sorted(qrels.values(), reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg(gains, k) / idcg


def mean_ndcg(run, qrels_by_qid, k):
    """run: dict qid -> ranked list of doc_ids. Mean over queries with qrels."""
    scores = []
    for qid, qrels in qrels_by_qid.items():
        ranked = run.get(qid, [])
        scores.append(ndcg_cut(ranked, qrels, k))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
