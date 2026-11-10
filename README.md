# beir-subsample-check

An independent statistical check of small-sample retrieval benchmark comparisons,
built around WonderSearch's public benchmark page
(https://wondersearch.ai/benchmarks/, fetched 2026-11-10).

## Why

That page compares WonderSearch's internal scores, measured on "300 questions
from the BEIR test set", against published MTEB full-test-set numbers for other
systems. Two of the headline rows are separated by hundredths of a point:

| Row on the page (2026-11-10) | Gap (nDCG@10 points) |
| --- | --- |
| WonderSearch Small 74.68 vs bge-large-en-v1.5 74.64 | 0.04 |
| WonderSearch Medium 77.79 vs text-embedding-3-large 77.77 | 0.02 |

The page's own 95% bootstrap intervals are about 4 points wide. This repo
measures what a comparison of that shape can and cannot show, using real BEIR
data and two genuinely close retrieval systems.

## Method

- Datasets: BEIR `scifact` (300 test queries, 5,183 docs) and `nfcorpus`
  (323 test queries, 3,633 docs), from the public BEIR mirror.
- Systems: two standard BM25 configurations (Okapi k1=1.5 vs k1=1.2, b=0.75).
  Neither is tuned to any benchmark; on the pooled full test sets they land
  0.02 nDCG@10 points apart (47.31 vs 47.33) - the same scale as the gaps
  quoted above.
- Metric: nDCG@10 with trec_eval semantics (implementation verified against
  `pytrec_eval` in `tests/`).
- Subsamples: 20 seeds; per seed, 37 queries per dataset (their published
  shape: "300 questions ... 8 datasets"), pooled and averaged.

## Results (results.json, reproducible)

- Full-test pooled scores differ by 0.02 points; the two systems are
  effectively tied.
- Across the 20 subsample runs, the winner **flips 10 times out of 20**.
  Picking the better system from one 300-question subsample is a coin flip.
- Subsample scores for the *same system* swing across seeds from 44.27 to
  53.70 - a 9.4-point spread, roughly 200x the cited 0.04-point gap.
- The full-set bootstrap 95% interval on ~620 pooled queries is already
  ~6 points wide (44.15-50.42); subsampling to 74 pooled queries makes it
  far worse.

## What this means for the benchmark page

As published, rows separated by hundredths of a point cannot rank systems;
the comparison mixes an internal 300-question subsample with full-test
published numbers, and the subsample is not identified. The fix is cheap and
standard: publish the question IDs and the harness, and score every compared
system on the same questions. This repo is a working template for that - point
`run_check.py` at the released question list and it produces full-set and
matched-subsample numbers with intervals.

Nothing here says WonderSearch retrieval is good or bad. It says the current
page's closest comparisons are statistically undecidable as published, and
shows exactly how to make them decidable.

## Run it

```
pip install -r requirements.txt
python3 -m pytest tests/ -q     # 11 tests, incl. nDCG cross-checked vs pytrec_eval
python3 src/run_check.py        # downloads BEIR scifact + nfcorpus, writes results.json
```

Runtime: about a minute on 2 CPU cores. No GPU, no API keys, no paid services.

## Layout

- `src/ndcg.py` - nDCG@k (trec_eval semantics)
- `src/beir_data.py` - BEIR download/load
- `src/retrievers.py` - the two BM25 systems
- `src/subsample.py` - seeded subsampling + bootstrap CI
- `src/run_check.py` - the experiment
- `tests/` - 11 tests
- `results.json` - the run this README reports
