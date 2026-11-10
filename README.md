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

The page's own 95% bootstrap intervals span about 8 points (half-width ~4).
This repo is an illustrative sensitivity demonstration: it shows, on real BEIR
data with two genuinely close retrieval systems, how unstable small-subsample
nDCG@10 comparisons can be.

## Method

- Datasets: BEIR `scifact` (300 test queries, 5,183 docs) and `nfcorpus`
  (323 test queries, 3,633 docs), from the public BEIR mirror.
- Systems: two standard BM25 configurations (Okapi k1=1.5 vs k1=1.2, b=0.75).
  Neither is tuned to any benchmark; on the pooled full test sets they land
  0.02 nDCG@10 points apart (47.31 vs 47.33) - the same scale as the gaps
  quoted above.
- Metric: nDCG@10 with trec_eval semantics (implementation verified against
  `pytrec_eval` in `tests/`).
- Subsamples: 20 seeds; per seed, 37 queries per dataset, pooled across the
  2 datasets (74 queries per run) and averaged. This is deliberately smaller
  than their 300 questions across 8 datasets. Their per-dataset allocation and
  weighting are not published, so 37-per-dataset is a uniform illustrative
  choice, not a reconstruction of their setup. Smaller samples are noisier,
  so the swings measured here do not transfer as magnitudes to their
  evaluation; they illustrate the mechanism, not their numbers.

## Results (results.json, reproducible)

- Full-test pooled scores differ by 0.02 points; the two systems are
  effectively tied.
- Across the 20 subsample runs (74 pooled queries each), the winner
  **flips 10 times out of 20**. At this sample size, picking the better of
  two systems this close is a coin flip.
- Subsample scores for the *same system* swing across seeds from 44.27 to
  53.70 - a 9.4-point spread, in a demo where the full-test gap is 0.02.
- The full-set bootstrap 95% interval on ~620 pooled queries spans ~6.3
  points (44.15-50.42); subsampling to 74 pooled queries widens it further.
- These magnitudes describe this demo's systems and sample sizes only.

## What this does and does not show

This is an illustrative sensitivity demo on unrelated systems (two BM25
configurations) and a different, smaller sample than WonderSearch's. It does
not evaluate their systems, their harness, or their data, and it is not
evidence that their published ranking is right or wrong. A paired comparison
on identical questions can be decisive even at n=300, and their internal
evaluation may well be one; nothing here rules that out.

The narrower, artifact-level point: as published, the page does not let an
independent reader reproduce the comparison or assess the significance of its
closest rows (0.02-0.04 nDCG@10 gaps). The question IDs, per-dataset
allocation, weighting, and harness are not public, and their internal
subsample scores sit next to externally published full-test scores, so the
two sides cannot be paired from public materials. Making the comparison
assessable is standard: publish the question IDs and the harness and score
every compared system on the same questions with a paired test. This repo is
a working template for that - point `run_check.py` at a released question
list and it produces full-set and matched-subsample numbers with intervals.

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
