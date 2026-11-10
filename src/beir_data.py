"""Download and load BEIR datasets (scifact, nfcorpus) from the public mirror."""
import io
import json
import os
import zipfile
import urllib.request

BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


def download(name, data_dir):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        url = BASE.format(name)
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
        zf.extractall(data_dir)
    return path


def load(name, data_dir, split="test"):
    path = download(name, data_dir)
    corpus, queries, qrels = {}, {}, {}
    with open(os.path.join(path, "corpus.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = (d.get("title", "") + " " + d.get("text", "")).strip()
    with open(os.path.join(path, "queries.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    qrels_path = os.path.join(path, "qrels", f"{split}.tsv")
    with open(qrels_path) as f:
        header = f.readline()
        for line in f:
            qid, docid, score = line.rstrip("\n").split("\t")
            score = int(score)
            if score > 0:
                qrels.setdefault(qid, {})[docid] = score
    return corpus, queries, qrels
