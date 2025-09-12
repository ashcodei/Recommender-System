# Two-Tower Visual Similarity (PyTorch + FAISS)

End-to-end **two-tower retrieval** for **image similarity search**:
- Train a **query tower** and a **candidate tower** using **in-batch InfoNCE** contrastive learning
- Evaluate with **Recall@K**
- Build a **FAISS** index for fast nearest-neighbor retrieval
- Query images and measure latency

> Embeddings are **L2-normalized**, so using FAISS **inner product (IP)** equals **cosine similarity**.

---

`run_pipeline.py` — a single orchestrator that calls the stages internally.

---

## Repo layout

```
two_tower_visual_similarity/
  run_pipeline.py          # single entrypoint
  train.py                 # trains two-tower model
  eval.py                  # recall@k evaluation
  build_index.py           # builds & saves FAISS index + idmap
  query.py                 # loads index, runs topK query + latency
  data/
  models/
  utils/
```

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

FAISS notes:
- `faiss-cpu` is installed by default via requirements.
- For GPU FAISS, install via conda (recommended):
  ```bash
  conda install -c pytorch faiss-gpu
  ```

---

## Data format (required)

Set paths in `config.yaml`.

### 1) Training positives: `pairs.csv`
Columns: `query_path,cand_path` (relative to `images_root`)

```csv
query_path,cand_path
q/0001.jpg,c/0451.jpg
q/0002.jpg,c/1203.jpg
```

### 2) Candidate set to index: `candidates.csv`
One column: `cand_path`

```csv
cand_path
c/0001.jpg
c/0002.jpg
```

### 3) Evaluation ground truth: `query_gt.csv`
Columns: `query_path,gt_cand_path`

```csv
query_path,gt_cand_path
q/0001.jpg,c/0451.jpg
q/0002.jpg,c/1203.jpg
```

---

## Run (single entrypoint)

### Train
```bash
python run_pipeline.py --config config.yaml --stage train
```

### Evaluate Recall@K
```bash
python run_pipeline.py --config config.yaml --stage eval --k 1 5 10
```

### Build FAISS index
```bash
python run_pipeline.py --config config.yaml --stage index
```

### Query + latency
```bash
python run_pipeline.py --config config.yaml --stage query --image /abs/path/to/query.jpg --topk 10
```

### Full pipeline
```bash
python run_pipeline.py --config config.yaml --stage all --k 1 5 10
```
(Optional) include `--image ...` if you want to run a query at the end.

---

## How it works

### Two-tower model
- **Query tower** encodes queries → `q ∈ R^D`
- **Candidate tower** encodes candidates → `c ∈ R^D`
- Both outputs are **L2-normalized**: `||q||=||c||=1`

Similarity:
- cosine similarity ≡ inner product: `sim(q,c) = q · c`

### Training loss: in-batch InfoNCE
For a batch size `B`:
- positives are aligned pairs (diagonal)
- negatives are all other candidates in the batch

Logits:
- `L = (Q @ C^T) / temperature`

Cross entropy pushes each query to score its positive candidate highest.

---

## FAISS indexing

### `flat` (exact)
- Best recall
- Often fast enough for ~100K embeddings

### `ivf_pq` (approx)
- Better for very large scales (millions)
- Trades some recall for speed/memory

You can tune `nprobe` in `utils/faiss_utils.py` (higher = better recall, slower).

---

## Troubleshooting

- **Broken images / PIL errors**: check that CSV paths exist under `images_root`
- **CUDA OOM**: reduce `train.batch_size` or use `resnet18`
- **Low recall**: more epochs, better positives, try `vit_b_16`, tune augmentations
- **Slow queries**: keep index in RAM; consider `ivf_pq` at larger scales
