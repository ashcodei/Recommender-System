import argparse, os
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data.pairs_dataset import ImageListDataset
from data.transforms import build_eval_tfms
from models.towers import TwoTower
from utils.metrics import recall_at_k
from utils.faiss_utils import build_faiss_index, search_index

def embed_all(model, tower_name: str, dl, device):
    model.eval()
    outs = []
    paths = []
    tower = getattr(model, tower_name)
    with torch.no_grad():
        for x, p in tqdm(dl, desc=f"embed {tower_name}"):
            x = x.to(device, non_blocking=True)
            z = tower(x).detach().cpu().numpy().astype(np.float32)
            outs.append(z)
            paths.extend(p)
    return np.vstack(outs), paths

def load_gt(cfg):
    import csv
    q, gt = [], []
    with open(cfg["data"]["query_gt_csv"], "r", newline="") as f:
        r = csv.DictReader(f)
        if "query_path" not in (r.fieldnames or []) or "gt_cand_path" not in (r.fieldnames or []):
            raise ValueError("query_gt_csv must have columns: query_path,gt_cand_path")
        for row in r:
            q.append(row["query_path"])
            gt.append(row["gt_cand_path"])
    return q, gt

def main(cfg, ks):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(cfg["train"]["ckpt_path"], map_location="cpu")
    model = TwoTower(
        cfg["model"]["backbone"],
        cfg["model"]["embedding_dim"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        shared=False
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    tfm = build_eval_tfms(image_size=224)

    cand_ds = ImageListDataset(
        cfg["data"]["images_root"],
        cfg["data"]["candidate_list_csv"],
        transform=tfm,
        col_preference=("cand_path","path")
    )
    cand_dl = DataLoader(cand_ds, batch_size=256, shuffle=False,
                         num_workers=cfg["train"]["num_workers"], pin_memory=True)

    cand_emb, cand_paths = embed_all(model, "cand", cand_dl, device)
    path_to_idx = {p: i for i, p in enumerate(cand_paths)}

    q_paths, gt_paths = load_gt(cfg)

    # create temporary query list csv
    import csv as _csv
    tmp_csv = "._tmp_eval_queries.csv"
    with open(tmp_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["query_path"])
        w.writeheader()
        for qp in q_paths:
            w.writerow({"query_path": qp})

    q_ds = ImageListDataset(
        cfg["data"]["images_root"],
        tmp_csv,
        transform=tfm,
        col_preference=("query_path",)
    )
    q_dl = DataLoader(q_ds, batch_size=256, shuffle=False,
                      num_workers=cfg["train"]["num_workers"], pin_memory=True)

    q_emb, q_paths_loaded = embed_all(model, "query", q_dl, device)
    os.remove(tmp_csv)

    gt_map = {qp: gp for qp, gp in zip(q_paths, gt_paths)}
    gt_ids = []
    missing = 0
    for qp in q_paths_loaded:
        gp = gt_map.get(qp)
        if gp is None or gp not in path_to_idx:
            missing += 1
            gt_ids.append(-1)
        else:
            gt_ids.append(path_to_idx[gp])
    gt_ids = np.array(gt_ids, dtype=np.int64)
    valid = gt_ids >= 0
    if valid.sum() == 0:
        raise RuntimeError("No valid query->gt candidate matches found. Check your CSVs.")

    index = build_faiss_index(
        cand_emb,
        cfg["index"]["type"],
        cfg["index"]["metric"],
        use_gpu=cfg["index"]["use_gpu"],
        nlist=cfg["index"]["nlist"],
        m_pq=cfg["index"]["m_pq"],
        nbits=cfg["index"]["nbits"]
    )

    maxk = max(ks)
    _, ids = search_index(index, q_emb[valid], maxk)
    metrics = recall_at_k(ids, gt_ids[valid], ks=tuple(ks))

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Valid queries: {valid.sum()} / {len(gt_ids)} (missing {missing})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", nargs="+", type=int, default=[1,5,10])
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.k)
