import argparse, time
import yaml
import numpy as np

import torch
from PIL import Image

from data.transforms import build_eval_tfms
from models.towers import TwoTower

def load_faiss(index_path: str, idmap_path: str, use_gpu: bool):
    import faiss
    index = faiss.read_index(index_path)
    idmap = np.load(idmap_path, allow_pickle=True)

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    return index, idmap

def embed_one(model, img_path: str, device):
    tfm = build_eval_tfms(224)
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.query(x).detach().cpu().numpy().astype(np.float32)
    return z

def main(cfg, image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(cfg["train"]["ckpt_path"], map_location="cpu")
    model = TwoTower(
        cfg["model"]["backbone"],
        cfg["model"]["embedding_dim"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        shared=False
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    index, idmap = load_faiss(cfg["index"]["index_path"], cfg["index"]["idmap_path"], cfg["index"]["use_gpu"])

    q = embed_one(model, image_path, device)
    topk = int(cfg["query"]["topk"])

    for _ in range(int(cfg["query"]["warmup"])):
        index.search(q, topk)

    runs = int(cfg["query"]["runs"])
    t0 = time.perf_counter()
    for _ in range(runs):
        D, I = index.search(q, topk)
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000.0 / runs
    print(f"Avg search latency over {runs} runs: {avg_ms:.3f} ms")

    print("\nTop matches:")
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        path = str(idmap[idx])
        print(f"{rank:02d}. idx={idx} score={dist:.4f}  path={path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.image)
