import argparse, os
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data.pairs_dataset import ImageListDataset
from data.transforms import build_eval_tfms
from models.towers import TwoTower
from utils.faiss_utils import build_faiss_index

def embed_all_cands(model, dl, device):
    model.eval()
    outs, paths = [], []
    with torch.no_grad():
        for x, p in tqdm(dl, desc="embed candidates"):
            x = x.to(device, non_blocking=True)
            z = model.cand(x).detach().cpu().numpy().astype(np.float32)
            outs.append(z)
            paths.extend(p)
    return np.vstack(outs), paths

def main(cfg):
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

    tfm = build_eval_tfms(224)

    cand_ds = ImageListDataset(
        cfg["data"]["images_root"],
        cfg["data"]["candidate_list_csv"],
        transform=tfm,
        col_preference=("cand_path","path")
    )
    cand_dl = DataLoader(cand_ds, batch_size=256, shuffle=False,
                         num_workers=cfg["train"]["num_workers"], pin_memory=True)

    emb, paths = embed_all_cands(model, cand_dl, device)

    os.makedirs(os.path.dirname(cfg["index"]["index_path"]), exist_ok=True)

    index = build_faiss_index(
        emb,
        cfg["index"]["type"],
        cfg["index"]["metric"],
        use_gpu=cfg["index"]["use_gpu"],
        nlist=cfg["index"]["nlist"],
        m_pq=cfg["index"]["m_pq"],
        nbits=cfg["index"]["nbits"]
    )

    import faiss
    if cfg["index"]["use_gpu"]:
        index = faiss.index_gpu_to_cpu(index)

    faiss.write_index(index, cfg["index"]["index_path"])
    np.save(cfg["index"]["idmap_path"], np.array(paths, dtype=object))

    print(f"Saved index: {cfg['index']['index_path']}")
    print(f"Saved id map: {cfg['index']['idmap_path']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
