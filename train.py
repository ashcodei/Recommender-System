import argparse, os, time
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.pairs_dataset import PairsDataset
from data.transforms import build_train_tfms
from models.towers import TwoTower
from utils.seed import set_seed

def info_nce_loss(qz, cz, temperature: float):
    logits = (qz @ cz.t()) / temperature
    labels = torch.arange(qz.size(0), device=qz.device)
    return F.cross_entropy(logits, labels)

def main(cfg):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.dirname(cfg["train"]["ckpt_path"]), exist_ok=True)

    tfm = build_train_tfms(image_size=224)
    ds = PairsDataset(cfg["data"]["images_root"], cfg["data"]["pairs_csv"], transform=tfm)
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True
    )

    model = TwoTower(
        cfg["model"]["backbone"],
        cfg["model"]["embedding_dim"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        shared=False
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]))

    model.train()
    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{cfg['train']['epochs']}")
        running = 0.0
        t0 = time.time()

        for (q, c, *_paths) in pbar:
            q = q.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["amp"])):
                qz, cz = model(q, c)
                loss = info_nce_loss(qz, cz, cfg["train"]["temperature"])

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            global_step += 1

            if global_step % cfg["train"]["log_every"] == 0:
                avg = running / cfg["train"]["log_every"]
                running = 0.0
                pbar.set_postfix(loss=f"{avg:.4f}")

        print(f"Epoch {epoch+1} done in {time.time()-t0:.1f}s")
        torch.save({"model": model.state_dict(), "cfg": cfg}, cfg["train"]["ckpt_path"])
        print(f"Saved: {cfg['train']['ckpt_path']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
