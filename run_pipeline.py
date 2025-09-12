#!/usr/bin/env python
"""
Single entrypoint to run pipeline stages.

Examples:
  python run_pipeline.py --config config.yaml --stage train
  python run_pipeline.py --config config.yaml --stage eval --k 1 5 10
  python run_pipeline.py --config config.yaml --stage index
  python run_pipeline.py --config config.yaml --stage query --image /abs/path/to/q.jpg --topk 10
  python run_pipeline.py --config config.yaml --stage all --k 1 5 10
"""

import argparse
import yaml

import train as train_mod
import eval as eval_mod
import build_index as index_mod
import query as query_mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["train","eval","index","query","all"])
    ap.add_argument("--k", nargs="+", type=int, default=[1,5,10])
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--topk", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.topk is not None:
        cfg.setdefault("query", {})
        cfg["query"]["topk"] = int(args.topk)

    if args.stage == "train":
        train_mod.main(cfg)
    elif args.stage == "eval":
        eval_mod.main(cfg, args.k)
    elif args.stage == "index":
        index_mod.main(cfg)
    elif args.stage == "query":
        if not args.image:
            raise SystemExit("--image is required for query stage")
        query_mod.main(cfg, args.image)
    elif args.stage == "all":
        train_mod.main(cfg)
        eval_mod.main(cfg, args.k)
        index_mod.main(cfg)
        if args.image:
            query_mod.main(cfg, args.image)

if __name__ == "__main__":
    main()
