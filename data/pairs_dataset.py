import csv
import os
from dataclasses import dataclass
from typing import List

from PIL import Image
from torch.utils.data import Dataset

@dataclass
class PairRow:
    query_path: str
    cand_path: str

class PairsDataset(Dataset):
    """
    Positive pairs dataset.
    pairs_csv must contain: query_path,cand_path (relative to images_root).
    """
    def __init__(self, images_root: str, pairs_csv: str, transform=None):
        self.images_root = images_root
        self.transform = transform
        self.rows: List[PairRow] = []

        with open(pairs_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "query_path" not in (reader.fieldnames or []) or "cand_path" not in (reader.fieldnames or []):
                raise ValueError("pairs_csv must have columns: query_path,cand_path")
            for r in reader:
                self.rows.append(PairRow(r["query_path"], r["cand_path"]))

        if len(self.rows) == 0:
            raise ValueError("No rows found in pairs_csv")

    def __len__(self):
        return len(self.rows)

    def _load_image(self, rel_path: str):
        path = os.path.join(self.images_root, rel_path)
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        q = self._load_image(row.query_path)
        c = self._load_image(row.cand_path)
        if self.transform:
            q = self.transform(q)
            c = self.transform(c)
        return q, c, row.query_path, row.cand_path

class ImageListDataset(Dataset):
    """
    For embedding candidates/queries from a CSV list.
    list_csv: one column `path` OR `cand_path` OR `query_path`.
    """
    def __init__(self, images_root: str, list_csv: str, transform=None, col_preference=("cand_path","query_path","path")):
        self.images_root = images_root
        self.transform = transform
        self.paths: List[str] = []

        with open(list_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            col = None
            for c in col_preference:
                if c in (reader.fieldnames or []):
                    col = c
                    break
            if col is None:
                raise ValueError(f"list_csv must have one of columns: {col_preference}")

            for r in reader:
                self.paths.append(r[col])

        if len(self.paths) == 0:
            raise ValueError("No paths found in list_csv")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        rel_path = self.paths[idx]
        img = Image.open(os.path.join(self.images_root, rel_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, rel_path
