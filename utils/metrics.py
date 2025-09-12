import numpy as np

def recall_at_k(retrieved_ids: np.ndarray, gt_ids: np.ndarray, ks=(1,5,10)):
    """
    retrieved_ids: [num_queries, max_k] candidate indices
    gt_ids: [num_queries] ground-truth candidate index for each query
    """
    out = {}
    for k in ks:
        hit = (retrieved_ids[:, :k] == gt_ids[:, None]).any(axis=1).mean()
        out[f"recall@{k}"] = float(hit)
    return out
