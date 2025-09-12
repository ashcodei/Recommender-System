import numpy as np

def build_faiss_index(embeddings: np.ndarray, index_type: str, metric: str,
                      use_gpu: bool, nlist: int, m_pq: int, nbits: int):
    """
    embeddings: float32 [N, D], assumed L2-normalized if using IP cosine similarity.
    """
    import faiss

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    N, D = embeddings.shape

    if metric == "ip":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        faiss_metric = faiss.METRIC_L2
    else:
        raise ValueError("metric must be 'ip' or 'l2'")

    index_type = index_type.lower()
    if index_type == "flat":
        index = faiss.IndexFlatIP(D) if metric == "ip" else faiss.IndexFlatL2(D)

    elif index_type == "ivf_pq":
        quantizer = faiss.IndexFlatIP(D) if metric == "ip" else faiss.IndexFlatL2(D)
        index = faiss.IndexIVFPQ(quantizer, D, nlist, m_pq, nbits, faiss_metric)
        index.train(embeddings)
        # speed/recall tradeoff
        index.nprobe = min(64, nlist)
    else:
        raise ValueError("index_type must be 'flat' or 'ivf_pq'")

    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings)
    return index

def search_index(index, queries: np.ndarray, topk: int):
    if queries.dtype != np.float32:
        queries = queries.astype(np.float32)
    distances, ids = index.search(queries, topk)
    return distances, ids
