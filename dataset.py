from typing import Tuple, Dict, List
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, y_train, X_test, y_test

def dirichlet_noniid_split(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.3,
    seed: int = 42,
    min_total_per_client: int = 1,
) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(idx_by_class[c])

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in classes:
        n = len(idx_by_class[c])
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = (proportions * n).astype(int)
        for _ in range(n - counts.sum()):
            counts[rng.integers(0, num_clients)] += 1
        start = 0
        for cid, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[cid].extend(idx_by_class[c][start:start+cnt].tolist())
            start += cnt

    lengths = {cid: len(v) for cid, v in client_indices.items()}
    need = [cid for cid, L in lengths.items() if L < min_total_per_client]
    have = [cid for cid, L in lengths.items() if L > min_total_per_client]
    for cid in need:
        for donor in have:
            if lengths[donor] <= min_total_per_client:
                continue
            moved = client_indices[donor].pop()
            client_indices[cid].append(moved)
            lengths[donor] -= 1
            lengths[cid] += 1
            if lengths[cid] >= min_total_per_client:
                break

    return {cid: np.array(sorted(idxs)) for cid, idxs in client_indices.items()}
