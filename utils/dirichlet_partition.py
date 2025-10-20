import numpy as np
import torch
from torch.utils.data import Subset

def _get_targets_array(dataset):
    y = getattr(dataset, "targets", getattr(dataset, "labels", None))
    if y is None:
        _, y = zip(*[dataset[i] for i in range(len(dataset))])
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    return np.asarray(y)

def dirichlet_partition(train, num_clients, alpha=0.5, seed=42, ensure_min_size=True):
    rng = np.random.default_rng(seed)
    y = _get_targets_array(train)
    num_classes = int(np.max(y)) + 1

    idx_by_class = [np.where(y == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    props = rng.dirichlet([alpha] * num_clients, size=num_classes)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        cls_idx = idx_by_class[c]
        n_c = len(cls_idx)
        if n_c == 0:
            continue
        frac = props[c] / props[c].sum()
        alloc = (frac * n_c).astype(int)
        leftovers = n_c - int(alloc.sum())
        if leftovers > 0:
            p = frac / frac.sum()
            extra = rng.choice(num_clients, size=leftovers, replace=True, p=p)
            for k in extra:
                alloc[k] += 1
        start = 0
        for k, s in enumerate(alloc):
            if s > 0:
                client_indices[k].extend(cls_idx[start:start + s])
                start += s

    if ensure_min_size:
        sizes = np.array([len(ix) for ix in client_indices])
        empties = np.where(sizes == 0)[0]
        for r in empties:
            d = int(np.argmax(sizes))
            if sizes[d] > 1:
                client_indices[r].append(client_indices[d].pop())
                sizes[r] += 1
                sizes[d] -= 1

    client_indices = [sorted(ix) for ix in client_indices]
    return [Subset(train, ix) for ix in client_indices]