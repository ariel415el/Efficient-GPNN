import torch

def get_patch_NNS(queries, keys, alpha):
    """
    Get the nearest neighbor index from keys patches for each query patch
    """
    queries = queries.reshape(queries.size(0), -1)
    keys = keys.reshape(keys.size(0), -1)

    dist = compute_distances_batch(queries, keys, b=min(100, len(queries)))
    # dist = torch.cdist(queries.view(len(queries), -1), keys.view(len(queries), -1)) # Not enough memory
    dist = (dist / (torch.min(dist, dim=0)[0] + alpha))  # compute_normalized_scores
    NNs = torch.argmin(dist, dim=1)  # find_NNs
    return NNs


def efficient_compute_distances(x, y):
    """
    Pytorch efficient way of computing distances between all vectors in x and y, i.e (x[:, None] - y[None, :])**2
    """
    dist = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def compute_distances_batch(queries, keys, b):
    dist_mat = torch.zeros((queries.shape[0], keys.shape[0]), dtype=torch.float32, device=queries.device)
    n_batches = len(queries) // b
    for i in range(n_batches):
        dist_mat[i * b:(i + 1) * b] = efficient_compute_distances(queries[i * b:(i + 1) * b], keys)
        # if mins is not None:
    if len(queries) % b != 0:
        dist_mat[n_batches * b:] = efficient_compute_distances(queries[n_batches * b:], keys)

    return dist_mat


def get_patch_NNS_low_memory(queries, keys, alpha, b=512):
    queries = queries.reshape(queries.size(0), -1)
    keys = keys.reshape(keys.size(0), -1)
    mins = get_mins_efficient(queries, keys, b=b)
    NNs = torch.zeros(queries.shape[0], dtype=torch.long, device=queries.device)
    n_batches = len(queries) // b
    for i in range(n_batches):
        dists = efficient_compute_distances(queries[i * b:(i + 1) * b], keys) / (alpha + mins[None, :])
        NNs[i * b:(i + 1) * b] = dists.min(1)[1]
    if len(queries) % b != 0:
        dists = efficient_compute_distances(queries[n_batches * b:], keys) / (alpha + mins[None, :])
        NNs[n_batches * b:] = dists.min(1)[1]
    return NNs


def get_mins_efficient(queries, keys, b):
    mins = torch.zeros(keys.shape[0], dtype=torch.float32, device=queries.device)
    n_batches = len(queries) // b
    for i in range(n_batches):
        mins[i * b:(i + 1) * b] = efficient_compute_distances(queries, keys[i * b:(i + 1) * b]).min(0)[0]
    if len(queries) % b != 0:
        mins[n_batches * b:] = efficient_compute_distances(queries, keys[n_batches * b:]).min(0)[0]

    return mins


