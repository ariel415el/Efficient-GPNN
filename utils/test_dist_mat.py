import torch
from time import time

def compute_dists_efficient(x, y):
    dist = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def compute_dists(x, y):
    dist = torch.sum((x[:, None] - y[None, :]) **2, -1)
    return dist

def cdist(x,y):
    return torch.cdist(x, y)**2

def compute_distances_batch_with_func(x, y, func, b):
    x = x.reshape(x.size(0), -1)
    y = y.reshape(y.size(0), -1)
    dist_mat = torch.zeros((x.shape[0], y.shape[0]), dtype=x.dtype, device=device)
    n_batches = len(x) // b
    for i in range(n_batches):
        dist_mat[i * b:(i + 1) * b] = func(x[i * b:(i + 1) * b], y)
    if len(x) % b != 0:
        dist_mat[n_batches * b:] = func(x[n_batches * b:], y)

    return dist_mat

def compute_distances_batch(x,y, b=32):
    return compute_distances_batch_with_func(x,y, compute_dists, b=b)

def compute_distances_batch_efficient(x, y, b=32):
    return compute_distances_batch_with_func(x,y, compute_dists_efficient, b=b)

def compute_distances_batch_cdist(x, y, b=32):
    func = lambda x,y : torch.cdist(x, y)**2
    return compute_distances_batch_with_func(x,y, func, b=b)

def time_func(func, x ,y):
    start = time()
    for i in range(10):
        func(x,y)
    return (time() - start) #/ 100



# for device in [torch.device("cpu"), torch.device("cuda")]:
for device in [torch.device("cuda")]:
    for dtype in [torch.float16, torch.float32, torch.float64]:
        x = torch.randn(2**13,147).to(dtype).to(device)
        y = torch.randn(2**13,147).to(dtype).to(device)
        # d1 = compute_distances_batch(x,y, b=32)
        d1 = compute_distances_batch_efficient(x,y, b=32)
        d2 = cdist(x,y)
        print(f"############## {device}, {x.dtype} ############")
        print(f"Bit-exact={torch.allclose(d1, d2)}, Max_diff={torch.max(torch.abs(d1-d2))}")

        for func in [compute_distances_batch, compute_distances_batch_efficient, compute_distances_batch_cdist]:
            print(f"{func.__name__}: time={time_func(func, x, y)}")


