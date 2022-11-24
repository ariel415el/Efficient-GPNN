import faiss
import numpy as np
import torch

from utils.NN import get_NN_indices_low_memory, get_NN_indices


class FaissNNModule:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.index = None

    def _get_index(self, n, d):
        raise NotImplemented

    def init_index(self, index_vectors):
        self.index_vectors = np.ascontiguousarray(index_vectors.cpu().numpy(), dtype='float32')
        self.index = self._get_index(*self.index_vectors.shape)

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        if not self.index.is_trained:
            self.index.train(self.index_vectors)

        self.index.add(self.index_vectors)

    def search(self, queries):
        assert self.index is not None
        queries_np = np.ascontiguousarray(queries.cpu().numpy(), dtype='float32')
        _, I = self.index.search(queries_np, 1)  # actual search

        NNs = torch.from_numpy(I[:, 0])

        return NNs

class FaissFlat(FaissNNModule):
    def __str__(self):
        return "FaissFlat(" + ("GPU" if self.use_gpu else "CPU") + ")"
        
    def _get_index(self, n, d):
        return faiss.IndexFlatL2(d)


class FaissIVF(FaissNNModule):
    def __str__(self):
        return "FaisIVF(" + ("GPU" if self.use_gpu else "CPU") + ")"
        
    def _get_index(self, n, d):
        return faiss.IndexIVFFlat(faiss.IndexFlat(d), d, int(np.sqrt(n)))


class FaissIVFPQ(FaissNNModule):
    def __str__(self):
        return "FaisIVFPQ(" + ("GPU" if self.use_gpu else "CPU") + ")"
        
    def _get_index(self, n, d):
        return faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, int(np.sqrt(n)), 8, 8)


class PytorchNN:
    def __init__(self, batch_size=256, alpha=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = torch.device("cuda:0" if use_gpu else 'cpu')

    def init_index(self, index_vectors):
        self.index_vectors = index_vectors.to(self.device)

    def search(self, queries):
        return get_NN_indices(queries.to(self.device), self.index_vectors, self.alpha, self.batch_size).cpu()

    def __str__(self):
        return "PytorchNN(" + ("GPU" if self.use_gpu else "CPU") + f",alpha={self.alpha})"

class PytorchNNLowMemory:
    def __init__(self, batch_size=256, alpha=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.alpha = alpha
        self.device = torch.device("cuda:0" if use_gpu else 'cpu')

    def init_index(self, index_vectors):
        self.index_vectors = index_vectors.to(self.device)
        
    def search(self, queries):
        return get_NN_indices_low_memory(queries.to(self.device), self.index_vectors, self.alpha, self.batch_size).cpu()

    def __str__(self):
        return "PytorchNNLowMem(" + ("GPU" if self.use_gpu else "CPU") + f",alpha={self.alpha})"