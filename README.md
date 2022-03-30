# Efficient-GPNN
An efficient Pytorch implementation of GPNN as depicted in ["Drop the GAN: In Defense of Patches Nearest Neighbors as Single Image Generative Models"](https://arxiv.org/abs/2103.15545)

This is the version of GPNN I used to as reference for the reaserach done for writing the paper ["Generating Natural Images with Direct Patch Distribution Matching"](https://arxiv.org/abs/2203.11862).
The Paper's repo is available here: https://github.com/ariel415el/GPDM.

While writing this implementation I consulted the implementaion in https://github.com/iyttor/GPNN.git.
My implementation offers more simplicity, a faster pytorch computiion of the NN matrix and and a low memory version of the computation done in O(N+M) as 
suggested in the suplementary of the paper: https://www.wisdom.weizmann.ac.il/~vision/gpnn/.

I've also implemented aproximated NN with Faiss (cpu/gpu) with various indices like IVF and IVFPQ.


# NN computation options
- Pytorch : Batched Fast pytorch nn computations

  `NN_module = PytorchNN(alpha, batch_size, use_gpu=True)`
- Pytorch_low_memory: Batched Efficient pytorch implementation that avoids holding a distance matrix on memory

  `NN_module = PytorchNNLowMemory(alpha, batch_size, use_gpu=True)`
- FaissFlat: uses faiss exact-NN computations (Cpu and GPU, no alpha)

  `NN_module = FaissFlat(use_gpu=True)`
- FaissIVF: uses faiss inverted index approximate-nn (Cpu and GPU, no alpha)

  `NN_module = FaissIVF(use_gpu=True)`
- FaissIVFPQ: uses faiss inverted index with product quantization approximate-nn (Cpu and GPU, no alpha)
  
  `NN_module = FaissIVFPQ(use_gpu=True)`


# Image reshuffling
`# python3 scripts/reshuffle.py`
![reshuffle](/Readme_images/reshuffle.png)

# Image Retargeting
`# python3 scripts/retarget.py`
![retarget](/Readme_images/retarget.png)

# Image style transfer
`# python3 scripts/style_transfer.py`
![style_transfer](/Readme_images/style_transfer.png)
