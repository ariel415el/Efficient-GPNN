# Efficient-GPNN
An efficient implementation of GPNN depicted in "Drop the GAN: In Defense of Patches Nearest Neighbors as Single Image Generative Models"

This is the version of GPNN I used to compare with my model in the reaserach done for writing the paper "Generating natural images with direct patch distributions matching [CVPR 2022]"

This implementaion is inspired uses parts of the code from the implementaion in https://github.com/iyttor/GPNN.git.
It offers more simplicity, an faster pytorch computiion of the NN matrix and and a low memory version of the computation done in O(N+M) as suggested in the suplementary of the paper: https://www.wisdom.weizmann.ac.il/~vision/gpnn/

