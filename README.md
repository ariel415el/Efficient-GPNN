# Efficient-GPNN
An efficient implementation of GPNN as depicted in ["Drop the GAN: In Defense of Patches Nearest Neighbors as Single Image Generative Models"](https://arxiv.org/abs/2103.15545)

This is the version of GPNN I used to compare with my model in the reaserach done for writing the paper "Generating natural images with direct patch distributions matching [CVPR 2022]"

While writing this implementation I consulted the implementaion in https://github.com/iyttor/GPNN.git.
My implementation offers more simplicity, a faster pytorch computiion of the NN matrix and and a low memory version of the computation done in O(N+M) as suggested in the suplementary of the paper: https://www.wisdom.weizmann.ac.il/~vision/gpnn/


# Image reshuffling
`# python3 scripts/reshuffle.py`
![reshuffle](/Readme_images/reshuffle.png)

# Image Retargeting
`# python3 scripts/retarget.py`
![retarget](/Readme_images/retarget.png)

# Image style transfer
`# python3 scripts/style_transfer.py`
![style_transfer](/Readme_images/style_transfer.png)
