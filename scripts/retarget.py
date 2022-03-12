import os
from time import time
import sys

import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NN_modules import *
from GPNN import GPNN
from utils.image import save_image


def retarget(image_paths, out_dir):
    with torch.no_grad():

        for scale_factor in [(1, 2), (1, 0.5), (2, 0.5)]:

            NN_module = PytorchNNLowMemory(alpha=0.005, use_gpu=True)

            GPNN_module = GPNN(NN_module,
                               patch_size=7,
                               stride=1,
                               scale_factor=scale_factor,
                               resize=256,
                               num_steps=10,
                               pyr_factor=0.85,
                               coarse_dim=14,
                               noise_sigma=1.5,
                               single_iteration_in_first_pyr_level=True)

            for im_path in image_paths:
                start = time()
                output_image = GPNN_module.run(target_img_path=im_path, init_mode="target")
                print(f"took {time() - start} s")
                save_image(output_image, os.path.join(out_dir, GPNN_module.name, f"{scale_factor}_{os.path.basename(im_path)}"))


if __name__ == '__main__':
    dataset_dir = 'images/SIGD16'
    dataset_name = os.path.basename(dataset_dir)
    image_paths = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
    retarget(image_paths, out_dir=os.path.join("outputs", 'retarget', f"{dataset_name}"))