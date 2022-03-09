import os
from time import time
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPNN import PNN, GPNN
from utils.image import save_image


def retarget(image_paths, out_dir):
    with torch.no_grad():
        PNN_moduel = PNN(patch_size=7,
                         stride=1,
                         alpha=0.005,
                         reduce_memory_footprint=True)
        for scale_factor in [(1, 2), (1,0.5), (2, 1)]:
            GPNN_module = GPNN(PNN_moduel,
                               scale_factor=(1, 1.5),
                               resize=0, num_steps=10,
                               pyr_factor=0.75,
                               coarse_dim=14,
                               noise_sigma=1.5,
                               device="cuda:0")

            for im_path in image_paths:
                start = time()
                output_image = GPNN_module.run(target_img_path=im_path, init_mode="target")
                print(f"took {time() - start} s")
                save_image(output_image, os.path.join(out_dir, f"{scale_factor}_{os.path.basename(im_path)}"))


if __name__ == '__main__':
    dataset_dir = f'images/SIGD16'
    out_dir = os.path.join("outputs", "retarget")
    image_paths = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
    retarget(image_paths, out_dir)