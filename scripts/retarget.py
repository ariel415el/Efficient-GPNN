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
                         alpha=1,
                         reduce_memory_footprint=True)
        # for scale_factor in [(1, 2), (1, 0.5), (2, 0.5)]:
        for scale_factor in [(1, 1.5), (1.5, 1)]:
            GPNN_module = GPNN(PNN_moduel,
                               scale_factor=scale_factor,
                               resize=256, num_steps=10,
                               pyr_factor=0.85,
                               coarse_dim=14,
                               noise_sigma=0,
                               device="cuda:0")

            for im_path in image_paths:
                start = time()
                output_image = GPNN_module.run(target_img_path=im_path, init_mode="blured_target")
                print(f"took {time() - start} s")
                save_image(output_image, os.path.join(out_dir, f"{scale_factor}_{os.path.basename(im_path)}"))


if __name__ == '__main__':
    dataset_dir = f'images/SIGD16'
    out_dir = os.path.join("outputs", "retarget")
    image_paths = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
    image_paths = ['/home/ariel/university/GPDM/images/HQ_16/pizza_tower.jpeg']
    retarget(image_paths, out_dir)