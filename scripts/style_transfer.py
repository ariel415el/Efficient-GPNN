import os
from time import time
import sys

import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.NN_modules import *
from GPNN import GPNN
from utils.image import save_image


def style_transfer(contents_and_styles):
    with torch.no_grad():

        # NN_module = PytorchNNLowMemory(alpha=0.005, batch_size=256, use_gpu=True); resize=256; coarse_dim=128
        NN_module = FaissIVFPQ(use_gpu=True); resize=1024; coarse_dim=1024

        GPNN_module = GPNN(NN_module, patch_size=8,
                                      resize=resize,
                                      coarse_dim=coarse_dim,
                                      num_steps=100,
                                      pyr_factor=0.75,
                                      noise_sigma=0,
                                      single_iteration_in_first_pyr_level=False)

        out_dir = f"outputs/style_transfer"
        for (content_image_path, style_iamge_path) in contents_and_styles:
            start = time()
            output_image = GPNN_module.run(target_img_path=style_iamge_path, init_mode=content_image_path)
            print(f"took {time() - start} s")

            content_fname, ext = os.path.splitext(os.path.basename(content_image_path))[:2]
            style_fname = os.path.basename(style_iamge_path)
            output_path = os.path.join(out_dir, f'{GPNN_module.NN_module}_{GPNN_module.resize}x{GPNN_module.pyr_factor}->{GPNN_module.coarse_dim}', f"{content_fname}-to-{style_fname}")
            save_image(output_image, output_path)


if __name__ == '__main__':
    contents_and_styles = [
        ('images/style_transfer/bin2.jpg', 'images/style_transfer/brick.jpg'),
        ('images/style_transfer/hillary1.jpg', 'images/style_transfer/thick_oil.jpg'),
        ('images/style_transfer/duck_mosaic.jpg', 'images/style_transfer/S_char.jpg'),
        ('images/style_transfer/S_char.jpg', 'images/style_transfer/duck_mosaic.jpg'),
        ('images/style_transfer/kanyon2.jpg', 'images/style_transfer/tower.jpg'),
        ('images/style_transfer/tower.jpg', 'images/style_transfer/kanyon2.jpg'),
        ('images/style_transfer/trump.jpg', 'images/style_transfer/mondrian.jpg'),
    ]
    style_transfer(contents_and_styles, )