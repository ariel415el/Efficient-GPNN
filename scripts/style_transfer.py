import os
from time import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPNN import PNN, GPNN
from utils.image import save_image

contents_and_styles = [
    ('images/style_transfer/bin2.jpg', 'images/style_transfer/brick.jpg'),
    ('images/style_transfer/hillary1.jpg', 'images/style_transfer/thick_oil.jpg'),
    ('images/style_transfer/duck_mosaic.jpg', 'images/style_transfer/S_char.jpg'),
    ('images/style_transfer/S_char.jpg', 'images/style_transfer/duck_mosaic.jpg'),
    ('images/style_transfer/kanyon2.jpg', 'images/style_transfer/tower.jpg'),
    ('images/style_transfer/tower.jpg', 'images/style_transfer/kanyon2.jpg'),
]


if __name__ == '__main__':
    PNN_moduel = PNN(patch_size=7, stride=1, alpha=0.005, reduce_memory_footprint=True)
    GPNN_module = GPNN(PNN_moduel, scale_factor=(1, 1), resize=256, num_steps=10, pyr_factor=0.75, coarse_dim=128,
                       noise_sigma=0, device="cuda:0")

    out_dir = f"outputs/style_transfer"
    for (content_image_path, style_iamge_path) in contents_and_styles:
        style_fname, ext = os.path.splitext(os.path.basename(content_image_path))[:2]
        content_fname, _ = os.path.splitext(os.path.basename(style_iamge_path))[:2]
        for i in range(1):
            start = time()
            output_image = GPNN_module.run(target_img_path=style_iamge_path, init_mode=content_image_path)
            print(f"took {time() - start} s")
            save_image(output_image, os.path.join(out_dir, f'{GPNN_module.resize}x{GPNN_module.pyr_factor}->{GPNN_module.coarse_dim}', f"{content_fname}-to-{style_fname}${i}{ext}"))
