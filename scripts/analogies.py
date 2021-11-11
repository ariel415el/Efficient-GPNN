import os
from time import time
from GPNN import PNN, GPNN
from utils.image import save_image

STYLE_DIR = '../images/style_transfer/style/'
CONTENT_DIR = '../images/style_transfer/content/'

contents_and_styles = [
    (f'{CONTENT_DIR}/bin2.jpg',f'{STYLE_DIR}brick.jpg'),
    (f'{CONTENT_DIR}/chicago.jpg', f'{STYLE_DIR}/starry_night.jpg'),
    (f'{CONTENT_DIR}/home_alone.jpg', f'{STYLE_DIR}/scream.jpg'),
    (f'{CONTENT_DIR}/hillary1.jpg', f'{STYLE_DIR}thick_oil.jpg'),
    (f'{CONTENT_DIR}/trump.jpg', f'{STYLE_DIR}mondrian.jpg'),
    (f'{CONTENT_DIR}/golden_gate.jpg', f'{STYLE_DIR}/scream.jpg'),
    (f'{CONTENT_DIR}/hotel_bedroom2.jpg', f'{STYLE_DIR}/Muse.jpg'),
    (f'{CONTENT_DIR}/cat1.jpg', f'{STYLE_DIR}/olive_Trees.jpg'),
    (f'{CONTENT_DIR}/cornell.jpg', f'{STYLE_DIR}/rug.jpeg'),
    (f'{CONTENT_DIR}/man1.jpg', f'{STYLE_DIR}/drawing.jpg'),
    (f'{CONTENT_DIR}/cornell.jpg', f'{STYLE_DIR}/rug.jpeg'),
    ('../images/analogies/duck_mosaic.jpg', '../images/analogies/S_char.jpg'),
    ('../images/analogies/S_char.jpg', '../images/analogies/duck_mosaic.jpg'),
    ('../images/analogies/kanyon2.jpg', '../images/analogies/tower.jpg'),
    ('../images/analogies/tower.jpg', '../images/analogies/kanyon2.jpg'),
]


if __name__ == '__main__':
    PNN_moduel = PNN(patch_size=11, stride=1, alpha=0.5, reduce_memory_footprint=True)
    GPNN_module = GPNN(PNN_moduel, scale_factor=(1, 1), resize=256, num_steps=10, pyr_factor=0.75, coarse_dim=200,
                       noise_sigma=0, device="cuda:0")

    out_dir = f"outputs/style_transfer"
    for (content_image_path, style_iamge_path) in contents_and_styles:
        style_fname, ext = os.path.splitext(os.path.basename(content_image_path))[:2]
        content_fname, _ = os.path.splitext(os.path.basename(style_iamge_path))[:2]
        for i in range(1):
            start = time()
            output_image = GPNN_module.run(target_img_path=style_iamge_path, init_mode=content_image_path)
            print(f"took {time() - start} s")
            save_image(output_image, os.path.join(out_dir, f"{content_fname}-to-{style_fname}${i}{ext}"))
