import os
from time import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPNN import PNN, GPNN
from utils.image import save_image

def reshuffle(dataset_dir, out_dir, use_faiss, alpha):
    image_paths = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir)]
    PNN_moduel = PNN(patch_size=7,
                     stride=1,
                     alpha=alpha,
                     use_faiss=use_faiss)
    GPNN_module = GPNN(PNN_moduel,
                       scale_factor=(1, 1),
                       resize=0,
                       num_steps=10,
                       pyr_factor=0.75,
                       coarse_dim=14,
                       noise_sigma=0.75,
                       device="cuda:0")

    for im_path in image_paths:
        for i in range(5):
            start = time()
            output_image = GPNN_module.run(target_img_path=im_path, init_mode="target")
            print(f"took {time() - start} s")
            save_image(output_image, os.path.join(out_dir, f"{i}/{os.path.basename(im_path)}"))


if __name__ == '__main__':
    dataset_name = "Places50"
    # reshuffle(f'/home/ariel/university/GPDM/images/{dataset_name}', out_dir=f"outputs/reshuffle/{dataset_name}_target_faissIvf-50", use_faiss=True, alpha=1)
    reshuffle(f'/home/ariel/university/GPDM/images/{dataset_name}', out_dir=f"outputs/reshuffle/{dataset_name}_target_alpha=1", use_faiss=False, alpha=1)
    reshuffle(f'/home/ariel/university/GPDM/images/{dataset_name}', out_dir=f"outputs/reshuffle/{dataset_name}_target_alpha=0.005", use_faiss=False, alpha=0.005)
