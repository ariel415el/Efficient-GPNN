import os

import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image
import torch.nn.functional as F


def load_image(path):
    return cv2pt(cv2.imread(path))


def dump_images(images, out_dir):
    if os.path.exists(out_dir):
        i = len(os.listdir(out_dir))
    else:
        i = 0
        os.makedirs(out_dir)
    for j in range(images.shape[0]):
        save_image(images[j], os.path.join(out_dir, f"{i}.png"), normalize=True)
        i += 1


def get_pyramid_scales(max_height, min_height, step):
    cur_scale = max_height
    scales = [cur_scale]
    while cur_scale > min_height:
        if type(step) == float:
            cur_scale = int(cur_scale * step)
        else:
            cur_scale -= step
        scales.append(cur_scale)

    return scales[::-1]


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    return img


def downscale(img, pyr_factor):
    assert 0 < pyr_factor < 1
    new_w = int(pyr_factor * img.shape[-1])
    new_h = int(pyr_factor * img.shape[-2])
    return transforms.Resize((new_h, new_w), antialias=True)(img)


def blur(img, pyr_factor):
    """Blur image by downscaling and then upscaling it back to original size"""
    if pyr_factor < 1:
        d_img = downscale(img, pyr_factor)
        img = transforms.Resize(img.shape[-2:], antialias=True)(d_img)
    return img

def extract_patches(src_img, patch_size, stride):
    """
    Splits the image to overlapping patches and returns a pytorch tensor of size (N_patches, 3*patch_size**2)
    """
    channels = src_img.shape[1]
    patches = F.unfold(src_img, kernel_size=patch_size, stride=stride) # shape (b, 3*p*p, N_patches)
    patches = patches.squeeze(dim=0).permute((1, 0)).reshape(-1, channels * patch_size**2)
    return patches


def combine_patches(patches, patch_size, stride, img_shape):
    """
    Combines patches into an image by averaging overlapping pixels
    :param patches: patches to be combined. pytorch tensor of shape (N_patches, 3*patch_size**2)
    :param img_shape: an image of a shape that if split into patches with the given stride and patch_size will give
                      the same number of patches N_patches
    returns an image of shape img_shape
    """
    patches = patches.permute(1,0).unsqueeze(0)
    combined = F.fold(patches, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    # normal fold matrix
    input_ones = torch.ones(img_shape, dtype=patches.dtype, device=patches.device)
    divisor = F.unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    divisor = F.fold(divisor, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    divisor[divisor == 0] = 1.0
    return (combined / divisor).squeeze(dim=0).unsqueeze(0)
