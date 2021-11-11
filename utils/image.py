import os

import cv2
import torch
from torch.nn import functional as F
from torchvision import transforms
import torchvision.utils


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(torch.clip(img, -1, 1), path, normalize=True)


def cv2pt(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()

    return img


def aspect_ratio_resize(img, max_dim=256):
    y, x, c = img.shape
    if x > y:
        return cv2.resize(img, (max_dim, int(y/x*max_dim)))
    else:
        return cv2.resize(img, (int(x/y*max_dim), max_dim))


def downscale(img, pyr_factor):
    assert 0 < pyr_factor < 1
    new_w = int(pyr_factor * img.shape[-1])
    new_h = int(pyr_factor * img.shape[-2])

    return transforms.Resize((new_h, new_w), antialias=True)(img)


def blur(img, pyr_factor):
    """Blur image by downscaling and then upscaling it back to original size"""
    assert pyr_factor <= 1
    if pyr_factor < 1:
        d_img = downscale(img, pyr_factor)
        img = transforms.Resize(img.shape[-2:], antialias=True)(d_img)
    return img


def get_pyramid(img, min_height, pyr_factor, device):
    res = [img]
    while True:
        img = downscale(img, pyr_factor)
        if img.shape[-2] < min_height:
            break
        res = [img] + res

    # ensure smallest size is of min_height
    if res[0].shape[-2] != min_height:
        new_width = int(min_height * res[0].shape[-1] / float(res[0].shape[-2]))
        res[0] = transforms.Resize((min_height, new_width), antialias=True)(res[0])

    res = [x.unsqueeze(0).to(device) for x in res]
    return res


def match_image_sizes(input, target):
    """resize and crop input image so that it has the same aspect ratio as target"""
    assert(len(input.shape) == len(target.shape) and len(target.shape) == 4)
    input_h, input_w = input.shape[-2:]
    target_h, target_w = target.shape[-2:]
    input_scale_factor = input_h / input_w
    target_scale_factor = target_h / target_w
    if target_scale_factor > input_scale_factor:
        input = transforms.Resize((target_h, int(input_w/input_h*target_h)), antialias=True)(input)
        pixels_to_cut = input.shape[-1] - target_w
        if pixels_to_cut > 0:
            input = input[:, :, :, int(pixels_to_cut / 2):-int(pixels_to_cut / 2)]

    else:
        input = transforms.Resize((int(input_h/input_w*target_w), target_w), antialias=True)(input)
        pixels_to_cut = input.shape[-2] - target_h
        if pixels_to_cut > 1:
            input = input[:, :, int(pixels_to_cut / 2):-int(pixels_to_cut / 2)]

    input = transforms.Resize(target.shape[-2:], antialias=True)(input)

    return input


def extract_patches(src_img, patch_size, stride):
    channels = 3
    return F.unfold(src_img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) \
        .squeeze(dim=0).permute((1, 0)).reshape(-1, channels, patch_size, patch_size)


def combine_patches(O, patch_size, stride, img_shape):
    """Combines patches into an image by averaging overlapping pixels"""
    channels = 3
    O = O.permute(1, 0, 2, 3).unsqueeze(0)
    patches = O.contiguous().view(O.shape[0], O.shape[1], O.shape[2], -1) \
        .permute(0, 1, 3, 2) \
        .contiguous().view(1, channels * patch_size * patch_size, -1)
    combined = F.fold(patches, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    # normal fold matrix
    input_ones = torch.ones(img_shape, dtype=O.dtype, device=O.device)
    divisor = F.unfold(input_ones, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0))
    divisor = F.fold(divisor, output_size=img_shape[-2:], kernel_size=patch_size, stride=stride)

    divisor[divisor == 0] = 1.0
    return (combined / divisor).squeeze(dim=0).unsqueeze(0)