import sys
import os

from tqdm import tqdm
import torch
from torchvision.transforms import Resize as tv_resize


sys.path.append('.')
from utils.image import extract_patches, combine_patches, save_image, load_image, blur


def generate(reference_images,
             nn_module,
             patch_size=7,
             stride=1,
             init_from: str = 'zeros',
             pyramid_scales=(32, 64, 128, 256),
             aspect_ratio=(1, 1),
             additive_noise_sigma=0.0,
             num_iters: int = 10,
             initial_level_num_iters: int = 1,
             keys_blur_factor=1,
             device=torch.device("cpu"),
             debug_dir=None):
    """
    Run the GPNN model to generate an image using coarse to fine patch replacements.
    """
    logger = Logger(initial_level_num_iters, num_iters, len(pyramid_scales))

    reference_images = reference_images.to(device)
    synthesized_images = get_fist_initial_guess(reference_images, init_from, additive_noise_sigma).to(device)
    original_image_shape = synthesized_images.shape[-2:]

    for i, scale in enumerate(pyramid_scales):
        logger.new_lvl()
        lvl_references = tv_resize(scale, antialias=True)(reference_images)
        lvl_output_shape = get_output_shape(original_image_shape, scale, aspect_ratio)
        synthesized_images = tv_resize(lvl_output_shape, antialias=True)(synthesized_images)

        synthesized_images = replace_patches(synthesized_images, lvl_references, nn_module,
                                             patch_size,
                                             stride,
                                             initial_level_num_iters if i == 0 else num_iters,
                                             keys_blur_factor=keys_blur_factor,
                                             pbar=logger)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            save_image(lvl_references, os.path.join(debug_dir, f'references-lvl-{logger.lvl}.png'), normalize=True)
            save_image(synthesized_images, os.path.join(debug_dir, f'outputs-lvl-{logger.lvl}.png'), normalize=True)

    logger.pbar.close()
    return synthesized_images


def replace_patches(queries_image, values_image, nn_module, patch_size, stride, num_iters, keys_blur_factor=1, pbar=None):
    """
    Repeats n_steps iterations of repalcing the patches in "queries_image" by thier nearest neighbors from "values_image".
    The NN matrix is calculated with "keys" wich are a possibly blurred version of the patches from "values_image"
    :param values_image: The target patches to extract possible pathces or replacement
    :param queries_image: The synthesized image who's patches are to be replaced
    :param num_iters: number of repeated replacements for each patch
    :param keys_blur_factor: the factor with which to blur the values to get keys (image is downscaled and then upscaled with this factor)
    """
    keys_image = blur(values_image, keys_blur_factor)
    keys = extract_patches(keys_image, patch_size, stride)

    nn_module.init_index(keys)

    values = extract_patches(values_image, patch_size, stride)
    for i in range(num_iters):
        queries = extract_patches(queries_image, patch_size, stride)

        NNs = nn_module.search(queries)

        queries_image = combine_patches(values[NNs], patch_size, stride, queries_image.shape)
        if pbar:
            pbar.step()
            pbar.print()

    return queries_image


def get_output_shape(initial_image_shape, size, aspect_ratio):
    """Get the size of the output pyramid level"""
    h, w = initial_image_shape
    h, w = int(size * aspect_ratio[0]), int((w * size / h) * aspect_ratio[1])
    return h, w


def get_fist_initial_guess(reference_images, init_from, additive_noise_sigma):
    if init_from == "zeros":
        synthesized_images = torch.zeros_like(reference_images)
    elif init_from == "target":
        synthesized_images = reference_images.clone()
        import torchvision
        synthesized_images = torchvision.transforms.GaussianBlur(7, sigma=7)(synthesized_images)
    elif os.path.exists(init_from):
        synthesized_images = load_image(init_from)
    else:
        raise ValueError("Bad init mode", init_from)
    if additive_noise_sigma:
        synthesized_images += torch.randn_like(synthesized_images) * additive_noise_sigma

    return synthesized_images





class Logger:
    """Keeps track of the levels and steps of optimization. Logs it via TQDM"""

    def __init__(self, n_steps_first_level, n_steps, n_lvls):
        self.n_steps = n_steps
        self.n_lvls = n_lvls
        self.lvl = -1
        self.lvl_step = 0
        self.steps = 0
        self.pbar = tqdm(total=(self.n_lvls - 1) * self.n_steps + n_steps_first_level, desc='Starting')

    def step(self):
        self.pbar.update(1)
        self.steps += 1
        self.lvl_step += 1

    def new_lvl(self):
        self.lvl += 1
        self.lvl_step = 0

    def print(self):
        self.pbar.set_description(f'Lvl {self.lvl}/{self.n_lvls - 1}, step {self.lvl_step}/{self.n_steps}')
