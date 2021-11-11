import sys
import os
from typing import Tuple
import torch.nn.functional as F
import cv2
import torch
from torchvision import transforms

from utils.NN import get_patch_NNS_low_memory, get_patch_NNS

sys.path.append('.')
from utils.image import aspect_ratio_resize, get_pyramid, cv2pt, match_image_sizes, blur


class PNN:
    def __init__(self,
                 patch_size: int = 7,
                 stride: int = 1,
                 alpha: float = 0.005,
                 keys_scale_factor: float = 0.75,
                 reduce_memory_footprint: bool = True
                 ):
        """
        :param patch_size:
        :param stride:
        :param alpha:
        :param keys_scale_factor:
        :param reduce_memory_footprint:

        """
        self.patch_size = patch_size
        self.stride = stride
        self.alpha = alpha
        self.keys_scale_factor = keys_scale_factor
        self.reduce_memory_footprint = reduce_memory_footprint

    def replace_patches(self, values_image, queries_image, n_steps, blur_keys=True):
        keys_image = blur(values_image, self.keys_scale_factor) if blur_keys else values_image
        keys = extract_patches(keys_image, self.patch_size, self.stride)
        values = extract_patches(values_image, self.patch_size, self.stride)
        for i in range(n_steps):
            queries = extract_patches(queries_image, self.patch_size, self.stride)

            if self.reduce_memory_footprint:
                NNs = get_patch_NNS_low_memory(queries, keys, self.alpha)
            else:
                NNs = get_patch_NNS(queries, keys, self.alpha)

            queries_image = combine_patches(values[NNs], self.patch_size, self.stride, queries_image.shape)

        return queries_image

class GPNN:
    """An image generation model according to "Generating natural images with direct Patch Distributions Matching"""
    def __init__(self,
                    PNN_module,
                    scale_factor: Tuple[float, float] = (1., 1.),
                    resize: int = None,
                    num_steps: int = 10,
                    pyr_factor: float = 0.7,
                    coarse_dim: int = 32,
                    noise_sigma: float = 0.75,
                    device: str = 'cuda:0',
    ):
        """
        :param PNN_module:
        :param scale_factor: scale of the output in relation to input
        :param resize: max size of input image dimensions
        :param num_steps: number of PNN steps in each level
        :param pyr_factor: Downscale ratio of each pyramid level
        :param coarse_dim: minimal height for pyramid level
        :param noise_sigma: standard deviation of the zero mean normal noise added to the initialization
        :param device: cuda/cpu
        """
        self.PNN_module = PNN_module
        self.scale_factor = scale_factor
        self.resize = resize
        self.num_steps = num_steps
        self.pyr_factor = pyr_factor
        self.coarse_dim = coarse_dim
        self.noise_sigma = noise_sigma
        self.device = torch.device(device)

        self.name = f'R-{resize}_S-{pyr_factor}->{coarse_dim}+I(0,{noise_sigma})'

    def _build_pyramid(self, cv_img):
        """Reads an image and create a pyraimd out of it. Ordered in increasing image size"""
        if self.resize:
            cv_img = aspect_ratio_resize(cv_img, max_dim=self.resize)
        pt_img = cv2pt(cv_img)
        pt_pyramid = get_pyramid(pt_img, self.coarse_dim, self.pyr_factor)
        pt_pyramid = [x.unsqueeze(0).to(self.device) for x in pt_pyramid]
        return pt_pyramid

    def get_synthesis_size(self, lvl):
        """Get the size of the output pyramid level"""
        lvl_img = self.target_pyramid[lvl]
        h, w = lvl_img.shape[-2:]
        h, w = int(h * self.scale_factor[0]), int(w * self.scale_factor[1])
        return h, w

    def _get_initial_image(self, mode):
        """Prepare the initial image for optimization"""
        target_img = self.target_pyramid[-1]
        h, w = self.get_synthesis_size(lvl=0)
        if os.path.exists(mode):
            initial_iamge = cv2pt(cv2.imread(mode)).unsqueeze(0)
            initial_iamge = match_image_sizes(initial_iamge, target_img)
            initial_iamge = transforms.Resize((h, w), antialias=True)(initial_iamge).to(self.device)
        elif mode == 'target':
            initial_iamge = transforms.Resize((h, w), antialias=True)(target_img)
        else:
            initial_iamge = torch.zeros(1, 3, h, w).to(self.device)

        initial_iamge = initial_iamge

        if self.noise_sigma > 0:
            initial_iamge += torch.normal(0, self.noise_sigma, size=(h, w)).reshape(1, 1, h, w).to(self.device)

        return initial_iamge.to(self.device)

    def run(self, target_img_path, init_mode):
        """
        Run the GPDM model to generate an image with a similar patch distribution to target_img_path with a given criteria.
        This manages the coarse to fine optimization steps.
        """
        self.target_pyramid = self._build_pyramid(cv2.imread(target_img_path))
        self.synthesized_image = self._get_initial_image(init_mode)

        for lvl, lvl_target_img in enumerate(self.target_pyramid):
            if lvl > 0:
                h, w = self.get_synthesis_size(lvl=lvl)
                self.synthesized_image = transforms.Resize((h, w), antialias=True)(self.synthesized_image)

            self.synthesized_image = self.PNN_module.replace_patches(values_image=self.target_pyramid[lvl],
                                                         queries_image=self.synthesized_image,
                                                         n_steps=self.num_steps if lvl > 0 else 1,
                                                         blur_keys=lvl>0)

        return self.synthesized_image


def get_synthesis_size(lvl_size, scale_factor):
    """Get the size of the output pyramid level"""
    h, w = int(lvl_size[-2] * scale_factor[0]), int(lvl_size[-1] * scale_factor[1])
    return lvl_size[:-2] + (h, w)


def extract_patches(src_img, patch_size, stride):
    channels = 3
    return F.unfold(src_img, kernel_size=patch_size, dilation=(1, 1), stride=stride, padding=(0, 0)) \
        .squeeze(dim=0).permute((1, 0)).reshape(-1, channels, patch_size, patch_size)


def combine_patches(O, patch_size, stride, img_shape):
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
