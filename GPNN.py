import sys
import os
from typing import Tuple
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

from utils.image import save_image

from utils.NN import get_NN_indices_low_memory, get_NN_indices, get_NN_indices_faiss

sys.path.append('.')
from utils.image import aspect_ratio_resize, get_pyramid, cv2pt, match_image_sizes, blur, extract_patches, \
    combine_patches

class logger:
    """Keeps track of the levels and steps of optimization. Logs it via TQDM"""
    def __init__(self, n_steps, n_lvls):
        self.n_steps = n_steps
        self.n_lvls = n_lvls
        self.lvl = -1
        self.lvl_step = 0
        self.steps = 0
        self.pbar = tqdm(total=self.n_lvls * self.n_steps, desc='Starting')

    def step(self):
        self.pbar.update(1)
        self.steps += 1
        self.lvl_step += 1

    def new_lvl(self):
        self.lvl += 1
        self.lvl_step = 0

    def print(self):
        # pass
        self.pbar.set_description(f'Lvl {self.lvl}/{self.n_lvls}, step {self.lvl_step}/{self.n_steps}')

    def close(self):
        self.pbar.close()

class PNN:
    """Patch nearest neighbor module. Replaces the patches of an initial guess by their nearest neighbor from a target image"""
    def __init__(self,
                 patch_size: int = 7,
                 stride: int = 1,
                 alpha: float = 0.005,
                 reduce_memory_footprint: bool = True,
                 use_faiss=False,
                 batch_size: int = 512,
                 ):
        """
        :param patch_size: size of the patches to be replaced
        :param stride: stride in which the patches are extarcted
        :param alpha: 'alpha parameter of the normalizing distance matrix. small alpha encourages completeness. default is 0.005 (float)'
        :param reduce_memory_footprint:
        :param batch_size: batch size for computing the NNs

        """
        self.patch_size = patch_size
        self.stride = stride
        self.alpha = alpha
        self.reduce_memory_footprint = reduce_memory_footprint
        self.use_faiss = use_faiss
        self.batch_size = batch_size

    def replace_patches(self, values_image, queries_image, n_steps, keys_blur_factor=1, logger=None):
        """
        Repeats n_steps iterations of repalcing the patches in "queries_image" by thier nearest neighbors from "values_image".
        The NN matrix is calculated with "keys" wich are a possibly blurred version of the patches from "values_image"
        :param values_image: The target patches to extract possible pathces or replacement
        :param queries_image: The synthesized image who's patches are to be replaced
        :param n_steps: number of repeated replacements for each patch
        :param keys_blur_factor: the factor with which to blur the values to get keys (image is downscaled and then upscaled with this factor)
        """
        keys_image = blur(values_image, keys_blur_factor)
        keys = extract_patches(keys_image, self.patch_size, self.stride)
        values = extract_patches(values_image, self.patch_size, self.stride)
        for i in range(n_steps):
            queries = extract_patches(queries_image, self.patch_size, self.stride)

            if self.use_faiss:
                NNs = get_NN_indices_faiss(queries, keys)

            elif self.reduce_memory_footprint:
                NNs = get_NN_indices_low_memory(queries, keys, self.alpha, b=self.batch_size)
            else:
                NNs = get_NN_indices(queries, keys, self.alpha, b=self.batch_size)

            queries_image = combine_patches(values[NNs], self.patch_size, self.stride, queries_image.shape)
            if logger:
                logger.step()
                logger.print()

        return queries_image

class GPNN:
    """An image generation model that runs a PNN module in a multi-scale setting as described in the paper "Drop-The-Gan"""
    def __init__(self,
                    PNN_module: PNN,
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

    def _process_target_image(self, np_img):
        """Create a pyraimd of pytorch tensors from a np image. Ordered in increasing image size"""
        if self.resize:
            np_img = aspect_ratio_resize(np_img, max_dim=self.resize)
        pt_img = cv2pt(np_img)
        pt_pyramid = get_pyramid(pt_img, self.coarse_dim, self.pyr_factor, self.device)
        return pt_pyramid

    def _get_synthesis_size(self, lvl):
        """Get the size of the output pyramid at a specific level"""
        lvl_img = self.target_pyramid[lvl]
        h, w = lvl_img.shape[-2:]
        h, w = int(h * self.scale_factor[0]), int(w * self.scale_factor[1])
        return h, w

    def _get_initial_image(self, init_mode):
        """
        Prepare the initial image for the synthesis
        :param init_mode: <image_path>: start from an image specified the path.
                          Target: start from the target image (at the according pyramid level)
                          O.W: Start from Zero image
        Note: pixel-level white Noise is injected to initial image. Control its intesity by changing self.noise_sigma.
        noise_sigma==0 means no noise.

        """
        target_img = self.target_pyramid[-1]
        h, w = self._get_synthesis_size(lvl=0)
        if os.path.exists(init_mode):
            # Read an image as the input and match its size to the
            initial_iamge = cv2pt(cv2.imread(init_mode)).unsqueeze(0)
            initial_iamge = match_image_sizes(initial_iamge, target_img)
            initial_iamge = transforms.Resize((h, w), antialias=True)(initial_iamge).to(self.device)
        elif init_mode == 'target':
            initial_iamge = transforms.Resize((h, w), antialias=True)(target_img)
        else:
            initial_iamge = torch.zeros(1, 3, h, w).to(self.device)

        initial_iamge = initial_iamge

        if self.noise_sigma > 0:
            initial_iamge += torch.normal(0, self.noise_sigma, size=(h, w)).reshape(1, 1, h, w).to(self.device)

        return initial_iamge.to(self.device)

    def run(self, target_img_path, init_mode, debug_dir=None):
        """
        Run the GPNN model to generate an image with a similar patch distribution to target_img_path.
        This manages the coarse to fine NN steps.
        :param target_img_path: path to a target image to match patches with
        :param init_mode: Intialization mode for the process. (<patch to image> / target / noise)
        """
        self.target_pyramid = self._process_target_image(cv2.imread(target_img_path))
        self.synthesized_image = self._get_initial_image(init_mode)
        self.logger = logger(self.num_steps, len(self.target_pyramid))

        for lvl, lvl_target_img in enumerate(self.target_pyramid):
            self.logger.new_lvl()
            if lvl > 0:
                h, w = self._get_synthesis_size(lvl=lvl)
                self.synthesized_image = transforms.Resize((h, w), antialias=True)(self.synthesized_image)


            lvl_output = self.PNN_module.replace_patches(values_image=self.target_pyramid[lvl],
                                                         queries_image=self.synthesized_image,
                                                         n_steps=self.num_steps if lvl > 0 else 1,
                                                         keys_blur_factor=self.pyr_factor if lvl > 0 else 1,
                                                         logger=self.logger)
            if debug_dir:
                save_image(self.synthesized_image, f"{debug_dir}/input{lvl}.png")
                save_image(self.target_pyramid[lvl], f"{debug_dir}/target{lvl}.png")
                save_image(lvl_output, f"{debug_dir}/output{lvl}.png")

            self.synthesized_image = lvl_output

        self.logger.close()
        return self.synthesized_image
