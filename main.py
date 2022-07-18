from os.path import join, basename, dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import GPNN
from utils.NN_modules import *
from utils.image import load_image, dump_images, get_pyramid_scales
import argparse

# IO
parser = argparse.ArgumentParser(description='Run GPDM')
parser.add_argument('reference_image', help="This image from which to copy patches")
parser.add_argument('--output_dir', default="Outputs", help="Where to put the results")
parser.add_argument('--debug_dir', default=None, help="If not None, debug images are dumped to this path")
parser.add_argument('--use_gpu', default=True)

# NN parameters
parser.add_argument('--patch_size', type=int, default=7)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--NN_type', default='Exact', help="Use one of 'Exact'/'Exact-low-memory'/'Faiss-IVF'")
parser.add_argument('--alpha', type=float, default=0.005, help="Relevant only for 'Exact' NN', "
                                                               "set to None in order to turn off use of alpha")

# Pyramids parameters
parser.add_argument('--fine_dim', type=int, default=None,
                    help="Height of the largest pyramid scale (can be used to get smaller output)."
                                                                "If None use the target_image height")
parser.add_argument('--coarse_dim', type=int, default=14,
                    help="Height of the smallest pyramid scale, When starting from noise,"
                    " bigger coarse dim lets the images outputs go more diverse (coarse_dim==~patch_size) "
                    "will probably output a copyof the input")
parser.add_argument('--pyr_factor', type=float, default=0.75, help="Downscale factor of the pyramid")
parser.add_argument('--height_factor', type=float, default=1., help="Controls the aspect ratio of the result: factor of height")
parser.add_argument('--width_factor', type=float, default=1., help="Controls the aspect ratio of the result: factor of width")

# GPNN parameters
parser.add_argument('--init_from', default='zeros', help="Defines the intial guess for the first level. Can one of ('zeros', 'target', '<path-to-image>')")
parser.add_argument('--noise_sigma', type=float, default=0.75, help="Std of noise added to the first initial image")
parser.add_argument('--num_iters', type=int, default=10, help="Number of iterations at each scale")
parser.add_argument('--initial_level_num_iters', type=int, default=1, help="Running too many iteration in the first"
                                                                 " level can lead to copying of the reference image")

args = parser.parse_args()


def get_nn_module(NN_type, alpha, use_gpu):
    if NN_type == "Exact":
        nn_module = PytorchNNLowMemory(alpha=alpha, use_gpu=use_gpu)
    elif args.NN_type == "Exact-low-memory":
        nn_module = PytorchNNLowMemory(alpha=alpha, use_gpu=use_gpu)
    else:
        if args.alpha is not None:
            raise ValueError("Can't use an alpha parameter with approximate nearest neighbor")
        nn_module = FaissIVF(use_gpu=args.use_gpu)

    return nn_module


if __name__ == '__main__':
    refernce_images = load_image(args.reference_image)

    nn_module = get_nn_module(args.NN_type, args.alpha, args.use_gpu)

    fine_dim = args.fine_dim if args.fine_dim is not None else refernce_images.shape[-2]

    new_image = GPNN.generate(refernce_images, nn_module,
                               pyramid_scales=get_pyramid_scales(fine_dim, args.coarse_dim, args.pyr_factor),
                               aspect_ratio=(args.height_factor, args.width_factor),
                               init_from=args.init_from,
                               num_iters=args.num_iters,
                               initial_level_num_iters=args.initial_level_num_iters,
                               keys_blur_factor=args.pyr_factor,
                               additive_noise_sigma=args.noise_sigma,
                               debug_dir=args.debug_dir,
                               device=torch.device("cuda:0" if args.use_gpu else "cpu")
    )
    dump_images(new_image, join(args.output_dir, basename(args.reference_image)))