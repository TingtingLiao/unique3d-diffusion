from PIL import Image
import numpy as np
import torch
import os 
from rembg import remove
import torch.nn as nn

from unique3d_diffusion import Unique3dDiffuser
 
 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="path to the checkpoint")
    parser.add_argument("--img_dir", type=str, default=None, help="path to image dir")
    parser.add_argument("--img", type=str, default="data/image.png", help="path to the image")
    parser.add_argument("--seed", type=int, default=-1, help="random seed") 
    args = parser.parse_args()

    model = Unique3dDiffuser(args.ckpt_dir, args.seed, save_dir="output")
    if args.img_dir is not None:  
        for im_file in sorted(os.listdir(args.img_dir)):
            image = model.load_image(os.path.join(args.img_dir, im_file))
            model(image)
    else:
        image = model.load_image(args.img)
        model(image)
    
    