from PIL import Image
import numpy as np
import torch
import os 
from rembg import remove, new_session

from unique3d_diffusion.model_zoo import build_model 
  
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 8 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'HEURISTIC',
    })
] 
session = new_session(providers=providers) 

def rgba_to_rgb(rgba: Image.Image, bkgd="WHITE"):
    new_image = Image.new("RGBA", rgba.size, bkgd)
    new_image.paste(rgba, (0, 0), rgba)
    new_image = new_image.convert('RGB')
    return new_image

def change_rgba_bg(rgba: Image.Image, bkgd="WHITE"):
    rgb_white = rgba_to_rgb(rgba, bkgd)
    new_rgba = Image.fromarray(np.concatenate([np.array(rgb_white), np.array(rgba)[:, :, 3:4]], axis=-1))
    return new_rgba


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def simple_preprocess(input_image, rembg_session=session, background_color=255):
    RES = 2048
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if input_image.mode != 'RGBA':
        image_rem = input_image.convert('RGBA')
        input_image = remove(image_rem, alpha_matting=False, session=rembg_session)

    arr = np.asarray(input_image)
    alpha = np.asarray(input_image)[:, :, -1]
    x_nonzero = np.nonzero((alpha > 60).sum(axis=1))
    y_nonzero = np.nonzero((alpha > 60).sum(axis=0))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    arr = arr[x_min: x_max, y_min: y_max]
    input_image = Image.fromarray(arr)
    input_image = expand2square(input_image, (background_color, background_color, background_color, 0))
    return input_image


def run_mvprediction(trainer, pipeline, image_path: str, remove_bg=True, guidance_scale=1.5, seed=1145):
    input_image = Image.open(image_path) 
    if input_image.mode == 'RGB' or np.array(input_image)[..., -1].mean() == 255.:
        # still do remove using rembg, since simple_preprocess requires RGBA image
        print("RGB image not RGBA! still remove bg!")
        remove_bg = True

    if remove_bg: 
        input_image = remove(input_image, session=session)

    # make front_pil RGBA with white bg
    input_image = change_rgba_bg(input_image, "white")
    single_image = simple_preprocess(input_image)

    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed >= 0 else None 
    img = rgba_to_rgb(single_image) if single_image.mode == 'RGBA' else single_image
    rgb_pils = trainer.pipeline_forward(
        pipeline=pipeline,
        image=img,
        guidance_scale=guidance_scale, 
        generator=generator,
        width=256,
        height=256,
        num_inference_steps=30,
    ).images
 
    return rgb_pils, single_image 

 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpt/img2mvimg/unet_state_dict.pth", help="path to the checkpoint")
    parser.add_argument("--img", type=str, default="data/image.png", help="path to the image")
    args = parser.parse_args()

    # checkpoint_path = "ckpt/img2mvimg/unet_state_dict.pth"
    trainer, pipeline = build_model("img2mvimg", args.ckpt) 
    rgb_pils, single_image = run_mvprediction(trainer, pipeline, args.img, remove_bg=True, guidance_scale=1.5, seed=1145)

    # concate images
    width, height = rgb_pils[0].size  
    combined_image = Image.new('RGB', (width * len(rgb_pils), height))
    for i, img in enumerate(rgb_pils):
        combined_image.paste(img, (i * width, 0))

    combined_image.save("output.png")
   
 