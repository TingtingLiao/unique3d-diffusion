from PIL import Image
import numpy as np
import torch
import os 
from rembg import remove

from unique3d_diffusion.model_zoo import build_model 
from lib.utils import simple_preprocess, change_rgba_bg, rgba_to_rgb, session
  

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
    parser.add_argument("--img_dir", type=str, default=None, help="path to image dir")
    parser.add_argument("--img", type=str, default="data/image.png", help="path to the image")
    parser.add_argument("--seed", type=int, default=-1, help="random seed")
    args = parser.parse_args()
 
    if args.img_dir is None: 
        images = [args.img]
    else:
        images = [os.path.join(args.img_dir, img) for img in os.listdir(args.img_dir)]

    for im_path in images: 
        img_name = os.path.basename(im_path).split(".")[0]
        # if os.path.exists(f"output/{img_name}/images.png"):
        #     continue

        print("processing ", im_path)
        # checkpoint_path = "ckpt/img2mvimg/unet_state_dict.pth"
        trainer, pipeline = build_model("img2mvimg", args.ckpt) 
        rgb_pils, single_image = run_mvprediction(trainer, pipeline, im_path, remove_bg=True, guidance_scale=1.5, seed=args.seed)

        os.makedirs(f"output/{img_name}/images", exist_ok=True)
        for i, img in enumerate(rgb_pils):
            img.save(f"output/{img_name}/images/{i}.png")
        # concate images
        width, height = rgb_pils[0].size  
        combined_image = Image.new('RGB', (width * len(rgb_pils), height))
        for i, img in enumerate(rgb_pils):
            combined_image.paste(img, (i * width, 0))
        combined_image.save(f"output/{img_name}/images.png")
   
    