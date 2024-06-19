import sys
from PIL import Image
from typing import List

# sys.path.append("..")
from unique3d_diffusion.model_zoo import build_model 
from lib.utils import rgba_to_rgb, simple_remove, rotate_normals_torch, change_rgba_bg

 
def predict_normals(trainer, pipeline, image: List, guidance_scale=2., do_rotate=True, num_inference_steps=30, **kwargs):
    img_list = image if isinstance(image, list) else [image]
    img_list = [rgba_to_rgb(i) if i.mode == 'RGBA' else i for i in img_list]
    images = trainer.pipeline_forward(
        pipeline=pipeline,
        image=img_list,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, 
        **kwargs
    ).images
    images = simple_remove(images)
    if do_rotate and len(images) > 1:
        images = rotate_normals_torch(images, return_types='pil')
    return images

if __name__ == "__main__":
    import os 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ckpt/image2normal/unet_state_dict.pth", help="path to the checkpoint")
    parser.add_argument("--img_dir", type=str, default="output/image/rgb", help="path to the image")
    args = parser.parse_args()

    trainer, pipeline = build_model("img2normal", args.ckpt) 

    image_files = sorted(os.listdir(args.img_dir))
    images = [Image.open(os.path.join(args.img_dir, image_file)) for image_file in image_files]
    normal_pils = predict_normals(trainer, pipeline, images)

    save_dir = os.path.join(os.path.dirname(args.img_dir), "normals")
    os.makedirs(save_dir, exist_ok=True)
    for i, normal in enumerate(normal_pils):
        normal.save(f"{save_dir}/{i}.png")
    
    # width, height = normal_pils[0].size  
    # combined_image = Image.new('RGB', (width * len(normal_pils), height))
    # for i, img in enumerate(normal_pils):
    #     img = rgba_to_rgb(img, "white")
    #     combined_image.paste(img, (i * width, 0))
    # combined_image.save(f"{os.path.dirname(args.img_dir)}/normals.png")