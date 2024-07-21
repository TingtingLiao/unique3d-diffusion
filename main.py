from PIL import Image
import numpy as np
import torch
import os 
from rembg import remove
import torch.nn as nn

from unique3d_diffusion.model_zoo import build_model 
from lib.utils import simple_preprocess, remove_color, rotate_normals_torch, change_rgba_bg, rgba_to_rgb, session, make_image_grid, split_image, erode_alpha
from lib.sd_model_zoo import load_common_sd15_pipe
from lib.upsampler import RealESRGANer 
from diffusers import StableDiffusionControlNetImg2ImgPipeline

class Unique3dDiffuser(nn.Module):
    def __init__(self, ckpt_dir, seed, save_dir):
        super(Unique3dDiffuser, self).__init__()
        self.save_dir = save_dir
          
        # generator seed 
        self.generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed >= 0 else None

        self.mvimg_trainer, self.mvnml_pipeline = build_model("img2mvimg", f"{ckpt_dir}/img2mvimg/unet_state_dict.pth") 
        self.mvnml_trainer, self.mvnml_pipeline = build_model("img2normal", f"{ckpt_dir}/image2normal/unet_state_dict.pth") 
         
        # refine tile images 
        self.pipe_disney_controlnet_tile_ipadapter_i2i = load_common_sd15_pipe(
            base_model="runwayml/stable-diffusion-v1-5", 
            ip_adapter=True, 
            plus_model=False, 
            controlnet="ckpt/controlnet-tile", 
            pipeline_class=StableDiffusionControlNetImg2ImgPipeline
        ) 
        self.neg_prompt_list = ["sketch, sculpture, hand drawing, outline, single color, NSFW, lowres, bad anatomy,bad hands, text, error, missing fingers, yellow sleeves, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry,(worst quality:1.4),(low quality:1.4)"]
        self.prompt_list=["4views, multiview"]

        # upsampler
        self.upsampler = RealESRGANer(
            scale=4,
            onnx_path="ckpt/realesrgan-x4.onnx",
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=0,
        )
    
    @torch.no_grad()
    def img2mvimg(self, front_image, refine=False, guidance_scale=1.5): 
        img = rgba_to_rgb(front_image) if front_image.mode == 'RGBA' else front_image
        rgb_pils = self.mvimg_trainer.pipeline_forward(
            pipeline=self.mvnml_pipeline,
            image=img,
            guidance_scale=guidance_scale, 
            generator=self.generator,
            width=256,
            height=256,
            num_inference_steps=30,
        ).images 

        if refine:
            rgb_pil = make_image_grid(rgb_pils, rows=2)   
            control_image = rgb_pil.resize((1024, 1024)) 
            refined_rgbs = self.pipe_disney_controlnet_tile_ipadapter_i2i(
                image=[rgb_pil],
                ip_adapter_image=[rgba_to_rgb(front_image)],
                prompt=self.prompt_list,
                neg_prompt=self.neg_prompt_list,
                num_inference_steps=50,
                strength=0.2,
                height=1024,
                width=1024,
                control_image=[control_image],
                guidance_scale=5.0,
                controlnet_conditioning_scale=1.0,
                generator=torch.manual_seed(233),
            ).images[0] 
            refined_rgbs = split_image(refined_rgbs, rows=2)
            return refined_rgbs
        
        return rgb_pils

    def run_sr_fast(self, images):  
        return [Image.fromarray(self.upsampler.enhance(np.array(img), outscale=4)[0]) for img in images]
  

    def load_image(self, im_path): 
        input_image = Image.open(im_path) 
        
        if input_image.mode == 'RGB' or np.array(input_image)[..., -1].mean() == 255.:
            # still do remove using rembg, since simple_preprocess requires RGBA image
            print("RGB image not RGBA! still remove bg!")
            remove_bg = True

        if remove_bg: 
            input_image = remove(input_image, session=session)
        
        input_image = change_rgba_bg(input_image, "white")
        front_image = simple_preprocess(input_image)
        return front_image
    
    def img2normal(self, img_list, guidance_scale=2., do_rotate=True, num_inference_steps=30, **kwargs): 
        img_list = [rgba_to_rgb(i) if i.mode == 'RGBA' else i for i in img_list]
        
        # image-to-normal 
        images = self.mvnml_trainer.pipeline_forward(
            pipeline=self.mvnml_pipeline,
            image=img_list,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, 
            **kwargs
        ).images

        images = self.run_sr_fast(images)
        
        # remove bg  
        images = [Image.fromarray(remove_color(np.array(image)).astype(np.uint8)) for image in images]
        
        # rotate normal 
        if do_rotate and len(images) > 1:
            images = rotate_normals_torch(images, return_types='pil')
        return images

    def forward(self, im_path, remove_bg=False):
        im_name = os.path.basename(im_path).split(".")[0]  
        front_pil = self.load_image(im_path)
        rgb_pils = self.img2mvimg(front_pil, refine=True)

        # upscale rgbs
        if front_pil.size[0] <= 512:
            front_pil = self.run_sr_fast([front_pil])[0] 
        img_list = [front_pil] + self.run_sr_fast(rgb_pils[1:])

        print('predicting normal maps...')  
        mv_normals = self.img2normal([img.resize((512, 512), resample=Image.LANCZOS) for img in img_list], guidance_scale=1.5)

        # transfer front rgb alpha to mv_normals[0]
        mv_normals[0] = Image.fromarray(
            np.concatenate([
                np.array(mv_normals[0])[:, :, :3], 
                np.array(img_list[0].resize((2048, 2048)))[:, :, 3:4]
                ], axis=-1)
        ) 
        # transfer the alpha channel of mv_normals to img_list
        for idx, img in enumerate(mv_normals):
            if idx > 0: 
                img_list[idx] = Image.fromarray(np.concatenate([np.array(img_list[idx]), np.array(img)[:, :, 3:4]], axis=-1))
        assert img_list[0].mode == "RGBA"
        assert np.mean(np.array(img_list[0])[..., 3]) < 250
        
        img_list = [img_list[0]] + erode_alpha(img_list[1:])
        
        # save img_list and mv_normals
        normal_dir = os.path.join(self.save_dir, im_name, "normals")
        image_dir = os.path.join(self.save_dir, im_name, "images")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        for i, (img, normal) in enumerate(zip(img_list, mv_normals)):
            img.save(os.path.join(image_dir, f"{i}.png"))
            normal.save(os.path.join(normal_dir, f"{i}.png"))
             
        return img_list, mv_normals

 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="ckpt", help="path to the checkpoint")
    parser.add_argument("--img_dir", type=str, default=None, help="path to image dir")
    parser.add_argument("--img", type=str, default="data/image.png", help="path to the image")
    parser.add_argument("--seed", type=int, default=-1, help="random seed") 
    args = parser.parse_args()

    model = Unique3dDiffuser(args.ckpt_dir, args.seed, save_dir="output")
    if args.img_dir is None:  
        for im_file in sorted(os.listdir(args.img_dir)):
            model(os.path.join(args.img_dir, im_file))
    else:
        model(args.img)
 
    