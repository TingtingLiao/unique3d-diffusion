# Unique3D Diffusion Models


## Install 
```bash 
pip install git+https://github.com/TingtingLiao/unique3d-diffuser.git 
```
or 
```
git clone https://github.com/TingtingLiao/unique3d-diffuser.git 
cd unique3d-diffuser
pip install -e .
```

## Usage 
```bash
# image to multi-view image 
python3 -m scripts.img2mvimg --ckpt {path-to-img2mvimg-pth} --img "data/image.png" 

# image to multi-view normal  
python3 -m scripts.img2normal 

# upsampling 

```
**Load model**
```bash 
# img2mvimg 
from unique3d_diffusion.model_zoo import build_model 
checkpoint_path = "ckpt/img2mvimg/unet_state_dict.pth"
trainer, pipeline = build_model("img2mvimg", checkpoint_path)

# img2normal 
from unique3d_diffusion.model_zoo import build_model 
checkpoint_path = "ckpt/image2normal/unet_state_dict.pth"
trainer, pipeline = build_model("img2normal", checkpoint_path)
```

**Inference**
```bash 
seed = 100 
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
```

## Acknowledgement 
The original paper: 
```
@misc{wu2024unique3d,
      title={Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image}, 
      author={Kailu Wu and Fangfu Liu and Zhihan Cai and Runjie Yan and Hanyang Wang and Yating Hu and Yueqi Duan and Kaisheng Ma},
      year={2024},
      eprint={2405.20343},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```