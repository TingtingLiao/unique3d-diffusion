# Unique3D Diffusion Models
This project contains the image-to-multiview diffusion and image-to-normal diffusion models from [Unique3D](https://github.com/AiuniAI/Unique3D).
![demo](https://github.com/TingtingLiao/unique3d-diffuser/assets/45743512/960c1a21-7972-4ea2-924e-a387773f2d47)


## Install 
```bash 
pip install git+https://github.com/TingtingLiao/unique3d-diffusion.git 
```
or 
```
git clone https://github.com/TingtingLiao/unique3d-diffuser.git 
cd unique3d-diffuser
pip install -e .
```

## Usage 
Download unique3d models from [huggingface](https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt). 
```bash
# single image 
python3 demo.py --ckpt_dir ./ckpt --img data/disney/belle.jpeg 

# processing img_dir  
python3 demo.py --ckpt_dir ./ckpt --img_dir data/disney 
```
or 
```bash 
from unique3d_diffusion import Unique3dDiffuser

seed = 0 
ckpt_dir = "./ckpt"
save_dir = "./output"
model = Unique3dDiffuser(ckpt_dir, seed, save_dir)
images, normals = model(args.img, save=True) 
```

## Acknowledgement 

The code is adapted from [unique3d](https://github.com/AiuniAI/Unique3D). Please consiter cite: 
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