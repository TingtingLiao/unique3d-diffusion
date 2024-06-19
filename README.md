# Unique3D Diffusion Models
This project contains the image-to-multiview diffusion and image-to-normal diffusion models from [Unique3D](https://github.com/AiuniAI/Unique3D).
![demo](https://github.com/TingtingLiao/unique3d-diffuser/assets/45743512/960c1a21-7972-4ea2-924e-a387773f2d47)


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
python3 -m scripts.img2mvimg --ckpt "ckpt/img2mvimg/unet_state_dict.pth" --img "data/belle.jpeg" 

# image to multi-view normal  
python3 -m scripts.img2normal --ckpt "ckpt/image2normal/unet_state_dict.pth" --img_dir "output/belle/images"
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