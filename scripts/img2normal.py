from unique3d_diffusion.model_zoo import build_model 

checkpoint_path = "ckpt/image2normal/unet_state_dict.pth"
trainer, pipeline = build_model("img2normal", checkpoint_path)

# from diffusers import DiffusionPipeline
# import os
# model = "lambdalabs/sd-image-variations-diffusers"

# model = "~/.cache/huggingface/hub/models--lambdalabs--sd-image-variations-diffusers/snapshots/42bc0ee1726b141d49f519a6ea02ccfbf073db2e/"
# pipeline = DiffusionPipeline.from_pretrained(os.path.expanduser(model))