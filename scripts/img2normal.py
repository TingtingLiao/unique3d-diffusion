checkpoint_path = "/media/mbzuai/Tingting/projects/unique3d-diffuser/unique3d_diffusion/ckpt/image2normal/unet_state_dict.pth"
trainer, pipeline = build_model("img2normal", checkpoint_path)
