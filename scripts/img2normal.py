checkpoint_path = "ckpt/image2normal/unet_state_dict.pth"
trainer, pipeline = build_model("img2normal", checkpoint_path)
