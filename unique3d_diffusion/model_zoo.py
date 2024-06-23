import os 
import torch  
from typing import List
from dataclasses import dataclass
# from app.utils import rgba_to_rgb
from .trainings.config_classes import ExprimentConfig, TrainerSubConfig
from . import modules
from .custum_modules.unifield_processor import AttnConfig, ConfigurableUNet2DConditionModel
from .trainings.base import BasicTrainer
from .trainings.utils import load_config



@dataclass
class FakeAccelerator:
    device: torch.device = torch.device("cuda")


def init_trainers(cfg_path: str, weight_dtype: torch.dtype, extras: dict, ckpt_path: str):
    accelerator = FakeAccelerator()
    cfg: ExprimentConfig = load_config(ExprimentConfig, cfg_path, extras)

    if 'img2mvimg' in ckpt_path:
        cfg.pretrained_model_name_or_path = os.path.dirname(ckpt_path)
        cfg.init_config.init_unet_path = os.path.dirname(ckpt_path) 
        cfg.trainers[0].trainer.pretrained_model_name_or_path = os.path.dirname(ckpt_path) 
    
    init_config: AttnConfig = load_config(AttnConfig, cfg.init_config)  
    configurable_unet = ConfigurableUNet2DConditionModel(init_config, weight_dtype)
    configurable_unet.enable_xformers_memory_efficient_attention()
    trainer_cfgs: List[TrainerSubConfig] = [load_config(TrainerSubConfig, trainer) for trainer in cfg.trainers]
    trainers: List[BasicTrainer] = [modules.find(trainer.trainer_type)(accelerator, None, configurable_unet, trainer.trainer, weight_dtype, i) for i, trainer in enumerate(trainer_cfgs)]
    return trainers, configurable_unet


def load_pipeline(config_path, ckpt_path, pipeline_filter=lambda x: True, weight_dtype = torch.bfloat16):
    training_config = config_path
    load_from_checkpoint = ckpt_path
    extras = []
    device = "cuda"
    trainers, configurable_unet = init_trainers(training_config, weight_dtype, extras, ckpt_path)
    shared_modules = dict() 
    for trainer in trainers:
        shared_modules = trainer.init_shared_modules(shared_modules)

    if load_from_checkpoint is not None:
        state_dict = torch.load(load_from_checkpoint)
        configurable_unet.unet.load_state_dict(state_dict, strict=False)
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    configurable_unet.unet.to(device, dtype=weight_dtype)

    pipeline = None
    trainer_out = None
    for trainer in trainers:
        if pipeline_filter(trainer.cfg.trainer_name):
            pipeline = trainer.construct_pipeline(shared_modules, configurable_unet.unet)
            pipeline.set_progress_bar_config(disable=False)
            trainer_out = trainer
    pipeline = pipeline.to(device)
    return trainer_out, pipeline


def build_model(model_name="img2mvimg", ckpt_path=None): 
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    if model_name == "img2mvimg": 
        # get the directory of the ckpt_path  
        training_config = f"{cur_dir}/configs/image2mvimage.yaml"
        return load_pipeline(training_config, ckpt_path)
    elif model_name == "img2normal": 
        training_config = f"{cur_dir}/configs/image2normal.yaml" 
        return load_pipeline(training_config, ckpt_path)
