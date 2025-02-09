import hydra
import torch
from omegaconf import OmegaConf
from pathlib import Path
from pkg_resources import resource_filename
from safetensors.torch import load_file


def get_v2_large_709(use_gpu: bool=False):
    conf_path = resource_filename('lasaft', 'pretrained/v2_large/')
    conf_path = Path(conf_path)
    ckpt_path = conf_path.joinpath('lasaft_v2_large_709.safetensors')
    device = "cuda" if use_gpu else "cpu"
    with open(conf_path.joinpath('config.yaml')) as f:
        train_config = OmegaConf.load(f)
        model_config = train_config['model']

        model = hydra.utils.instantiate(model_config).to(device)

        try:
            loaded_state_dict = load_file(str(ckpt_path))
            model.load_state_dict(loaded_state_dict)

            print('checkpoint is loaded '.format(ckpt_path))
        except FileNotFoundError:
            print('FileNotFoundError.\n\t {} not exists\ntest mode'.format(ckpt_path))  # issue 10: fault tolerance

    return model


def get_mdx_light_v2_699(use_gpu: bool=False):
    conf_path = resource_filename('lasaft', 'pretrained/v2_light/')
    conf_path = Path(conf_path)
    ckpt_path = conf_path.joinpath('lasaft_v2_light_669_for_mdx.safetensors')
    device = "cuda" if use_gpu else "cpu"
    with open(conf_path.joinpath('config.yaml')) as f:
        train_config = OmegaConf.load(f)
        model_config = train_config['model']

        model = hydra.utils.instantiate(model_config).to(device)

        try:
            loaded_state_dict = load_file(str(ckpt_path))
            model.load_state_dict(loaded_state_dict)

            print('checkpoint is loaded.'.format(ckpt_path))
        except FileNotFoundError:
            print('FileNotFoundError.\n\t {} not exists\ntest mode'.format(ckpt_path))  # issue 10: fault tolerance

    return model
