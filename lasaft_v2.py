import hydra
import torch

from omegaconf import OmegaConf
from pathlib import Path
from pkg_resources import resource_filename
from safetensors.torch import load_file


class LasaftV2:
    def __init__(self, use_gpu: bool = False, light_model:bool = False):
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"

        self._model = self.load_light_model() if light_model else self.load_large_model()

    def load_large_model(self):
        conf_path = resource_filename('lasaft', 'pretrained/v2_large/')
        conf_path = Path(conf_path)
        ckpt_path = conf_path.joinpath('lasaft_v2_large_709.safetensors')

        with open(conf_path.joinpath('config.yaml')) as f:
            train_config = OmegaConf.load(f)
            model_config = train_config['model']

            model = hydra.utils.instantiate(model_config).to(self.device)

            try:
                loaded_state_dict = load_file(str(ckpt_path))
                model.load_state_dict(loaded_state_dict)

                print('checkpoint is loaded '.format(ckpt_path))
            except FileNotFoundError:
                print('FileNotFoundError.\n\t {} not exists\ntest mode'.format(ckpt_path))  # issue 10: fault tolerance

        return model

    def load_light_model(self):
        conf_path = resource_filename('lasaft', 'pretrained/v2_light/')
        conf_path = Path(conf_path)
        ckpt_path = conf_path.joinpath('lasaft_v2_light_669_for_mdx.safetensors')

        with open(conf_path.joinpath('config.yaml')) as f:
            train_config = OmegaConf.load(f)
            model_config = train_config['model']

            model = hydra.utils.instantiate(model_config).to('cpu')

            try:
                loaded_state_dict = load_file(str(ckpt_path))
                model.load_state_dict(loaded_state_dict)

                print('checkpoint is loaded.'.format(ckpt_path))
            except FileNotFoundError:
                print('FileNotFoundError.\n\t {} not exists\ntest mode'.format(ckpt_path))  # issue 10: fault tolerance

        return model

    @property
    def lasfat_model(self):
        return self._model


if __name__ == "__main__":
    model = LasaftV2(use_gpu=True, light_model=False).lasfat_model

    print(model)
