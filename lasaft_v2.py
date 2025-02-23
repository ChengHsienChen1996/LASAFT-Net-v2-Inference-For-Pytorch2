import hydra
import torch

from lasaft.load_pretrained import get_v2_large_709, get_mdx_light_v2_699


class LasaftV2:
    def __init__(self, use_gpu: bool = False, light_model:bool = False):
        self._use_gpu = use_gpu

        self._model = self.load_light_model() if light_model else self.load_large_model()

    def load_large_model(self):
        _model = get_v2_large_709(self._use_gpu)
        return _model

    def load_light_model(self):
        _model = get_mdx_light_v2_699(self._use_gpu)
        return _model

    def separate_vocals(self, target_voice) -> torch.Tensor:
        result = self._model.separate_tracks(target_voice, ['vocals', 'drums', 'bass', 'other'],
                                             overlap_ratio=0.5,
                                             batch_size=4)

        return torch.from_numpy(result["vocals"]).transpose(0, 1)

    @property
    def lasfat_model(self):
        return self._model


if __name__ == "__main__":
    model = LasaftV2(use_gpu=True, light_model=False).lasfat_model

    print(model)
