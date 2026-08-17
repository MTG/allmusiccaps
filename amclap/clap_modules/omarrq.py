import gin
import gin.config as gin_config
from omar_rq import get_model
from torch.nn import Module
from typing import Optional, Set


@gin.configurable
class OMARRQ(Module):
    def __init__(
        self,
        model_id: str,
        load_weights: bool = True,
        layers: Optional[Set[int]] = None,
    ):
        super(OMARRQ, self).__init__()

        # Track keys before get_model() mutates gin config via gin.parse_config_files_and_bindings
        _keys_before = set(gin_config._CONFIG.keys())

        self.model = get_model(model_id=model_id, load_weights=load_weights)

        # Remove only the keys that omar_rq added, leaving existing entries untouched
        with gin.unlock_config():
            for k in set(gin_config._CONFIG.keys()) - _keys_before:
                del gin_config._CONFIG[k]
        self.layers = layers

        if hasattr(self.model, "embed_dim"):
            self.embed_dim = self.model.embed_dim

        if hasattr(self.model.net, "embed_dim"):
            self.embed_dim = self.model.net.embed_dim

        self.sr = self.model.sr = 24000
        self.patch_size = self.model.patch_size

        if self.layers is None:
            # By default, use the last layer (-1) for embedding extraction
            self.layers = set([-1])
        else:
            self.layers = layers
            print(f"OMAR-RQ: Using layers {self.layers}")

    def forward(self, input_values):
        # Average pooling over layers
        return self.model.extract_embeddings(input_values, layers=self.layers).mean(
            axis=0
        )
