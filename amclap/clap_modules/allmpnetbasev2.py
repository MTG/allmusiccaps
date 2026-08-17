import os

import gin
from sentence_transformers import SentenceTransformer
from torch.nn import Module


@gin.configurable
class AllMPNetBaseV2(Module):
    def __init__(
        self,
        model_id: str,
        tokenizers_parallelism: bool = False,
        local_files_only: bool = False,
    ):
        super(AllMPNetBaseV2, self).__init__()

        if tokenizers_parallelism:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model = SentenceTransformer(model_id, local_files_only=local_files_only)
