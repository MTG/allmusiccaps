"""Transformer-based text encoder wrappers for CLAP training."""

import os

import gin
import torch
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer
import transformers.tokenization_utils_base as tub


def _no_patch(*args, **kwargs):
    return args[0]


# Workaround for transformers bug: _patch_mistral_regex calls model_info()
# which requires network access even when loading from local files.
tub.PreTrainedTokenizerBase._patch_mistral_regex = staticmethod(_no_patch)


def mean_pooling(model_output, attention_mask):
    """Mean pooling over token embeddings with attention mask."""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class TransformerTextEncoder(Module):
    """Base class for transformer-based text encoders.

    Wraps HuggingFace AutoModel models with mean pooling for sentence embeddings.
    Provides a SentenceTransformer-like interface for compatibility with CLAP.
    """

    def __init__(
        self,
        model_id: str,
        tokenizers_parallelism: bool = False,
        local_files_only: bool = False,
        max_length: int = 512,
    ):
        super(TransformerTextEncoder, self).__init__()

        if tokenizers_parallelism:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
        self.max_length = max_length

        # Get embedding dimension from model config
        self._embed_dim = self._model.config.hidden_size

    @property
    def model(self):
        """Return self to provide SentenceTransformer-like interface."""
        return self

    def encode(self, sentences, convert_to_tensor=True, device=None):
        """Encode sentences to embeddings (SentenceTransformer-compatible interface)."""
        single_input = isinstance(sentences, str)
        if single_input:
            sentences = [sentences]

        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            # Move to same device as model
            model_device = next(self._model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        embeddings = mean_pooling(outputs, inputs["attention_mask"])

        # Match SentenceTransformer behavior: return 1D tensor for single input
        if single_input:
            embeddings = embeddings.squeeze(0)

        if not convert_to_tensor:
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def tokenize(self, sentences):
        """Tokenize sentences (for training with gradients)."""
        return self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, inputs):
        """Forward pass for training with gradients."""
        outputs = self._model(**inputs)
        embeddings = mean_pooling(outputs, inputs["attention_mask"])
        return {"sentence_embedding": embeddings}


@gin.configurable
class RoBERTaTextEncoder(TransformerTextEncoder):
    """RoBERTa-base text encoder.

    Reference: https://huggingface.co/roberta-base
    Embedding dimension: 768
    """

    def __init__(
        self,
        model_id: str = "roberta-base",
        tokenizers_parallelism: bool = False,
        local_files_only: bool = False,
        max_length: int = 512,
    ):
        super(RoBERTaTextEncoder, self).__init__(
            model_id=model_id,
            tokenizers_parallelism=tokenizers_parallelism,
            local_files_only=local_files_only,
            max_length=max_length,
        )


@gin.configurable
class XLMRobertaTextEncoder(TransformerTextEncoder):
    """XLM-RoBERTa-base text encoder (multilingual).

    Reference: https://huggingface.co/xlm-roberta-base
    Embedding dimension: 768
    """

    def __init__(
        self,
        model_id: str = "xlm-roberta-base",
        tokenizers_parallelism: bool = False,
        local_files_only: bool = False,
        max_length: int = 512,
    ):
        super(XLMRobertaTextEncoder, self).__init__(
            model_id=model_id,
            tokenizers_parallelism=tokenizers_parallelism,
            local_files_only=local_files_only,
            max_length=max_length,
        )


@gin.configurable
class ModernBERTTextEncoder(TransformerTextEncoder):
    """ModernBERT-base text encoder.

    Reference: https://huggingface.co/answerdotai/ModernBERT-base
    Embedding dimension: 768
    """

    def __init__(
        self,
        model_id: str = "answerdotai/ModernBERT-base",
        tokenizers_parallelism: bool = False,
        local_files_only: bool = False,
        max_length: int = 512,
    ):
        super(ModernBERTTextEncoder, self).__init__(
            model_id=model_id,
            tokenizers_parallelism=tokenizers_parallelism,
            local_files_only=local_files_only,
            max_length=max_length,
        )
