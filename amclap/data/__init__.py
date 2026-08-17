"""Training datamodules, imported on demand.

Resolved lazily so that importing `amclap.data` does not drag in every
datamodule's dependencies. The MSD one needs `pandas` and `datasets`, which ship
in the `[train]` extra, and pulling those in eagerly made the whole package
unimportable for anyone who installed only the inference dependencies.

`DATASETS` keeps its original keys and iteration behaviour: `amclap.train`
iterates it to register each name with gin, and the configs refer to those names.
"""

from collections.abc import Mapping
from importlib import import_module

# gin name -> (submodule, class name)
_DATASET_SPECS = {
    "discotube": (".discotube", "DiscotubeAudioDataModule"),
    "discotube_multiview": (".discotube", "DiscotubeMultiViewAudioDataModule"),
    "discotube_text_audio": (".discotube_text_audio", "DiscotubeTextAudioDataModule"),
    "discotube_text_audio_clean": (
        ".discotube_text_audio_clean",
        "DiscotubeTextAudioCleanDataModule",
    ),
    "discotube_structured_text_audio": (
        ".discotube_structured_text_audio",
        "DiscotubeStructuredTextAudioDataset",
    ),
    "freesound_text_audio": (".freesound_text_audio", "FreesoundTextAudioDataModule"),
    "concat_text_audio": (".concat_text_audio", "ConcatTextAudioDataModule"),
    "pse_text_audio": (".pse_text_audio", "PSETextAudioDataModule"),
    "msd_text_audio": (".msd_text_audio", "MSDTextAudioDataModule"),
    "dummy_text_audio": (".dummy_text_audio", "DummyTextAudioDataModule"),
    "m4rag_text_audio": (".m4rag_text_audio", "M4RAGTextAudioDataModule"),
}

_BY_CLASS_NAME = {cls: key for key, (_, cls) in _DATASET_SPECS.items()}


def _load(key: str):
    submodule, class_name = _DATASET_SPECS[key]
    return getattr(import_module(submodule, __name__), class_name)


class _LazyDatasets(Mapping):
    """Mapping of gin name -> datamodule class, imported on first access."""

    def __init__(self, specs):
        self._specs = specs
        self._cache = {}

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = _load(key)
        return self._cache[key]

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return len(self._specs)


DATASETS = _LazyDatasets(_DATASET_SPECS)

__all__ = ["DATASETS", *_BY_CLASS_NAME]


def __getattr__(name: str):
    """Expose the datamodules as attributes without importing them up front."""
    if name in _BY_CLASS_NAME:
        return _load(_BY_CLASS_NAME[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
