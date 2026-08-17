"""CLAP model variants, imported on demand.

Modules are resolved lazily so that `import amclap` does not pull in every
research variant. That matters for two reasons: the baseline encoders (MERT,
HTSAT, MAEST) need heavy optional dependencies, and `LeJEPA`/`ClapJepa` import
the `lejepa` package, which is only available from git and therefore ships as
the `[train]` extra rather than a core dependency.

`MODULES` keeps its original keys and iteration behaviour, because
`amclap.def_module` iterates it to register every name with
`gin.external_configurable`. The published `config.gin` files reference those
names unqualified (`@CLAP`, `@OMARRQ`, `@AllMPNetBaseV2`), so the keys and the
registered class names must not change.
"""

from collections.abc import Mapping
from importlib import import_module

# gin name -> (submodule, class name)
_MODULE_SPECS = {
    "clap": (".clap", "CLAP"),
    "omar_rq": (".omarrq", "OMARRQ"),
    "all_mpnet_base_v2": (".allmpnetbasev2", "AllMPNetBaseV2"),
    "clap_jepa": (".clap_jepa", "ClapJepa"),
    "lejepa": (".lejepa", "LeJEPA"),
    "slap": (".slap", "SLAP"),
    "lpjepa": (".lpjepa", "LpJEPA"),
    "mert": (".mert", "MERT"),
    "htsat": (".htsat", "HTSAT"),
    "maest": (".maest", "MAEST"),
    "roberta": (".transformer_text_encoder", "RoBERTaTextEncoder"),
    "xlm_roberta": (".transformer_text_encoder", "XLMRobertaTextEncoder"),
    "modern_bert": (".transformer_text_encoder", "ModernBERTTextEncoder"),
}

# Class name -> gin name, for attribute access (`clap_modules.CLAP`).
_BY_CLASS_NAME = {cls: key for key, (_, cls) in _MODULE_SPECS.items()}


def _load(key: str):
    submodule, class_name = _MODULE_SPECS[key]
    return getattr(import_module(submodule, __name__), class_name)


class _LazyModules(Mapping):
    """Mapping of gin name -> class, importing each class on first access."""

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

    def spec(self, key):
        """(submodule, class name) for `key`, without importing it."""
        return self._specs[key]


MODULES = _LazyModules(_MODULE_SPECS)

__all__ = ["MODULES", "get_module", *_BY_CLASS_NAME]


def __getattr__(name: str):
    """Expose the classes as attributes without importing them up front."""
    if name in _BY_CLASS_NAME:
        return _load(_BY_CLASS_NAME[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_module(module_name: str):
    """Get module by name."""

    return MODULES[module_name]()
