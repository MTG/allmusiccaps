import gin
import lightning as L
import torch

from .clap_modules import MODULES
from .nets import NETS

# Register every module and net name with gin.
#
# The published config.gin files bind parameters on the *class* name and select
# the module by its *gin* name -- `CLAP.audio_encoder = @OMARRQ` alongside
# `def_module.module = @clap`. Both refer to the same object, so the class
# itself must be what gets registered: wrapping it in a factory would give gin
# two distinct configurables and the `CLAP.*` bindings would never be applied.
#
# Registration therefore resolves the classes, and only the variants no config
# references stay unimported. `_INFERENCE_MODULES` covers everything the
# published models select; anything else (SLAP, MERT, HTSAT, MAEST, and the
# LeJEPA family, which needs the optional `lejepa` package) is registered on
# demand by `ensure_registered`, called from `def_module` below.
_INFERENCE_MODULES = ("clap", "omar_rq", "all_mpnet_base_v2")

_registered: set[str] = set()


def ensure_registered(*module_names: str) -> None:
    """Register the named modules with gin, importing them if needed."""
    for name in module_names or tuple(MODULES):
        if name in _registered:
            continue
        gin.external_configurable(MODULES[name], name)
        _registered.add(name)


ensure_registered(*_INFERENCE_MODULES)

for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)


def _load_submodule(parent, key, state_dict, label=None) -> None:
    """Load one submodule's weights from `state_dict` by key prefix.

    Which submodules exist depends on the config, so an absent one is a normal
    configuration rather than a failure; only a genuine mismatch is reported.
    """
    label = label or key
    submodule = getattr(parent, key, None)
    weights = {
        k[len(key) + 1 :]: v for k, v in state_dict.items() if k.startswith(key + ".")
    }

    if submodule is None and not weights:
        # The attention-pooler pair (`proj_att_query`, `att_pooler`) is only
        # built when `aggregation_type == "attention_pooler"`, so under mean
        # pooling it is absent from both the model and the checkpoint. Nothing
        # to load and nothing wrong: stay quiet.
        return

    # `weights` is non-empty here: the early return above took the other case.
    if submodule is None:
        print(
            f"AMCLAP: skipping `{label}`: the checkpoint has {len(weights)} "
            "weights for it, but this model does not define it."
        )
        return

    if not weights:
        print(
            f"AMCLAP: `{label}` got no weights: the checkpoint contains no "
            f"`{key}.*` keys, so it keeps its initialisation."
        )
        return

    try:
        submodule.load_state_dict(weights, strict=True)
    except Exception as exc:  # noqa: BLE001 - one bad submodule must not abort the rest
        print(f"AMCLAP: error loading weights for `{label}`:", exc)
        return

    print(f"AMCLAP: {len(weights)} weights loaded for `{label}`")


def _check_text_encoder_loaded(module, state_dict) -> None:
    """Fail if the checkpoint's fine-tuned text weights were silently ignored.

    Models trained with a trainable text encoder ship their `text_encoder.*`
    weights in the checkpoint. Those key names track the internal layout of
    `sentence_transformers`, which renamed its inner module from `auto_model` to
    `model` in 5.2. Under a mismatched version every key fails to match, the
    text tower quietly falls back to stock MPNet, and the model returns
    plausible-looking but wrong text embeddings. Better to refuse to load.
    """
    ckpt_text_keys = {k for k in state_dict if k.startswith("text_encoder.")}
    if not ckpt_text_keys:
        return

    model_keys = set(module.state_dict())
    if ckpt_text_keys & model_keys:
        return

    import sentence_transformers

    raise RuntimeError(
        f"This checkpoint carries {len(ckpt_text_keys)} fine-tuned text-encoder "
        "weights, but none of their names exist in the instantiated model, so the "
        "text tower would silently revert to stock all-mpnet-base-v2 and produce "
        "wrong embeddings.\n"
        f"Installed sentence-transformers is {sentence_transformers.__version__}; "
        "this layout needs >=5.0,<5.2. Pin it with "
        "`pip install 'sentence-transformers<5.2'`."
    )


@gin.configurable
def def_module(
    module: L.LightningModule,
    ckpt_path: str | None = None,
    weights_only: bool = True,
) -> L.LightningModule:
    """Default module to use when no module is specified in the gin config."""
    if ckpt_path is None:
        module = module()
    else:
        if weights_only:
            module = module()

            state_dict = torch.load(ckpt_path, map_location=module.device)

            # In case it is a checkpoint from a LightningModule, we need to extract the state_dict
            if hasattr(state_dict, "state_dict"):
                raise ValueError(
                    "The provided checkpoint is a LightningModule checkpoint. Please provide a state_dict checkpoint or set `weights_only` to False."
                )

            try:
                module.load_state_dict(state_dict, strict=True)

            except RuntimeError:
                # Expected for the published checkpoints: they store the audio
                # encoder under `net.*` / `embedding_layer.*`, while the
                # instantiated model nests it under `audio_encoder.model.*`, so
                # a strict whole-model load can never match. Fall back to
                # loading each submodule by prefix.

                for key in ["proj_a", "proj_t", "proj_att_query", "att_pooler"]:
                    _load_submodule(module, key, state_dict)

                for key in ["net", "embedding_layer"]:
                    _load_submodule(
                        module.audio_encoder.model,
                        key,
                        state_dict,
                        label=f"audio_encoder.model.{key}",
                    )

                _check_text_encoder_loaded(module, state_dict)

        else:
            module = module.load_from_checkpoint(ckpt_path, strict=True)

    return module
