"""Evaluate SSL models using gin configs. This script is used to extract
embeddings from a pre-trained SSL model for downstream tasks. The embeddings are
saved in the embeddings_dir specified in the downstream task's gin config.
By default, the embeddings won't be aggregated and will be saved as they are.
(L, B, T, C,)
    where L = len(layers),
    B = number of audio chunks
    T = number of melspec frames the model can accomodate
    C = model output dimension
"""

import shutil
import traceback
import argparse
import json
import os
from pathlib import Path

# Add src directory to path for imports

import gin.torch
import torch
import lightning.pytorch as L

torch.set_float32_matmul_precision("medium")

from .. import get_model
from ..data import DATASETS
from ..nets import NETS
from ..data.audio_embedding_datamodule import AudioEmbeddingDataModule
from embedding_writer import EmbeddingWriter

for data_name, data in DATASETS.items():
    gin.external_configurable(data, data_name)

for net_name, net in NETS.items():
    gin.external_configurable(net, net_name)


@gin.configurable
def predict(
    config_file: Path,
    dataset_name: str,
    embeddings_dir: Path,
    params: dict,
    overlap_ratio: float,
    output_dir: str = None,
    segment_size: float = 10.0,
    new_freq: int = 24000,
    use_audio_type_token: bool = False,
    force_recompute: bool = False,
    ckpt_step: int = None,
    avg_last_n: int = None,
):
    """Wrapper function. Basically overrides some train parameters."""

    # We use the following structure for the embeddings directory:
    # root_output_dir/ssl_model_id/dataset_name/ Inside dataset_name,
    # we have the following structure: dataset_name/audio_name[:3]/audio_name.pt
    ssl_model_id = Path(config_file).parent.parent.name
    embeddings_dir = Path(embeddings_dir) / ssl_model_id / dataset_name

    # Optionally wipe existing embeddings to force re-extraction
    if force_recompute and embeddings_dir.exists():
        print(f"force_recompute=True: deleting existing embeddings in {embeddings_dir}")
        shutil.rmtree(embeddings_dir)

    # Determine output directory for logs
    if output_dir:
        log_dir = output_dir
    else:
        log_dir = embeddings_dir

    # Add the callback to write the embeddings
    callbacks = [EmbeddingWriter(embeddings_dir)]

    # Create the trainer first
    trainer = L.Trainer(callbacks=callbacks, default_root_dir=log_dir, **params)

    # Build the module and load the weights
    module = get_model(
        config_file=config_file,
        device="cuda",
        weights_only=False,
        ckpt_step=ckpt_step,
        avg_last_n=avg_last_n,
    )
    module.eval()

    # Set the overlap ratio
    module.overlap_ratio = overlap_ratio

    print(f"Using segment size: {segment_size}s at {new_freq}Hz")
    print(f"Use audio type token: {use_audio_type_token}")

    # Data module with embedding filter: skips files already extracted
    data_module = AudioEmbeddingDataModule(
        n_seconds=segment_size, new_freq=new_freq, embeddings_dir=embeddings_dir
    )
    data_module.setup(stage="predict")

    if len(data_module.dataset) == 0:
        print("All embeddings already computed. Skipping extraction.")
    else:
        trainer.predict(module, datamodule=data_module)

    # Separate dataset without filter to iterate all files for evaluation
    eval_data_module = AudioEmbeddingDataModule(
        n_seconds=segment_size, new_freq=new_freq, embeddings_dir=None
    )
    eval_data_module.setup(stage="predict")

    # Get embedding paths
    embedding_paths = []
    y_true = []
    classes = set()
    for audio_path, *_ in eval_data_module.dataset.index.values():
        audio_name = Path(audio_path).stem
        _output_dir = embeddings_dir / audio_name[:3]
        output_path = _output_dir / f"{audio_name}.pt"
        embedding_paths.append(output_path)
        label = audio_name.split(".")[0]
        classes.add(label)
        y_true.append(label)

    classes = list(classes)

    prompt = "This is a *** song."

    # Add audio type token prefix if required by the model
    if use_audio_type_token:
        captions = [f"[MUSIC] {prompt.replace('***', label)}" for label in classes]
    else:
        captions = [prompt.replace("***", label) for label in classes]

    module.text_encoder.eval()
    with torch.no_grad():
        class_embeddings = module.forward_text(captions)

    y_pred = []
    for embedding_path in embedding_paths:
        # Load the embedding
        embedding = torch.load(embedding_path).squeeze()
        # embedding.unsqueeze_(0)

        if embedding.ndim == 2:
            # Average over time dimension if needed
            embedding = torch.mean(embedding, dim=1)

        # Put embedding in the same device as the model
        embedding = embedding.to(module.device)

        # Get the cosine similarity

        cos_sim = [
            torch.nn.functional.cosine_similarity(
                class_embeddings[i], embedding, dim=-1
            )
            for i in range(len(class_embeddings))
        ]
        cos_sim = torch.stack(cos_sim, dim=0)

        # Get the predicted label
        pred_label = classes[torch.argmax(cos_sim)]
        y_pred.append(pred_label)

    # compute accuracy
    acc = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)
    print(f"{ssl_model_id} accuracy: {acc:.3f}")

    # Save results to output directory
    results = {
        "model_id": ssl_model_id,
        "dataset": dataset_name,
        "accuracy": acc,
    }

    save_dir = output_dir if output_dir else str(embeddings_dir)
    os.makedirs(save_dir, exist_ok=True)
    results_path = os.path.join(save_dir, "results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the model config of a trained model.",
    )
    parser.add_argument(
        "predict_config",
        type=Path,
        help="Path to the config file of the downstream task's dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--segment_size",
        type=float,
        default=10.0,
        help="Audio segment size in seconds",
    )
    parser.add_argument(
        "--new_freq",
        type=int,
        default=24000,
        help="Audio sample rate in Hz",
    )
    parser.add_argument(
        "--use_audio_type_token",
        action="store_true",
        help="Add [MUSIC] prefix to text queries",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Delete existing embeddings and recompute from scratch",
    )
    parser.add_argument(
        "--ckpt_step",
        type=int,
        default=None,
        help="Load checkpoint at this specific training step",
    )
    parser.add_argument(
        "--avg_last_n",
        type=int,
        default=None,
        help="Average the last N checkpoints",
    )

    args = parser.parse_args()

    try:
        # Parse the predict config (train config is parsed by get_model)
        gin.parse_config_file(args.predict_config, skip_unknown=True)

        predict(
            config_file=args.train_config,
            output_dir=args.output_dir,
            segment_size=args.segment_size,
            new_freq=args.new_freq,
            use_audio_type_token=args.use_audio_type_token,
            force_recompute=args.force_recompute,
            ckpt_step=args.ckpt_step,
            avg_last_n=args.avg_last_n,
        )

        print("Embedding extraction completed successfully!")

    except Exception:
        traceback.print_exc()
