from argparse import ArgumentParser
from typing import List
import gin
import torch
from torch import nn
from pathlib import Path
from lightning.pytorch.utilities import rank_zero_info
from .data.audio_embedding_datamodule import AudioEmbeddingDataModule
from tqdm import tqdm

from .clap_modules import MODULES

for module_name, module in MODULES.items():
    gin.external_configurable(module, module_name)


def extract(
    model: nn.Module,
    audio: torch.Tensor,
    layers: List[int],
) -> torch.Tensor:
    """
    Extract embeddings for each segment of audio from specified layers,
    then aggregate layer-wise embeddings by averaging.

    Args:
        audio (torch.Tensor): Audio tensor of shape (B, T)
        layers (List[int]): The layers to extract embeddings from.

    Returns:
        torch.Tensor: Aggregated embeddings (B, C)
    """

    # Iterate through specified layers to collect embeddings
    # model.model is the underlying OMARRQ wrapper which has extract_embeddings
    embeddings = model.model.extract_embeddings(audio, layers=layers)

    # Average over layers: (B, Time, C) -> implying previous dim was layers
    # But wait, OMARRQ.extract_embeddings in src/clap_modules/omarrq.py calls self.model.extract_embeddings
    # and existing code suggests it returns something that needs averaging.
    # Let's assume the previous logic was correct for the model structure.
    # The previous code:
    # embeddings = model.model.extract_embeddings(audio, layers=layers)
    # embeddings = torch.mean(embeddings, dim=0) # Average over layers
    # embeddings = torch.mean(embeddings, dim=1) # Average over time

    embeddings = torch.mean(embeddings, dim=0)
    embeddings = torch.mean(embeddings, dim=1)

    return embeddings


def save_embeddings(
    embeddings_list: List[torch.Tensor], file_path: Path, data_dir: Path, out_dir: Path
):
    """
    Aggregates embeddings for a single file and saves to disk.
    """
    if not embeddings_list:
        return

    # Stack segments: (N_segments, C)
    embeddings = torch.stack(embeddings_list)

    # Average segments to get one embedding per file: (C,)
    aggregated_embedding = torch.mean(embeddings, dim=0)

    # Compute save path preserving directory structure
    try:
        # Resolve to handle ../ or symlinks if needed, but usually relative_to works on simple paths
        rel_path = file_path.relative_to(data_dir)
    except ValueError:
        # Fallback if file_path is not relative to data_dir
        rel_path = Path(file_path.name)

    save_path = (out_dir / rel_path).with_suffix(".pt")

    if save_path.exists():
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(aggregated_embedding, save_path)
    print(f"Saved embeddings to {save_path}")


@gin.configurable
def extract_embeddings(
    model: nn.Module,
    layers: List[int],
    out_dir: Path = "embeddings",
):
    """
    Orchestrates the embedding extraction pipeline by loading data,
    processing audio files into batches, and saving embeddings to disk.
    """
    # Load gin configuration
    rank_zero_info("Initializing the embedding extraction pipeline.")

    # Initialize data module
    datamodule = AudioEmbeddingDataModule()
    datamodule.setup(stage="predict")

    data_dir = datamodule.data_dir
    out_dir = Path(out_dir)

    # Load model
    model = model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    rank_zero_info("Starting embedding extraction.")

    # We need to handle multiple segments per file.
    # Since DataLoader(shuffle=False), segments for the same file come sequentially.

    current_file_path = None
    embeddings_buffer = []

    dataloader = datamodule.predict_dataloader()

    for batch in tqdm(dataloader):
        audio, file_paths = batch
        # audio: (B, T)
        # file_paths: tuple of strings, length B

        audio = audio.to(device)

        with torch.no_grad():
            # Get embeddings for this batch of segments
            # shape: (B, C)
            batch_embeddings = extract(model, audio, layers)

        batch_embeddings = batch_embeddings.cpu()

        for i, file_path_str in enumerate(file_paths):
            file_path = Path(file_path_str)
            emb = batch_embeddings[i]  # (C,)

            # Check if we moved to a new file
            if current_file_path is not None and file_path != current_file_path:
                save_embeddings(embeddings_buffer, current_file_path, data_dir, out_dir)
                embeddings_buffer = []

            current_file_path = file_path
            embeddings_buffer.append(emb)

    # Flush the last file
    if current_file_path is not None and embeddings_buffer:
        save_embeddings(embeddings_buffer, current_file_path, data_dir, out_dir)

    rank_zero_info("Embedding extraction completed.")


if __name__ == "__main__":
    parser = ArgumentParser("Extract audio embeddings using OMARRQ model.")
    parser.add_argument(
        "train_config",
        type=Path,
        help="Path to the gin config file for training.",
    )
    parser.add_argument(
        "--gin-binds",
        action="append",
        help="Gin bindings to override config file.",
    )

    args = parser.parse_args()
    gin.parse_config_files_and_bindings(
        [args.train_config], args.gin_binds or [], skip_unknown=True
    )

    extract_embeddings()
