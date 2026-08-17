from pathlib import Path
import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class EmbeddingWriter(BasePredictionWriter):
    def __init__(self, output_dir: Path, write_interval: str = "batch"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        # Track expected n_segments per file
        self._n_segments = {}

    def _write_and_flush(self, pl_module, audio_path):
        """Write embeddings for a completed file and free memory."""
        embeddings = pl_module.predict_data[audio_path]
        audio_name = Path(audio_path).stem
        _output_dir = self.output_dir / audio_name[:3]
        _output_dir.mkdir(parents=True, exist_ok=True)
        output_path = _output_dir / f"{audio_name}.pt"

        if embeddings is not None:
            try:
                prediction = torch.stack(embeddings, dim=1)
                torch.save(prediction, output_path)
            except Exception as e:
                print(f"Error saving embeddings for {audio_name}: {e}")

        del pl_module.predict_data[audio_path]

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        # Unpack batch to get n_segments info
        _, filenames, _, n_segments_batch = batch

        # Update expected n_segments for each file in this batch
        for fname, n_seg in zip(filenames, n_segments_batch):
            fname = str(fname) if not isinstance(fname, str) else fname
            self._n_segments[fname] = int(n_seg)

        # Check which files are complete and write them out
        completed = []
        for fname, expected in self._n_segments.items():
            if fname in pl_module.predict_data:
                if len(pl_module.predict_data[fname]) >= expected:
                    completed.append(fname)

        for fname in completed:
            self._write_and_flush(pl_module, fname)
            del self._n_segments[fname]

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ):
        # Safety net: write any remaining files that weren't flushed during batches
        remaining = list(pl_module.predict_data.keys())
        if remaining:
            print(
                f"EmbeddingWriter: writing {len(remaining)} remaining files at epoch end"
            )
        for audio_path in remaining:
            self._write_and_flush(pl_module, audio_path)
        self._n_segments.clear()
