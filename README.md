<div align="center">

# AllMusicCaps: Album Reviews as Complementary Supervision for Music CLAP

_Pablo Alonso-Jiménez, Xavier Lizarraga-Seijas, Xavier Serra, Dmitry Bogdanov_

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.en.html)
[![PyPI version](https://img.shields.io/pypi/v/amclap.svg)](https://pypi.org/project/amclap/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-AllMusicCaps-yellow)](https://huggingface.co/datasets/mtg-upf/allmusiccaps)
[![Tests](https://github.com/MTG/allmusiccaps/actions/workflows/python-tests.yml/badge.svg)](https://github.com/MTG/allmusiccaps/actions/workflows/python-tests.yml)

</div>

Music-text contrastive (CLAP) models trained with captions derived from professional album reviews,
plus the **AllMusicCaps** caption dataset.

## Install

### From PyPI

```bash
pip install amclap
```

### From source

For embedding extraction or fine-tuning:

```bash
pip install .
```

For development including pre-training your own models:

```bash
pip install -e .[train]
```

## Inference

Load a model by specifying its [Hugging Face model ID](#hugging-face-model-ids):

```python
import torch
from amclap import get_model

x_a = torch.randn(1, 24000 * 10).cpu()   # mono audio at 24 kHz
x_t = ["dreamy shoegaze with washed-out guitars"]

model_id = "mtg-upf/allmusiccaps_te_trained_sigreg"
model = get_model(model_id=model_id, device="cpu")

with torch.no_grad():
    z_a = model.forward_audio(x_a)       # torch.Size([1, 512])
    z_t = model.forward_text(x_t)        # torch.Size([1, 512])
```

Audio must be mono at 24 kHz. Both towers output 512-dimensional embeddings in a shared space,
comparable with cosine similarity.

> **Note:** `pip install amclap` is enough for inference on every released model. Training with
> the SigReg objective additionally needs [`lejepa`](https://github.com/rbalestr-lab/lejepa),
> which has no PyPI release and so is not part of the `[train]` extra — install it from git:
>
> ```bash
> pip install "lejepa @ git+https://github.com/rbalestr-lab/lejepa.git"
> ```

## Available models

All models use the OMAR-RQ audio encoder and an `all-mpnet-base-v2` text encoder (_TE_). _Step_ is
the checkpoint the paper reports.

| Model | Data | Layers | Objective | TE | Step |
|---|---|---|---|---|---|
| **baseline** | baseline | last | InfoNCE | frozen | 147k |
| **amcquotes** | quotes | last | InfoNCE | frozen | 150k |
| **amcstruct** | struct | last | InfoNCE | frozen | 150k |
| **baseline_quotes** | baseline+quotes | last | InfoNCE | frozen | 150k |
| **baseline_struct** | baseline+struct | last | InfoNCE | frozen | 150k |
| **layer6** | baseline+quotes | 6 | InfoNCE | frozen | 150k |
| **all_layers** | baseline+quotes | all | InfoNCE | frozen | 150k |
| **sigmoid** | baseline+quotes | all | sigmoid | trained | 150k |
| **lejepa** | baseline+quotes | all | LeJEPA (cosine) | frozen | 150k |
| **infonce_sigreg** | baseline+quotes | all | LeJEPA (InfoNCE) | frozen | 150k |
| **te_trained** | baseline+quotes | all | InfoNCE | trained | 60k |
| **te_trained_sigreg** | baseline+quotes | all | InfoNCE+SigReg | trained | 60k |

**te_trained_sigreg** is the best overall model; **all_layers** is the frozen-TE recipe it builds on.

> **Note:** models with a trainable text encoder overfit past ~40--80k steps, so they are released
> at their 60k checkpoint rather than the final one.

### Hugging Face Model IDs

- [mtg-upf/allmusiccaps_baseline](https://huggingface.co/mtg-upf/allmusiccaps_baseline)
- [mtg-upf/allmusiccaps_amcquotes](https://huggingface.co/mtg-upf/allmusiccaps_amcquotes)
- [mtg-upf/allmusiccaps_amcstruct](https://huggingface.co/mtg-upf/allmusiccaps_amcstruct)
- [mtg-upf/allmusiccaps_baseline_quotes](https://huggingface.co/mtg-upf/allmusiccaps_baseline_quotes)
- [mtg-upf/allmusiccaps_baseline_struct](https://huggingface.co/mtg-upf/allmusiccaps_baseline_struct)
- [mtg-upf/allmusiccaps_layer6](https://huggingface.co/mtg-upf/allmusiccaps_layer6)
- [mtg-upf/allmusiccaps_all_layers](https://huggingface.co/mtg-upf/allmusiccaps_all_layers)
- [mtg-upf/allmusiccaps_sigmoid](https://huggingface.co/mtg-upf/allmusiccaps_sigmoid)
- [mtg-upf/allmusiccaps_lejepa](https://huggingface.co/mtg-upf/allmusiccaps_lejepa)
- [mtg-upf/allmusiccaps_infonce_sigreg](https://huggingface.co/mtg-upf/allmusiccaps_infonce_sigreg)
- [mtg-upf/allmusiccaps_te_trained](https://huggingface.co/mtg-upf/allmusiccaps_te_trained)
- [mtg-upf/allmusiccaps_te_trained_sigreg](https://huggingface.co/mtg-upf/allmusiccaps_te_trained_sigreg)

## The AllMusicCaps dataset

540,454 rows pairing YouTube tracks with captions derived from AllMusic album reviews, in two
styles: review _quotes_ and LLM-filled _structured_ attributes. It contains identifiers and
captions, **no audio**.

```python
from datasets import load_dataset

ds = load_dataset("mtg-upf/allmusiccaps", split="train")
print(ds[0]["generated_quotes_captions"])
```

**v1 is the default.** `allmusiccaps_v0.jsonl` is what the paper's models were trained on, kept for
exact reproducibility: it holds the raw LLM output, in which `generated_quotes_captions` is not
consistently typed, so Arrow-backed readers reject it. v1 normalizes that field to `list[string]`
and changes nothing else. See the [dataset card](https://huggingface.co/datasets/mtg-upf/allmusiccaps)
for the mapping and how to read v0, and
[`normalize_allmusiccaps_v1.py`](scripts/preprocess_am_discotube/normalize_allmusiccaps_v1.py) to
reproduce it. A 50-row sample lives in [`data/allmusiccaps/_samples/`](data/allmusiccaps/_samples/).

## Training

1. Install development dependencies:

```bash
pip install -e .[train]
```

1. Prepare the data

Audio is stored downsampled to 24 kHz mono as 16-bit raw bytes
([numpy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) files); captions
come from the JSONL above. Check the [preprocessing scripts](scripts/preprocess_am_discotube/).

1. Configuration

Experiment configuration is controlled with [gin-config](https://github.com/google/gin-config); see
[`cfg/README.md`](cfg/README.md). At least the dataset paths need to point at your own data.

1. Run the experiment

```bash
python -m amclap.train cfg/<config>.gin
```

## Citation

If you find this work useful, please cite the paper:

```bibtex
@inproceedings{alonso2026allmusiccaps,
  title = {{AllMusicCaps}: Album Reviews as Complementary Supervision for Music {CLAP}},
  author = {Alonso-Jim{\'e}nez, Pablo and Lizarraga-Seijas, Xavier and Serra, Xavier and Bogdanov, Dmitry},
  booktitle = {International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2026},
}
```

## Licensing information

The code in this repository is available under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html) license.
The model weights are available under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license for non-commercial applications.
The AllMusicCaps dataset is released for non-commercial scientific research purposes only, and any
publication of results based on it must cite AllMusic as the source of the data.
[Contact us](https://www.upf.edu/web/mtg/contact) for more information.
