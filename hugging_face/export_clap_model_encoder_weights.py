from pathlib import Path
from shutil import copyfile

import torch

torch.manual_seed(2)

base = Path("/mnt/projects/BSC_A3/projects-<group>/<group>/logs/text-audio/")

for model_name, model_id in {
    "clap_omarrq_mp_small_music": "bayc7zny",
    "clap_omarrq_mp_large_music": "pmn4dy8p",
    "clap_omarrq_att_large_music": "lunrdjvy",
}.items():
    print(f"Processing model {model_id}...")

    weights_dir = base / model_id
    cfg_file = list(weights_dir.rglob("*.gin"))[0]
    weights_file = list(weights_dir.rglob("*.ckpt"))[0]

    state_dict = torch.load(weights_file, map_location="cpu")["state_dict"]
    # print(state_dict.keys())

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("audio_encoder.net."):
            nk = k.replace("audio_encoder.", "")
            new_state_dict[nk] = v
        elif k.startswith("audio_encoder.embedding_layer."):
            nk = k.replace("audio_encoder.", "")
            new_state_dict[nk] = v
        elif k.startswith("proj_"):
            new_state_dict[k] = v
        elif k.startswith("att_pooler."):
            new_state_dict[k] = v
        else:
            continue

    print("Loaded state dict with", len(new_state_dict), "keys.")

    weights_dir = Path(f"weights_light/{model_name}")
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, weights_dir / "model.ckpt")

    copyfile(cfg_file, weights_dir / "config.gin")
    print("ok!")
