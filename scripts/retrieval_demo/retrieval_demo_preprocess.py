import gin
import argparse
from pathlib import Path

import torch
from essentia.standard import MonoLoader

from tqdm import tqdm


parser = argparse.ArgumentParser(description="")
parser.add_argument("--data_dir", type=str, default="../../dataset")
parser.add_argument("--audio_loader", type=str, default="ffmpeg")
parser.add_argument("--eval_query", type=str, default="caption")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--id_list", type=str)

# ttmr_pp config
parser.add_argument(
    "--model_type", type=str, default="mtg-upf/clap_omarrq_att_large_music"
)
parser.add_argument("--caption_type", type=str, default="meta_tag_caption_sim")
# train confing
parser.add_argument("--tid", default="base", type=str)
parser.add_argument("--ckpt_type", default="last", type=str)
parser.add_argument("--cfg_file", type=str, default=None)
args = parser.parse_args()

duration = 30


def get_audio(path, sr, duration):
    n_samples = sr * duration

    audio = MonoLoader(filename=path, sampleRate=sr, resampleQuality=4)()

    if len(audio) > n_samples:
        mid_point = len(audio) // 2
        start = mid_point - (n_samples // 2)
        audio = audio[start : start + n_samples]
    elif len(audio) < n_samples:
        pad = torch.zeros(n_samples)
        pad[: len(audio)] = torch.tensor(audio)
        audio = pad

    audio = torch.tensor(audio).unsqueeze(0)
    return audio


def main(args):
    if args.cfg_file is not None:
        from amclap import get_model

        model = get_model(config_file=args.cfg_file, device=args.device)

        cfg_id = Path(args.cfg_file).parent.parent.stem
        save_dir = f"embeddings/id_{cfg_id}/"
        sr = 24000
        duration = 30

    elif args.model_type == "laion_clap":
        import laion_clap

        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()  # download the default pretrained checkpoint.
        model = model.to(args.device)
        model.eval()
        save_dir = f"embeddings/id_{cfg_id}/"
        sr = 48000
        duration = 10

    elif args.model_type is not None:
        from amclap import get_model

        model = get_model(model_id=args.model_type, device=args.device)
        sr = 24000
        save_dir = f"embeddings/id_{cfg_id}/"

    else:
        raise ValueError("Either cfg_file or model_type should be provided.")

    model.eval()

    with open(args.id_list, "r") as f:
        id_list = [line.rstrip() for line in f.readlines() if line.rstrip() != ""]

    print(f"Number of audio files: {len(id_list)}")
    print(id_list[:5])

    for id in tqdm(id_list):
        try:
            audio_path = f"{args.data_dir}/{id[:2]}/{id}.mp4"

            embedding_path = Path(save_dir) / id[:2] / f"{id}.npy"
            embedding_path.parent.mkdir(parents=True, exist_ok=True)

            if embedding_path.exists():
                print(f"Skipping {id} as it already exists.")
                continue

            audio = get_audio(audio_path, sr, duration)

            if args.model_type == "laion_clap":
                with torch.no_grad():
                    audio_embs = model.get_audio_embedding_from_data(
                        x=audio.to(args.device), use_tensor=True
                    ).mean(0, True)
            else:
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    audio_embs = model.forward_audio(audio.to(args.device)).float()
                    audio_embs = audio_embs.mean(0, True)

            audio_embs.detach().cpu().numpy().tofile(embedding_path)
        except Exception as e:
            print(f"Error processing {id}: {e}")


if __name__ == "__main__":
    main(args)
