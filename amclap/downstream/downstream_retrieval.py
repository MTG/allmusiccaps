import argparse
import os
import json
from pathlib import Path

import torch

torch.set_float32_matmul_precision("medium")

# import omar_rq
from sklearn import metrics
from tqdm import tqdm

from retrieval.eval_utils import get_query2target_idx, get_task_predictions
from retrieval.metrics import (
    recall,
    mean_average_precision,
    mean_reciprocal_rank,
    median_rank,
)
from retrieval.musicaps_dataset import MusicCaps
from retrieval.allmusic_dataset import AllMusic
from retrieval.song_describer_dataset import SongDescriber
from retrieval.query_utils import query_processor

parser = argparse.ArgumentParser(description="")
parser.add_argument("--data_dir", type=str, default="../../dataset")
parser.add_argument("--audio_loader", type=str, default="ffmpeg")
parser.add_argument("--eval_query", type=str, default="caption")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dataset", type=str, default="music_caps")

# ttmr_pp config
parser.add_argument(
    "--model_type", type=str, default="mtg-upf/clap_omarrq_att_large_music"
)
parser.add_argument("--caption_type", type=str, default="meta_tag_caption_sim")
# train confing
parser.add_argument("--tid", default="base", type=str)
parser.add_argument("--ckpt_type", default="last", type=str)
parser.add_argument("--cfg_file", type=str, default=None)
parser.add_argument("--audio-setup", type=str, choices=["clap", "clamp3", "frozen"])
parser.add_argument(
    "--output_dir", type=str, default=None, help="Directory to save results"
)
parser.add_argument(
    "--segment_size", type=float, default=10.0, help="Audio segment size in seconds"
)
parser.add_argument(
    "--new_freq", type=int, default=24000, help="Audio sample rate in Hz"
)
parser.add_argument(
    "--use_audio_type_token",
    action="store_true",
    help="Add [MUSIC] prefix to text queries",
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

audio_setup = args.audio_setup
print(f"Audio setup: {audio_setup}")


def main(args):
    # Use args for audio config
    sr = args.new_freq
    segment_size = args.segment_size
    use_audio_type_token = args.use_audio_type_token

    print(f"Using segment size: {segment_size}s at {sr}Hz")
    print(f"Use audio type token: {use_audio_type_token}")

    if args.cfg_file is not None:
        from amclap import get_model

        model = get_model(
            config_file=args.cfg_file,
            device=args.device,
            weights_only=False,
            ckpt_step=args.ckpt_step,
            avg_last_n=args.avg_last_n,
        )

        cfg_id = Path(args.cfg_file).parent.parent.stem
        if args.output_dir:
            save_dir = args.output_dir
        else:
            save_dir = f"exp/id_{cfg_id}/{args.caption_type}"

        if audio_setup == "clamp3":
            import omar_rq

            model_id = "mtg-upf/omar-rq-multifeature-25hz-fsq"
            layers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]

            feature_extractor = omar_rq.get_model(model_id=model_id, device=args.device)

            sr = feature_extractor.sr
            segment_size = 5.0  # seconds

            print(f"Using CLAMP3-style feature extractor: {model_id}")
            print(f"Layers: {layers}")
        elif audio_setup == "frozen":
            sr = None
            segment_size = None

    elif args.model_type is not None:
        from amclap import get_model

        model = get_model(model_id=args.model_type, device=args.device)
        save_dir = f"exp/{args.model_type}/{args.caption_type}"

    elif args.model_type == "laion_clap":
        import laion_clap

        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()  # download the default pretrained checkpoint.
        model = model.to(args.device)
        model.eval()
        save_dir = f"exp/{args.model_type}/{args.caption_type}"
        sr = 48000
        segment_size = 10.0  # seconds
        use_audio_type_token = False

    else:
        raise ValueError("Either cfg_file or model_type should be provided.")

    print(f"Sampling rate: {sr}")
    print(f"Segment size: {segment_size} seconds")

    if args.dataset == "music_caps":
        data_holder = MusicCaps
        split = "test"
        audio_enc = ".wav"
    elif args.dataset == "song_describer":
        data_holder = SongDescriber
        split = "is_valid_subset"
        audio_enc = ".mp3"
    elif args.dataset == "all_music":
        data_holder = AllMusic
        split = "val"
        audio_enc = ".pt"  # .mmap
    else:
        raise NotImplementedError()

    dataset = data_holder(
        data_dir=args.data_dir,
        split=split,
        audio_loader=args.audio_loader,
        caption_type="caption",
        sr=sr,
        duration=segment_size,
        audio_enc=audio_enc,
    )
    binary_matrix, track2query, query2track = query_processor(dataset, args.eval_query)
    unique_track = list(track2query.keys())
    unique_query = list(query2track.keys())

    track2idx = {i: idx for idx, i in enumerate(unique_track)}
    query2idx = {i: idx for idx, i in enumerate(unique_query)}
    # query <-> track
    query2track_idx = get_query2target_idx(query2track, track2idx)

    audio_features, query_features = [], []
    for audio_id in tqdm(unique_track):
        audio = dataset.get_audio(audio_id)  # (S, T)
        audio = audio.to(args.device)

        if args.model_type == "laion_clap":
            with torch.inference_mode():
                rep = model.get_audio_embedding_from_data(x=audio, use_tensor=True)
        else:
            if audio_setup == "frozen":
                x = audio.unsqueeze(0)  # preprocessed features (1, S, C)

            elif audio_setup == "clamp3":
                with torch.inference_mode():
                    # Audio to reps (B, T') -> (L, S, T, C)
                    x = feature_extractor.extract_embeddings(audio, layers=layers)

                # Average over layers: (S, T, C)
                x = torch.mean(x, dim=0)
                # # Average over time: (S, C)
                x = torch.mean(x, dim=1)

                x = x.unsqueeze(0)  # (1, S, C)

            elif audio_setup == "clap":
                x = audio
            else:
                raise ValueError(f"Unknown audio setup: {audio_setup}")

            with torch.inference_mode():
                rep = model.forward_audio(x)

        rep = rep.mean(0, keepdim=True)  # (1, C)

        audio_features.extend(rep.detach().cpu())

    # extract unique_query
    for query in tqdm(unique_query):
        # Add audio type token prefix if required by the model
        if use_audio_type_token:
            query_text = f"[MUSIC] {query}"
        else:
            query_text = query

        if args.model_type == "laion_clap":
            with torch.no_grad():
                query_embs = model.get_text_embedding([query], use_tensor=True)
        else:
            with torch.no_grad():
                query_embs = model.forward_text([query_text])

        query_features.extend(query_embs.detach().cpu())

    model_output = {
        "audio_features": torch.stack(audio_features),
        "query_features": torch.stack(query_features),
        "audio_ids": unique_track,
        "querys": unique_query,
    }

    query2audio_matrix = get_task_predictions(
        model_output["query_features"], model_output["audio_features"]
    )
    binary_matrix = binary_matrix.loc[unique_track][unique_query].T  # ordering
    medrank, query2rank = median_rank(unique_query, query2track_idx, query2audio_matrix)
    os.makedirs(os.path.join(save_dir, args.dataset), exist_ok=True)
    if args.eval_query == "caption":
        query_to_audio_results = {
            "recall@1": recall(
                query2audio_matrix, unique_query, query2track_idx, top_k=1
            ),
            "recall@5": recall(
                query2audio_matrix, unique_query, query2track_idx, top_k=5
            ),
            "recall@10": recall(
                query2audio_matrix, unique_query, query2track_idx, top_k=10
            ),
            "map@10": mean_average_precision(
                query2audio_matrix, unique_query, query2track_idx, top_k=10
            ),
            "mean_reciprocal_rank": mean_reciprocal_rank(
                query2audio_matrix, unique_query, query2track_idx
            ),
            "median_rank": medrank,
        }
        print(query_to_audio_results)
        with open(
            os.path.join(save_dir, args.dataset, f"caption2rank.json"), "w"
        ) as json_file:
            json.dump(query2rank, json_file, indent=4)
    else:
        query_to_audio_results = {
            "rocauc": metrics.roc_auc_score(
                binary_matrix, query2audio_matrix, average="samples"
            ),
            "prauc": metrics.average_precision_score(
                binary_matrix, query2audio_matrix, average="samples"
            ),
        }
        tag_wise_results = {}
        for query, gt, pred in zip(
            unique_query, binary_matrix.to_numpy(), query2audio_matrix
        ):
            # tag wise performance
            tag_wise_results[query] = {
                "rocauc": metrics.roc_auc_score(gt, pred, average=None),
                "prauc": metrics.average_precision_score(gt, pred, average=None),
            }
        print(query_to_audio_results)
        print(tag_wise_results)
        with open(
            os.path.join(save_dir, args.dataset, f"tag_wise.json"), "w"
        ) as json_file:
            json.dump(tag_wise_results, json_file, indent=4)
    with open(
        os.path.join(save_dir, args.dataset, f"{args.eval_query}.json"), "w"
    ) as json_file:
        json.dump(query_to_audio_results, json_file, indent=4)

    print(f"Evaluation Done! Results are saved in {save_dir}/{args.dataset}/")


if __name__ == "__main__":
    main(args)
