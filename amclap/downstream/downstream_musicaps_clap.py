import os
import argparse

import json
import laion_clap
import torch
from tqdm import tqdm
from sklearn import metrics

from retrieval.musicaps_dataset import MusicCaps
from retrieval.query_utils import query_processor
from retrieval.metrics import (
    recall,
    mean_average_precision,
    mean_reciprocal_rank,
    median_rank,
)
from retrieval.eval_utils import get_query2target_idx, get_task_predictions

parser = argparse.ArgumentParser(description="")
parser.add_argument("--data_dir", type=str, default="../../dataset")
# parser.add_argument("--data_type", type=str, default="music_caps")
parser.add_argument("--audio_loader", type=str, default="ffmpeg")
parser.add_argument("--eval_query", type=str, default="caption")
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--caption_type", type=str, default="meta_tag_caption_sim")
# train confing
parser.add_argument("--tid", default="base", type=str)
parser.add_argument("--ckpt_type", default="last", type=str)
args = parser.parse_args()

duration = 10
sr = 48000


def main(args):
    save_dir = f"exp/lainon_clap/{args.caption_type}"
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()  # download the default pretrained checkpoint.
    model = model.to(args.device)
    model.eval()

    # model, sr, duration = load_ttmr_pp(save_dir, model_types=args.ckpt_type)
    # print_model_params(model)
    # if args.data_type == "music_caps":
    data_type = "music_caps"
    data_holder = MusicCaps
    split = "test"
    # elif args.data_type == "song_describer":
    # data_holder = SongDescriber
    # split = "is_valid_subset"

    dataset = data_holder(
        data_dir=args.data_dir,
        split=split,
        audio_loader=args.audio_loader,
        caption_type="caption",
        sr=sr,
        duration=duration,
        audio_enc=".wav",
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
        # try:
        audio = dataset.get_audio(audio_id)
        # print("audio shape", audio.shape)
        if audio.shape[0] == int(sr * duration):
            audio = audio.unsqueeze(0)  # for pre-batch
        else:
            audio = audio.squeeze(0)  # for pre-chunk
        with torch.no_grad():
            audio_embs = model.get_audio_embedding_from_data(
                x=audio.to(args.device), use_tensor=True
            ).mean(0, True)
        # except Exception as e:
        #     audio_embs = torch.zeros(1, 512)  # dummy for debug

        audio_features.extend(audio_embs.detach().cpu())

    # extract unique_query
    for query in tqdm(unique_query):
        with torch.no_grad():
            query_embs = model.get_text_embedding([query], use_tensor=True)
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
    os.makedirs(os.path.join(save_dir, data_type), exist_ok=True)
    if args.eval_query == "caption":
        query_to_audio_results = {
            "recall@1": recall(query2audio_matrix, query2track_idx, top_k=1),
            "recall@5": recall(query2audio_matrix, query2track_idx, top_k=5),
            "recall@10": recall(query2audio_matrix, query2track_idx, top_k=10),
            "map@10": mean_average_precision(
                query2audio_matrix, query2track_idx, top_k=10
            ),
            "mean_reciprocal_rank": mean_reciprocal_rank(
                query2audio_matrix, query2track_idx
            ),
            "median_rank": medrank,
        }
        with open(
            os.path.join(save_dir, data_type, f"caption2rank.json"), "w"
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
        with open(
            os.path.join(save_dir, data_type, f"tag_wise.json"), "w"
        ) as json_file:
            json.dump(tag_wise_results, json_file, indent=4)
    with open(
        os.path.join(save_dir, data_type, f"{args.eval_query}.json"), "w"
    ) as json_file:
        json.dump(query_to_audio_results, json_file, indent=4)

    print(f"Evaluation Done! Results are saved in {save_dir}/{data_type}/")


if __name__ == "__main__":
    main(args)
