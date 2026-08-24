from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
names = [
    "baseline",
    "amcquotes",
    "amcstruct",
    "baseline_quotes",
    "baseline_struct",
    "layer6",
    "all_layers",
    "sigmoid",
    "lejepa",
    "infonce_sigreg",
    "te_trained",
    "te_trained_sigreg",
]

CITATION = """
## Citation

```bibtex
@inproceedings{alonso2026allmusiccaps,
  title = {{AllMusicCaps}: Album Reviews as Complementary Supervision for Music {CLAP}},
  author = {Alonso-Jim{\\'e}nez, Pablo and Lizarraga-Seijas, Xavier and Serra, Xavier and Bogdanov, Dmitry},
  booktitle = {International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2026},
}
```
"""

for n in names:
    repo_id = f"mtg-upf/allmusiccaps_{n}"
    with open(hf_hub_download(repo_id, "README.md")) as f:
        card = f.read()

    # Re-running must not append a second copy.
    if "## Citation" in card:
        print(f"already has citation: {repo_id}")
        continue

    api.upload_file(
        path_or_fileobj=(card.rstrip() + "\n" + CITATION).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add citation",
    )
    print(f"updated: {repo_id}")
