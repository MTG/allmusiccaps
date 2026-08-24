from huggingface_hub import HfApi

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
for n in names:
    api.update_repo_settings(f"mtg-upf/allmusiccaps_{n}", private=False)
api.update_repo_settings("mtg-upf/allmusiccaps", repo_type="dataset", private=False)
