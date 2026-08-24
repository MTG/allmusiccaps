from huggingface_hub import HfApi

COLLECTION = "mtg-upf/allmusiccaps-6a8cd5806a5d53fb80bdf016"

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

# Adding an item that is already in the collection raises 409, so skip the ones
# already present rather than let a re-run fail part way through.
existing = {item.item_id for item in api.get_collection(COLLECTION).items}

for n in names:
    repo_id = f"mtg-upf/allmusiccaps_{n}"
    if repo_id in existing:
        print(f"already in collection: {repo_id}")
        continue
    api.add_collection_item(COLLECTION, repo_id, item_type="model")
    print(f"added: {repo_id}")

if "mtg-upf/allmusiccaps" in existing:
    print("already in collection: mtg-upf/allmusiccaps (dataset)")
else:
    api.add_collection_item(COLLECTION, "mtg-upf/allmusiccaps", item_type="dataset")
    print("added: mtg-upf/allmusiccaps (dataset)")
