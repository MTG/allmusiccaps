import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset("seungheondoh/enrich-msd", split="train")
df = dataset.to_pandas().set_index("track_id")
filtered = df[df["pseudo_caption"].apply(lambda x: x != "")]

ids = list(filtered.index)
print(f"Loaded {len(ids)} track IDs from the dataset.")
base_path = Path("/projects/<group>/audio/incoming/millionsong-audio/mp3")
metadata_path = Path("/projects/<group>/projects/mtg_text_audio/metadata/msd/")

split_ratio = 0.9

filenames = []
for id in tqdm(ids):
    rel_path = f"{id[2]}/{id[3]}/{id[4]}/{id}.mp3"
    abs_path = base_path / rel_path

    if abs_path.exists():
        filenames.append(rel_path)

print(f"Found {len(filenames)} out of {len(ids)} files.")

random.shuffle(filenames)

train_filenames = filenames[: int(len(filenames) * split_ratio)]
val_filenames = filenames[int(len(filenames) * split_ratio) :]

metadata_path.mkdir(parents=True, exist_ok=True)

with open(metadata_path / "filelist_mp3.txt", "w") as f:
    for fn in filenames:
        f.write(f"{fn}\n")

with open(metadata_path / "filelist_train_mp3.txt", "w") as f:
    for fn in train_filenames:
        f.write(f"{fn}\n")

with open(metadata_path / "filelist_val_mp3.txt", "w") as f:
    for fn in val_filenames:
        f.write(f"{fn}\n")
