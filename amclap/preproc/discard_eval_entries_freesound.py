import json
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("freesound_file", type=str, help="Path to the freesound filelist")

args = parser.parse_args()
freesound_file = Path(args.freesound_file)

fsd50k_eval_file = (
    "/data/shared/<user>/data/src/clap-mtg/src/preproc/eval_clips_info_FSD50K.json"
)


with open(fsd50k_eval_file, "r") as f:
    keys = json.load(f).keys()
    keys = set(keys)

print(f"Number of eval entries in FSD50K: {len(keys)}")


with open(freesound_file, "r") as f:
    filelist = f.readlines()
print(f"Data before filtering: {len(filelist)}")

filtered_data = []
for line in tqdm(filelist, desc="Filtering eval entries"):
    name = Path(line).stem
    fid = name.split("_")[0]
    if fid not in keys:
        filtered_data.append(line)
print(f"Data after filtering: {len(filtered_data)}")


freesound_output_file = freesound_file.with_stem(freesound_file.stem + "_noeval")
print(f"Writing filtered data to {freesound_output_file}")
with open(freesound_output_file, "w") as f:
    f.writelines(filtered_data)
print("Done!")
