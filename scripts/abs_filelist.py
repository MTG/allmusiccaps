from pathlib import Path

input_file = "yt_ids"
output_file = "yt_filelist"
base_path = Path("/mnt/projects/discotube/youtube-downloads/")

keys = list()
with open(input_file, "r") as f:
    for line in f.readlines():
        line = line.strip()
        path = base_path / line[:2] / f"{line}.mp4"
        keys.append(str(path))

with open(output_file, "w") as f:
    for key in keys:
        f.write(f"{key}\n")
