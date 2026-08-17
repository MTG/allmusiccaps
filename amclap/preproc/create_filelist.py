import json

input_file = "/data/shared/<user>/data/discotube/metadata/Qwen_Qwen2.5-32B__chatgpt_v2__t0.5__1.1.jsonl"
output_file = "filelist_dummy.txt"

keys = list()
with open(input_file, "r") as f:
    for line in f.readlines():
        key = list(json.loads(line).keys())[0]
        keys.append(key)

with open(output_file, "w") as f:
    for key in keys:
        f.write(f"{key}\n")
