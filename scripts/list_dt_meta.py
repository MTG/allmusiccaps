import argparse
import json

parser = argparse.ArgumentParser(
    description="Count Discogs/YouTube ID overlap between the AllMusic dump and the YouTube->Discogs mapping."
)
parser.add_argument(
    "--allmusic-dump",
    required=True,
    help="Path to allmusic-discogs-dump.json (keys: discogs release IDs)",
)
parser.add_argument(
    "--yt2discogs",
    required=True,
    help="Path to youtube_to_discogs_total_clean.jsonl (one {youtube_id: [discogs_id, ...]} per line)",
)
args = parser.parse_args()

with open(args.allmusic_dump, "r") as f:
    allm = json.load(f)

dids = set(allm.keys())
print(f"Number of discogs ids in allmusic-discogs-dump: {len(dids)}")

yt2dt_file = args.yt2discogs

yt2dt = dict()
with open(yt2dt_file, "r") as f:
    lines = f.readlines()
    for line in lines:
        yd = json.loads(line)
        for k, v in yd.items():
            inter = set(v).intersection(dids)
            if len(inter) > 0:
                yt2dt[k] = list(inter)

yids = set(yt2dt.keys())
print(f"Number of youtube ids in youtube_to_discogs_total_clean {len(yids)}")


allm_rev = {k: v for k, v in allm.items() if not v["review"]}
dids_rev = set(allm_rev.keys())

yids_rev = {k: v for k, v in yt2dt.items() if len(set(v).intersection(dids_rev)) > 0}

print(f"Number of discogs ids in allmusic-discogs-dump with review: {len(dids_rev)}")
print(
    f"Number of youtube ids in youtube_to_discogs_total_clean with review {len(yids_rev)}"
)
