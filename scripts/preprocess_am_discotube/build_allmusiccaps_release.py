"""Build the preliminary AllMusicCaps release JSONL.

Joins three inputs into a single JSONL with exactly these fields:

    youtube_id, discogs_release_id, allmusic_album_id,
    youtube_url, discogs_release_url, allmusic_review_link,
    generated_quotes_captions, generated_structured_captions

Inputs (on the SLURM cluster):

    --reviews PATH    JSONL, one object per line:
                      {"youtube_id", "releaseid_discogs", "releaseid_allmusic",
                       "review": {"heading", "text"}, ...}

    --quotes PATH     JSONL, each line is a single-key dict
                      {youtube_id: {discogs_release_id: [caption, caption, ...]}}

    --struct PATH     JSONL, each line is a single-key dict
                      {youtube_id: {music_style, mood, tempo, energy,
                                    instrumentation, production_style}}

Output:

    --out PATH        JSONL, one row per (youtube_id, discogs_release_id)
                      pair that appears in the reviews file AND has at least
                      one of {quotes, struct} captions available.

A row is emitted even if only one of the two caption sources is present;
the missing field is set to null. This lets reviewers see coverage gaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


YOUTUBE_URL_TMPL = "https://www.youtube.com/watch?v={}"
DISCOGS_URL_TMPL = "https://www.discogs.com/release/{}"
ALLMUSIC_URL_TMPL = "https://www.allmusic.com/album/{}"


def load_quotes(path: Path) -> dict[tuple[str, str], list[str]]:
    """Returns {(youtube_id, discogs_release_id): [caption, ...]}."""
    out: dict[tuple[str, str], list[str]] = {}
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            for yt_id, releases in obj.items():
                for rel_id, captions in releases.items():
                    out[(yt_id, str(rel_id))] = captions
    return out


def load_struct(path: Path) -> dict[str, dict[str, str]]:
    """Returns {youtube_id: {music_style, mood, tempo, energy, instrumentation, production_style}}.

    Note: struct captions are keyed on youtube_id only (no discogs split).
    """
    out: dict[str, dict[str, str]] = {}
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            for yt_id, fields in obj.items():
                out[yt_id] = fields
    return out


def iter_reviews(path: Path):
    """Yields (youtube_id, discogs_release_id, allmusic_album_id, review_dict) tuples."""
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            yt_id = obj.get("youtube_id")
            rel_discogs = obj.get("releaseid_discogs")
            rel_allmusic = obj.get("releaseid_allmusic")
            if not (yt_id and rel_discogs and rel_allmusic):
                continue
            yield yt_id, str(rel_discogs), rel_allmusic, obj.get("review")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--reviews", type=Path, required=True, help="youtube-to-allmusic-review.json"
    )
    p.add_argument(
        "--quotes", type=Path, required=True, help="Qwen2.5-32B quotes JSONL"
    )
    p.add_argument(
        "--struct", type=Path, required=True, help="LLaMA-3.3 promptv3 structured JSONL"
    )
    p.add_argument("--out", type=Path, required=True, help="Output JSONL")
    p.add_argument(
        "--require-both",
        action="store_true",
        help="Drop rows missing either caption source",
    )
    args = p.parse_args()

    print(f"Loading quotes from {args.quotes}", flush=True)
    quotes = load_quotes(args.quotes)
    print(f"  {len(quotes):,} (youtube_id, discogs_release_id) keys", flush=True)

    print(f"Loading struct from {args.struct}", flush=True)
    struct = load_struct(args.struct)
    print(f"  {len(struct):,} youtube_id keys", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_only_quotes = 0
    n_only_struct = 0
    n_both = 0
    n_neither = 0

    with args.out.open("w") as out_f:
        for yt_id, rel_discogs, rel_allmusic, _review in iter_reviews(args.reviews):
            n_in += 1
            quotes_caps = quotes.get((yt_id, rel_discogs))
            struct_caps = struct.get(yt_id)

            if quotes_caps and struct_caps:
                n_both += 1
            elif quotes_caps:
                n_only_quotes += 1
            elif struct_caps:
                n_only_struct += 1
            else:
                n_neither += 1
                continue

            if args.require_both and not (quotes_caps and struct_caps):
                continue

            row = {
                "youtube_id": yt_id,
                "discogs_release_id": rel_discogs,
                "allmusic_album_id": rel_allmusic,
                "youtube_url": YOUTUBE_URL_TMPL.format(yt_id),
                "discogs_release_url": DISCOGS_URL_TMPL.format(rel_discogs),
                "allmusic_review_link": ALLMUSIC_URL_TMPL.format(rel_allmusic),
                "generated_quotes_captions": quotes_caps,
                "generated_structured_captions": struct_caps,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Reviews scanned:        {n_in:,}")
    print(f"  both caption sources: {n_both:,}")
    print(f"  quotes only:          {n_only_quotes:,}")
    print(f"  struct only:          {n_only_struct:,}")
    print(f"  neither (skipped):    {n_neither:,}")
    print(f"Rows written to {args.out}: {n_out:,}")


if __name__ == "__main__":
    main()
