"""
Prompt builders for music description generation.

Keeping prompts in a dedicated module makes it easy to compare and swap variants.
"""

from __future__ import annotations

import pandas as pd


def _fmt_list(x) -> str:
    """Normalize list-like and set-like metadata fields to readable strings."""
    if x is None or x == "" or (isinstance(x, (list, set, frozenset)) and len(x) == 0):
        return "N/A"
    if isinstance(x, (list, set, frozenset)):
        # Convert to list if needed and filter None values
        items = list(x) if isinstance(x, (set, frozenset)) else x
        if None in items:
            items = [i for i in items if i is not None]
        return ", ".join(str(i) for i in items)
    return str(x)


prompt_v1 = """
Read the metadata carefully and write short sentences describing the musical piece.
Use JSON format.

## RULES

* Write complete sentences quoting the Review excerpt when possible.
* Avoid track or album titles in the sentences.
* Avoid proper or artist names in the sentences.
* Return N/A if no relevant information is available.
* Only extract factual descriptions related to musical and acoustic characteristics.
* Ignore subjective opinions, biographical information, release dates, or commercial performance mentions.

# OUTPUT INSTRUCTIONS

Answer in valid JSON. Here are the different objects relevant for the output:

Description
    music_style (str): description of the music style and genre
    vibe (str): description of the overall vibe or mood of the music
    tempo_energy (str): description of the tempo and energy level
    instrumentation (str): description of the main instruments and sounds used
    production_notes (str): description of the production style and techniques


# OUTPUT EXAMPLE

    {{
        "music_style": "A fusion of jazz and funk with electronic elements",
        "vibe": "Energetic and upbeat with a danceable groove",
        "tempo_energy": "Fast tempo with high energy throughout",
        "instrumentation": "N/A",
        "production_notes": "Clean production with layered textures and effects"
    }}
"""


prompt_v2 = """
    You are extracting factual musical descriptors from metadata and reviews.

You MUST output a single JSON object with ALL fields defined below.
Do NOT add extra fields.
Do NOT omit any field.
If no relevant information is available for a field, return "N/A".

## OUTPUT SCHEMA (JSON)

{
  "music_style": string,          // genre or stylistic classification
  "vibe": string,                 // overall mood or atmosphere
  "tempo_energy": string,         // tempo and energy level
  "instrumentation": string,      // main instruments or sound sources
  "production_notes": string      // production style or techniques
}

## CONTENT RULES

- Use complete, concise sentences (1–2 short sentences per field).
- Only include factual musical or acoustic characteristics.
- Avoid artist names, track titles, album titles, or proper names.
- Avoid subjective opinions or evaluations.
- Do NOT include biographical, commercial, or release information.
- If quoting the review, paraphrase rather than copy verbatim.

## METADATA AND REVIEW
"""


prompt_v3 = """
You are extracting musical descriptors (genre/style, mood, energy, tempo, instrumentation, production style) from metadata and reviews.

Discogs and AllMusic describe the album; YouTube metadata may provide limited track-level context. Identify the track contextually (without naming it) and prioritize track-level information when available.

Output a single JSON object with all fields defined.
Do not add or omit fields.
Return "N/A" only if no musical information (track- or album-level) is available for that field.

If track-level information is missing, you may conservatively infer from album-level Discogs or AllMusic metadata, consistent with the album’s dominant style and mood. Avoid "N/A" when reasonable musical inference is possible. Evaluate each field independently.

## OUTPUT SCHEMA (JSON)

{
  "music_style": string,       // short sentence describing the genre or stylistic classification inferred from Discogs, AllMusic, and YouTube tags
  "mood": string,              // short sentence describing the overall atmosphere or emotional character
  "energy": string,            // short sentence describing the perceived intensity or dynamic drive (independent of tempo)
  "tempo": string,             // short sentence describing the perceived speed or BPM (independent of intensity)
  "instrumentation": string,   // short sentence describing the main instruments or sound sources
  "production_style": string   // short sentence describing the recording, mixing, or arrangement characteristics
}

## CONTENT RULES

Use complete, concise sentences (1–2 per field).

Infer style from Discogs genres/styles, AllMusic moods/themes/styles, and YouTube tags.

Do not include artist names, track titles, album titles, dates, studios, or people.

Do not include biographical, commercial, or release information.

Production style may include terms such as organic, polished, lo-fi, acoustic-forward, electronic, layered, or minimal when supported by metadata or reviews.

## METADATA AND REVIEW
"""


def format_prompt(row: pd.Series, instruction=prompt_v2, max_len: int = 2000) -> str:
    """Build the base prompt string for one metadata row."""
    # Limit long text fields to 2000 characters
    description = str(row.get("description", "N/A"))[:max_len]
    review_text = str(row.get("review_text", "N/A"))[:max_len]

    prompt = (
        instruction
        + f"""
YouTube title: {_fmt_list(row.get("title", "N/A"))}
YouTube description: {_fmt_list(description)}
YouTube tags: {_fmt_list(row.get("tags", "N/A"))}
Discogs genres: {_fmt_list(row.get("genres", "N/A"))}
Discogs styles: {_fmt_list(row.get("styles", "N/A"))}
AllMusic moods: {_fmt_list(row.get("moods", "N/A"))}
AllMusic themes: {_fmt_list(row.get("themes", "N/A"))}
AllMusic styles: {_fmt_list(row.get("styles_all_music", "N/A"))}
Countries of release: {_fmt_list(row.get("countries", "N/A"))}
Expert review:
{review_text}
""".strip()
    )

    return prompt
