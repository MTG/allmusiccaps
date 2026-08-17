"""Prompt + schema for 5-level figurative rewrite of MusicCaps captions.

The schema is intentionally flat (5 string fields) so Outlines / vLLM can
constrain generation deterministically and we can round-trip through JSON.
"""

from pydantic import BaseModel, Field


class FigurativeLevels(BaseModel):
    """Five versions of the same caption along a literal→figurative axis.

    Semantics to preserve across all 5 levels:
      - genre, subgenre, era references
      - instrumentation
      - tempo / energy / mood bucket
      - production cues (lo-fi, polished, live, ...)

    What changes across levels:
      - level_1: bare, literal, descriptive. No metaphor.
      - level_2: slightly evocative, still concrete.
      - level_3: mixed literal + figurative.
      - level_4: mostly figurative / evaluative / comparative language.
      - level_5: highly figurative, metaphorical, poetic; could appear in
                 an album review.
    """

    level_1: str = Field(..., description="Bare, literal, descriptive. No metaphor.")
    level_2: str = Field(..., description="Slightly evocative, still concrete.")
    level_3: str = Field(..., description="Mixed literal and figurative.")
    level_4: str = Field(
        ..., description="Mostly figurative / evaluative / comparative."
    )
    level_5: str = Field(..., description="Highly figurative, poetic review-style.")


LEVELS = ("level_1", "level_2", "level_3", "level_4", "level_5")


PROMPT_TEMPLATE = """You are rewriting a music caption into 5 progressively more figurative versions.

CONSTRAINTS:
- Preserve the EXACT same musical content across all 5 versions:
  the same genre/subgenre, instruments, tempo, energy, mood, and production style.
- Do NOT add new instruments, new genres, new artists, or new era references
  that were not present (directly or implied) in the original caption.
- Do NOT change the factual content. Only change the STYLE of language.
- Each rewrite must stand alone as a standalone music description (1–3 sentences).
- Keep each rewrite roughly the same length as the original (±50%).

LEVELS (increasing figurativeness):
- level_1: bare, literal, descriptive. Plain clinical prose. No metaphor, no simile,
           no evaluative adjectives like "haunting" or "lush".
- level_2: slightly evocative but still concrete. At most one mild figurative touch.
- level_3: mixed literal and figurative. Some metaphor or evaluative language,
           but still grounded in concrete descriptors.
- level_4: mostly figurative / evaluative / comparative. The kind of language a
           music critic would use. Metaphor, simile, sensory imagery.
- level_5: highly figurative, metaphorical, poetic. The kind of sentence you would
           find in a long-form album review. Rich imagery, figurative tone.

You must output a single JSON object with exactly these keys:
  "level_1", "level_2", "level_3", "level_4", "level_5".

Do not include any other text, explanation, preamble, or postscript.

ORIGINAL CAPTION:
{caption}

JSON OUTPUT:
"""


def format_prompt(caption: str) -> str:
    """Render the 5-level rewrite prompt for a single caption.

    The caller is responsible for length-checking the caption against the
    model's max_model_len — MusicCaps captions are short (~50 tokens),
    so in practice this is never an issue.
    """
    caption = (caption or "").strip().replace("\r\n", "\n").replace("\r", "\n")
    return PROMPT_TEMPLATE.format(caption=caption)
