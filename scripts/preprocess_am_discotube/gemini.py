from google import genai
from google.genai import types

client = genai.Client()

query = """

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
YouTube video description: Leslie West - Got Blooze (2005)
Guitar hero Leslie West has issued countless albums over the years, mostly either as a member of Mountain or as a solo artist. But he has never set out to record an album comprised entirely of classic blues rockers. Then 2005's Got Blooze came along. While this sort of thing has been done by countless fellow veteran classic rock acts of late (the best-known of the bunch being Aerosmith's Honkin' on Bobo), it turns out to be custom-made for a player like West. Throughout the 12-track set, West keeps things raw and gritty, as evidenced by such standouts as the extraordinary "Third Degree" and a cover of the oft-overlooked Cream gem, "Politician." West also surrounds himself with a fantastic rhythm section, comprised of Vanilla Fudge's Tim Bogert on bass and session ace Aynsley Dunbar on drums (turns out the pair pulled double duty, as they also backed ex-Ozzy Osbourne guitarist Jake E. Lee on an additional album full of covers, Retraced). While we probably could have done without the umpteenth cover of "Baby Please Don't Go" (AC/DC, Aerosmith, Ted Nugent, and countless others have played it over the years), overall, Got Blooze is a welcome return for West, and easily his strongest, most focused work in quite some time.

Tracks

01. Baby Please Don't Go      4:09
02. Third Degree      4:17
03. Louisiana Blues      4:53
04. I Can't Quit You      4:49
05. Riot in Cell Block # 9      3:51
06. House of the Rising Sun      4:34
07. ( Look Over ) Yonder's Wall      3:33
08. The Sky Is Crying      5:39
09. Politician      4:44
10. The Thrill Is Gone      5:36
11. Walk in My Shadow      3:38

Credits

Mae Boren Axton Composer
Tim Bogert Bass
Eddie Boyd Composer
P. Brown Composer
Jack Bruce Composer
Kevin Curry Arranger, Guitar (Acoustic), Guitar (Rhythm), Guitar Engineer
Rick Darnell Composer
Willie Dixon Composer
Aynsley Dunbar Drums
Tommy Durden Composer
Rob Fraboni Mastering
Andy Fraser Composer
Roy Hawkins Composer
E. James Composer
Elmore James Composer
Michael Lardie Bass Engineer, Drum Engineering
Jerry Leiber Composer
Clarence Lewis Composer
Chris Marksbury Photography
McKinley Morganfield Composer
Elvis Presley Composer
Morgan Robinson Composer
Paul Rodgers Composer
Dave Stephens Cover Design, Graphic Design
Mike Stoller Composer
Mike Varney Producer
Leslie West Guitar, Guitar (Rhythm), Producer, Slide Guitar, Vocals
YouTube tags: Leslie, West, Heartbreak, Hotel
Discogs genres: Rock
Discogs styles: Rock---Blues Rock, Rock---Classic Rock, Rock---Hard Rock
AllMusic moods: nan
AllMusic themes: nan
AllMusic styles: Hard Rock
Countries of release: Russia, Netherlands, US
Expert review:
Guitar hero Leslie West has issued countless albums over the years, mostly either as a member of Mountain or as a solo artist. But he has never set out to record an album comprised entirely of classic blues rockers. Then 2005's Got Blooze came along. While this sort of thing has been done by countless fellow veteran classic rock acts of late (the best-known of the bunch being Aerosmith's Honkin' on Bobo), it turns out to be custom-made for a player like West. Throughout the 12-track set, West keeps things raw and gritty, as evidenced by such standouts as the extraordinary "Third Degree" and a cover of the oft-overlooked Cream gem, "Politician." West also surrounds himself with a fantastic rhythm section, comprised of Vanilla Fudge's Tim Bogert on bass and session ace Aynsley Dunbar on drums (turns out the pair pulled double duty, as they also backed ex-Ozzy Osbourne guitarist Jake E. Lee on an additional album full of covers, Retraced). While we probably could have done without the umpteenth cover of "Baby Please Don't Go" (AC/DC, Aerosmith, Ted Nugent, and countless others have played it over the years), overall, Got Blooze is a welcome return for West, and easily his strongest, most focused work in quite some time.
"""

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=[query],
    config=types.GenerateContentConfig(temperature=0.1),
)
print(response.text)
