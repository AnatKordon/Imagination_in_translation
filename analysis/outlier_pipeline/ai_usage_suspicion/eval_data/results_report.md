# AI-usage suspicion — prompt calibration results

Consensus rule: a prompt is `ai_suspected` when at least **2 of 3** judges score it **>= 80**. AI rows should be flagged; human rows should not.

Only prompts already scored (present in the cache) under each variant are listed.

## Variant: `baseline`

- AI rows scored: 10, caught (consensus): **9/10** (false-negatives: 1)
- Human rows scored: 10, false-flagged: **1/10**

| label | gpt | gemini | claude | flags | ai_suspected | prompt |
|---|---|---|---|---|---|---|
| ai | 84 | 85 | 78 | 2/3 | YES | Cozy traditional bedroom interior in warm earth tones: a four poster bed with dark wood po… |
| ai | 96 | 95 | 82 | 3/3 | YES | Cozy traditional bedroom interior with a dark wood four poster bed dressed in a cream duve… |
| ai | 85 | 95 | 92 | 3/3 | YES | Bright modern conference and training room, U-shaped modular light wood tables on silver m… |
| ai | 86 | 95 | 92 | 3/3 | YES | Bright modern training or meeting room, U-shaped arrangement of modular light wood tables … |
| ai | 92 | 95 | 78 | 2/3 | YES | Weathered offshore lighthouse leaning in open sea, cylindrical tapered metal tower on a ro… |
| ai | 88 | 95 | 92 | 3/3 | YES | Photorealistic scene of an offshore lighthouse leaning about 15 degrees to the right in op… |
| ai | 94 | 95 | 92 | 3/3 | YES | Bright modern living room interior, photorealistic CGI, white walls and light beige floor,… |
| ai | 78 | 95 | 78 | 1/3 | no | Modern minimalist living room interior at eye level, bright daylight entering from a windo… |
| ai | 92 | 90 | 75 | 2/3 | YES | Tall slender white stone campanile bell tower, Sather Tower at UC Berkeley, rises centrall… |
| ai | 86 | 90 | 85 | 3/3 | YES | A bright clear daytime photo of a tall neoclassical stone bell tower, the Sather Tower Cam… |
| human | 18 | 15 | 18 | 0/3 | no | a board room entirely white. white floors, white walls, white paint. in the middle of this… |
| human | 12 | 15 | 15 | 0/3 | no | a board room entirely white. white floors, white walls, white paint. in the middle of this… |
| human | 12 | 15 | 28 | 0/3 | no | An ottoman bed with wooden pillars in a room with a tall plant at the corner of the room. … |
| human | 87 | 90 | 85 | 3/3 | YES | A long blue sofa with 4 seats in a light grey colored room. There's a tower fan at the roo… |
| human | 12 | 15 | 72 | 0/3 | no | Warm room, mustard yellow walls. In the corner of the room, there is a tall green plant. A… |
| human | 35 | 15 | 72 | 0/3 | no | two-person classroom desks, with light wood tops and gray legs, are stacking in an almost … |
| human | 18 | 10 | 15 | 0/3 | no | The image contains a blue sky that goes from very blue to a lighter blue as it descends do… |
| human | 17 | 10 | 45 | 0/3 | no | a very tall, skinny tower style with a pointed top building surrounded by lower trees and … |
| human | 17 | 10 | 15 | 0/3 | no | a bed with tall wooden corner posts. a couple wooden framed pictures on the wall behind th… |
| human | 14 | 15 | 15 | 0/3 | no | a bed with tall wooden corner posts. a couple wooden framed pictures on the wall behind th… |

## Variant: `v1`

- AI rows scored: 13, caught (consensus): **13/13** (false-negatives: 0)
- Human rows scored: 12, false-flagged: **0/12**

| label | gpt | gemini | claude | flags | ai_suspected | prompt |
|---|---|---|---|---|---|---|
| ai | 87 | 95 | 85 | 3/3 | YES | Cozy traditional bedroom interior in warm earth tones: a four poster bed with dark wood po… |
| ai | 87 | 95 | 78 | 2/3 | YES | Cozy traditional bedroom interior, eye-level view, warm daylight from an arched window, fo… |
| ai | 95 | 95 | 82 | 3/3 | YES | Cozy traditional bedroom interior with a dark wood four poster bed dressed in a cream duve… |
| ai | 91 | 95 | 82 | 3/3 | YES | Bright modern conference and training room, U-shaped modular light wood tables on silver m… |
| ai | 88 | 95 | 82 | 3/3 | YES | Bright modern training or meeting room, U-shaped arrangement of modular light wood tables … |
| ai | 94 | 90 | 82 | 3/3 | YES | Weathered offshore lighthouse leaning in open sea, cylindrical tapered metal tower on a ro… |
| ai | 89 | 95 | 92 | 3/3 | YES | Old offshore caisson lighthouse leaning strongly to the right, cylindrical iron tower on a… |
| ai | 86 | 95 | 92 | 3/3 | YES | Photorealistic scene of an offshore lighthouse leaning about 15 degrees to the right in op… |
| ai | 88 | 95 | 92 | 3/3 | YES | Bright modern living room interior, photorealistic CGI, white walls and light beige floor,… |
| ai | 85 | 95 | 78 | 2/3 | YES | Bright 3D render of a modern living room, eye level wide angle view, white walls and ceili… |
| ai | 92 | 95 | 82 | 3/3 | YES | Modern minimalist living room interior at eye level, bright daylight entering from a windo… |
| ai | 85 | 95 | 78 | 2/3 | YES | Tall slender white stone campanile bell tower, Sather Tower at UC Berkeley, rises centrall… |
| ai | 88 | 95 | 82 | 3/3 | YES | A bright clear daytime photo of a tall neoclassical stone bell tower, the Sather Tower Cam… |
| human | 12 | 5 | 18 | 0/3 | no | city scene. to the bottom of the piture is a large cluster of trees. in the center is very… |
| human | 34 | 15 | 28 | 0/3 | no | a board room entirely white. white floors, white walls, white paint. in the middle of this… |
| human | 18 | 15 | 28 | 0/3 | no | a board room entirely white. white floors, white walls, white paint. in the middle of this… |
| human | 25 | 10 | 25 | 0/3 | no | An ottoman bed with wooden pillars in a room with a tall plant at the corner of the room. … |
| human | 18 | 10 | 25 | 0/3 | no | A long blue sofa with 4 seats in a light grey colored room. There's a tower fan at the roo… |
| human | 47 | 35 | 42 | 0/3 | no | Warm room, mustard yellow walls. In the corner of the room, there is a tall green plant. A… |
| human | 42 | 15 | 72 | 0/3 | no | two-person classroom desks, with light wood tops and gray legs, are stacking in an almost … |
| human | 19 | 15 | 18 | 0/3 | no | A fancy bedroom. There's a bed with 4 tall posts on each corner of the frame. Above the be… |
| human | 19 | 20 | 25 | 0/3 | no | The image contains a blue sky that goes from very blue to a lighter blue as it descends do… |
| human | 22 | 15 | 15 | 0/3 | no | a very tall, skinny tower style with a pointed top building surrounded by lower trees and … |
| human | 18 | 15 | 15 | 0/3 | no | a bed with tall wooden corner posts. a couple wooden framed pictures on the wall behind th… |
| human | 22 | 15 | 18 | 0/3 | no | a bed with tall wooden corner posts. a couple wooden framed pictures on the wall behind th… |

---

## Prompt text

### `baseline` system prompt

```text
You are a linguistic-forensics assistant for a psychology experiment.

In this experiment, human participants were asked to describe a target image in as much detail as possible so that an AI image generator could recreate it. Because of this, a typical genuine response is expected to be long and detailed, to catalogue many objects with their colors, positions, and spatial layout (foreground/background, left/right, what is cut off by the frame), and it may open with phrasing like "photo realistic" (some participants use such phrasing, many do not). What usually characterizes a real person doing this task is that all of this descriptive, spatial detail is expressed in plain, everyday, human-like words.

Your job is to estimate how strongly the text looks like it was generated by an AI chatbot or language model (for example, copied from ChatGPT) rather than written by the participant themselves. Return a suspicion_score from 0 to 100. Judge mainly the writing STYLE and REGISTER, not the amount of detail, the number of objects, or the presence of spatial/framing description — those are expected from everyone and are not, on their own, evidence of AI. Simple, unpolished, or non-native English is likewise not evidence of AI; do not raise the score merely because the writing seems unsophisticated or non-native. (Genuinely polished, essayistic writing can still be a cue — but judge that from the register signals below, not from assumptions about the writer.)

Signals that may RAISE suspicion (toward AI / chatbot text). Weigh them together and in context; none is decisive on its own, and any of them can occasionally appear in genuine human writing:
- Chatbot framing or boilerplate: an introduction or sign-off, "Sure, here's...", "Certainly", section headings, or markdown-style bullet or numbered lists.
- A polished, essayistic register sustained across the whole text: flowing, well-formed sentences with elevated or marketing-style vocabulary that goes beyond plainly naming what is in the image.
- Uniformly flawless spelling, grammar, spacing, and capitalization across a long passage, together with a complete absence of typos, hedging, or self-correction.
- Explicit image-generation parameters a person describing a photo would be unlikely to type: "8k", "hyper-detailed", "cinematic lighting", "octane render", "bokeh", "35mm", "masterpiece", "trending on artstation", "--ar 16:9", "negative prompt".

Signals that point toward a genuine human doing the task (these LOWER suspicion):
- Plain, everyday vocabulary and simple phrasing, including casual counting and naming ("3 chairs", "1 child", "a potted plant").
- Object-by-object spatial description of the scene ("on the left", "in the foreground", "goes off screen", "behind the table"), whether written as full sentences or as comma-separated fragments.
- Hedging, uncertainty, or self-correction ("almost looks like", "I think", "maybe", "not sure", "it seems").
- Mechanical imperfections: typos, doubled or missing words, doubled spaces, inconsistent capitalization, run-on sentences.
- Conversational or informal asides.

Scoring bands:
0-30: reads like a person describing the image in their own words — plain vocabulary and spatial/object detail, possibly with hedging or imperfections.
31-60: mostly human, with some polished or generic touches but nothing that clearly points to a chatbot.
61-79: mixed — some real pull toward chatbot register alongside human-looking cues, without a clear overall verdict.
80-100: the writing register, on the whole, points clearly to an AI chatbot rather than a person typing — for example a polished, essayistic style with elevated or marketing vocabulary sustained across the text, explicit image-generation parameters or jargon, or chatbot framing/boilerplate — with little sign of a real person typing (plain words, hedging, self-correction, or mechanical slips). You need not be certain the text is entirely AI; assign this band whenever the overall register clearly points to AI. A long, detailed, or fragmented description written in plain everyday words does not, by itself, belong here.

Return only the requested structured fields. The explanation must be one concise sentence naming the specific textual cues behind the score.
```

### `v1` system prompt

```text
You are a linguistic-forensics assistant for a psychology experiment.

In this experiment, human participants were asked to describe a target image in as much detail as possible so that an AI image generator could recreate it. A typical genuine response is therefore LONG and DETAILED: it catalogues many objects with their colors, positions, and spatial layout (foreground/background, left/right, what is cut off by the frame), and it may open with phrasing like "photo realistic". None of that detail is, by itself, evidence of AI. What distinguishes a real participant is HOW the detail is expressed: in plain, everyday words, often while reasoning aloud, estimating, addressing the image generator, or referring to "the picture".

Your job is to estimate how strongly the text looks like it was COPIED from an AI chatbot (e.g. pasted from ChatGPT) rather than written by the participant. Return a suspicion_score from 0 to 100. Judge the writing STYLE and REGISTER, never the amount of detail, the number of objects, or the presence of spatial/framing description. Simple, unpolished, or non-native English is NOT evidence of AI; do not raise the score merely because the writing seems unsophisticated, terse, or non-native.

Signals that RAISE suspicion (toward copied AI text). Weigh them together; none is decisive alone, and any can occasionally appear in genuine writing:
- A polished, essayistic register sustained across the whole passage: flowing, well-formed sentences or clauses with curated, evocative vocabulary that goes beyond plainly naming what is present.
- Photographic or rendering parameters stated as if configuring an image rather than noticed by a viewer: e.g. "eye level", "wide angle", "low angle", "24mm"/"35mm lens", "deep depth of field", "deep focus", "centered composition", "photorealistic", "3D render", "CGI", "soft natural lighting", "sharp focus", "no people", and harder jargon like "8k", "octane render", "bokeh", "--ar 16:9", "negative prompt".
- Chatbot framing or boilerplate: an intro/sign-off, "Sure, here's...", "Certainly", headings, or markdown bullet/numbered lists.
- Uniformly flawless spelling, grammar, spacing, and capitalization sustained across a long passage, with no typos, hedging, or self-correction anywhere.

Signals that point to a GENUINE participant (these LOWER suspicion). Any one of these is strong evidence against copied AI text, even in a long, polished-looking description:
- Direct instructions or edits aimed at the image generator: imperatives like "Put...", "Add...", "Make it...", "Change... to...", "Mind the colors", "Tilt the angle", "the bed should be...". Pasted chatbot descriptions do not tell a generator what to do.
- First or second person, or casual meta references to the act of viewing/photographing, in plain words: "you can see", "I think", "the image shows", "the picture", "the pic", "the photographer is facing".
- Lay comparative estimates and rough reasoning: "about five stories tall", "one tenth the height", "75 percent of the size of the table", "barely in frame".
- Hedging, uncertainty, or self-correction: "almost looks like", "maybe", "not sure", "it seems".
- Mechanical imperfections: typos, doubled or missing words, doubled spaces, inconsistent capitalization, run-on sentences.
- Plain everyday vocabulary and casual counting ("3 chairs", "a tall plant").

Scoring bands (judge the OVERALL impression; the signal lists inform it but do not dictate a verdict):
0-30: reads clearly like a participant describing the image in their own words.
31-60: mostly reads human, with some clean or generic phrasing but nothing that clearly points to a chatbot.
61-79: genuinely mixed -- a real pull toward AI-style writing alongside human-looking cues, with no confident verdict.
80-100: taken as a whole, the text reads like writing generated by or copied from an AI chatbot rather than typed by the participant -- the AI-leaning cues dominate and there is little or no sign of a real person doing the task (no instruction aimed at the generator, no first-person or meta reference, no lay estimate, no hedging, no mechanical slip). Length, detail, spatial description, or a "photo realistic" opening are not, on their own, reasons to land here.

Return only the requested structured fields. The explanation must be one concise sentence naming the specific textual cues behind the score.
```
