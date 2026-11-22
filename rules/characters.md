# RULE: Characters — Bible, Card, Dossier, LoRA

## Pipeline (automatic from first appearance)

1. **Bible first** (`characters/characters.json`): prompt_tags +
   danbooru_tags per character. New character = bible entry BEFORE any
   generation. No bible entry, no identity. Enforced by the characters
   gate (a bible entry with no tags aborts before GPU spend).
2. **Identity** is the bible's tags: they go into the text prompt of every
   shot that character is in, and they are what keeps a face on-model
   across an episode. A trained Qwen LoRA is a STRONGER identity source
   layered on top of them — never a requirement.
3. **Character card** (`characters/sheets/<key>_sheet.png` + 6 labeled
   views): a multi-view portrait of ONE character, generated from the
   bible's tags. It is a **TRAINING SET, not a render input** — the still
   path never loads it (an anchored scene repeated one picture and dragged
   the card's flat backdrop into the scene). It exists so a character who
   has never been rendered can still get a LoRA, and it is generated ONLY
   when that character has no training images yet.
4. **Dossier** (`characters/pages/<key>.html` + index): card, views, bio
   table, copyable prompt tags, voice player. One page = full profile.
5. **LoRA** (`models/loras/<key>*.safetensors`, trigger = key): 10-20
   captioned 1024px training images STAGED in `dataset/<key>_lora/` —
   a PNG plus a `.txt` caption each (the bible's danbooru_tags is written
   for any missing caption). Stage them with
   `python _jobs/stage_dataset_from_stills.py <key>`. rank16/alpha8,
   unet 1e-4 / TE 5e-5, cosine, bf16, isolated `_training/` env.
   Eye-verify trigger+description before wiring.

## Trigger discipline

- LoRA fires ONLY with `<key>` in the prompt + description support
  (trigger alone draws empty streets — verified).
- The trigger is PER CHARACTER (`lora_trigger` in the bible). The global
  `qwen_lora_trigger` is the fallback for `qwen_lora_trigger_owner` only,
  and must never ride another character's LoRA.
- The pipeline attaches the subject's LoRA at `lora_strength` for every
  shot that character is in, on the Qwen-Image-2.1 path as well as SDXL.
- Caption pruning is load-bearing: hair/eye/coat words OUT of captions
  (bound to trigger), pose/setting/background words IN.

## Frequent vs low-appearance characters (owner, 2026-11-18)

- **Frequent characters get a LoRA** — trained from their staged stills, as
  above. This is the identity source for Alaric, Amon, and the recurring cast.
- **Low-appearance characters use a SHEET, not a LoRA.** Generate a 3-image
  sheet — close-up, front view, back view — on a plain light gray background
  (`python _jobs/gen_3view_sheet.py <key>`), staged in
  `characters/sheets/<key>_*.png`. Describe the front view once with the VLM
  (`python _jobs/gen_sheet_identity.py <key>`) and cache the identity words in
  `characters/sheets/<key>_identity.txt`. `_sheet_identity_phrase()` injects
  those words into BOTH the still and H3 clip prompts when the focal character
  has a sheet and no Qwen LoRA — normal scenes, consistent identity.
- **Never feed the sheet image to the generator.** Measured: VAEEncode-as-latent
  copied the gray sheet verbatim into the shot (even at 0.92 denoise), and
  ReferenceLatent composited a gray panel into the middle of the frame. Text
  identity is the only path that keeps both the character and the scene.
- Once a character earns a LoRA, the LoRA wins automatically: the helper returns
  '' so no stale sheet identity competes with the trained adapter.
