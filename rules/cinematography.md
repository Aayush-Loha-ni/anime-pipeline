# RULE: Cinematography — direct every shot like a camera, not a caption

Every chapter is shot with an intentional lens, camera move and light. The
director's `scene_plan` owns these decisions; `_apply_director_plan` copies them
onto the scenes, `_cine_phrase()` turns them into prompt language, and the same
phrase reaches BOTH the Qwen still and the H3 clip so a shot and its motion
share one camera.

## Lens (perspective only — never bokeh)
The whole frame is drawn sharp (script rule). Describe perspective/compression,
never "blurry background", "bokeh" or "shallow depth of field".

- close-up → 85mm portrait lens, compressed perspective, tight framing
- extreme close-up → 100mm macro lens, compressed perspective, intimate framing
- medium → 50mm lens, natural perspective
- wide → 35mm wide-angle lens, deep focus
- extreme-wide → 24mm wide-angle lens, deep focus, expansive scale

A scene may override with an explicit `lens` field in the director package.

## Light
From the director's `lighting` field, else the scene's `mood`/`time_of_day`:

- menacing/tense → hard directional key light, deep chiaroscuro shadows, high contrast
- somber/melancholic → soft diffused light, low contrast, muted palette
- hopeful → soft warm key light, gentle rim light
- night → warm gaslamp key light, deep shadows, cool moonlight rim

## Camera movement
Exactly ONE move per shot, from `camera_movement` (single value, never a comma
list): `static|push-in|pull-out|pan-left|pan-right|tracking|handheld`. It reaches
the H3 prompt through `_h3_prompt()`; a still never gets a move, only the lens
and light. No consecutive duplicate compositions (shot_budget rule still holds).

## Why this exists
Before this rule the still prompt carried shot size + mood but no lens or light,
and the H3 prompt carried motion but no camera identity — clips drifted from
their stills and many frames read as the same flat candlelit interior. Lens and
light are now deterministic and versioned with the director package, so all
following videos stay consistent. Do not remove them to "simplify" a prompt.
