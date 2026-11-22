#!/usr/bin/env python3
"""Director's Package generator — the creative bridge between novel and scenes.

Produces a human-editable JSON + Markdown package in scenes/ for every episode.
Never overwrites manual edits unless force=True.
"""

import json
import re
import time
from pathlib import Path

from pipeline.schemas import validate_director_package

# stdlib-only, so this cannot cycle into pipeline.*
from json_util import dump_json_atomic


def _llm_generate(prompt, llm_chat_fn, timeout=900, temperature=0.3,
                  max_tokens=8192):
    """Call the LLM and return parsed JSON or None.

    900/8192 match the script generator's budget: the director package is the
    other long-JSON producer, and it truncates or times out on the old
    300/4096 pair exactly as break_into_scenes did. No CONFIG import here —
    the module is standalone, so the budget is a parameter.
    """
    raw = llm_chat_fn([{"role": "user", "content": prompt}],
                      timeout=timeout, temperature=temperature,
                      max_tokens=max_tokens)
    if not raw:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def generate_director_package(chapter_text, chapter_num, scenes,
                              llm_chat_fn, char_bible=None):
    """Generate a director's package from chapter text and existing scenes.

    Returns (package_dict, errors_list). Package is schema-validated.
    """
    char_context = ""
    if char_bible and char_bible.get("characters"):
        names = ", ".join(char_bible["characters"].keys())
        char_context = f"Characters available: {names}."

    scene_summaries = []
    for s in scenes[:30]:
        scene_summaries.append({
            "scene_number": s.get("scene_number", 0),
            "narration": (s.get("narration") or "")[:100],
            "characters": s.get("characters", []),
            "shot_size": s.get("shot_size", ""),
            "mood": s.get("mood", ""),
        })

    prompt = f"""You are a film director creating a production package for an anime episode.

Given this novel chapter and the scene breakdown, create a DIRECTOR'S PACKAGE as valid JSON.

{char_context}

CHAPTER (first 8000 chars):
{chapter_text[:8000]}

SCENE BREAKDOWN:
{json.dumps(scene_summaries, indent=2)[:4000]}

Return ONLY a valid JSON object with these keys:
{{
  "episode_promise": "one sentence describing what this episode delivers",
  "target_audience": "who this is for",
  "target_emotion": "primary emotion to leave viewers with",
  "cold_open_hook": {{
    "description": "the hook moment (0-20 seconds)",
    "duration_range": "e.g. 0-15s",
    "visual": "what we see",
    "audio_cue": "sound design hint"
  }},
  "central_dramatic_question": "the question this episode answers or poses",
  "midpoint_reversal": "what changes at the midpoint",
  "climax": "the peak moment",
  "ending": "how the episode resolves or hooks next",
  "beat_sheet": [
    {{
      "beat_number": 1,
      "description": "what happens",
      "source_chapter_ref": "which part of the chapter",
      "emotion": "emotional tone"
    }}
  ],
  "scene_plan": [
    {{
      "scene_number": 1,
      "purpose": "HOOK|TURN|LAND",
      "conflict_or_change": "what shifts",
      "focal_character": "who drives this scene",
      "location": "where",
      "time_of_day": "when",
      "narration_or_dialogue": "narration|dialogue|both",
      "visual_motif": "recurring visual element",
      "intended_shot_type": "close-up|medium|wide|extreme-wide",
      "camera_movement": "push-in|pull-out|pan-left|pan-right|tracking|handheld|static",
      "lens": "e.g. 35mm wide-angle, deep focus | 85mm portrait, compressed perspective",
      "lighting": "e.g. warm gaslamp key with deep chiaroscuro shadows | hard high-contrast key",
      "target_duration": "seconds"
    }}
  ],
  "camera_move_diversity": "no single camera_movement may be used in more than 40% of scenes",
  "shot_budget": {{
    "total_shots": {len(scenes)},
    "shot_type_limits": {{"close-up": "max 30%", "wide": "min 20%", "medium": "max 40%"}},
    "composition_rules": "no consecutive duplicate compositions",
    "max_consecutive_same_type": 2,
    "max_same_location": 5
  }},
  "continuity_ledger": {{
    "character_states": {{}},
    "unresolved_promises": [],
    "prohibited_contradictions": []
  }}
}}

RULES:
- Preserve the novel's facts; summarize rather than inventing major plot changes.
- The beat sheet must reference the source chapter.
- Scene plan must cover ALL scenes from the breakdown.
- Shot budget must ban consecutive duplicate compositions.
- Continuity ledger tracks character state, wardrobe, injuries, props, relationships.
- Return ONLY the JSON object, no prose."""

    pkg = _llm_generate(prompt, llm_chat_fn, timeout=900, temperature=0.3)
    if pkg is None:
        return None, ["LLM returned unparseable director package"]

    errors = validate_director_package(pkg)
    if errors:
        # Stamp identity BEFORE the early return. It used to return here, so
        # the caller saved a package with no `version`, and the next run's
        # "is there a package?" check (version != "placeholder") treated the
        # broken file as final — a bad package was permanent and the LLM
        # never got to retry.
        pkg["episode_number"] = chapter_num
        pkg["source_chapter"] = chapter_num
        pkg["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        pkg["version"] = "1.0-invalid"
        return pkg, errors

    pkg["episode_number"] = chapter_num
    pkg["source_chapter"] = chapter_num
    pkg["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    pkg["version"] = "1.0"

    return pkg, []


def save_director_package(pkg, scenes_dir, chapter_num, force=False):
    """Save director's package as JSON + Markdown. Returns paths.

    Never overwrites manual edits unless force=True. A package that failed
    schema validation is never written, whatever force says — a broken
    package on disk is worse than no package, because the load path treats
    any existing file as final.
    """
    scenes_dir = Path(scenes_dir)
    scenes_dir.mkdir(parents=True, exist_ok=True)

    json_path = scenes_dir / f"ch{chapter_num:03d}_director.json"
    md_path = scenes_dir / f"ch{chapter_num:03d}_director.md"

    if str(pkg.get("version", "")).endswith("-invalid"):
        return json_path, md_path, False

    if json_path.exists() and not force:
        return json_path, md_path, False

    # Atomic: load_director_package returns None on a parse error, so a torn
    # write made the whole package silently disappear and get regenerated from
    # scratch next render — losing exactly the manual edits the "never overwrite
    # manual edits" contract above exists to protect.
    dump_json_atomic(json_path, pkg, indent=2, ensure_ascii=False)

    md = _director_to_markdown(pkg, chapter_num)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    return json_path, md_path, True


def load_director_package(scenes_dir, chapter_num):
    """Load an existing director's package. Returns dict or None."""
    json_path = Path(scenes_dir) / f"ch{chapter_num:03d}_director.json"
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _director_to_markdown(pkg, chapter_num):
    """Convert a director package to human-readable Markdown."""
    lines = [f"# Director's Package — Chapter {chapter_num}\n"]

    lines.append(f"**Episode Promise:** {pkg.get('episode_promise', 'TBD')}\n")
    lines.append(f"**Target Audience:** {pkg.get('target_audience', 'TBD')}")
    lines.append(f"**Target Emotion:** {pkg.get('target_emotion', 'TBD')}\n")

    hook = pkg.get("cold_open_hook", {})
    lines.append("## Cold Open Hook\n")
    lines.append(f"- **Description:** {hook.get('description', 'TBD')}")
    lines.append(f"- **Duration:** {hook.get('duration_range', 'TBD')}")
    if hook.get("visual"):
        lines.append(f"- **Visual:** {hook['visual']}")
    if hook.get("audio_cue"):
        lines.append(f"- **Audio:** {hook['audio_cue']}")
    lines.append("")

    lines.append(f"## Central Dramatic Question\n{pkg.get('central_dramatic_question', 'TBD')}\n")
    lines.append(f"## Midpoint Reversal\n{pkg.get('midpoint_reversal', 'TBD')}\n")
    lines.append(f"## Climax\n{pkg.get('climax', 'TBD')}\n")
    lines.append(f"## Ending\n{pkg.get('ending', 'TBD')}\n")

    beats = pkg.get("beat_sheet", [])
    if beats:
        lines.append("## Beat Sheet\n")
        lines.append("| # | Beat | Source | Emotion |")
        lines.append("|---|------|--------|---------|")
        for b in beats:
            lines.append(f"| {b.get('beat_number', '?')} | "
                         f"{b.get('description', '')} | "
                         f"{b.get('source_chapter_ref', '')} | "
                         f"{b.get('emotion', '')} |")
        lines.append("")

    plan = pkg.get("scene_plan", [])
    if plan:
        lines.append("## Scene Plan\n")
        for sp in plan:
            lines.append(f"### Scene {sp.get('scene_number', '?')}: "
                         f"{sp.get('purpose', '?')}\n")
            lines.append(f"- **Conflict/Change:** {sp.get('conflict_or_change', '')}")
            lines.append(f"- **Focal Character:** {sp.get('focal_character', '')}")
            lines.append(f"- **Location:** {sp.get('location', '')} ({sp.get('time_of_day', '')})")
            lines.append(f"- **Shot:** {sp.get('intended_shot_type', '')} / "
                         f"{sp.get('camera_movement', '')}")
            lines.append(f"- **Duration:** {sp.get('target_duration', '?')}s")
            lines.append(f"- **Visual Motif:** {sp.get('visual_motif', '')}")
            lines.append("")

    budget = pkg.get("shot_budget", {})
    if budget:
        lines.append("## Shot Budget\n")
        lines.append(f"- **Total Shots:** {budget.get('total_shots', '?')}")
        limits = budget.get("shot_type_limits", {})
        if limits:
            for k, v in limits.items():
                lines.append(f"- **{k}:** {v}")
        lines.append(f"- **Max consecutive same type:** "
                     f"{budget.get('max_consecutive_same_type', '?')}")
        lines.append(f"- **Max same location:** "
                     f"{budget.get('max_same_location', '?')}")
        lines.append("")

    ledger = pkg.get("continuity_ledger", {})
    if ledger:
        lines.append("## Continuity Ledger\n")
        states = ledger.get("character_states", {})
        if states:
            lines.append("### Character States\n")
            for char, state in states.items():
                lines.append(f"- **{char}:** {json.dumps(state)}")
        unresolved = ledger.get("unresolved_promises", [])
        if unresolved:
            lines.append("\n### Unresolved Promises\n")
            for p in unresolved:
                lines.append(f"- {p}")
        prohibited = ledger.get("prohibited_contradictions", [])
        if prohibited:
            lines.append("\n### Prohibited Contradictions\n")
            for p in prohibited:
                lines.append(f"- {p}")
        lines.append("")

    lines.append(f"\n---\n*Generated: {pkg.get('generated_at', 'unknown')} "
                 f"| Version: {pkg.get('version', '?')}*\n")

    return "\n".join(lines)
