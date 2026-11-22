#!/usr/bin/env python3
"""JSON schemas for pipeline upgrade artifacts — v2.

Every schema is a dict-of-dicts keyed by artifact name. Each value has:
  - required: list of required top-level keys
  - optional: list of allowed optional keys
  - nested: dict of nested object schemas (key -> {required, optional})

Validation is lightweight: check presence + types, never deep recursion.
"""

DIRECTOR_PACKAGE = {
    "required": ["episode_promise", "target_audience", "target_emotion",
                 "cold_open_hook", "central_dramatic_question",
                 "midpoint_reversal", "climax", "ending",
                 "beat_sheet", "scene_plan", "shot_budget", "continuity_ledger"],
    "optional": ["episode_number", "source_chapter", "generated_at", "version",
                 "owner_approved", "approved_by", "approved_at",
                 "source_spans", "narration_word_target"],
}

COLD_OPEN_HOOK = {
    "required": ["description", "duration_range"],
    "optional": ["visual", "audio_cue"],
}

BEAT_SHEET_ITEM = {
    "required": ["beat_number", "description", "source_chapter_ref"],
    "optional": ["timestamp_estimate", "emotion", "character_focus"],
}

SCENE_PLAN_ITEM = {
    "required": ["scene_number", "purpose", "conflict_or_change",
                 "focal_character", "location", "time_of_day",
                 "narration_or_dialogue", "visual_motif",
                 "intended_shot_type", "camera_movement", "target_duration"],
    "optional": ["characters", "mood", "props", "continuity_notes",
                 "narration_text", "source_beat", "transition_from",
                 "lens", "lighting"],
}

SHOT_BUDGET = {
    "required": ["total_shots", "shot_type_limits", "composition_rules"],
    "optional": ["max_consecutive_same_type", "max_same_location",
                 "min_wide_shots", "min_closeups", "min_inserts"],
}

CONTINUITY_LEDGER = {
    "required": ["character_states", "unresolved_promises", "prohibited_contradictions"],
    "optional": ["wardrobe", "injuries", "props", "relationships",
                 "location_state", "time_progression"],
}

CHARACTER_STATE = {
    "required": ["character_id", "status"],
    "optional": ["wardrobe", "injuries", "mood", "props_held",
                 "location", "last_seen_scene"],
}

STYLE_BIBLE = {
    "required": ["art_direction", "palette", "lighting", "lens_language",
                 "aspect_ratio", "recurring_motifs", "prohibited_visual_traits",
                 "negative_prompts"],
    "optional": ["line_weight", "color_temperature", "reference_shots",
                 "character_design_rules", "series_title", "version"],
}

CHARACTER_BIBLE = {
    "required": ["character_id", "full_name", "role", "canonical_description",
                 "approved_references", "approval_state"],
    "optional": ["age_category", "wardrobe_variants", "voice_notes",
                 "safety_restrictions", "lora_details", "change_history",
                 "danbooru_tags", "prompt_tags", "negative_tags"],
}

RIGHTS_ENTRY = {
    "required": ["asset_id", "asset_type", "source", "license",
                 "commercial_use", "local_path_or_hash"],
    "optional": ["owner_confirmation", "notes", "expires", "attribution",
                 "version", "reviewer", "reviewed_at", "used",
                 "model_version", "reference_file_hash", "trigger"],
}

RIGHTS_MANIFEST = {
    "required": ["novel_adaptation_rights", "entries"],
    "optional": ["last_verified", "version", "notes",
                 "voice", "image_models", "video_models",
                 "image_model_licenses", "audio_assets",
                 "llm_models", "vlm_models", "loras",
                 "custom_nodes", "upscalers", "interpolators",
                 "music", "sfx", "fonts", "stock_assets",
                 "character_references"],
}

AUDIO_PLAN_ITEM = {
    "required": ["scene_number", "voice_priority", "intended_silence",
                 "ambience_cue", "music_cue", "sfx_cue",
                 "emotional_intensity", "asset_id"],
    "optional": ["notes", "duration_hint", "ducking_amount",
                 "music_start_offset", "music_volume"],
}

SHOT_MANIFEST_ITEM = {
    "required": ["scene_number", "shot_number", "prompt", "negative_prompt",
                 "seed", "checkpoint", "generation_settings", "timestamp"],
    "optional": ["loras", "reference_image_hashes", "character_ids",
                 "output_hash", "dimensions", "frame_rate",
                 "director_package_version", "character_bible_version",
                 "qa_result", "qa_confidence"],
}

RELEASE_REPORT = {
    "required": ["chapter", "verdict", "issues"],
    "optional": ["timestamp", "generated_at", "summary", "checks",
                 "contact_sheets", "scores", "notes"],
}

RELEASE_PACKAGE = {
    "required": ["chapter", "video_path", "video_sha256",
                 "subtitles", "metadata", "disclosure_checklist",
                 "thumbnail_concepts", "release_report", "rights_manifest",
                 "credit_note"],
    "optional": ["shorts_cut", "analytics_log_path",
                 "edl_path", "srt_path", "vtt_path"],
}

ANALYTICS_ENTRY = {
    "required": ["chapter", "added_at"],
    "optional": ["impressions", "ctr", "avg_view_duration",
                 "retention_30s", "completion_rate", "subscribers_gained",
                 "comments_themes", "manual_lesson"],
}

EDL_ITEM = {
    "required": ["scene_number", "shot_number", "in_point", "out_point",
                 "asset_id", "asset_type"],
    "optional": ["transition", "duration", "narration_path",
                 "music_path", "sfx_path", "caption_text",
                 "caption_start_ms", "caption_end_ms",
                 "fallback_used", "sidecar_path"],
}


def validate_dict(data, schema, name="artifact", strict=True):
    """Validate a dict against a schema. Returns list of error strings.

    strict=True rejects unknown keys (additionalProperties:false equivalent).
    """
    errors = []
    if not isinstance(data, dict):
        return [f"{name}: expected dict, got {type(data).__name__}"]
    for key in schema.get("required", []):
        if key not in data:
            errors.append(f"{name}: missing required key '{key}'")
    if strict:
        allowed = set(schema.get("required", [])) | set(schema.get("optional", []))
        allowed |= set(schema.get("nested", {}).keys())
        for key in data:
            if key not in allowed:
                errors.append(f"{name}: unexpected key '{key}'")
    for key, sub_schema in schema.get("nested", {}).items():
        if key in data and isinstance(data[key], dict):
            sub_errors = validate_dict(data[key], sub_schema, f"{name}.{key}")
            errors.extend(sub_errors)
        elif key in data and isinstance(data[key], list):
            for i, item in enumerate(data[key]):
                if isinstance(item, dict):
                    sub_errors = validate_dict(item, sub_schema, f"{name}.{key}[{i}]")
                    errors.extend(sub_errors)
    return errors


def validate_director_package(pkg):
    """Validate a director package. Returns list of error strings."""
    return validate_dict(pkg, DIRECTOR_PACKAGE, "director_package")


def validate_rights_manifest(manifest):
    """Validate a rights manifest. Returns list of errors."""
    errors = validate_dict(manifest, RIGHTS_MANIFEST, "rights_manifest")
    novel = manifest.get("novel_adaptation_rights") if isinstance(manifest, dict) else None
    if isinstance(novel, dict):
        errors.extend(validate_dict(novel, RIGHTS_ENTRY, "rights_manifest.novel_adaptation_rights"))
    voice = manifest.get("voice") if isinstance(manifest, dict) else None
    if isinstance(voice, dict):
        errors.extend(validate_dict(voice, RIGHTS_ENTRY, "rights_manifest.voice"))
    # Validate all category lists
    for category in ("entries", "image_models", "video_models", "llm_models",
                     "vlm_models", "loras", "music", "sfx", "character_references"):
        for i, entry in enumerate(manifest.get(category, []) if isinstance(manifest, dict) else []):
            if isinstance(entry, dict):
                errors.extend(validate_dict(entry, RIGHTS_ENTRY,
                                           f"rights_manifest.{category}[{i}]"))
    return errors


def validate_audio_plan(plan):
    """Validate an audio plan list. Returns list of errors."""
    if not isinstance(plan, list):
        return [f"audio_plan: expected list, got {type(plan).__name__}"]
    errors = []
    for i, item in enumerate(plan):
        errors.extend(validate_dict(item, AUDIO_PLAN_ITEM, f"audio_plan[{i}]"))
    return errors


def validate_style_bible(bible):
    """Validate a style bible. Returns list of errors."""
    return validate_dict(bible, STYLE_BIBLE, "style_bible")


def validate_character_bible(bible):
    """Validate a character bible. Returns list of errors."""
    if not isinstance(bible, dict):
        return [f"character_bible: expected dict, got {type(bible).__name__}"]
    errors = []
    for char_id, char_data in bible.items():
        if char_id.startswith("_"):
            continue
        if isinstance(char_data, dict):
            errors.extend(validate_dict(char_data, CHARACTER_BIBLE, f"character_bible.{char_id}"))
    return errors


def validate_release_report(report):
    """Validate a release report. Returns list of errors."""
    return validate_dict(report, RELEASE_REPORT, "release_report")


def validate_release_package(pkg):
    """Validate a release package. Returns list of errors."""
    return validate_dict(pkg, RELEASE_PACKAGE, "release_package")


def validate_edl_item(item):
    """Validate an EDL item. Returns list of errors."""
    return validate_dict(item, EDL_ITEM, "edl_item")
