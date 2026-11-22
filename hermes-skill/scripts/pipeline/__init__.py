"""Pipeline package — asset library, task manager, script reviewer, camera presets.

Upgrade modules:
  director_package - creative bridge between novel and scenes
  rights_manifest  - provenance and commercial-use tracking
  audio_planner    - per-scene audio design
  subtitle_generator - SRT/VTT generation
  release_report   - structured QA verdict
  release_package  - YouTube deliverable set
  analytics_log    - performance tracking
  schemas          - JSON validation schemas
  quality_evolve   - episode rating + the closed-loop auto-tuner
                     (moved out of episode_producer.py 2026-09-29; post-render
                     analytics only, nothing on the critical render path)

Deleted 2026-09-29 as unreachable from the render path (import-graph check;
each was referenced only by its own test): style_bible, migrate, continuity,
visual_provenance, motion_detect, integration.
"""
