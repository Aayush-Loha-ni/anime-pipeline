"""Best-image selector for visual consistency grading.

After generating candidate images for a scene, a vision LLM judges
which candidate best matches the reference(s) and text description.
Replaces the crude grade_clip heuristic with a structured judge.
"""
import base64
import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

_SYSTEM = """You are a professional visual assessment expert. Grade candidate images against a reference image and text description.

Assess:
- Character Consistency: gender, ethnicity, age, facial features, body shape, outlook, hairstyle vs reference
- Spatial Consistency: relative positions, scene layout, perspective vs reference
- Description Accuracy: matches the text description (not editing instructions)
- Quality: no white borders, black edges, framing artifacts, deformities

Guidelines:
- Prioritize Character Consistency over all
- Strictly compare with text description
- If multiple candidates partially meet criteria, pick highest overall
- Base all analysis on objective comparison, not subjective preference"""

_FORMAT = """{
  "best_image_index": <int>,
  "reason": "<brief explanation>",
  "anime_look": 0-100,
  "composition": 0-100,
  "issues": ["short", "phrases"]
}"""


class BestImageSelector:
    """Pick the best candidate image from a batch."""

    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url
        self.model = model

    def _b64(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def select(self, ref_image_path: str, ref_text: str,
               frame_description: str,
               candidate_paths: List[str], candidate_texts: List[str]) -> Tuple[int, str, dict]:
        """Pick the best candidate.

        Returns (index, reason, metadata). Falls back to (0, "unavailable", {})
        on error so the pipeline never blocks on one failed grade.
        """
        import re as _re
        if not candidate_paths:
            return 0, "no candidates", {}
        if len(candidate_paths) == 1:
            return 0, "single-candidate", {}

        try:
            from json_util import oai_chat, oai_image_part
            parts = [{"type": "text", "text": _build_human(
                ref_image_path, ref_text, frame_description,
                candidate_paths, candidate_texts)}]
            parts += [oai_image_part(self._b64(path))
                      for path in [ref_image_path] + candidate_paths]
            txt = oai_chat(self.ollama_url, [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": parts}], timeout=120)
            m = _re.search(r"\{.*\}", txt, flags=_re.DOTALL)
            if not m:
                return 0, "no JSON in response", {}
            parsed = json.loads(m.group(0))
            idx = parsed.get("best_image_index", 0)
            if type(idx) is not int or idx < 0 or idx >= len(candidate_paths):
                idx = 0
            reason = parsed.get("reason", "")
            metadata = {k: parsed.get(k) for k in
                        ("anime_look", "composition", "issues") if k in parsed}
            logger.info(f"Best-image selector: best={idx} ({reason})")
            return idx, reason, metadata
        except Exception as e:
            logger.warning(f"Best-image selection failed: {e}")
            return 0, f"selector error: {e}", {}


def _build_human(ref_path: str, ref_text: str, frame_desc: str,
                 candidates: List[str], candidate_texts: List[str]) -> str:
    lines = [f"Reference Image: {ref_text}", ""]
    for i, path in enumerate(candidates):
        text = candidate_texts[i] if i < len(candidate_texts) else Path(path).name
        lines.append(f"Candidate Image {i}: {text}")
    lines.append(f"\n<FRAME_DESC>\n{frame_desc}\n</FRAME_DESC>")
    lines.append(f"\nSelect the best candidate and reply ONLY valid JSON:\n{_FORMAT}")
    return "\n".join(lines)
