"""Multimodal reference-image selector for video consistency.

Chooses the best reference images from a growing library (character
portraits + prior scene frames) for the next frame's text description.
Posts OpenAI-style vision messages to the swapped VLM profile on :8080;
all production call sites gate on vlm_available().
"""
import base64
import json
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

_SYSTEM = """You are a professional visual assessment expert. Identify Character Consistency and Spatial Consistency between candidate images and reference images, and semantic consistency between candidate images and text descriptions.

Task: Based on the reference images, the text description of the target image, and several candidate images, evaluate which candidate image best matches in:
- Character Consistency: gender, ethnicity, age, facial features, body shape, outlook, hairstyle
- Spatial Consistency: relative positions between characters, scene layout, perspective
- Description Accuracy: accurately reflects the text (editing instructions, not directives)

Guidelines:
- Prioritize Character Consistency
- Focus on Spatial Consistency
- Strictly compare with text description
- If multiple images partially meet criteria, select the one with highest overall consistency
- Prioritize images without white borders, black edges, or additional framing"""

_HUMAN_TEMPLATE = """<FRAME_DESC>
{frame_description}
</FRAME_DESC>

<CANDIDATES>
{candidates}
</CANDIDATES>

{format_instructions}"""

_FORMAT = """{
  "best_image_index": <int>,
  "reason": "<brief explanation>"
}"""


class ReferenceImageSelector:
    """Selects the most relevant reference images for a frame."""

    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url
        self.model = model

    def _b64(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _messages(self, frame_desc: str, ref_pairs: List[Tuple[str, str]],
                  candidates: List[Tuple[str, str]]) -> list:
        # OpenAI vision format: text part first, then image parts in order
        # (Reference 0..N, then Candidate 0..M).
        from json_util import oai_image_part
        text_parts = []
        for i, (_, text) in enumerate(ref_pairs):
            text_parts.append(f"Reference Image {i}: {text}")
        for i, (_, text) in enumerate(candidates):
            text_parts.append(f"Candidate Image {i}: {text}")
        text_parts.append(f"Frame: {frame_desc}")
        text_parts.append(f"Select best candidate. Reply ONLY JSON:\n{_FORMAT}")
        parts = [{"type": "text", "text": "\n".join(text_parts)}]
        parts += [oai_image_part(self._b64(path)) for path, _ in ref_pairs + candidates]
        return [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": parts}]

    def _autofill_texts(self, paths: List[str], texts: List[str]) -> List[str]:
        """Fill empty prompt texts via local BLIP captioner. Never raises."""
        try:
            from pipeline.captioner import available, describe_image
            if not available():
                return texts
            out = list(texts) + [""] * max(0, len(paths) - len(texts))
            for i, (p, t) in enumerate(zip(paths, out)):
                if not (t or "").strip():
                    try:
                        out[i] = describe_image(p)
                    except Exception:
                        pass
            return out
        except Exception as e:
            logger.warning(f"Caption autofill skipped: {e}")
            return texts

    def select(self, frame_description: str,
               ref_image_paths: List[str], ref_texts: List[str],
               candidate_paths: List[str], candidate_texts: List[str],
               max_ref: int = 8) -> Tuple[int, str]:
        """Pick the best candidate image index and reason.

        Returns (index, reason). Falls back to (0, "selector unavailable")
        on any error so the pipeline never blocks on one failed selection.
        """
        if not candidate_paths:
            logger.warning("No candidate images; skipping reference selector")
            return 0, "no candidates"

        # BLIP autofill: empty prompt texts get local captions (never blocks)
        ref_texts = self._autofill_texts(ref_image_paths, list(ref_texts or []))
        candidate_texts = self._autofill_texts(candidate_paths, list(candidate_texts or []))

        max_ref = max(0, max_ref)
        available = list(zip(ref_image_paths, ref_texts))
        if len(available) > max_ref:
            # Text-only pre-filter to trim to max_ref
            try:
                available = self._filter_by_text(available, frame_description, max_ref)
            except Exception as e:
                logger.warning(f"Reference text-filter failed: {e}")
        available = available[:max_ref]

        try:
            from json_util import oai_chat
            txt = oai_chat(self.ollama_url, self._messages(
                frame_description, available,
                list(zip(candidate_paths, candidate_texts))), timeout=120)
            m = __import__("re").search(r"\{.*\}", txt, flags=__import__("re").DOTALL)
            if not m:
                return 0, "no JSON in response"
            parsed = json.loads(m.group(0))
            idx = parsed.get("best_image_index", 0)
            if type(idx) is not int or idx < 0 or idx >= len(candidate_paths):
                idx = 0
            reason = parsed.get("reason", "")
            logger.info(f"Reference selector: best={idx} ({reason})")
            return idx, reason
        except Exception as e:
            logger.warning(f"Reference selection failed: {e}")
            return 0, f"selector error: {e}"

    def _filter_by_text(self, pairs: List[Tuple[str, str]],
                        desc: str, max_ref: int) -> List[Tuple[str, str]]:
        """Text-only pre-filter to trim reference library to max_ref."""
        import re as _re
        from json_util import oai_chat
        human = "\n".join(f"Image {i}: {t}" for i, (_, t) in enumerate(pairs))
        txt = oai_chat(self.ollama_url, [
            {"role": "system", "content": "Select the most relevant image indices for this frame description. Reply ONLY valid JSON with a 'indices' integer array."},
            {"role": "user", "content": f"Frame:\n{desc}\n\nImages:\n{human}\n\nSelect up to {max_ref} most relevant indices as JSON: {{\"indices\": []}}"}],
            timeout=60)
        max_ref = max(0, max_ref)
        m = _re.search(r"\{.*\}", txt, flags=_re.DOTALL)
        try:
            parsed = json.loads(m.group(0) if m else txt)
        except (ValueError, TypeError):
            return pairs[:max_ref]
        idxs = parsed.get("indices") if isinstance(parsed, dict) else None
        if not isinstance(idxs, list):
            return pairs[:max_ref]
        valid = list(dict.fromkeys(i for i in idxs if type(i) is int and 0 <= i < len(pairs)))
        return [pairs[i] for i in valid[:max_ref]] if valid else pairs[:max_ref]
