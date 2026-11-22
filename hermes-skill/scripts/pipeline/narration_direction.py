"""Direction for a narrator: pace, breath, and emphasis per line.

Why this exists. Measured on this machine, the cloned voice scored UTMOS 3.85
while a premium narrator hit 4.45 on the same engine and the same reference. The
audio was fine; the TIMING was not. Every segment was rendered at speed 1.0 and
stitched with an identical 0.12 s gap, so every line carried the same weight and
the same length. Creator-side research names exactly this as the leading cause of
"robotic" AI narration ("the rhythm never changes... the fix is not a better
voice, the fix is timing"), and it is the same defect that makes a viewer read a
channel as mass-produced.

So this module supplies what the engine cannot infer. It does NOT try to control
pitch or emotion: Qwen3-TTS's Base model exposes neither through
generate_voice_clone, and pretending otherwise would be theatre. What it does is
decide, per line, from the TEXT ALONE:

  rate     a speed multiplier. Hooks and short punchy lines run slightly fast,
           long expository lines ease back. This is the highest-impact lever in
           the published workflow (hook at 1.04-1.08x, body at 0.98-1.0x).
  gap      the silence AFTER the line. Commas inside a line are the engine's
           business; the join between lines is ours, and a constant gap is what
           produces the metronome effect.
  emphasis the payoff words. Every source agrees creators over-stress
           everything and so land nothing; only payoff words get marked.

Deliberately deterministic. A narrator's timing is part of the performance, not
noise to be reseeded per render: the same script must produce the same read, or
the cache and the listener both suffer. Everything here is a pure function of
the text, so it is testable without a GPU and stable across re-renders.

Deliberately conservative. These multipliers are small. A narrator who sounds
manic is worse than a flat one, and there is no measurement yet that justifies a
large excursion. If the A/B disagrees, raise them one notch at a time.
"""

import re

# Words a narrator would actually lean on, split by how safe they are to mark.

# STRONG: emphatic anywhere, including sentence-initial. These are absolutes and
# negations, and in English they are almost always the point of the line whether
# they open it or close it. "Nobody had ever seen it." leads with the punch.
_PAYOFF_STRONG = re.compile(
    r"\b(never|nothing|everything|everyone|nobody|always|only|dead|alive|"
    r"gone|forever|alone|betrayal|impossible|zero|billion|million)\b",
    re.IGNORECASE)

# WEAK: worth marking only near the end of a line. They carry the line when they
# land, but leading with one is usually setup — "It was still raining when he
# left" — and capitalising it there just makes the sentence look like shouting.
_PAYOFF_WEAK = re.compile(
    r"\b(first|last|until|exactly|instead|immediately|suddenly|finally|"
    r"still|wrong|true|real|remember|forget|listen|wait|stop|turn)\b",
    re.IGNORECASE)

# A weak payoff only counts near the end, where a narrator would land it.
_PAYOFF_TAIL = 0.45

_HOOK_MAX_CHARS = 120          # the first line of a scene
_HOOK_FAST = 1.06              # published hook range is 1.04-1.08
_PUNCHY_MAX_WORDS = 9          # "This was different." -> lift it
_PUNCHY_FAST = 1.04
_LONG_MIN_WORDS = 26           # expository run-on -> ease back
_LONG_SLOW = 0.98

# Gaps between stitched segments INSIDE one scene (multiple sentences).
#
# This is a different thing from the gap between scenes, which is
# episode_producer.NARRATION_GAP (0.20s). A scene with one sentence has no
# internal joins at all, so these values never fire on ch001 - which is exactly
# why the 1.24s between-scene gap went unnoticed: the metronome constant this
# module was written to fix was not the one producing it.
#
# Where both apply, an internal join (between two sentences of the same scene)
# is a smaller beat than a scene change, so these stay below NARRATION_GAP.
_GAP_TAIL = 0.12               # ordinary join
_GAP_PAYOFF = 0.20             # the line ended on its point: let it land
_GAP_SHORT = 0.09              # very short line, keep momentum



def _clamp(v, lo=0.90, hi=1.10):
    return max(lo, min(hi, round(v, 3)))


def line_rate(text, is_first_segment=False):
    """Speed multiplier for one segment. Pure function of the text.

    is_first_segment marks the opening of a scene, which is where retention is
    won or lost and therefore the one place a faster read pays off.
    """
    t = (text or "").strip()
    if not t:
        return 1.0
    words = len(t.split())
    if is_first_segment and len(t) <= _HOOK_MAX_CHARS:
        return _clamp(_HOOK_FAST)
    if words <= _PUNCHY_MAX_WORDS:
        return _clamp(_PUNCHY_FAST)
    if words >= _LONG_MIN_WORDS:
        return _clamp(_LONG_SLOW)
    return 1.0


def has_payoff(text):
    """Does this line land on a point worth a beat of silence after it?

    STRONG payoff words count anywhere. WEAK ones count only in the tail, so
    "still" at the end of a line is the point and "still" at the start is
    weather.
    """
    t = (text or "").strip()
    if not t:
        return False
    if _PAYOFF_STRONG.search(t):
        return True
    words = t.split()
    if len(words) < 3:
        return False
    tail = " ".join(words[-max(1, int(len(words) * _PAYOFF_TAIL)):])
    return bool(_PAYOFF_WEAK.search(tail))


def gap_after(text, is_last_segment=False):
    """Silence, in seconds, to insert AFTER this segment.

    The last segment gets no trailing gap: it is the end of the scene, and
    padding it only moves the next scene's start, which assembly already handles.
    """
    if is_last_segment:
        return 0.0
    t = (text or "").strip()
    if not t:
        return _GAP_TAIL
    if has_payoff(t):
        return _GAP_PAYOFF
    if len(t.split()) <= 4:
        return _GAP_SHORT
    return _GAP_TAIL


def emphasise(text):
    """NO-OP. Emphasis by capitalisation was cut on 2026-10-04.

    It used to capitalise one payoff word per line, on the claim that Qwen3-TTS
    reads CAPITALS as emphasis. That is false on this engine, and the only
    controlled measurement of it was a coin flip.

    1. THE MECHANISM IS WRONG. The tokenizer does not case-fold; capitals
       FRAGMENT the word. Measured token ids: never -> [36493], 1 token, but
       NEVER -> [3944, 3763] = ['NE', 'VER'], 2 tokens; NOTHING -> ['NOT',
       'HING']; DEAD -> ['DE', 'AD']. 7 of 7 pairs differed. That is an
       off-distribution spelling, not an emphasis convention.
    2. IT WAS NEVER MEASURED BEFORE SHIPPING. A seeded paired A/B (seed reset
       per call so the null arm is bit-identical) gave dUTMOS -0.040 with no
       consistent sign: [-1.073, +0.872, +0.152, -0.111].
    3. IT CAN DROP WORDS, SILENTLY. The -1.073 case transcribed as "It opened
       and nothing came through it." - a missing word - and there is no
       intelligibility gate anywhere in the render path, so it would ship.
    4. IT UNDID THE PIPELINE'S OWN NORMALISATION. clean_tts_text de-shouts
       ALL-CAPS hook words ("ALWAYS watch" -> "Always watch"); this put the
       caps straight back, so the isupper() skip was dead code in production,
       and a script's deliberate EMPTY came out as "Empty" while an unrelated
       "dead" came out shouted.

    Kept as a function, returning its input unchanged, because the direction
    plan has an "emphasised" key and the tests pin the contract. The alternative
    - deleting the key - would have meant changing the plan shape and the test
    suite for no behavioural gain. Gaps stay: they are genuine post-processing
    of audio that is synthesised once. This never met that standard.
    """
    return (text or "").strip()

def direct(text, is_first=False, is_last=False):
    """Full direction for one segment.

    is_first marks the scene's opening line, where a faster read is the single
    highest-impact change in the published workflow. It has to be threaded down
    into line_rate: calling line_rate(t) with the default False silently skipped
    the hook branch, which left the most valuable rule dead while the tests still
    passed on the punchy and long branches.
    """
    t = (text or "").strip()
    return {
        "text": t,
        "rate": line_rate(t, is_first_segment=is_first),
        "gap_after": gap_after(t, is_last_segment=is_last),
        "emphasised": emphasise(t),
    }


def direct_all(segments, base_rate=1.0):
    """Direct a whole scene's worth of segments.

    base_rate is the user's global voice_speed. Two rules protect it:

    * A SINGLE segment gets no rate change at all. There is no rhythm to vary in
      one line, so a "hook boost" on a two-word fragment is meaningless — and it
      would silently rewrite an explicit voice_speed the user set on purpose.
    * With multiple segments the per-line variation rides on top as a multiplier,
      and the product is deliberately NOT re-clamped. Clamping the combined value
      capped voice_speed=1.25 at 1.10, which is the config key doing nothing the
      user asked for.
    """
    n = len(segments)
    out = []
    for i, seg in enumerate(segments):
        d = direct(seg, is_first=(i == 0 and n > 1), is_last=(i == n - 1))
        d["rate"] = round(d["rate"] * (base_rate or 1.0), 4) if n > 1 \
            else round(base_rate or 1.0, 4)
        d["index"] = i
        out.append(d)
    return out
