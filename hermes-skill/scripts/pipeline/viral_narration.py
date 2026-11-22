"""Viral narration scoring — deterministic retention checks for scene narration.

Adapted techniques (patterns, not copied text) from battle-tested
open-source projects:
- 10-trigger hook taxonomy (curiosity gap, contrarian, question, negation,
  specificity, story-open, confession, stakes...) — cf.
  rediumvex/viral-hooks-skill (100 hook formulas; mix one stop-scroll
  trigger with one retention trigger).
- Niche-profile rules: forbidden openers, word targets, CTA placement — cf.
  rushindrasinha/youtube-shorts-pipeline niches/*.yaml.
- Deterministic pacing/readability QA (grade level, word caps) — cf.
  YouTube-Script-to-Storyboard-Agent PacingReviewAgent.
- Viewer-simulator retention pass as deterministic checks (hook, open loops,
  payoff promise) instead of extra LLM calls — cf. script2vid 3-critic loop.
"""
from __future__ import annotations

import re

FORBIDDEN_OPENERS = (
    "in this video", "hey guys", "hey everyone", "welcome back", "welcome to",
    "today we", "in today's", "like and subscribe", "before we start",
    "make sure to subscribe", "smash that",
)
FORBIDDEN_ANYWHERE = (
    "smash that", "hit the bell", "don't forget to subscribe",
    "as an ai language", "as a language model",
)
POWER_WORDS = (
    "secret", "forbidden", "dead", "betrayed", "hidden", "final", "escaped",
    "caught", "warning", "last", "never", "truth", "lies", "lied", "buried",
    "vanished", "hunted", "trapped", "curse", "blood",
)
CURIOSITY_CUES = (
    "secret", "hidden", "truth", "mystery", "nobody", "never knew", "what if",
    "something was wrong", "but ", "however", "until ",
)
FORWARD_TAILS = ("...", "?", "!")
FORWARD_WORDS = ("then", "next", "but", "until", "when", "suddenly")
# ponytail: was 25 ("one breath"), which is what flattened the ch002 script to
# ~5 words/line and grade-2 fragments — the audit measured mean 5.3 words. 45
# keeps a line speakable in one breath while letting it carry an actual plot
# fact (a name, a number, a cause) instead of a telegraphic fragment. Warn-only
# above this; nothing enforces a hard ceiling.
# 90 (2026-11-13): a real chapter (1021 words) reached the viewer at ~30%
# across 22 scenes (mean 14 words) because the 45 cap told the model to make
# every line a breath. The frame can show ONE image; the narration has to
# carry the rest of the beat, or 70% of what the chapter establishes is lost.
MAX_WORDS = 90

_WORD_RE = re.compile(r"[A-Za-z']+")
# ponytail: \b, not substring — "blast"/"deadline"/"families" must not trip
# "last"/"dead"/"lies", and "but."/"but," must still trip "but".
_CURIOSITY_RE = re.compile(
    r"\b(secret|hidden|truth|mystery|nobody|never knew|what if|"
    r"something was wrong|but|however|until)\b")
_POWER_RE = re.compile(
    r"\b(secret|forbidden|dead|betrayed|hidden|final|escaped|caught|"
    r"warning|last|never|truth|lies|lied|buried|vanished|hunted|"
    r"trapped|curse|blood)\b")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def count_syllables(word: str) -> int:
    vowels = set("aeiouy")
    w = word.lower().strip()
    if not w:
        return 0
    n, prev_vowel = 0, False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            n += 1
        prev_vowel = is_vowel
    if w.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def fk_grade(text: str) -> float:
    """Flesch-Kincaid grade level; narration should read at grade ~6."""
    words = _words(text)
    if not words:
        return 0.0
    sents = max(1, len(re.findall(r"[.!?…]+", text or "")))
    syl = sum(count_syllables(w) for w in words)
    return 0.39 * (len(words) / sents) + 11.8 * (syl / len(words)) - 15.59


def detect_triggers(text: str) -> list[str]:
    """Hook triggers present in a line, priority-ordered. Empty = weak open."""
    t = (text or "").strip()
    low = t.lower()
    out = []
    if re.match(r"^(not |no |never |don't |stop )", low):
        out.append("negation")
    if t.endswith("?") or re.match(r"^(who|what|where|when|why|how|can|could|is|are|do|does|did|have you|what if)\b", low):
        out.append("question")
    if re.search(r"\d", t):
        out.append("specificity")
    if re.match(r"^(he|she|they|it|that night|that morning|suddenly|then|yesterday|last night)\b", low):
        out.append("story")
    if _CURIOSITY_RE.search(low):
        out.append("curiosity")
    if _POWER_RE.search(low):
        out.append("stakes")
    if re.match(r"^(i |i'll |i'm |i've |my |confession)", low):
        out.append("confession")
    return out


def score_hook(text: str) -> tuple[int, str | None]:
    """Score an opening line 0-100. Two different triggers beat one."""
    trig = detect_triggers(text)
    if not trig:
        score = 20
    else:
        score = 55
        if len(set(trig)) > 1:
            score += 15
    low = (text or "").lower()
    if re.search(r"\b(you|your|yours)\b", low):
        score += 10  # direct address: viewer enters dialogue mode
    if any(low.startswith(o) for o in FORBIDDEN_OPENERS):
        score -= 40
    return max(0, min(100, score)), (trig[0] if trig else None)


# ponytail: an ABSENCE marker is a withheld detail, which is the same thing
# a forward cue is doing. "The Registry never sent cards." withholds why, so
# the viewer needs the next line — but it has no cue word and no punctuation,
# so the old check scored it 0 while the cue-word version that put "..." on
# 24 of 25 lines scored 96. That gap is the whole bug: the check could only
# see punctuation, so it rewarded a tic over actual restraint.
# ponytail: `n't` is matched as a suffix, not \bn't\b — "didn't" has no word
# boundary before the n, so the \b form silently never fires and every
# "didn't/don't/couldn't" line read as a plain statement.
# ponytail: enumerated, not \bun\w{3,}. A generic un- prefix also matches
# "under", "until" and "unusual", which are ordinary scene-setting and would
# let flat narration pass. Named words only, so the check stays honest.
_ABSENCE_RE = re.compile(
    r"(\bnever\b|\bno\b|\bnobody\b|\bnone\b|\bnot\b|n't\b|"
    r"\bunnamed\b|\bunknown\b|\bunexplained\b|\bunnoticed\b|\bunheard\b|"
    r"\bunseen\b|\bunread\b|\btoo\b|\balready\b|\bonly\b|\bwithout\b|"
    r"\bneither\b|\bnor\b|\bnothing\b|\bno one\b|of uncertain)")


def has_open_loop(text: str) -> bool:
    """Does the line end mid-tension so skipping feels like missing out?"""
    t = (text or "").strip()
    if not t:
        return False
    if t.endswith(FORWARD_TAILS):
        return True
    tail = " ".join(_words(t)[-8:]).lower()
    # ponytail: tail is punctuation-stripped, so last-word match covers
    # "then.", single-word "Suddenly", etc. — no endswith needed.
    if _CURIOSITY_RE.search(tail):
        return True
    if (tail.split()[-1:] or [""])[0] in FORWARD_WORDS:
        return True
    return bool(_ABSENCE_RE.search(tail))


def scene_grip(text: str, is_first: bool = False) -> str:
    """One-token hook mark for dashboards: trigger, loop, or weak."""
    trig = detect_triggers(text)
    if trig:
        return trig[0]
    if not is_first and has_open_loop(text):
        return "loop"
    return "weak"


def _line(scene: dict) -> str:
    if not isinstance(scene, dict):
        return ""
    return (scene.get("narration") or scene.get("dialogue") or "").strip()


def _num(scene: dict, fallback: int) -> int:
    if isinstance(scene, dict):
        n = scene.get("scene_number", fallback)
        return n if type(n) is int and n > 0 else fallback
    return fallback


def score_narration(scenes: list[dict]) -> dict:
    """Score a full episode's narration. Severities stay warning/info so this
    never flips review_script validity — it only drags the score."""
    findings: list[dict] = []
    if not scenes:
        return {"score": 0, "hook": {"score": 0, "trigger": None}, "findings": findings}

    def add(scene, check, severity, message):
        findings.append({"scene": scene, "check": check,
                         "severity": severity, "message": message})

    texts = [_line(s) for s in scenes]
    first = texts[0]
    hscore, htrig = score_hook(first)
    if hscore < 45:
        add(1, "hook", "warning",
            f"Weak opening hook ({hscore}/100, trigger: {htrig or 'none'}) — "
            "open mid-mystery: a question, a negation, a number, or mid-action.")
    elif hscore < 70:
        add(1, "hook", "info", f"Serviceable hook ({hscore}/100, {htrig}) — one more trigger would grip harder.")
    if any(first.lower().startswith(o) for o in FORBIDDEN_OPENERS):
        add(1, "hook", "warning", "Forbidden opener — never introduce, start mid-mystery.")

    for i, (s, text) in enumerate(zip(scenes, texts)):
        num = _num(s, i + 1)
        n = len(_words(text))
        if not text:
            continue
        if n > MAX_WORDS:
            add(num, "pacing", "warning",
                f"{n} words — over ~{MAX_WORDS} words the line is a monologue; "
                f"split the beat instead of the fact.")
        elif n < 3:
            add(num, "pacing", "info", "Very short line — fine only as a punch beat.")
    # ponytail: ONE aggregate retention finding, not one per scene. The
    # per-scene version fired on ~75% of lines (ch020: 33 of 40) and each
    # info cost 2 points of the review score, so the score WAS the proxy rate
    # in disguise - ch020 scored 31/100 on a script that holds a name back
    # for 13 scenes (s8 "Edvyn says only one name" -> s21 "The man's name
    # was Caelen"). The check is a KEYWORD proxy: it sees "never", "until"
    # and "?" but not a line that withholds a name, a number or a reason, so
    # the rate is a floor, not a verdict. One note, with the rate, keeps the
    # signal and stops the score from meaning nothing.
    _body = [t for t in texts[:-1] if t]
    _flat = sum(1 for t in texts[:-1] if t and not has_open_loop(t))
    if _flat and _body:
        add(0, "retention", "info",
            f"{_flat} of {len(_body)} lines end on NO detected forward cue - a "
            "keyword proxy, not a verdict: a line that withholds a name, a "
            "number or a reason without a cue word reads as flat here")
    if texts and texts[-1] and not has_open_loop(texts[-1]):
        add(_num(scenes[-1], len(scenes)), "retention", "info",
            "Final scene answers instead of asking — end on the question, not the answer.")
    if not any(re.search(r"\b(you|your)\b", t.lower()) for t in texts[:3]):
        add(0, "retention", "info", "Nobody is talked TO in the first 3 scenes — add one 'you'.")
    if not _POWER_RE.search(" ".join(texts).lower()):
        add(0, "language", "info", "Zero power words in the episode — season with secret/forbidden/dead/betrayed.")
    for i, text in enumerate(texts):
        if i < len(texts) / 2 and any(p in text.lower() for p in FORBIDDEN_ANYWHERE):
            add(_num(scenes[i], i + 1), "language", "warning",
                "CTA/greeting filler before the payoff — move it to the end or cut it.")
            break
    grades = [fk_grade(t) for t in texts if t]
    if grades and sum(grades) / len(grades) > 6.5:
        add(0, "readability", "info",
            f"Reads at grade {sum(grades) / len(grades):.1f} — bedtime-story simple wins (aim ≤ 6).")

    w = sum(1 for f in findings if f["severity"] == "warning")
    i = sum(1 for f in findings if f["severity"] == "info")
    return {"score": max(0, 100 - 8 * w - 2 * i),
            "hook": {"score": hscore, "trigger": htrig}, "findings": findings}


def viral_prompt_block() -> str:
    """Compact LLM instruction block: the hook menu + bans. Merged into the
    script-writing prompt so generation itself goes viral-first."""
    return (
        "VIRAL HOOK MENU (scene 1 MUST match exactly one):\n"
        "- question (end on ? — 'Can you see it?')\n"
        "- negation ('Not a dream.' / 'Never again.')\n"
        "- specific number ('3:07am...', '73%...')\n"
        "- mid-action story open ('He ran...', 'That night...')\n"
        "- curiosity gap (secret / hidden / truth / nobody)\n"
        "- direct address ('Listen...', 'You felt it too.')\n"
        "BANNED openers anywhere near scene 1: 'in this video', 'hey guys', "
        "'welcome', 'today we', 'like and subscribe', 'before we start'.\n"
        "Every non-final scene ends mid-tension (..., ?, or a forward pointer "
        "like 'then' / 'until'). Final scene ends ON the question, never the answer. "
        f"Write 40-70 words per narration line — the frame shows ONE thing, so "
        f"the voice carries the WHOLE beat (names, numbers, causes, deadlines). "
        f"Grade-6 words; a one-line summary of a beat the chapter spends "
        f"sentences on is information thrown away."
    )
