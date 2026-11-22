"""Novel registry — one self-contained folder per novel, shared pipeline code.

Layout
------
    <project-root>/novels/index.json      registry (version + entries)
    <project-root>/novels/<slug>/         one novel
        novel.json                        title, art_direction, config overrides
        chapters/ scenes/ images/ voice/ videos/
        characters/                       characters.json, props.json, pages/
        dataset/ music/ sfx/
        pinned.json
        rights_manifest.json

The built-in ``sinners_world`` novel is the LEGACY layout at the project root
(no folder of its own) so every existing render, cache and test keeps working
unchanged. New novels live under ``novels/<slug>/`` and are fully isolated.

Path rules (kept deliberately small):
  * per-novel — chapters, scenes, images, voice, videos, characters, dataset,
    music, sfx, pinned.json, rights_manifest.json
  * shared    — models, ComfyUI, rules/, cosy-server/ (the voice clone),
    _models/, _training/, issues.json, tokens.json, render state, tuning
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_NOVEL_ID = "sinners_world"
DEFAULT_NOVEL_TITLE = "Example Novel"
INDEX_VERSION = 1

# novel.json keys that are safe to merge over the global CONFIG for a render.
# Paths are rewritten separately in episode_producer._novel_context.
_PER_NOVEL_DIRS = ("chapters", "scenes", "images", "voice", "videos",
                   "characters", "dataset", "music", "sfx")


def project_root() -> Path:
    """The Anime Series root (the folder that holds chapters/, videos/, ...)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def novels_dir(root: Path | str | None = None) -> Path:
    base = Path(root) if root is not None else project_root()
    return base / "novels"


def index_path(root: Path | str | None = None) -> Path:
    return novels_dir(root) / "index.json"


def slugify(title: str) -> str:
    # The apostrophe goes, it does not become a separator: "Example Novel"
    # must slug to "sinners_world", not "sinner_s_world".
    cleaned = str(title or "").replace("'", "").replace("’", "")
    slug = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_")
    return slug or "novel"


@dataclass
class Novel:
    id: str
    title: str
    root: Path
    legacy: bool = False
    # True when the LEGACY novel was relocated into novels/Example Novel/:
    # same id, but the render must treat it as its own folder (paths, bibles,
    # media) exactly like a registered novel.
    relocated: bool = False
    config: dict = field(default_factory=dict)
    art_direction: dict = field(default_factory=dict)

    # -- directories ---------------------------------------------------------
    def chapters_dir(self) -> Path:
        return self.root / "chapters"

    def chapter_files(self) -> list[Path]:
        """Every chapter text file, .txt or .md (novels arrive as either)."""
        out = []
        for pattern in ("*.txt", "*.md"):
            out.extend(self.chapters_dir().glob(pattern))
        return sorted(out, key=lambda p: p.name)

    def chapter_file(self, number: int) -> Path | None:
        """The text file for a chapter NUMBER (the first number in the name)."""
        for cand in self.chapter_files():
            m = re.search(r"(\d+)", cand.stem)
            if m and int(m.group(1)) == int(number):
                return cand
        return None

    def scenes_dir(self) -> Path:
        return self.root / "scenes"

    def images_dir(self) -> Path:
        return self.root / "images"

    def voice_dir(self) -> Path:
        return self.root / "voice"

    def videos_dir(self) -> Path:
        return self.root / "videos"

    def characters_dir(self) -> Path:
        return self.root / "characters"

    def characters_file(self) -> Path:
        return self.root / "characters" / "characters.json"

    def music_dir(self) -> Path:
        return self.root / "music"

    def sfx_dir(self) -> Path:
        return self.root / "sfx"

    def pins_file(self) -> Path:
        return self.root / "pinned.json"

    def rights_manifest_file(self) -> Path:
        return self.root / "rights_manifest.json"

    def novel_file(self) -> Path:
        return self.root / "novel.json"

    def config_overrides(self) -> dict:
        """The CONFIG path entries this novel owns (absolute, per novel)."""
        return {
            "output_dir": str(self.videos_dir()),
            "chapters_dir": str(self.chapters_dir()),
            "images_dir": str(self.images_dir()),
            "scenes_dir": str(self.scenes_dir()),
            "characters_file": str(self.characters_file()),
            "music_dir": str(self.music_dir()),
            "sfx_dir": str(self.sfx_dir()),
        }

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "root": str(self.root),
                "legacy": bool(self.legacy)}


def _legacy_novel(root: Path | None = None) -> Novel:
    base = Path(root) if root is not None else project_root()
    title = DEFAULT_NOVEL_TITLE
    art: dict = {}
    # The legacy novel now lives at novels/Example Novel/ when that folder
    # exists: same id and art direction as ever, resolved against its own
    # folder instead of the project root.
    loc = base / "novels" / "Example Novel"
    relocated = (loc / "characters" / "characters.json").exists()
    root = loc if relocated else base
    for bible in (root / "series_bible.json", base / "series_bible.json"):
        try:
            data = json.loads(bible.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            title = str(data.get("series_title") or title)
            if isinstance(data.get("art_direction"), dict):
                art = data["art_direction"]
        break
    return Novel(id=DEFAULT_NOVEL_ID, title=title, root=root, legacy=True,
                 relocated=relocated, config={}, art_direction=art)


def load_index(root: Path | str | None = None) -> list[dict]:
    """Read novels/index.json. Missing/corrupt file -> [] (the legacy novel is
    always implied, so an unreadable index must not take the pipeline down)."""
    try:
        data = json.loads(index_path(root).read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = data.get("novels") if isinstance(data, dict) else None
    return [e for e in entries if isinstance(e, dict) and e.get("id")] \
        if isinstance(entries, list) else []


def save_index(entries: list[dict], root: Path | str | None = None) -> Path:
    path = index_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps({"version": INDEX_VERSION, "novels": entries},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def _entry_root(entry: dict, base: Path) -> Path:
    rel = str(entry.get("path") or "").strip()
    if not rel or rel == ".":
        return base
    return (base / rel).resolve()


def _load_novel_json(path: Path) -> tuple[str, dict, dict]:
    title, art, cfg = "", {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return title, art, cfg
    if isinstance(data, dict):
        title = str(data.get("title") or "")
        if isinstance(data.get("art_direction"), dict):
            art = data["art_direction"]
        if isinstance(data.get("config"), dict):
            cfg = data["config"]
    return title, art, cfg


def list_novels(root: Path | str | None = None) -> list[Novel]:
    """Every novel: the legacy root novel first, then novels/<slug>."""
    base = Path(root) if root is not None else project_root()
    out = [_legacy_novel(base)]
    seen = {DEFAULT_NOVEL_ID}
    for entry in load_index(base):
        nid = str(entry.get("id") or "").strip()
        if not nid or nid in seen:
            continue
        nroot = _entry_root(entry, base)
        novel = Novel(id=nid, title=str(entry.get("title") or nid), root=nroot)
        if nid != DEFAULT_NOVEL_ID:
            t, art, cfg = _load_novel_json(novel.novel_file())
            novel.title = t or novel.title
            novel.art_direction = art
            novel.config = cfg
        out.append(novel)
        seen.add(nid)
    return out


def get_novel(novel_id: str | None = None,
              root: Path | str | None = None) -> Novel | None:
    """Resolve a novel id. None/"" -> the legacy novel. Unknown -> None."""
    if novel_id in (None, ""):
        return _legacy_novel(root)
    for novel in list_novels(root):
        if novel.id == novel_id:
            return novel
    return None


def create_novel(title: str, novel_id: str | None = None,
                 root: Path | str | None = None) -> Novel:
    """Create novels/<slug>/ with its folders, novel.json, empty bibles and a
    rights manifest stub. Returns the Novel. Raises ValueError on collision."""
    base = Path(root) if root is not None else project_root()
    existing = list_novels(base)
    used = {n.id for n in existing}
    slug = slugify(novel_id or title)
    if not slug or slug == DEFAULT_NOVEL_ID:
        raise ValueError(f"invalid novel id: {novel_id or title!r}")
    candidate = slug
    n = 2
    while candidate in used:
        candidate = f"{slug}_{n}"
        n += 1
    slug = candidate
    nroot = base / "novels" / slug
    for d in _PER_NOVEL_DIRS:
        (nroot / d).mkdir(parents=True, exist_ok=True)

    novel = Novel(id=slug, title=str(title or slug), root=nroot)
    novel_file = novel.novel_file()
    novel_file.write_text(json.dumps({
        "id": slug, "title": novel.title,
        # style: one of the shared ANIMATION_STYLES names. still/motion/
        # negative: extra prompt fragments for that novel's book.
        "art_direction": {"style": "", "still": "", "motion": "", "negative": ""},
        "config": {}},
        indent=2, ensure_ascii=False), encoding="utf-8")

    chars = nroot / "characters" / "characters.json"
    if not chars.exists():
        chars.write_text(json.dumps({"characters": {}, "settings": {}},
                                    indent=2, ensure_ascii=False), encoding="utf-8")
    props = nroot / "characters" / "props.json"
    if not props.exists():
        props.write_text(json.dumps({"props": {}}, indent=2), encoding="utf-8")
    rights = novel.rights_manifest_file()
    if not rights.exists():
        # Same stub a fresh project gets: preflight blocks until the owner
        # approves it from the dashboard, exactly like the legacy novel.
        from pipeline.rights_manifest import _empty_manifest, save_rights_manifest
        save_rights_manifest(_empty_manifest(), nroot)

    entries = [e for e in load_index(base) if str(e.get("id")) != slug]
    entries.append({"id": slug, "title": novel.title,
                    "path": f"novels/{slug}"})
    save_index(entries, base)
    return novel
