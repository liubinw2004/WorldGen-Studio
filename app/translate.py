"""Chinese -> English prompt translation for FLUX.

Why this exists: FLUX.1-dev's text encoder is T5-XXL, trained on English only,
and the panorama prompt baked into ``pano_gen.py`` is English too. A Chinese
description tokenises to ``<unk>`` -- measured on a typical sentence, 4 of 9
tokens were unknown, so the model receives punctuation and nothing else and
falls back to a generic panorama. Translating first is the only way to make
Chinese input actually mean something.

The translator is MarianMT ``opus-mt-zh-en`` (~300 MB) and runs on CPU, so it
costs no VRAM. First use loads it (~1 s); afterwards a short prompt takes
100-300 ms.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "mt" / "opus-mt-zh-en"

# CJK unified ideographs + extension A + compatibility block.
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# MarianMT is a general-purpose model and mangles scene vocabulary: measured on
# real prompts it turned 锦鲤池 into "cascading pool", 电影感光线 into "the movie
# senses" and -- worse -- transliterated 赛博朋克 into the meaningless
# "Saberpenk".
#
# Two layers fix that, and the split matters:
#
# * PRE substitutes only proper nouns that have no chance of surviving
#   translation (loanwords get transliterated, brand-ish terms get invented).
#   An earlier version substituted *every* known term; the sentence then became
#   mostly Latin and the translator hallucinated ("When you're ready, you'll be
#   able to use it"). Keep this list short.
# * POST rewrites specific wrong phrases the model reliably produces. It is a
#   plain string table, so it can never make a sentence worse than the MT output.
GLOSSARY_PRE = {
    "赛博朋克": "cyberpunk",
    "蒸汽朋克": "steampunk",
    "丁达尔效应": "god rays",
    "超广角": "ultra wide angle",
    "高动态范围": "HDR",
    "日式庭院": "japanese garden",
    "锦鲤池": "koi pond",
    "锦鲤": "koi",
    "苔藓": "moss",
}

POSTFIX = (
    ("the movie senses", "cinematic lighting"),
    ("movie senses", "cinematic lighting"),
    ("the cascading pool", "a koi pond"),
    ("cascading pool", "koi pond"),
    ("the wide corners", "wide angle"),
    ("wide corners", "wide angle"),
    ("rain nights", "rainy night"),
    ("wet reflection ground", "wet reflective ground"),
    ("the fog of the morning", "morning mist"),
    ("the warm lights", "warm lanterns"),
    ("warm lights", "warm lanterns"),
    ("a wood library", "a wooden library"),
)
_GLOSSARY_RE = re.compile(
    "|".join(map(re.escape, sorted(GLOSSARY_PRE, key=len, reverse=True))))


def _apply_glossary(text: str) -> str:
    return _GLOSSARY_RE.sub(lambda m: " " + GLOSSARY_PRE[m.group(0)] + " ", text)


def _apply_postfix(text: str) -> str:
    low = text.lower()
    for wrong, right in POSTFIX:
        if wrong in low:
            text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
            low = text.lower()
    return text

_lock = threading.Lock()
_tok = None
_model = None
_failed: str = ""


def has_chinese(text: str) -> bool:
    """True when the string contains at least one CJK character."""
    return bool(text and _CJK_RE.search(text))


def model_ready() -> bool:
    if not (MODEL_DIR / "config.json").exists():
        return False
    return any(MODEL_DIR.glob("*.safetensors")) or any(
        MODEL_DIR.glob("pytorch_model*.bin"))


def _load() -> None:
    """Load tokenizer + weights once, on first use, under a lock."""
    global _tok, _model, _failed
    if _model is not None or _failed:
        return
    if not model_ready():
        _failed = f"翻译模型缺失: {MODEL_DIR}"
        raise RuntimeError(_failed)

    from transformers import MarianMTModel, MarianTokenizer

    _tok = MarianTokenizer.from_pretrained(str(MODEL_DIR))
    _model = MarianMTModel.from_pretrained(str(MODEL_DIR))
    _model.eval()


def _clean(text: str) -> str:
    """Collapse whitespace and the join artefacts Marian likes to emit."""
    text = text.replace(">>cmn_Hans<<", " ").strip()
    # Marian occasionally wraps fragments in musical notes or emits stray
    # control markers when the input is short -- they mean nothing to FLUX.
    text = text.replace("\u266a", " ")
    text = re.sub(r"\s+", " ", text)
    # Marian sometimes leaves a leading comma or a dangling one.
    return text.strip(" ,.;:")


def translate(text: str) -> str:
    """Translate Chinese to English. Non-Chinese input is returned unchanged."""
    src = (text or "").strip()
    if not src or not has_chinese(src):
        return src

    with _lock:
        _load()
        import torch

        prepared = _apply_glossary(src)
        batch = _tok([prepared], return_tensors="pt", padding=True, truncation=True,
                     max_length=512)
        with torch.no_grad():
            ids = _model.generate(**batch, max_length=512, num_beams=4,
                                  no_repeat_ngram_size=3)
        out = _tok.batch_decode(ids, skip_special_tokens=True)[0]
    return _clean(_apply_postfix(out)) or src


def translate_prompt(text: str) -> tuple[str, bool]:
    """Return ``(english, was_translated)``.

    Failures degrade to the original text rather than blocking a generation:
    a bad prompt is better than a failed run.
    """
    if not has_chinese(text):
        return text, False
    try:
        return translate(text), True
    except Exception:  # noqa: BLE001 - never break generation over this
        return text, False
