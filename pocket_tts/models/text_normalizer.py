"""Intelligent text normalization for TTS.

Converts common unpronounceable forms to spoken English while preserving
bracketed style/emotion tags like [happy].
"""
from __future__ import annotations

import re

_STYLE_TAG_RE = re.compile(r"\[(whisper|cheerful|happy|calm|serious)\]", re.IGNORECASE)

_ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
    13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen",
}
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
_SCALES = [(10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")]
_ORDINAL_EXCEPTIONS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


def _int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens * 10] + (f" {_ONES[rest]}" if rest else "")
    if n < 1000 and n % 100:
        return f"{_int_to_words(n // 100)} hundred {_int_to_words(n % 100)}"
    parts: list[str] = []
    rem = n
    for scale, name in _SCALES:
        if rem >= scale:
            q, rem = divmod(rem, scale)
            parts.append(f"{_int_to_words(q)} {name}")
    if rem:
        parts.append(_int_to_words(rem))
    return " ".join(parts).strip()


def _ordinal_word(word_number: str) -> str:
    if word_number in _ORDINAL_EXCEPTIONS:
        return _ORDINAL_EXCEPTIONS[word_number]
    if word_number.endswith("y"):
        return word_number[:-1] + "ieth"
    if word_number.endswith("e"):
        return word_number + "th" if word_number.endswith("ie") else word_number[:-1] + "th"
    return word_number + "th"


def number_to_ordinal(n: int) -> str:
    words = _int_to_words(n)
    toks = words.split()
    toks[-1] = _ordinal_word(toks[-1])
    return " ".join(toks)


# Avoid expanding titles/abbreviations that text_chunking treats as boundary-sensitive.
_ABBREV = {
    r"\bDr\.": "Doctor",
    r"\bMr\.": "Mister",
    r"\bMrs\.": "Missus",
    r"\bMs\.": "Miss",
    r"\bProf\.": "Professor",
    r"\bSt\.": "Saint",
    r"\bAve\.": "Avenue",
    r"\bRd\.": "Road",
    r"\bBlvd\.": "Boulevard",
    r"\betc\.": "etcetera",
    r"\bvs\.": "versus",
}


def _replace_currency(match: re.Match[str]) -> str:
    num = match.group(0).replace("$", "").replace(",", "").strip()
    if "." in num:
        dollars_s, cents_s = num.split(".", 1)
        dollars = int(dollars_s or "0")
        cents = int((cents_s + "00")[:2])
    else:
        dollars = int(num or "0")
        cents = 0
    out = f"{_int_to_words(dollars)} dollar{'s' if dollars != 1 else ''}"
    if cents:
        out += f" and {_int_to_words(cents)} cent{'s' if cents != 1 else ''}"
    return out


def normalize_text_for_tts(text: str, *, normalize_numbers: bool = True) -> str:
    if not text:
        return text
    tags: list[str] = []

    def _tag_repl(m: re.Match[str]) -> str:
        tags.append(m.group(0))
        return f" __STYLE_TAG_{len(tags)-1}__ "

    s = _STYLE_TAG_RE.sub(_tag_repl, text)
    s = re.sub(r"\$\s?\d[\d,]*(?:\.\d+)?", _replace_currency, s)
    for pat, repl in _ABBREV.items():
        s = re.sub(pat, repl, s)

    if normalize_numbers:
        s = re.sub(r"\b(\d+)\s*(?:st|nd|rd|th)\b", lambda m: number_to_ordinal(int(m.group(1))), s, flags=re.IGNORECASE)
        # Keep decimals/versions (3.14, 2.0) intact; they matter for chunking/pronunciation handling.
        def _num(m: re.Match[str]) -> str:
            raw = m.group(0)
            if "." in raw:
                return raw
            return _int_to_words(int(raw.replace(",", "")))

        s = re.sub(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b", _num, s)

    for i, tag in enumerate(tags):
        s = s.replace(f"__STYLE_TAG_{i}__", tag)
    s = re.sub(r"\s+", " ", s).strip()
    return s


__all__ = ["normalize_text_for_tts", "number_to_ordinal", "_int_to_words"]
