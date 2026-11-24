"""Find French dialogue turns whose forms of address warrant contextual review."""
from dataclasses import dataclass
import re

_SPEAKER = re.compile(r"^([^\W\d_][\w .’'\-]{0,34})\s*:\s*", re.UNICODE)
_FAMILIAR = re.compile(r"(?<!\w)(?:tu|te|toi|ton|ta|tes)\b|(?<!\w)t['’]", re.I)
_FORMAL = re.compile(r"(?<!\w)(?:vous|votre|vos)\b", re.I)
_PLURAL = re.compile(r"\b(?:you (?:both|all)|(?:both|all) of you|you two|vous (?:deux|tous|toutes))\b", re.I)


@dataclass(frozen=True)
class Turn:
    index: int
    speaker: str
    source: str
    pronoun: str
    register: str


@dataclass(frozen=True)
class AddressPair:
    earlier: Turn
    later: Turn
    reciprocal: bool


def _turn(index, row):
    source = _SPEAKER.match(row.source)
    target = _SPEAKER.match(row.target)
    if not source or not target or source[1].strip().casefold() != target[1].strip().casefold():
        return None
    dialogue = row.target[target.end():]
    familiar, formal = _FAMILIAR.search(dialogue), _FORMAL.search(dialogue)
    if bool(familiar) == bool(formal) or _PLURAL.search(row.source) or _PLURAL.search(dialogue):
        return None
    return Turn(index, source[1].strip(), row.source[source.end():],
                (familiar or formal).group(), 'tu' if familiar else 'vous')


def _vocative(text, name):
    return re.search(rf"(?:^\s*{re.escape(name)}\s*,|[,—]\s*{re.escape(name)}(?=\s*[,!?—.]|$))",
                     text, re.I)


def address_pairs(segments):
    turns = {i: turn for i, row in enumerate(segments) if (turn := _turn(i, row)) is not None}
    names = sorted({t.speaker for t in turns.values()}, key=str.casefold)
    pairs = []
    for index, later in turns.items():
        for other in range(index - 1, max(-1, index - 5), -1):
            earlier = turns.get(other)
            if not earlier or earlier.register == later.register:
                continue
            reciprocal = earlier.speaker.casefold() != later.speaker.casefold()
            if reciprocal:
                matched = other == index - 1 and _vocative(later.source, earlier.speaker)
            else:
                matched = any(name.casefold() != later.speaker.casefold() and
                              _vocative(earlier.source, name) and _vocative(later.source, name)
                              for name in names)
            if matched:
                pairs.append(AddressPair(earlier, later, reciprocal))
                break
        if len(pairs) == 24:
            break
    return pairs
