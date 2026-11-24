"""Choose repeat-source anchors for a focused comparison of bilingual segments."""
from collections import defaultdict
from dataclasses import dataclass
import re

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_TAG = re.compile(r'⟦[^⟧]*⟧')
_STOP = frozenset("""
about after again against almost along already also although among another around back
based been before being between both came come could does done down during each early
either even every from further gave give going gone had has have having here herself
himself into itself just keep last later least less like made make many might more most much
must next none only onto other over own part please previous rather really right same said
seen shall should since some soon still such take than that their them then there these
they thing this those through thus today together took toward under until upon used using
very want were what when where which while will with would years your
""".split())
_MAX_NEAR = 9
_MAX_OTHER = 9
_MAX_PAIR_CHARS = 2_200


@dataclass(frozen=True)
class TermCandidate:
    term: str
    left: str
    right: str
    source_quotes: tuple[str, str]


def verbatim_term_match(term: str, source: str, target: str) -> bool:
    """Find a capitalized Latin term preserved verbatim outside inline codes."""
    if not re.fullmatch(r"[a-z][a-z0-9'-]*(?: [a-z0-9][a-z0-9'-]*)*", term):
        return False
    # A Chinese character next to a Latin name is a boundary. A Latin suffix is not.
    before, after = r'(?<![A-Za-z0-9_-])', r'(?![A-Za-z0-9_-])'
    pattern = r'\s+'.join(map(re.escape, term.split()))
    source_text, target_text = (_TAG.sub(' ', text) for text in (source, target))
    return any(
        match[0][0].isupper() and re.search(before + re.escape(match[0]) + after, target_text)
        for match in re.finditer(before + pattern + after, source_text, re.ASCII | re.IGNORECASE)
    )


def _anchors(source):
    text = _TAG.sub(lambda m: ' ' * len(m.group()), source)
    words = list(_WORD.finditer(text))
    for i, first in enumerate(words):
        token = first.group().lower()
        if len(token) < 4 or token in _STOP:
            continue
        yield token, source[first.start():first.end()]
        for last in words[i + 1:i + 3]:
            phrase = source[first.start():last.end()]
            if not re.fullmatch(r"[A-Za-z'-]+(?: +[A-Za-z'-]+){1,2}", phrase):
                break
            if len(last.group()) >= 3 and last.group().lower() not in _STOP:
                yield phrase.lower(), phrase


def term_candidates(segments, source_language):
    if re.split(r'[-_]', source_language.casefold(), maxsplit=1)[0] != 'en':
        return ()
    occurrences = defaultdict(list)
    for position, segment in enumerate(segments):
        if not segment.target.strip():
            continue
        size = len(segment.source) + len(segment.target)
        if size >= _MAX_PAIR_CHARS:
            continue
        for term, quote in dict(_anchors(segment.source)).items():
            occurrences[term].append((position, segment, quote, size))
    nearby, other = [], []
    for term, hits in occurrences.items():
        if len(hits) < 2:
            continue
        choices = []
        for first, second in zip(hits, hits[1:]):
            a, left, quote1, size_a = first
            b, right, quote2, size_b = second
            if (size_a + size_b > _MAX_PAIR_CHARS
                    or left.source.casefold().strip() == right.source.casefold().strip()
                    or left.target.strip() == right.target.strip()):
                continue
            choices.append((b - a, a, TermCandidate(term, left.id, right.id, (quote1, quote2))))
        if not choices:
            continue
        distance, first, candidate = min(choices, key=lambda item: item[:2])
        if distance <= 3 and ' ' not in term and len(term) >= 5:
            nearby.append((distance, -len(term), first, candidate))
        else:
            other.append((-term.count(' '), -min(len(hits), 5), distance, first, candidate))
    picked = []
    seen = set()
    for ranked, limit in ((nearby, _MAX_NEAR), (other, _MAX_OTHER)):
        used = 0
        for *_, candidate in sorted(ranked, key=lambda item: (*item[:-1], item[-1].term)):
            pair = (candidate.left, candidate.right)
            if pair in seen:
                continue
            seen.add(pair)
            picked.append(candidate)
            used += 1
            if used == limit:
                break
    return tuple(picked)
