"""Small, public evidence checks for unambiguous bilingual cases."""
from collections import Counter
import re

from .inputs import same_native_file

_PRINTF = re.compile(r'%%|%(?:[1-9]\d*\$)?[-+ #0]*(?:\d+|\*)?(?:\.(?:\d+|\*))?(?:hh|ll|[hljztL])?[diouxXfFeEgGaAcspn]')
_SPACED_CONVERSION = re.compile(r'% (?:hh|ll|[hljztL])?[diouxXfFeEgGaAcspn]')
_NUMBER_WORDS = ('zero one two three four five six seven eight nine ten eleven twelve thirteen '
                 'fourteen fifteen sixteen seventeen eighteen nineteen twenty').split()
_CJK_DIGITS = '零一二三四五六七八九十'
_EQUIVALENT_INT = {**dict(zip(_NUMBER_WORDS, range(len(_NUMBER_WORDS)))),
                   **dict(zip(_CJK_DIGITS, range(len(_CJK_DIGITS))))}
_COLLAPSE = re.compile(r'\bcollapse\b', re.I)
_SYNCOPE = re.compile(r'\b(?:syncope|faint(?:ed|s|ing)?|passed out|lost consciousness)\b', re.I)


def _literal_percentage(text, match):
    if not _SPACED_CONVERSION.fullmatch(match.group()):
        return False
    before = text[:match.start()].rstrip()
    following = text[match.end():].lstrip("'’")
    return bool(before and before[-1].isdecimal() and following and following[0].isalpha())


def printf_tokens(text):
    return Counter(match.group() for match in _PRINTF.finditer(text)
                   if match.group() != '%%' and not _literal_percentage(text, match))


def printf_false_positive(issue, segment):
    """Ignore only a placeholder allegation specifically grounded in a literal percent."""
    if printf_tokens(segment.source) != printf_tokens(segment.target):
        return False
    detail = issue.explanation
    if not re.search(r'printf|placeholder|conversion specifier|占位符|佔位符', detail, re.I):
        return False
    quotes = [quote for quote in (issue.source_quote, issue.target_quote) if quote.strip()]
    if not quotes or any(printf_tokens(quote) for quote in quotes):
        # A short '% de' quote has no numeral; validate it in the full target.
        if not quotes or any(
            any(match.group() in quote and not _literal_percentage(text, match)
                for match in _PRINTF.finditer(text) if match.group() != '%%')
            for text, quote in ((segment.source, issue.source_quote), (segment.target, issue.target_quote)) if quote
        ):
            return False
    return (any(re.search(r'\d\s*%|%[ ]+[a-z]', quote, re.I) for quote in quotes)
            and any(re.search(r'\d\s*%', text) for text in (segment.source, segment.target)))


def _single_integer(text):
    token = text.strip().strip(' “…”"\'‘’「」.。').casefold()
    if re.fullmatch(r'[0-9]{1,6}', token):
        return int(token)
    return _EQUIVALENT_INT.get(token)


def numeric_false_positive(issue, segment):
    """Suppress a number-only finding when the original target already gives its value."""
    if issue.category not in ('accuracy', 'omission'):
        return False
    value = _single_integer(issue.source_quote)
    if value is None:
        return False
    if issue.target_quote.strip():
        return _single_integer(issue.target_quote) == value
    if not re.search(r'number|numeral|digit|数字|數字|数值|數值', issue.explanation, re.I):
        return False
    tokens = re.findall(r'\b(?:[0-9]{1,6}|[a-z]+)\b|[零一二三四五六七八九十]', segment.target.casefold())
    return any(_single_integer(token) == value for token in tokens)


def is_named_term_vs_participle(first_quote, second_quote, source_language, target_language):
    """A capitalized English label and the corresponding participle may translate differently."""
    if (source_language.split('-')[0].casefold(), target_language.split('-')[0].casefold()) != ('en', 'fr'):
        return False
    for label, participle in ((first_quote, second_quote), (second_quote, first_quote)):
        if not re.fullmatch(r'[A-Z][a-z]{2,}', label) or not re.fullmatch(r'[A-Za-z]{4,}', participle):
            continue
        singular = label.casefold()
        if singular.endswith('s'):
            singular = singular[:-1]
        if participle.casefold() in (singular + 'ed', singular + 'd'):
            return True
    return False


def collapse_pairs(segments, source_language, target_language):
    """Pinpoint unsupported narrowing within an explicit two-turn reference."""
    if (source_language.split('-')[0].casefold(), target_language.split('-')[0].casefold()) != ('en', 'zh'):
        return
    for left, right in zip(segments, segments[1:]):
        if (not same_native_file(left, right) or _SYNCOPE.search(left.source + ' ' + right.source)
                or not re.search(r'\ba collapse\b', left.source, re.I)
                or not re.search(r'\bthe collapse\b', right.source, re.I)):
            continue
        broad = next((word for word in ('倒下', '倒地') if word in left.target), '')
        narrow = next((word for word in ('晕厥', '暈厥') if word in right.target), '')
        if broad and narrow and not any(word in left.target for word in ('晕厥', '暈厥')):
            yield left, right, broad, narrow


def is_collapse_duplicate(issue, pairs):
    for left, right, broad, narrow in pairs:
        if issue.category not in ('terminology', 'accuracy', 'addition', 'omission') or issue.segment_id not in (left.id, right.id):
            continue
        term = issue.source_quote + ' ' + issue.related_source_quote
        mentions = term + ' ' + issue.explanation
        if not (_COLLAPSE.search(mentions) or '倒下' in mentions or '倒地' in mentions):
            continue
        if issue.origin == 'document' and {issue.segment_id, issue.related_id} == {left.id, right.id} and _COLLAPSE.search(term):
            return True
        if issue.segment_id == right.id and narrow in (issue.target_quote or issue.explanation):
            return True
        if issue.segment_id == left.id and broad in issue.target_quote and narrow in issue.explanation:
            return True
    return False
