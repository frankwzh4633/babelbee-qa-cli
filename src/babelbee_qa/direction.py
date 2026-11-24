"""Detect explicit transfer-direction reversals in aligned English–Chinese labels."""
import re

_FACILITY = r'(?:hospital|facility|ward|unit)'
_SOURCE = re.compile(
    rf'(?:'
    rf'(?P<incoming>{_FACILITY}\s+transferred\s+(?:in\s+)?from'
    rf'|transferred\s+in\s+from(?:\s+{_FACILITY})?'
    rf'|transfer(?:red)?\s+into\s+(?:the\s+)?{_FACILITY})'
    rf'|(?P<outgoing>{_FACILITY}\s+transferred\s+(?:out\s+)?to'
    rf'|transferred\s+out\s+to(?:\s+{_FACILITY})?'
    rf'|transfer(?:red)?\s+out\s+of\s+(?:the\s+)?{_FACILITY})'
    rf')\s*[:：]?',
    re.IGNORECASE,
)
_TARGET = re.compile(
    r'(?:(?P<incoming>转入|轉入)|(?P<outgoing>转出|轉出))'
    r'(?:的)?(?:来源|來源|目的地|目的)?(?:医院|醫院|机构|機構|病区|病區|病房|科室|单位|單位)?'
    r'\s*[:：]?',
)


def transfer_direction_mismatch(source, target):
    """Return the source direction and exact opposing label evidence if both are explicit."""
    source_label = _SOURCE.fullmatch(source.strip().strip('\u200b\u2060\ufeff').strip())
    target_label = _TARGET.fullmatch(target.strip().strip('\u200b\u2060\ufeff').strip())
    if not source_label or not target_label or source_label.lastgroup == target_label.lastgroup:
        return None
    return source_label.lastgroup, source_label.group(source_label.lastgroup), target_label.group(target_label.lastgroup)
