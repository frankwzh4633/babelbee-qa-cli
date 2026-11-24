"""Read aligned pairs without modifying the input."""
import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
import unicodedata
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

from defusedxml.ElementTree import fromstring
from defusedxml.common import DefusedXmlException
from openpyxl import load_workbook
from openpyxl.utils.exceptions import IllegalCharacterError, InvalidFileException

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 500
MAX_CHARS = 60_000
MAX_SEGMENT_CHARS = 8_000
XLIFF_1 = 'urn:oasis:names:tc:xliff:document:1.2'
XLIFF_2 = 'urn:oasis:names:tc:xliff:document:2.0'


class QAError(Exception):
    """A user-facing failure."""


@dataclass(frozen=True)
class Segment:
    id: str
    location: str
    source: str
    target: str
    source_codes: tuple[str, ...] = ()
    target_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Document:
    segments: tuple[Segment, ...]
    source_language: str | None
    target_language: str | None


def same_native_file(left: Segment, right: Segment):
    """Each XLIFF file has its own neighboring segments."""
    if left.location.startswith('File ') or right.location.startswith('File '):
        return left.location.split(' / ', 1)[0] == right.location.split(' / ', 1)[0]
    return True


def _table(data, suffix, sheet_name, source_column, target_column, header):
    columns = tuple(sorted((source_column, target_column)))
    if suffix == '.csv':
        if sheet_name:
            raise QAError('--sheet applies only to XLSX workbooks.')
        reader = csv.reader(StringIO(data.decode('utf-8-sig'), newline=''), strict=True)
        previous_line = 0
        for row in reader:
            location = f'CSV line {previous_line + 1}'
            previous_line = reader.line_num
            source, target = (row[index - 1] if index <= len(row) else '' for index in (source_column, target_column))
            if not (source.strip() or target.strip()):
                continue
            if header:
                header = False
                continue
            yield location, source, target
        return

    with ZipFile(BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 2_000 or sum(entry.file_size for entry in entries) > 32 * 1024 * 1024:
            raise QAError('Excel archive exceeds the expanded size limit.')
    workbook = load_workbook(BytesIO(data), read_only=True, data_only=False, keep_links=False)
    try:
        if sheet_name is None and len(workbook.worksheets) != 1:
            raise QAError('Use --sheet to select a worksheet from a multi-sheet workbook.')
        if sheet_name and sheet_name not in workbook.sheetnames:
            raise QAError('The selected worksheet does not exist.')
        sheet = workbook[sheet_name] if sheet_name else workbook.worksheets[0]
        # Ignore cached worksheet dimensions; some producers write stale dimensions.
        sheet.reset_dimensions()
        for number, cells in enumerate(sheet.iter_rows(min_col=1, max_col=columns[-1]), 1):
            if number > 20_000:
                raise QAError('Excel worksheet exceeds the physical row limit.')
            selected = tuple(cells[index - 1] for index in (source_column, target_column))
            if all(cell.value is None or not str(cell.value).strip() for cell in selected):
                continue
            if header:
                header = False
                continue
            if any(cell.data_type in {'f', 'e'} for cell in selected):
                raise QAError(f'Excel row {number}: replace formulas or cell errors with text values.')
            yield f'{sheet.title}!{number}', *(str(cell.value) if cell.value is not None else '' for cell in selected)
    finally:
        workbook.close()


def _inline(node, namespace):
    codes = []

    def read(element):
        parts = [element.text or '']
        for child in element:
            kind = child.tag.removeprefix(namespace)
            if child.tag in {f'{namespace}mrk', f'{namespace}g', f'{namespace}pc', f'{namespace}sub'}:
                if kind in {'g', 'pc'}:
                    codes.append(f'{kind}:{child.get("id", "")}:start')
                parts.append(read(child))
                if kind in {'g', 'pc'}:
                    codes.append(f'{kind}:{child.get("id", "")}:end')
            elif child.tag in {f'{namespace}{name}' for name in ('x', 'ph', 'bpt', 'ept', 'bx', 'ex', 'it', 'sc', 'ec', 'cp', 'sm', 'em')}:
                code_id = child.get('rid') or child.get('id') or child.get('startRef') or child.get('hex') or ''
                codes.append(f'{kind}:{code_id}')
                if code_id:
                    parts.append(f'⟦{kind}:{code_id}⟧')
            else:
                raise QAError('Unsupported XLIFF inline element.')
            parts.append(child.tail or '')
        return ''.join(parts)

    return (read(node), tuple(codes)) if node is not None else ('', ())


def _sdl_marks(container, namespace):
    if container is None:
        return {}
    marks = {}

    def walk(parent):
        if (parent.text or '').strip():
            raise QAError('Cannot align SDLXLIFF text outside segment markers.')
        for node in parent:
            if node.tag == f'{namespace}mrk' and node.get('mtype') == 'seg':
                mid = node.get('mid')
                if not mid or mid in marks or node.findall(f'.//{namespace}mrk[@mtype="seg"]'):
                    raise QAError('SDLXLIFF has ambiguous segment IDs.')
                marks[mid] = node
            elif node.tag == f'{namespace}g':
                walk(node)
            elif node.tag != f'{namespace}x' or len(node) or (node.text or '').strip():
                raise QAError('Cannot align SDLXLIFF content outside segment markers.')
            if (node.tail or '').strip():
                raise QAError('Cannot align SDLXLIFF text outside segment markers.')

    walk(container)
    return marks


def _xliff(root, suffix):
    version = root.get('version')
    if root.tag == f'{{{XLIFF_1}}}xliff' and version == '1.2':
        namespace = f'{{{XLIFF_1}}}'
    elif root.tag == f'{{{XLIFF_2}}}xliff' and version in {'2.0', '2.1'} and suffix != '.sdlxliff':
        namespace = f'{{{XLIFF_2}}}'
    else:
        raise QAError('Expected XLIFF 1.2, 2.0 or 2.1 (SDLXLIFF uses 1.2).')
    files = root.findall(f'{namespace}file')

    def language(attribute):
        values = {str(value).strip() for file in files if (value := file.get(attribute))}
        if len({value.lower() for value in values}) > 1:
            raise QAError('Import one source and target language pair at a time.')
        return next(iter(values), None)

    def rows():
        for number, file in enumerate(files, 1):
            if version != '1.2':
                for unit in file.iter(f'{namespace}unit'):
                    for index, segment in enumerate(unit.findall(f'{namespace}segment'), 1):
                        yield (f'File {number} / unit {unit.get("id", "?")} / segment {segment.get("id") or index}',
                               segment.find(f'{namespace}source'), segment.find(f'{namespace}target'))
                continue
            body = file.find(f'{namespace}body')
            if body is None:
                continue
            for unit in body.iter(f'{namespace}trans-unit'):
                location = f'File {number} / unit {unit.get("id", "?")}'
                target = unit.find(f'{namespace}target')
                segmented = unit.find(f'{namespace}seg-source')
                if segmented is None:
                    yield location, unit.find(f'{namespace}source'), target
                    continue
                sources = _sdl_marks(segmented, namespace)
                targets = _sdl_marks(target, namespace)
                if not sources and not targets:
                    # CAT exports can emit formatting-only units around real segments.
                    plain_source = unit.find(f'{namespace}source')
                    if plain_source is None or not ''.join(plain_source.itertext()).strip():
                        continue
                if not sources or not targets.keys() <= sources.keys():
                    raise QAError('SDLXLIFF source and target markers cannot be matched.')
                for mid, source in sources.items():
                    yield f'{location} / segment {mid}', source, targets.get(mid)

    def pairs():
        for location, source_node, target_node in rows():
            if source_node is None:
                raise QAError('An XLIFF segment is missing its source element.')
            source, source_codes = _inline(source_node, namespace)
            target, target_codes = _inline(target_node, namespace)
            yield location, source, target, source_codes, target_codes

    if version == '1.2':
        return pairs(), language('source-language'), language('target-language')
    return pairs(), root.get('srcLang'), root.get('trgLang')


def read_document(path: Path, *, sheet=None, source_column=1, target_column=2, header=True):
    suffix = path.suffix.lower()
    if suffix not in {'.csv', '.xlsx', '.xlf', '.xliff', '.sdlxliff'}:
        raise QAError('Choose a CSV, XLSX, XLF, XLIFF or SDLXLIFF file.')
    if source_column == target_column or not all(1 <= column <= 16_384 for column in (source_column, target_column)):
        raise QAError('Choose two different column numbers between 1 and 16384.')
    try:
        with path.open('rb') as file:
            data = file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise QAError('Input exceeds 5 MiB.')
        source_language = target_language = None
        if suffix in {'.csv', '.xlsx'}:
            rows = ((location, source, target, (), ()) for location, source, target in
                    _table(data, suffix, sheet, source_column, target_column, header))
        else:
            if sheet:
                raise QAError('--sheet applies only to XLSX workbooks.')
            root = fromstring(data, forbid_dtd=True)
            rows, source_language, target_language = _xliff(root, suffix)
        segments = []
        char_count = 0
        for location, source, target, source_codes, target_codes in rows:
            source_visible = ''.join(c for c in source if unicodedata.category(c) != 'Cf').strip()
            target_visible = ''.join(c for c in target if unicodedata.category(c) != 'Cf').strip()
            if not source_visible and not target_visible:
                # Formatting wrappers and invisible characters can be emitted as
                # independent CAT rows. Standalone codes include a visible marker.
                continue
            if not source_visible and not any(not c.endswith((':start', ':end')) for c in source_codes):
                raise QAError(f'{location}: translation has no corresponding source.')
            if not target_visible and not any(not c.endswith((':start', ':end')) for c in target_codes):
                target, target_codes = '', ()
            size = len(source) + len(target) + sum(map(len, (*source_codes, *target_codes)))
            char_count += size
            if size > MAX_SEGMENT_CHARS:
                raise QAError(f'{location}: a pair exceeds {MAX_SEGMENT_CHARS:,} characters.')
            if len(segments) >= MAX_ROWS or char_count > MAX_CHARS:
                raise QAError(f'Input exceeds {MAX_ROWS} segments or {MAX_CHARS:,} combined characters.')
            segments.append(Segment(f'SEG-{len(segments) + 1:03}', location, source, target, source_codes, target_codes))
    except (OSError, UnicodeError, csv.Error, ParseError, DefusedXmlException, BadZipFile,
            InvalidFileException, IllegalCharacterError, KeyError, IndexError, RecursionError, ValueError) as exc:
        raise QAError(f'Cannot read {suffix} input. Check the file format, encoding and worksheet data.') from exc
    if not segments:
        raise QAError('The document contains no aligned source segments.')
    return Document(tuple(segments), source_language, target_language)
