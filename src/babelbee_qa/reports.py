"""Render an issue report and the reviewed segments in HTML and Excel."""
from collections import Counter
from html import escape
from io import BytesIO
from pathlib import Path
import re

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .inputs import QAError
from .review import SEVERITIES

_TEXT = {
    'title': ('Translation quality report', '译文质检报告'),
    'context': ('Document context', '全文背景'),
    'coverage': ('Terminology check coverage', '术语检查覆盖提示'),
    'coverage_gaps': ('Unverified term occurrences', '待核实术语位置'),
    'term': ('Source term', '原文术语'),
    'findings': ('Findings', '问题清单'),
    'segments': ('Segments', '句段'),
    'summary': ('Summary', '摘要'),
    'source_language': ('Source language', '原文语言'),
    'target_language': ('Target language', '译文语言'),
    'model': ('AI model', 'AI 模型'),
    'notice': ('AI findings and suggested translations require human review.', '请人工核实 AI 发现的问题和建议译文。'),
    'segment': ('Segment', '句段号'),
    'location': ('File location', '文件位置'),
    'category': ('Category', '类别'),
    'severity': ('Severity', '严重程度'),
    'explanation': ('Explanation', '问题说明'),
    'source_quote': ('Source evidence', '原文证据'),
    'target_quote': ('Translation evidence', '译文证据'),
    'related_id': ('Comparison segment', '对照句段'),
    'related_source_quote': ('Comparison source', '对照原文'),
    'related_target_quote': ('Comparison translation', '对照译文'),
    'suggestion': ('Suggested translation', '建议译文'),
    'origin': ('Check', '检查方式'),
    'source': ('Source', '原文'),
    'target': ('Translation', '受检译文'),
    'critical': ('Critical', '严重错误'),
    'major': ('Major', '主要错误'),
    'minor': ('Minor', '轻微错误'),
    'advisory': ('Review advisory', '待核实提示'),
    'register': ('Dialogue address review', '对话称呼复核'),
    'accuracy': ('Accuracy', '准确性'),
    'omission': ('Omission', '漏译'),
    'addition': ('Addition', '增译'),
    'terminology': ('Terminology', '术语'),
    'fluency': ('Fluency', '语言表达'),
    'formatting': ('Formatting', '格式'),
    'ai': ('AI segment review', 'AI 逐句检查'),
    'rule': ('Rule check', '规则检查'),
    'document': ('AI terminology review', 'AI 全文术语检查'),
    'source_check': ('AI source-ending review', 'AI 原文截断复核'),
}


def _content(review):
    def label(key):
        return _TEXT[key][review.report_language == 'zh']

    counts = Counter(finding.severity for finding in review.findings)
    summary = [(label('segments'), len(review.segments)),
               (label('findings'), len(review.findings) - counts['advisory'])]
    summary.extend((label(severity), counts[severity]) for severity in SEVERITIES
                   if severity != 'advisory' or counts[severity])
    summary.extend([
        (label('source_language'), review.source_language),
        (label('target_language'), review.target_language),
        (label('model'), review.model),
    ])
    issue_headers = ['segment', 'category', 'severity', 'explanation', 'source_quote', 'target_quote',
                     'related_id', 'related_source_quote', 'related_target_quote', 'suggestion', 'origin']
    findings = [(issue.segment_id, label(issue.category), label(issue.severity), issue.explanation,
                 issue.source_quote, issue.target_quote, issue.related_id, issue.related_source_quote, issue.related_target_quote,
                 issue.suggestion, label(issue.origin)) for issue in review.findings]
    segment_headers = ['segment', 'location', 'source', 'target']
    segments = [(row.id, row.location, row.source, row.target) for row in review.segments]
    tables = [
        (label('findings'), [label(key) for key in issue_headers], findings),
        (label('segments'), [label(key) for key in segment_headers], segments),
    ]
    if review.terminology_coverage.gaps:
        locations = {segment.id: segment.location for segment in review.segments}
        tables.append((label('coverage_gaps'), [label('term'), label('segment'), label('location')],
                       [(term, sid, locations[sid]) for term, sid in review.terminology_coverage.gaps]))
    return label, summary, tables


def _coverage_notice(review):
    coverage = review.terminology_coverage
    zh = review.report_language == 'zh'
    if not coverage.reported:
        notice = ('模型未返回可用的术语覆盖记录，请人工核对重复术语。' if zh else
                  'The model did not return a usable coverage record. Verify recurring terms manually.')
    elif not (coverage.compared or coverage.limitations or coverage.unverified):
        notice = ('模型本次未返回可核对的重复术语样本，请人工核对关键术语。' if zh else
                  'The model returned no recurring term samples for verification. Check key terms manually.')
    else:
        count = ('纳入首轮比较的重复术语样本中，{compared} 项有两处译文证据，{limited} 项无法可靠比对，'
                 '{unverified} 项证据未通过核验。' if zh else
                 'Of the recurring terms sampled initially, {compared} had evidence from two translations, '
                 '{limited} could not be reliably compared, and {unverified} did not pass evidence validation.')
        notice = count.format(compared=coverage.compared, limited=len(coverage.limitations), unverified=coverage.unverified)
    if coverage.expected_occurrences:
        verified, total, gaps = (len(coverage.verified_occurrences), len(coverage.expected_occurrences), len(coverage.gaps))
        scope = (f'所选术语的出现位置：已核验 {verified}/{total} 处，{gaps} 处待人工核实。' if zh else
                 f'Occurrences of selected terms: {verified}/{total} verified, {gaps} for manual review.')
        notice += f' {scope}'
    if coverage.candidate_count:
        focused = (f'定向复核提交了 {coverage.candidate_count} 个重复术语候选。' if zh else
                   f'The focused review received {coverage.candidate_count} recurring-term candidates.')
        notice = f'{focused} {notice}'
    notes = [f'{term} — {reason}' for term, reason in coverage.limitations]
    return '\n'.join([notice, *notes])


def _html(review, label, summary, tables):
    def e(value):
        return escape(_literal_text(str(value)))

    sections = []
    for title, headers, rows in tables:
        heading = ''.join(f'<th scope="col">{e(value)}</th>' for value in headers)
        body = ''.join('<tr>' + ''.join(f'<td>{e(value)}</td>' for value in row) + '</tr>' for row in rows)
        sections.append(f'<h2>{e(title)}</h2><div class="scroll"><table>'
                        f'<thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>')
    metadata = ''.join(f'<div><dt>{e(key)}</dt><dd>{e(value)}</dd></div>' for key, value in summary)
    language = 'zh-CN' if review.report_language == 'zh' else 'en'
    return f"""<!doctype html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>{e(label('title'))}</title>
<style>
body {{font:16px/1.55 system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:84rem;color:#182838}}
dl {{display:flex;flex-wrap:wrap;gap:.75rem}} dl div {{background:#eef4fb;padding:.5rem .75rem}}
dt {{font-size:.85rem;color:#364d62}} dd {{margin:0;font-weight:600}}
.scroll {{overflow:auto}} table {{width:100%;border-collapse:collapse}}
th,td {{text-align:start;vertical-align:top;padding:.55rem;border:1px solid #d5dfe9;min-width:6rem;
        max-width:26rem;white-space:pre-wrap;overflow-wrap:anywhere}}
th {{background:#eef4fb}} p {{white-space:pre-wrap;overflow-wrap:anywhere}}
</style>
</head>
<body><main>
<h1>{e(label('title'))}</h1>
<p>{e(label('notice'))}</p>
<dl>{metadata}</dl>
<h2>{e(label('context'))}</h2><p>{e(review.context)}</p>
<h2>{e(label('coverage'))}</h2><p>{e(_coverage_notice(review))}</p>
{''.join(sections)}
</main></body>
</html>
"""


def _literal_text(value):
    return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]', '\uFFFD', value)


def _append(sheet, rows):
    for row_number, row in enumerate(rows, 1):
        for column, value in enumerate(row, 1):
            cell = sheet.cell(row_number, column)
            if isinstance(value, str):
                cell.value = _literal_text(value)
                cell.data_type = 's'
            else:
                cell.value = value
            cell.alignment = Alignment(vertical='top', wrap_text=True)


def _excel(review, label, summary, tables):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = label('summary')
    coverage = _coverage_notice(review)
    _append(sheet, [(label('title'), label('notice')), *summary, (label('context'), review.context),
                    (label('coverage'), coverage)])
    sheet.column_dimensions['A'].width = 26
    sheet.column_dimensions['B'].width = 90
    for row_number, paragraph in ((sheet.max_row - 1, review.context), (sheet.max_row, coverage)):
        lines = sum(len(line) // 55 + 1 for line in paragraph.splitlines())
        sheet.row_dimensions[row_number].height = min(409, max(42, lines * 16))
    for title, headers, rows in tables:
        sheet = workbook.create_sheet(title)
        _append(sheet, [headers, *rows])
        sheet.freeze_panes = 'B2'
        sheet.auto_filter.ref = sheet.dimensions
        for index in range(1, len(headers) + 1):
            sheet.column_dimensions[get_column_letter(index)].width = 16 if index == 1 else 30
    for sheet in workbook.worksheets:
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='253D58')
    stream = BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def write_reports(review, directory: Path):
    label, summary, tables = _content(review)
    html = _html(review, label, summary, tables)
    excel = _excel(review, label, summary, tables)
    try:
        directory.mkdir()
    except FileExistsError:
        raise QAError('The output directory already exists. Choose a new directory.') from None
    paths = directory / 'report.html', directory / 'report.xlsx'
    try:
        paths[0].write_text(html, encoding='utf-8')
        paths[1].write_bytes(excel)
    except (OSError, KeyboardInterrupt):
        for path in paths:
            path.unlink(missing_ok=True)
        directory.rmdir()
        raise
    return paths
