from contextlib import contextmanager
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread

from openpyxl import Workbook, load_workbook
import pytest

from babelbee_qa.inputs import QAError, read_document
from babelbee_qa.provider import Provider

X1 = 'urn:oasis:names:tc:xliff:document:1.2'
X2 = 'urn:oasis:names:tc:xliff:document:2.0'


@contextmanager
def endpoint(handler):
    requests = []

    class Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            data = json.loads(body['messages'][1]['content'])
            requests.append((self.path, self.headers.get('Authorization'), body, data))
            reply, status, headers = handler(data)
            if data.get('task') == 'document_overview' and status == 200 and isinstance(reply, dict):
                choice = reply.get('choices', [{}])[0]
                if choice.get('finish_reason') == 'stop':
                    try:
                        decoded = json.loads(choice['message']['content'])
                    except (ValueError, KeyError):
                        pass
                    else:
                        if isinstance(decoded, dict) and not decoded.pop('_omit_receipt', False):
                            by_id = {r['id']: r['source'].strip() for r in data['segments']}
                            decoded.setdefault('receipt', [[sid, by_id[sid][:36]] for sid in data['reading_anchors']])
                        choice['message']['content'] = json.dumps(decoded)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Stub)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def response(data):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(data)}}]}, 200, {}


def finding(segment_id, category, source, target, *, suggestion='', explanation='Needs review.', **extra):
    return dict(segment_id=segment_id, category=category, severity='major', explanation=explanation,
                source_quote=source, target_quote=target, suggestion=suggestion, **extra)


def model(data):
    if 'document_context' not in data:
        terminology = finding('SEG-019', 'terminology', 'Pump', '机组', suggestion='检查泵站',
                              related_id='SEG-001', related_source_quote='Pump', related_target_quote='泵站', error_segment_id='SEG-019')
        return response({'context': 'A maintenance manual for technicians.', 'findings': [terminology]})
    issues = []
    if any(segment['id'] == 'SEG-002' for segment in data['segments']):
        issues.append(finding('SEG-002', 'accuracy', 'at least', '最多',
                              suggestion='=HYPERLINK("https://example.invalid")'))
    return response({'reviewed_ids': [segment['id'] for segment in data['segments']], 'findings': issues})


@pytest.fixture
def pairs():
    return [
        ('Pump station', '泵站'),
        ('Wait at least 15 minutes', '最多等 15 分钟'),
        ('<script>document marker</script>', '文档标记'),
        ('Missing target', ''),
        *((f'Instruction {i}', f'指示 {i}') for i in range(5, 19)),
        ('Pump station inspection', '检查机组'),
    ]


def _write(path, pairs):
    if path.suffix == '.csv':
        import csv

        with path.open('w', encoding='utf-8-sig', newline='') as file:
            csv.writer(file).writerows([('Source', 'Target'), *pairs])
    elif path.suffix == '.xlsx':
        workbook = Workbook()
        workbook.active.append(('Source', 'Target'))
        for row in pairs:
            workbook.active.append(row)
        workbook.save(path)
        workbook.close()
    else:
        from xml.sax.saxutils import escape

        def unit(i, source, target):
            source, target = escape(source), escape(target)
            if path.suffix == '.sdlxliff':
                return (f'<trans-unit id="u{i}"><source>{source}</source>'
                        f'<seg-source><mrk mtype="seg" mid="{i}">{source}</mrk></seg-source>'
                        f'<target><mrk mtype="seg" mid="{i}">{target}</mrk></target></trans-unit>')
            return f'<trans-unit id="u{i}"><source>{source}</source><target>{target}</target></trans-unit>'

        units = ''.join(unit(index, *row) for index, row in enumerate(pairs, 1))
        path.write_text(f'<xliff xmlns="{X1}" version="1.2"><file source-language="en" '
                        f'target-language="zh-CN"><body>{units}</body></file></xliff>', encoding='utf-8')


def run_cli(path, url, *args):
    env = {**os.environ, 'BABELBEE_QA_API_KEY': 'test-secret', 'BABELBEE_QA_BASE_URL': url,
           'BABELBEE_QA_MODEL': 'synthetic-model'}
    return subprocess.run([sys.executable, '-m', 'babelbee_qa.cli', str(path),
                           '--source-language', 'en', '--target-language', 'zh-CN', *args],
                          env=env, capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize('suffix', ['.csv', '.xlsx', '.xliff', '.sdlxliff'])
def test_end_to_end_formats_and_reports(tmp_path, pairs, suffix):
    path = tmp_path / f'project{suffix}'
    _write(path, pairs)
    original = sha256(path.read_bytes()).digest()
    with endpoint(model) as (url, requests):
        result = run_cli(path, url, '--report-language', 'zh')
    assert result.returncode == 0, result.stderr
    assert sha256(path.read_bytes()).digest() == original
    assert all(route == '/v1/chat/completions' and auth == 'Bearer test-secret' for route, auth, _, _ in requests)
    assert len(requests) == 3
    overview = requests[0][3]
    assert [(row['source'], row['target']) for row in overview['segments']] == pairs
    assert [r['id'] for _, _, _, data in requests[1:] for r in data['segments']] == [
        f'SEG-{i:03}' for i in range(1, 20) if i != 4
    ]
    assert all(data['document_context'] == 'A maintenance manual for technicians.' for _, _, _, data in requests[1:])
    html = (tmp_path / 'project-qa/report.html').read_text()
    assert '<html lang="zh-CN">' in html and '对照句段' in html
    assert '&lt;script&gt;document marker&lt;/script&gt;' in html
    assert '<script>' not in html and 'test-secret' not in html
    with BytesIO((tmp_path / 'project-qa/report.xlsx').read_bytes()) as buffer:
        workbook = load_workbook(buffer)
        assert workbook.sheetnames == ['摘要', '问题清单', '句段']
        issues = workbook['问题清单']
        assert [issues.cell(i, 1).value for i in range(2, 5)] == ['SEG-002', 'SEG-004', 'SEG-019']
        assert issues['J2'].data_type == 's'
        assert issues['J2'].value.startswith('=HYPERLINK')
        assert issues['G4'].value == 'SEG-001'
        assert issues['H4'].value == 'Pump'
        assert issues['I4'].value == '泵站'
        assert workbook['句段'].max_row == 20
        assert '评分' not in [cell.value for cell in workbook['摘要']['A']]
        workbook.close()


@pytest.mark.parametrize('suffix', ['csv', 'xliff'])
def test_packaged_examples_export_reports(tmp_path, suffix):
    sample = Path(__file__).resolve().parents[1] / 'examples' / f'sensor-guide.{suffix}'
    before = sample.read_bytes()

    def stub(data):
        if data.get('task') == 'document_overview':
            return response({'context': 'A synthetic sensor guide.', 'findings': [], 'terminology_coverage': [
                {'term': 'pressure sensor', 'segment_ids': ['SEG-002', 'SEG-003'],
                 'target_quotes': ['压力传感器', '压力计'], 'limitation': ''}]})
        return response({'reviewed_ids': data['review_ids'], 'findings': []})

    folder = tmp_path / 'reports'
    with endpoint(stub) as (url, _):
        result = run_cli(sample, url, '--output-dir', str(folder))
    assert result.returncode == 0, result.stderr
    assert sample.read_bytes() == before
    html = (folder / 'report.html').read_text(encoding='utf-8')
    assert '2/2 verified' in html and 'SEG-004' in html and 'printf' in html
    workbook = load_workbook(folder / 'report.xlsx', read_only=True, data_only=True)
    try:
        assert workbook['Segments'].max_row == 5
        assert '2/2 verified' in dict(workbook['Summary'].values)['Terminology check coverage']
        findings = list(workbook['Findings'].values)[1:]
        assert len(findings) == 1 and findings[0][0] == 'SEG-004' and 'printf' in findings[0][3]
    finally:
        workbook.close()


def test_sdl_matches_markers_by_id_and_keeps_blank_target(tmp_path):
    path = tmp_path / 'reordered.sdlxliff'
    path.write_text(f"""<xliff version="1.2" xmlns="{X1}"><file source-language="en"
    target-language="fr"><body><trans-unit id="1"><source>One Two Three</source>
    <seg-source><g id="12"><mrk mtype="seg" mid="1">One</mrk><mrk mtype="seg" mid="2">Two</mrk>
    <mrk mtype="seg" mid="3">Three</mrk></g></seg-source>
    <target><mrk mtype="seg" mid="3">Trois</mrk><mrk mtype="seg" mid="1">Un</mrk></target>
    </trans-unit></body></file></xliff>""")
    parsed = read_document(path)
    assert [(row.source, row.target) for row in parsed.segments] == [('One', 'Un'), ('Two', ''), ('Three', 'Trois')]


@pytest.mark.parametrize('version', ['2.0', '2.1'])
def test_xliff_2_inline_and_multi_segment_unit(tmp_path, version):
    path = tmp_path / 'file.xlf'
    path.write_text(f"""<xliff version="{version}" xmlns="{X2}" srcLang="en" trgLang="fr">
    <file id="f"><unit id="u"><segment id="a"><source>Hi <ph id="1"/></source>
    <target>Salut <ph id="1"/></target></segment><segment id="b"><source>World</source>
    <target>Monde</target></segment></unit></file></xliff>""")
    parsed = read_document(path)
    assert parsed.source_language == 'en'
    assert parsed.segments[0].source_codes == parsed.segments[0].target_codes == ('ph:1',)
    assert parsed.segments[1].location == 'File 1 / unit u / segment b'


def test_xlsx_sheet_column_selection_and_formula_rejection(tmp_path):
    path = tmp_path / 'layout.xlsx'
    workbook = Workbook()
    sheet = workbook.create_sheet('Labels')
    sheet.append(('Name', 'Target', 'Source'))
    sheet.append(('Key', 'Texte', 'Text'))
    sheet.append(('Blank', None, None))
    workbook.save(path)
    with pytest.raises(QAError, match='--sheet'):
        read_document(path)
    result = read_document(path, sheet='Labels', source_column=3, target_column=2)
    assert result.segments[0].location == 'Labels!2'
    assert result.segments[0].source == 'Text' and result.segments[0].target == 'Texte'
    sheet['B2'] = '=1+2'
    workbook.save(path)
    workbook.close()
    with pytest.raises(QAError, match='formulas'):
        read_document(path, sheet='Labels', source_column=3, target_column=2)


def test_invalid_input_and_truncation_do_not_create_report(tmp_path):
    path = tmp_path / 'bad.xlf'
    path.write_text(f'<!DOCTYPE xliff><xliff xmlns="{X1}" version="1.2"/>')
    with pytest.raises(QAError, match='Cannot read'):
        read_document(path)
    path = tmp_path / 'project.csv'
    _write(path, [('Hello', '你好')])

    def incomplete(data):
        if 'document_context' not in data:
            return response({'context': 'A one-line greeting.', 'findings': []})
        return {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]}, 200, {}

    with endpoint(incomplete) as (url, _):
        result = run_cli(path, url)
    assert result.returncode == 1
    assert 'did not finish' in result.stderr
    assert not (tmp_path / 'project-qa').exists()


@pytest.mark.parametrize('kind', ['bad_evidence', 'missing_segment'])
def test_model_cannot_silently_report_invalid_findings(tmp_path, kind):
    path = tmp_path / 'project.csv'
    _write(path, [('Hello', '你好'), ('World', '世界')])

    def bad_response(data):
        if 'document_context' not in data:
            return response({'context': 'A greeting.', 'findings': []})
        return response({'reviewed_ids': ['SEG-001'] if kind == 'missing_segment' else ['SEG-001', 'SEG-002'],
                         'findings': [finding('SEG-001', 'accuracy', 'invented', '你好')]})

    with endpoint(bad_response) as (url, _):
        result = run_cli(path, url)
    assert result.returncode == 1
    assert not (tmp_path / 'project-qa').exists()


@pytest.mark.parametrize('language', ['zh', 'en'])
def test_bad_document_comparisons_are_counted_without_blocking_batch_review(tmp_path, language):
    path = tmp_path / 'comparisons.sdlxliff'
    _write(path, [('Inspect the pressure\u00a0sensor.', '检查压力传感器。'),
                  ('Calibrate the pressure\u00a0sensor.', '校准压力计。'),
                  ('Wait 15 minutes.', '等待 5 分钟。')])

    def stub(data):
        if data.get('task') == 'document_overview':
            valid = finding('SEG-002', 'terminology', 'pressure sensor', '压力计',
                            related_id='SEG-001', related_source_quote='pressure sensor',
                            related_target_quote='压力传感器', error_segment_id='SEG-002')
            invalid = [dict(valid, related_id='SEG-002'), dict(valid, related_target_quote='不存在的引文'),
                       dict(valid, source_quote='invented evidence'), None]
            return response({'context': 'A maintenance guide.', 'findings': [invalid[0], valid, *invalid[1:]],
                             'terminology_coverage': []})
        issue = finding('SEG-003', 'accuracy', '15 minutes', '5 分钟')
        return response({'reviewed_ids': data['review_ids'], 'findings': [issue]})

    with endpoint(stub) as (url, requests):
        completed = run_cli(path, url, '--report-language', language)
    assert completed.returncode == 0, completed.stderr
    assert [data.get('task', 'batch') for *_, data in requests] == ['document_overview', 'batch']
    folder = tmp_path / 'comparisons-qa'
    text = ('全文术语对照中有 4 条未通过校验，已剔除。' if language == 'zh' else
            'Rejected document-wide terminology comparisons: 4 (failed validation).')
    assert text in (folder / 'report.html').read_text(encoding='utf-8')
    workbook = load_workbook(folder / 'report.xlsx', read_only=True, data_only=True)
    try:
        summary, issues, label = (('摘要', '问题清单', '术语检查覆盖提示') if language == 'zh' else
                                  ('Summary', 'Findings', 'Terminology check coverage'))
        assert text in dict(workbook[summary].values)[label]
        rows = list(workbook[issues].values)[1:]
        assert [(r[0], r[4], r[6]) for r in rows] == [('SEG-002', 'pressure\u00a0sensor', 'SEG-001'),
                                                       ('SEG-003', '15 minutes', None)]
    finally:
        workbook.close()


def test_provider_does_not_follow_redirect_or_print_provider_response(tmp_path):
    path = tmp_path / 'project.csv'
    _write(path, [('Hello', '你好')])
    with endpoint(lambda _: (None, 500, {})) as (other, other_requests):
        with endpoint(lambda _: ({'secret': 'test-secret'}, 307, {'Location': other})) as (url, requests):
            result = run_cli(path, url)
    assert len(requests) == 1 and not other_requests
    assert result.returncode == 1
    assert 'HTTP 307' in result.stderr and 'test-secret' not in result.stderr
    with pytest.raises(QAError, match='HTTPS'):
        Provider('http://example.com/v1', 'test', 'secret')


def test_sdl_skips_layout_only_units_but_rejects_unpaired_text(tmp_path):
    path = tmp_path / 'layout.sdlxliff'
    document = f"""<xliff version="1.2" xmlns="{X1}"><file source-language="en" target-language="zh-CN">
    <body><trans-unit id="layout"><source><x id="1"/></source>
    <seg-source><g id="2"><x id="1"/></g><x id="3"/></seg-source>
    <target><g id="2"><x id="1"/></g><x id="3"/></target></trans-unit>
    <trans-unit id="content"><source>Label</source>
    <seg-source><mrk mtype="seg" mid="4">Label</mrk></seg-source>
    <target><mrk mtype="seg" mid="4">标签</mrk></target></trans-unit>
    </body></file></xliff>"""
    path.write_text(document)
    result = read_document(path)
    assert [(row.id, row.source, row.target) for row in result.segments] == [('SEG-001', 'Label', '标签')]

    path.write_text(document.replace('<source><x id="1"/></source>', '<source>Unpaired text<x id="1"/></source>'))
    with pytest.raises(QAError, match='cannot be matched'):
        read_document(path)


def test_source_ending_review_and_terminology_coverage_in_both_reports(tmp_path):
    path = tmp_path / 'clinical-note.csv'
    pairs = [
        ('Care plan: use the pump.', '照护计划：使用泵。'),
        ('The note reads "similar to condition A, but not"', '记录称“类似于病情 A，但并非如此”。'),
        ('Review the care plan and check the pump.', '复核照护计划并检查泵。'),
        ('Wait until the patient can eat or', '等到患者能吃饭或'),
        ('drink independently.', '独立饮水。'),
    ]
    _write(path, pairs)

    def stub(data):
        if data.get('check') == 'terminology_gaps':
            assert all(row['terms'] == ['pump'] for row in data['segments'])
            return response({'observations': []})
        if 'document_context' not in data:
            return response({
                'context': 'A clinical note with instructions.',
                'findings': [],
                'terminology_coverage': [
                    {'term': 'care plan', 'segment_ids': ['SEG-001', 'SEG-003'],
                     'target_quotes': ['照护计划', '照护计划'], 'limitation': ''},
                    {'term': 'pump', 'segment_ids': ['SEG-001', 'SEG-003'],
                     'target_quotes': [], 'limitation': '泵的用途不明确，需要人工对照。'},
                    {'term': 'patient', 'segment_ids': ['SEG-003', 'SEG-004'],
                     'target_quotes': ['患者', '患者'], 'limitation': ''},
                ],
            })
        ids = [segment['id'] for segment in data['segments']]
        if data.get('check') == 'source_ending':
            issue = finding('SEG-002', 'accuracy', 'similar to condition A, but not', '但并非如此',
                            suggestion='An invented ending that must never be shown.')
            return response({'reviewed_ids': ids, 'findings': [issue]})
        return response({'reviewed_ids': ids, 'findings': []})

    with endpoint(stub) as (url, requests):
        result = run_cli(path, url, '--report-language', 'zh')
    assert result.returncode == 0, result.stderr
    assert len(requests) == 4
    focus = next(data for _,_,_,data in requests if data.get('check') == 'source_ending')
    assert focus['check'] == 'source_ending'
    assert [s['id'] for s in focus['segments']] == ['SEG-002', 'SEG-004']
    assert focus['segments'][0]['previous_source'] == pairs[0][0]
    assert focus['segments'][1]['next_source'] == pairs[4][0]
    batch = next(data for _,_,_,data in requests if 'document_context' in data and not data.get('check'))
    assert [s['id'] for s in batch['segments'] if s.get('source_check_confirmed')] == ['SEG-002']

    directory = tmp_path / 'clinical-note-qa'
    html = (directory / 'report.html').read_text()
    expected = '1 项有两处译文证据，1 项无法可靠比对，1 项证据未通过核验'
    assert '术语检查覆盖提示' in html and expected in html and '泵的用途不明确' in html
    assert 'AI 原文截断复核' in html and 'An invented ending' not in html
    assert '待核实术语位置' in html
    workbook = load_workbook(directory / 'report.xlsx', read_only=True)
    summary = dict(workbook['摘要'].values)
    assert expected in summary['术语检查覆盖提示']
    assert '泵的用途不明确' in summary['术语检查覆盖提示']
    row = list(workbook['问题清单'].values)[1]
    assert row[0] == 'SEG-002' and row[9] is None and row[10] == 'AI 原文截断复核'
    assert workbook['问题清单'].max_row == 2
    assert [row[:2] for row in list(workbook['待核实术语位置'].values)[1:]] == [('pump', 'SEG-001'), ('pump', 'SEG-003')]
    workbook.close()


def test_missing_coverage_metadata_is_disclosed_not_reported_as_zero(tmp_path, pairs):
    path = tmp_path / 'project.csv'
    _write(path, pairs)
    with endpoint(model) as (url, _):
        result = run_cli(path, url)
    assert result.returncode == 0, result.stderr
    folder = tmp_path / 'project-qa'
    notice = 'The model did not return a usable coverage record.'
    assert notice in (folder / 'report.html').read_text()
    workbook = load_workbook(folder / 'report.xlsx', read_only=True)
    assert notice in dict(workbook['Summary'].values)['Terminology check coverage']
    workbook.close()


def test_split_date_context_crosses_batches_and_preserves_real_omissions(tmp_path):
    path = tmp_path / 'split-date.csv'
    _write(path, [
        *((f'Instruction {index}', f'指示 {index}') for index in range(1, 16)),
        ('Property owner outreach is planned in \u200b', '計劃聯絡物業業主'),
        ('early 2027', '2027年初'),
        ('Provide written notice before', '發出通知'),
        ('early 2028', '2028年初'),
    ])

    def stub(data):
        if 'document_context' not in data:
            assert len(data['segments']) == 19
            return response({'context': 'Public outreach schedule.', 'findings': [], 'terminology_coverage': []})
        ids = [segment['id'] for segment in data['segments']]
        issues = []
        if 'SEG-016' in ids:
            issues.append(finding('SEG-016', 'omission', 'in', '', suggestion='計劃在'))
        if 'SEG-018' in ids:
            issues.append(finding('SEG-018', 'omission', 'before', '', suggestion='在……之前發出通知'))
        return response({'reviewed_ids': ids, 'findings': issues})

    with endpoint(stub) as (url, requests):
        result = run_cli(path, url)
    assert result.returncode == 0, result.stderr
    assert len(requests) == 3
    first_batch = requests[1][3]
    assert first_batch['segments'][-1]['date_continues_into'] == 'SEG-017'
    assert 'SEG-017' not in first_batch['review_ids']
    assert first_batch['boundary_context'] == {'next': {'source': 'early 2027', 'target': '2027年初'}}
    second_batch = requests[2][3]
    assert second_batch['boundary_context']['previous']['target'] == '計劃聯絡物業業主'
    assert 'date_continues_into' not in second_batch['segments'][1]
    workbook = load_workbook(tmp_path / 'split-date-qa' / 'report.xlsx', read_only=True)
    assert [row[0] for row in list(workbook['Findings'].values)[1:]] == ['SEG-018']
    workbook.close()


def test_invalid_batch_receipt_is_retried_once_with_exact_review_ids(tmp_path):
    path = tmp_path / 'review.csv'
    _write(path, [('One', '一'), ('Two', '二')])

    def stub(data):
        if 'document_context' not in data:
            return response({'context': 'A numbered list.', 'findings': [], 'terminology_coverage': []})
        if not data.get('validation_retry'):
            return response({'reviewed_ids': ['SEG-001'], 'findings': []})
        assert data['review_ids'] == ['SEG-001', 'SEG-002']
        assert 'did not acknowledge every segment' in data['validation_retry']
        return response({'reviewed_ids': data['review_ids'], 'findings': []})

    with endpoint(stub) as (url, requests):
        result = run_cli(path, url)
    assert result.returncode == 0, result.stderr
    assert len(requests) == 3
    assert (tmp_path / 'review-qa' / 'report.html').exists()


def test_retries_when_all_terminology_evidence_is_invalid(tmp_path):
    path = tmp_path / 'terms.csv'
    _write(path, [('Check the pump', '检查泵'), ('Replace the pump', '更换泵')])

    def stub(data):
        if 'document_context' not in data:
            term = 'pump' if data.get('validation_retry') else '泵'
            return response({
                'context': 'A maintenance manual.',
                'findings': [],
                'terminology_coverage': [
                    {'term': term, 'segment_ids': ['SEG-001', 'SEG-002'],
                     'target_quotes': ['泵', '泵'], 'limitation': ''},
                ],
            })
        return response({'reviewed_ids': data['review_ids'], 'findings': []})

    with endpoint(stub) as (url, requests):
        result = run_cli(path, url, '--report-language', 'zh')
    assert result.returncode == 0, result.stderr
    assert len(requests) == 3
    assert requests[1][3]['validation_retry'].startswith('Terminology samples')
    assert '1 项有两处译文证据' in (tmp_path / 'terms-qa' / 'report.html').read_text()


def test_focused_terminology_compares_repeat_source_anchors(tmp_path):
    path = tmp_path / 'repeated-terms.csv'
    _write(path, [
        ('The patient had a collapse in the waiting room.', '患者在候诊室倒下。'),
        ('An examination investigated the collapse.', '进行了检查，以排查晕厥。'),
        ('The pressure gauge was installed.', '压力表已安装。'),
        ('Inspect the pressure gauge weekly.', '每周检查压力表。'),
        ('Use the rear entrance.', '使用后入口。'),
        ('The rear entrance closes at six.', '后入口六点关闭。'),
        ('Leave the ventilation system running.', '保持通风系统运行。'),
        ('Examine the ventilation system.', '检查通风系统。'),
    ])

    def stub(data):
        if data.get('check') == 'terminology_candidates':
            checks = []
            quotes = {'pressure': '压力表', 'gauge': '压力表', 'entrance': '后入口', 'ventilation': '通风系统', 'system': '通风系统'}
            for candidate in data['candidates']:
                term = candidate['term']
                if term == 'collapse':
                    assert candidate['left']['id'] == 'SEG-001' and candidate['right']['id'] == 'SEG-002'
                    checks.append({'id': candidate['id'], 'status': 'variant', 'quotes': ['倒下', '晕厥'],
                                   'note': '同一次事件的描述具体程度不同，请核实。'})
                else:
                    equivalent = next(value for key, value in quotes.items() if key in term)
                    checks.append({'id': candidate['id'], 'status': 'same',
                                   'quotes': [equivalent, equivalent], 'note': ''})
            return response({'checks': checks})
        if 'document_context' not in data:
            return response({'context': 'A clinical and maintenance log.', 'findings': [],
                             'terminology_coverage': []})
        return response({'reviewed_ids': data['review_ids'], 'findings': []})

    with endpoint(stub) as (url, requests):
        result = run_cli(path, url, '--report-language', 'zh')
    assert result.returncode == 0, result.stderr
    assert sum(data.get('check') == 'terminology_candidates' for _, _, _, data in requests) == 1
    folder = tmp_path / 'repeated-terms-qa'
    html = (folder / 'report.html').read_text()
    assert '定向复核提交了 4 个重复术语候选' in html
    assert '4 项有两处译文证据' in html
    workbook = load_workbook(folder / 'report.xlsx', read_only=True)
    issue = list(workbook['问题清单'].values)[1]
    assert issue[0] == 'SEG-002' and issue[5:9] == ('晕厥', 'SEG-001', 'collapse', '倒下')
    assert issue[9] is None
    workbook.close()


@pytest.mark.parametrize(('source', 'target', 'direction'), [
    ('Hospital Transferred in From:', '转出医院：', 'incoming'),
    ('Ward transferred out to:', '轉入病房：', 'outgoing'),
    ('Transferred in from:', '轉入來源醫院：', None),
    ('Hospital transferred from:', '来源医院：', None),
    ('Hospital transferred out to:', '轉出目的醫院：', None),
    ('Transfer into hospital:', '未转出医院：', None),
    ('Hospital transferred in from:', '由转出医院转入：', None),
    ('Hospital transferred in from:', '转入或转出医院：', None),
    ('The patient was transferred in from a hospital.', '转出医院：', None),
    ('Hospital transferred in from:', '转出医院：转入病房', None),
])
def test_transfer_direction_rule_requires_opposite_unambiguous_labels(source, target, direction):
    from babelbee_qa.direction import transfer_direction_mismatch

    match = transfer_direction_mismatch(source, target)
    assert (match[0] if match else None) == direction
    if match:
        assert match[1] in source and match[2] in target


def test_report_keeps_direction_finding_when_ai_is_silent_and_deduplicates(tmp_path):
    source = 'Hospital Transferred in From:'
    path = tmp_path / 'transfers.csv'
    _write(path, [(source, '转出医院：'), (source, '轉出醫院：'), (source, '转入来源医院：')])

    def stub(data):
        if 'document_context' not in data:
            return response({'context': 'Medical transfer form.', 'findings': [], 'terminology_coverage': []})
        issue = finding('SEG-001', 'accuracy', 'Transferred in From', '转出医院',
                        suggestion='转入来源医院')
        return response({'reviewed_ids': data['review_ids'], 'findings': [issue]})

    with endpoint(stub) as (url, requests):
        completed = run_cli(path, url, '--report-language', 'zh')
    assert completed.returncode == 0, completed.stderr
    assert len(requests) == 2
    folder = tmp_path / 'transfers-qa'
    html = (folder / 'report.html').read_text()
    assert html.count('方向相反') == 2
    workbook = load_workbook(folder / 'report.xlsx', read_only=True)
    rows = list(workbook['问题清单'].values)[1:]
    assert [row[0] for row in rows] == ['SEG-001', 'SEG-002']
    assert all(row[1:3] == ('准确性', '主要错误') and row[10] == '规则检查' for row in rows)
    assert rows[0][4:6] == ('Hospital Transferred in From', '转出')
    workbook.close()


def test_source_completion_is_counted_once_and_distinct_same_row_issue_is_kept(tmp_path):
    path = tmp_path / 'truncated-note.csv'
    _write(path, [
        ('The note is dated May. It says "similar to condition A, but not"',
         '记录日期为六月，称“类似于病情 A，但并非如此”。'),
        ('No further details were provided.', '没有提供进一步的细节。'),
    ])

    def stub(data):
        if 'document_context' not in data:
            return response({'context': 'An incomplete clinical note.',
                             'findings': [], 'terminology_coverage': []})
        if data.get('check') == 'source_ending':
            issue = finding('SEG-001', 'accuracy', '"similar to condition A, but not"',
                            '“类似于病情 A，但并非如此”')
            issue['explanation'] = '相邻句段未补全原文，译文擅自增加了否定结论。'
            return response({'reviewed_ids': ['SEG-001'], 'findings': [issue]})
        duplicate = finding('SEG-001', 'addition', 'similar to condition A, but not',
                            '但并非如此', suggestion='这种情况不是病情 A。')
        independent = finding('SEG-001', 'accuracy', 'dated May', '日期为六月')
        return response({'reviewed_ids': data['review_ids'], 'findings': [duplicate, independent]})

    with endpoint(stub) as (url, _):
        completed = run_cli(path, url, '--report-language', 'zh')
    assert completed.returncode == 0, completed.stderr
    folder = tmp_path / 'truncated-note-qa'
    html = (folder / 'report.html').read_text()
    assert '相邻句段未补全原文' in html
    assert '这种情况不是病情 A。' not in html
    workbook = load_workbook(folder / 'report.xlsx', read_only=True)
    rows = list(workbook['问题清单'].values)[1:]
    assert len(rows) == 2
    assert rows[0][4:6] == ('"similar to condition A, but not"', '“类似于病情 A，但并非如此”')
    assert rows[0][9] is None and rows[0][10] == 'AI 原文截断复核'
    assert rows[1][4:6] == ('dated May', '日期为六月')
    assert dict(workbook['摘要'].values)['问题清单'] == 2
    workbook.close()


@pytest.mark.parametrize(('status', 'source_clue', 'has_advisory'), [
    ('verify', '', True),
    ('explained', 'Address me formally before the council.', False),
    ('explained', 'An invented reason for the switch.', True),
])
def test_dialogue_register_checks_context_and_counts_advisories_separately(
        tmp_path, status, source_clue, has_advisory):
    path = tmp_path / 'dialogue.csv'
    _write(path, [
        ('Address me formally before the council.', 'Vouvoyez-moi devant le conseil.'),
        ('Mara: Are you coming?', 'Mara : Tu viens ?'),
        ('Elias: If you are ready, Mara, we should go.', 'Elias : Si vous êtes prête, Mara, partons.'),
        ('Pip: Are you both ready?', 'Pip : Vous êtes tous les deux prêts ?'),
    ])

    def stub(data):
        if data.get('check') == 'dialogue_register':
            assert len(data['candidates']) == 1
            candidate = data['candidates'][0]
            assert (candidate['earlier'], candidate['later'], candidate['relation']) == (
                'SEG-002', 'SEG-003', 'reciprocal')
            assert candidate['context'][0]['id'] == 'SEG-001'
            return response({'checks': [{'id': 1, 'status': status, 'source_clue': source_clue}]})
        if 'document_context' not in data:
            return response({'context': 'A dialogue before a meeting.', 'findings': [], 'terminology_coverage': []})
        generic = finding('SEG-003', 'fluency', 'you are ready', 'vous êtes prête')
        generic['explanation'] = 'tu 和 vous 的语域不一致。'
        return response({'reviewed_ids': data['review_ids'], 'findings': [generic]})

    with endpoint(stub) as (url, _):
        env = {**os.environ, 'BABELBEE_QA_API_KEY': 'test-secret', 'BABELBEE_QA_BASE_URL': url,
               'BABELBEE_QA_MODEL': 'synthetic-model'}
        completed = subprocess.run([sys.executable, '-m', 'babelbee_qa.cli', str(path),
                                    '--source-language', 'en', '--target-language', 'fr', '--report-language', 'zh'],
                                   env=env, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    directory = tmp_path / 'dialogue-qa'
    html = (directory / 'report.html').read_text()
    workbook = load_workbook(directory / 'report.xlsx', read_only=True)
    summary = dict(workbook['摘要'].values)
    assert summary['问题清单'] == 0
    rows = list(workbook['问题清单'].values)[1:]
    assert len(rows) == int(has_advisory)
    if has_advisory:
        assert summary['待核实提示'] == 1
        assert rows[0][0] == 'SEG-003' and rows[0][2] == '待核实提示'
        assert rows[0][5:7] == ('vous', 'SEG-002')
        assert rows[0][8:10] == ('Tu', None)
        assert '角色关系' in html
    workbook.close()


@pytest.mark.parametrize('first_reply', ['missing_receipt', 'bad_quote', 'malformed_json'])
def test_document_receipt_or_malformed_provider_response_retries_only_overview(tmp_path, first_reply):
    path = tmp_path / 'full-reading.csv'
    _write(path, [('The first source line is here.', '这是首行。'), ('The last source line is here.', '这是末行。')])

    def stub(data):
        if data.get('task') == 'document_overview':
            if not data.get('validation_retry'):
                if first_reply == 'missing_receipt':
                    return response({'context': 'Generic summary.', 'findings': [], '_omit_receipt': True})
                if first_reply == 'bad_quote':
                    return response({'context': 'Generic summary.', 'findings': [],
                                     'receipt': [['SEG-001', 'made up source'], ['SEG-002', 'The last source']]})
                return {'choices': [{'finish_reason': 'stop', 'message': {'content': '{invalid JSON'}}]}, 200, {}
            return response({'context': 'Verified context.', 'findings': [], 'terminology_coverage': []})
        return response({'reviewed_ids': data['review_ids'], 'findings': []})

    with endpoint(stub) as (url, requests):
        result = run_cli(path, url)
    assert result.returncode == 0, result.stderr
    assert [data.get('task', 'batch') for *_, data in requests] == ['document_overview', 'document_overview', 'batch']
    assert requests[0][3]['reading_anchors'] == ['SEG-001', 'SEG-002']
    assert 'Verified context.' in (tmp_path / 'full-reading-qa' / 'report.html').read_text()


def test_incomplete_full_document_receipt_never_creates_reports(tmp_path):
    path = tmp_path / 'no-full-read.csv'
    _write(path, [('A short greeting.', '简短问候。')])
    with endpoint(lambda _: response({'context': 'Generic context', 'findings': [], '_omit_receipt': True})) as (url, requests):
        result = run_cli(path, url)
    assert result.returncode == 1 and 'receipt' in result.stderr
    assert len(requests) == 2
    assert not (tmp_path / 'no-full-read-qa').exists()


@pytest.mark.parametrize('invalid', [None, 'receipt', 'source_quote', 'target_quote'])
def test_sdl_no_break_spaces_in_receipts_and_evidence(tmp_path, invalid):
    path = tmp_path / 'bus-route.sdlxliff'
    source = 'Transfer to\u00a0Line 52.'
    target = '换乘\u00a051 路。'
    _write(path, [(source, target)])

    def stub(data):
        if data.get('task') == 'document_overview':
            excerpt = 'Transfer to Line 51.' if invalid == 'receipt' else 'Transfer to Line 52.'
            return response({'context': 'A bus route guide.', 'receipt': [['SEG-001', excerpt]],
                             'findings': [], 'terminology_coverage': []})
        source_quote = 'to Line 52!' if invalid == 'source_quote' else 'to Line 52.'
        target_quote = '换乘 59 路' if invalid == 'target_quote' else '换乘 51 路'
        return response({'reviewed_ids': data['review_ids'], 'findings': [
            finding('SEG-001', 'accuracy', source_quote, target_quote, explanation='The bus number changed.')]})

    with endpoint(stub) as (url, requests):
        result = run_cli(path, url, '--report-language', 'zh')
    folder = tmp_path / 'bus-route-qa'
    if invalid:
        assert result.returncode == 1
        assert ('exact source excerpts' if invalid == 'receipt' else 'does not match') in result.stderr
        assert sum(bool(data.get('validation_retry')) for *_, data in requests) == 1
        assert not folder.exists()
        return

    assert result.returncode == 0, result.stderr
    assert len(requests) == 2 and all(not data.get('validation_retry') for *_, data in requests)
    html = (folder / 'report.html').read_text(encoding='utf-8')
    assert '<td>to\u00a0Line 52.</td>' in html and '<td>换乘\u00a051 路</td>' in html
    workbook = load_workbook(folder / 'report.xlsx', read_only=True, data_only=True)
    try:
        assert list(workbook['问题清单'].values)[1][4:6] == ('to\u00a0Line 52.', '换乘\u00a051 路')
        assert list(workbook['句段'].values)[1][2:4] == (source, target)
    finally:
        workbook.close()


def test_transient_failure_retries_only_the_affected_segment_batch(tmp_path):
    path = tmp_path / 'one-bad-batch.csv'
    _write(path, [(f'Hello {index}.', f'你好 {index}。') for index in range(17)])

    def stub(data):
        if data.get('task') == 'document_overview':
            return response({'context': 'A series of greetings.', 'findings': [], 'terminology_coverage': []})
        if data['review_ids'] == ['SEG-017'] and not data.get('validation_retry'):
            return {'error': 'temporary'}, 503, {}
        return response({'reviewed_ids': data['review_ids'], 'findings': []})

    with endpoint(stub) as (url, requests):
        result = run_cli(path, url)
    assert result.returncode == 0, result.stderr
    batches = [data for *_, data in requests if 'review_ids' in data]
    assert [len(data['review_ids']) for data in batches] == [16, 1, 1]
    assert batches[1]['review_ids'] == batches[2]['review_ids'] == ['SEG-017']


def test_term_gaps_request_only_unmapped_rows_and_record_remaining_gap_in_both_reports(tmp_path):
    path = tmp_path / 'terms-with-gaps.csv'
    _write(path, [('Inspect the valve.', '检查阀门。'), ('Replace the valve.', '更换阀门。'),
                  ('Clean the valve.', '清洁阀门。'), ('Photograph the valve.', '拍照。')])

    def stub(data):
        if data.get('check') == 'terminology_gaps':
            assert [(s['id'], s['terms']) for s in data['segments']] == [
                ('SEG-003', ['valve']), ('SEG-004', ['valve'])]
            return response({'observations': [
                {'segment_id': 'SEG-003', 'term': 'valve', 'quote': '阀门'},
                {'segment_id': 'SEG-004', 'term': 'valve', 'quote': '虚构证据'}]})
        if data.get('task') == 'document_overview':
            return response({'context': 'Maintenance of a valve.', 'findings': [], 'terminology_coverage': [
                {'term': 'valve', 'segment_ids': ['SEG-001', 'SEG-002'],
                 'target_quotes': ['阀门', '阀门'], 'limitation': ''}]})
        return response({'reviewed_ids': data['review_ids'], 'findings': []})

    with endpoint(stub) as (url, requests):
        result = run_cli(path, url, '--report-language', 'zh')
    assert result.returncode == 0, result.stderr
    assert sum(data.get('check') == 'terminology_gaps' for *_, data in requests) == 1
    folder = tmp_path / 'terms-with-gaps-qa'
    html = (folder / 'report.html').read_text()
    assert '已核验 3/4 处' in html and '待核实术语位置' in html
    with (folder / 'report.xlsx').open('rb') as stream:
        workbook = load_workbook(stream, read_only=True)
        assert list(workbook['待核实术语位置'].values)[1][:2] == ('valve', 'SEG-004')
        workbook.close()


def test_term_gap_uses_verbatim_names_and_sends_remaining_occurrences_to_model(tmp_path):
    path = tmp_path / 'names.csv'
    _write(path, [('Arrive at North Station.', '抵達北站。'),
                  ('Transfer at North Station.', '在北站轉車。'),
                  ('North Station opens at noon.', 'North Station中午開放。'),
                  ('Go to North Station.', '前往 North Stationary。'),
                  ('Look for North Station.', '請找北車站。'),
                  ('Check the North Station timetable.', '⟦North Station⟧查看時刻表。')])

    def stub(data):
        if data.get('check') == 'terminology_gaps':
            assert [(row['id'], row['terms']) for row in data['segments']] == [
                ('SEG-004', ['north station']), ('SEG-005', ['north station']), ('SEG-006', ['north station'])]
            return response({'observations': [{'segment_id': 'SEG-005', 'term': 'north station', 'quote': '北車站'}]})
        if data.get('task') == 'document_overview':
            return response({'context': 'Transit directions.', 'findings': [], 'terminology_coverage': [
                {'term': 'North Station', 'segment_ids': ['SEG-001', 'SEG-002'],
                 'target_quotes': ['北站', '北站'], 'limitation': ''}]})
        return response({'reviewed_ids': data['review_ids'], 'findings': []})

    with endpoint(stub) as (url, _):
        result = run_cli(path, url, '--report-language', 'zh')
    assert result.returncode == 0, result.stderr
    folder = tmp_path / 'names-qa'
    assert '已核验 4/6 处' in (folder / 'report.html').read_text()
    workbook = load_workbook(folder / 'report.xlsx', read_only=True)
    try:
        assert [row[:2] for row in list(workbook['待核实术语位置'].values)[1:]] == [
            ('north station', 'SEG-004'), ('north station', 'SEG-006')]
        assert dict(workbook['摘要'].values)['问题清单'] == 0
    finally:
        workbook.close()


def test_number_word_and_literal_percentage_are_not_reported_but_real_printf_changes_are(tmp_path):
    path = tmp_path / 'numeric.csv'
    _write(path, [('The count is 七.', 'Seven buildings.'),
                  ('Gain 20% damage.', 'Gagne 20 % de dégâts.'),
                  ('Player %1$s gained %d points.', 'Le joueur %1$d gagne des points.'),
                  ('The limit is 7.', 'La limite est 8.'),
                  ('This is 20% lower and 10% higher.', 'C’est 20 % de moins et 10 % de plus.')])

    def stub(data):
        if data.get('task') == 'document_overview':
            return response({'context': 'Game statistics and counts.', 'findings': [], 'terminology_coverage': []})
        return response({'reviewed_ids': data['review_ids'], 'findings': [
            finding('SEG-001', 'accuracy', '七', 'Seven', explanation='数字不一致。'),
            finding('SEG-002', 'formatting', '20%', '% de', explanation='printf 占位符变化。'),
            finding('SEG-003', 'formatting', '%1$s', '%1$d', explanation='printf 占位符变化。'),
            finding('SEG-004', 'accuracy', '7', '8', explanation='数字不一致。'),
            finding('SEG-005', 'formatting', '% lower', '% de', explanation='printf 占位符变化。'),
        ]})

    with endpoint(stub) as (url, _):
        result = run_cli(path, url, '--report-language', 'zh')
    assert result.returncode == 0, result.stderr
    folder = tmp_path / 'numeric-qa'
    workbook = load_workbook(folder / 'report.xlsx', read_only=True)
    rows = list(workbook['问题清单'].values)[1:]
    assert [row[0] for row in rows] == ['SEG-003', 'SEG-004']
    assert all(text in rows[0][3] for text in ('%1$s', '%1$d', '%d'))
    assert rows[0][10] == '规则检查'
    workbook.close()


def test_invisible_layout_rows_are_skipped_without_losing_blank_targets_or_codes(tmp_path):
    path = tmp_path / 'blank-layout.xlf'
    path.write_text(f'''<xliff xmlns="{X1}" version="1.2"><file source-language="en" target-language="zh"><body>
      <trans-unit id="1"><source>\u200b</source><target>\u200b</target></trans-unit>
      <trans-unit id="2"><source><g id="1">\u2060</g></source><target><g id="2"> </g></target></trans-unit>
      <trans-unit id="3"><source>Real source</source><target>\u200b</target></trans-unit>
      <trans-unit id="4"><source><x id="3"/></source><target><x id="3"/></target></trans-unit>
      </body></file></xliff>''')
    rows = read_document(path).segments
    assert len(rows) == 2 and rows[0].source == 'Real source' and rows[0].target == ''
    assert rows[1].source_codes and rows[1].target_codes


def test_split_date_requires_matching_year_qualifier():
    from babelbee_qa.inputs import Segment
    from babelbee_qa.review import _preserved_date_joins

    def find(target):
        return _preserved_date_joins([Segment('SEG-001', 'line 1', 'Work starts in', '工作开始'),
                                      Segment('SEG-002', 'line 2', 'early 2027', target)], 'en', 'zh-CN')
    assert find('2027 年初。') == {'SEG-001': 'in'}
    assert find('2027年末') == {}
    assert find('2028年初') == {}


def test_ambiguous_term_is_advisory_and_right_hand_error_keeps_left_as_comparison():
    from types import SimpleNamespace
    from babelbee_qa.inputs import Segment
    from babelbee_qa.review import _focused_terminology

    rows = {row.id: row for row in (Segment('SEG-001', 'line 1', 'The client signs.', '客户签名。'),
                                   Segment('SEG-002', 'line 2', 'The client pays.', '雇员付款。'))}
    candidate = SimpleNamespace(term='client', left='SEG-001', right='SEG-002', source_quotes=('client', 'client'))
    def classify(fault):
        stub = SimpleNamespace(evaluate=lambda *_: {'checks': [
            dict(id=1, status='variant', quotes=['客户', '雇员'], note='Different referents.', error_segment_id=fault)]})
        _, findings = _focused_terminology([candidate], rows, stub, {})
        return findings[0]
    assert classify('').severity == 'advisory'
    issue = classify('SEG-002')
    assert (issue.segment_id, issue.related_id, issue.severity) == ('SEG-002', 'SEG-001', 'minor')


def test_same_exact_ai_defect_reported_in_several_categories_is_counted_once():
    from babelbee_qa.review import Finding, _deduplicate_findings
    rows = [Finding('SEG-007', 'omission', 'minor', 'The term is mistranslated.', 'motto', 'slogan'),
            Finding('SEG-007', 'accuracy', 'major', 'The term is mistranslated.', 'motto', 'slogan'),
            Finding('SEG-007', 'formatting', 'minor', 'The colon is misplaced.', 'title', 'Title:')]
    findings = _deduplicate_findings(rows)
    assert [(r.category,r.severity) for r in findings] == [('accuracy','major'),('formatting','minor')]


def test_printf_with_real_space_flag_and_length_modifier_is_preserved():
    from babelbee_qa.evidence_rules import printf_tokens
    assert printf_tokens('At 20% lower, values are % ld and %1$lld.') == {'% ld': 1, '%1$lld': 1}


def test_collapse_uses_correct_reference_and_keeps_only_one_grounded_finding(tmp_path):
    path = tmp_path / 'same-event.csv'
    _write(path, [('The patient had a collapse in the hall.', '患者在走廊倒下。'),
                  ('The nurse investigated the collapse.', '护士调查了这次晕厥！！')])

    def stub(data):
        if data.get('task') == 'document_overview':
            return response({'context': 'A short medical note.', 'terminology_coverage': [],
                             'findings': [finding('SEG-001', 'terminology', 'collapse', '倒下',
                                 related_id='SEG-002', related_source_quote='collapse', related_target_quote='晕厥',
                                 explanation='The same collapse was rendered as 倒下 and 晕厥.',
                                 error_segment_id='SEG-001')]})
        return response({'reviewed_ids': data['review_ids'], 'findings': [
            finding('SEG-002', 'addition', 'collapse', '晕厥', explanation='The target adds syncope to a collapse.'),
            finding('SEG-002', 'formatting', '.', '！！', explanation='Repeated punctuation.')]})

    with endpoint(stub) as (url, _):
        result = run_cli(path, url, '--report-language', 'zh')
    assert result.returncode == 0, result.stderr
    folder = tmp_path / 'same-event-qa'
    assert '同一次倒下事件' in (folder / 'report.html').read_text()
    workbook = load_workbook(folder / 'report.xlsx', read_only=True)
    rows = list(workbook['问题清单'].values)[1:]
    assert [(row[0], row[1]) for row in rows] == [('SEG-002','格式'), ('SEG-002','术语')]
    assert rows[1][5:9] == ('晕厥','SEG-001','collapse','倒下')
    workbook.close()


def test_named_term_and_related_participle_do_not_establish_an_error(tmp_path):
    path = tmp_path / 'game-terms.csv'
    _write(path, [('Marked enemies receive more damage.', 'Les ennemis marqués subissent plus de dégâts.'),
                  ('A Mark lasts a turn.', 'Un Sceau dure un tour.'),
                  ('Marks expire at the end of the turn.', 'Les Repères expirent en fin de tour.')])

    def stub(data):
        if data.get('task') == 'document_overview':
            grammar = finding('SEG-002', 'terminology', 'Mark', 'Sceau', related_id='SEG-001',
                              related_source_quote='Marked', related_target_quote='marqués', error_segment_id='SEG-002')
            defect = finding('SEG-003', 'terminology', 'Marks', 'Repères', related_id='SEG-002',
                             related_source_quote='Mark', related_target_quote='Sceau', error_segment_id='SEG-003')
            return response({'context': 'Game mechanic.', 'findings': [grammar, defect]})
        return response({'reviewed_ids': data['review_ids'], 'findings': []})

    with endpoint(stub) as (url, _):
        result = run_cli(path, url, '--target-language', 'fr')
    assert result.returncode == 0, result.stderr
    workbook = load_workbook(tmp_path / 'game-terms-qa' / 'report.xlsx', read_only=True)
    try:
        rows = list(workbook['Findings'].values)[1:]
        assert [(r[0], r[2]) for r in rows] == [('SEG-002', 'Review advisory'), ('SEG-003', 'Major')]
        assert dict(workbook['Summary'].values)['Findings'] == 1
    finally:
        workbook.close()


def test_document_term_is_not_an_error_if_the_proposed_expression_is_already_used(tmp_path):
    path = tmp_path / 'mechanic.csv'
    _write(path, [('Breaking an enemy leaves them vulnerable.', 'Mettre un ennemi en Rupture le rend vulnérable.'),
                  ('Breaking an enemy cancels attacks.', 'La Rupture d’un ennemi annule les attaques.'),
                  ('Breaking an enemy limits healing.', 'Affaiblir un ennemi limite les soins.')])

    def stub(data):
        if data.get('task') == 'document_overview':
            common = dict(related_id='SEG-002', related_source_quote='Breaking', related_target_quote='Rupture',
                          suggestion='Mettre un ennemi en Rupture')
            already_used = finding('SEG-001', 'terminology', 'Breaking', 'Mettre', error_segment_id='SEG-001', **common)
            missing = finding('SEG-003', 'terminology', 'Breaking', 'Affaiblir', error_segment_id='SEG-003', **common)
            return response({'context': 'Game mechanic.', 'findings': [already_used, missing]})
        return response({'reviewed_ids': data['review_ids'], 'findings': []})

    with endpoint(stub) as (url, _):
        result = run_cli(path, url, '--target-language', 'fr')
    assert result.returncode == 0, result.stderr
    with (tmp_path / 'mechanic-qa' / 'report.html').open() as report:
        assert 'Affaiblir' in report.read()
    workbook = load_workbook(tmp_path / 'mechanic-qa' / 'report.xlsx', read_only=True)
    try:
        assert [row[0] for row in list(workbook['Findings'].values)[1:]] == ['SEG-003']
    finally:
        workbook.close()
