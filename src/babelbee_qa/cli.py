"""Run BabelBee QA against one aligned bilingual file."""
import argparse
import os
from pathlib import Path
import sys

from .inputs import QAError, read_document
from .provider import Provider
from .review import review_document
from .reports import write_reports


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='One bilingual XLSX, CSV, XLF, XLIFF or SDLXLIFF file')
    parser.add_argument('--base-url', default=os.environ.get('BABELBEE_QA_BASE_URL'),
                        help='Provider API base URL, e.g. https://api.example.com/v1')
    parser.add_argument('--model', default=os.environ.get('BABELBEE_QA_MODEL'), help='Provider model name')
    parser.add_argument('--timeout', type=int, default=180, help='Seconds per provider request (default: 180)')
    parser.add_argument('--source-language', help='Source language code, e.g. en')
    parser.add_argument('--target-language', help='Target language code, e.g. zh-CN')
    parser.add_argument('--report-language', choices=('en', 'zh'), default='en')
    parser.add_argument('--output-dir', type=Path, help='New output directory (default: <input stem>-qa)')
    table = parser.add_argument_group('CSV and XLSX options')
    table.add_argument('--source-column', type=int, default=1, help='One-based source column (default: 1)')
    table.add_argument('--target-column', type=int, default=2, help='One-based translation column (default: 2)')
    table.add_argument('--no-header', action='store_true', help='Treat the first populated row as content')
    table.add_argument('--sheet', help='Worksheet name for XLSX input')
    return parser


def _language(override, declared, role):
    if override and declared and override.replace('_', '-').casefold() != declared.replace('_', '-').casefold():
        raise QAError(f'The selected {role} language does not match the XLIFF declaration.')
    language = (override or declared or '').strip()
    if not language or len(language) > 80:
        raise QAError(f'Specify a valid --{role}-language for this file.')
    return language


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.base_url or not args.model:
        parser.error('configure --base-url and --model (or BABELBEE_QA_BASE_URL and BABELBEE_QA_MODEL)')
    destination = args.output_dir or args.input.with_name(args.input.stem + '-qa')
    try:
        if destination.exists() or not destination.parent.is_dir():
            raise QAError('Choose a new output directory inside an existing parent directory.')
        provider = Provider(args.base_url, args.model, os.environ.get('BABELBEE_QA_API_KEY', '').strip(), args.timeout)
        document = read_document(args.input, sheet=args.sheet, source_column=args.source_column,
                                 target_column=args.target_column, header=not args.no_header)
        source = _language(args.source_language, document.source_language, 'source')
        target = _language(args.target_language, document.target_language, 'target')
        result = review_document(document, provider, source_language=source, target_language=target,
                                 report_language=args.report_language, model=args.model,
                                 progress=lambda message: print(message, file=sys.stderr))
        for path in write_reports(result, destination):
            print(path)
    except QAError as exc:
        print(f'BabelBee QA: {exc}', file=sys.stderr)
        return 1
    except OSError:
        print('BabelBee QA: Cannot write reports to the chosen output directory.', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('BabelBee QA: Cancelled.', file=sys.stderr)
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
