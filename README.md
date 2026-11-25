# BabelBee QA CLI

A standalone command-line tool for checking an existing translation using your own model account. It generates HTML and Excel reports with findings, evidence, and terminology coverage.

## Quick start

[Download BabelBee QA CLI v0.1.1](https://github.com/frankwzh4633/babelbee-qa-cli/releases/tag/v0.1.1) for Python 3.10–3.14. On macOS or Linux, install the wheel in a virtual environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install https://github.com/frankwzh4633/babelbee-qa-cli/releases/download/v0.1.1/babelbee_qa-0.1.1-py3-none-any.whl
babelbee-qa --help
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. Then run the same install and help commands.

For a source install, extract the source archive from the release page or clone this repository, then run `python -m pip install .` from its root. The [examples](examples) are included in the source archive and checkout.

## Run a review

Use a model that offers an OpenAI-compatible `/chat/completions` endpoint, returns JSON reliably, and can read your entire bilingual file in one request. Set the base URL up to and including `/v1` where applicable. Keep the key in your shell environment. In Bash you can enter it without putting it into command history. Run the examples below from the extracted source archive or checkout.

```bash
read -r -s -p 'Model API key: ' BABELBEE_QA_API_KEY
echo
export BABELBEE_QA_API_KEY
export BABELBEE_QA_BASE_URL='https://model-provider.example/v1'
export BABELBEE_QA_MODEL='your-model-name'

babelbee-qa examples/sensor-guide.csv --source-language en --target-language zh-CN --report-language zh --output-dir demo-csv-qa
babelbee-qa examples/sensor-guide.xliff --report-language zh --output-dir demo-xliff-qa
```

In PowerShell, set the base URL and model with `$env:BABELBEE_QA_BASE_URL` and `$env:BABELBEE_QA_MODEL`. To prompt for the key, use `$env:BABELBEE_QA_API_KEY = (Get-Credential -UserName 'api-key' -Message 'Model API key').GetNetworkCredential().Password`.

Both files use fully synthetic sensor-maintenance text. The translation of “pressure sensor” varies across rows and one printf placeholder changes type. Open `report.html` in a browser and `report.xlsx` in Excel or a compatible editor.

The tool sends the entire bilingual text and a review prompt to the configured endpoint. Your provider's costs and data terms apply. A compatible endpoint on loopback (for example, `http://127.0.0.1:11434/v1`) can be used with a locally hosted model, without an API key if the endpoint permits it. The tool writes only local reports and does not submit files to BabelBee. Review reports contain original document text and should be handled like the source files.

- **XLSX or UTF-8 CSV**. Default layout is a header row followed by pairs in columns 1 and 2. Use `--source-column` and `--target-column` to select other columns, `--no-header` for headerless data and `--sheet` for workbooks with multiple sheets. Replace formulas with values before importing.
- **XLIFF 1.2, 2.0, 2.1 and SDLXLIFF**. Import paired segments directly. SDLXLIFF segments are joined by segment ID. Use language options when the file does not declare the language pair.
- **Limits**. One already-aligned file per run, up to 5 MiB, 500 usable pairs, 60,000 combined text characters and 8,000 characters per pair. Your model may have a smaller context limit. If it cannot read the complete file, the run fails without generating a partial report.
- **Document and segment review**. The model reads the full bilingual document for context. The tool validates exact source excerpts at sampled positions, then requires a receipt for every segment batch. Source-ending and dialogue checks use neighboring text. A recoverable API error or an invalid response gets one retry of the affected request. If a required reading or segment check still fails, the run stops before writing reports.
- **Terminology coverage**. The reports show verified occurrences of selected terms and positions still needing human review. Unchanged capitalized terms such as names and addresses can be verified directly in the bilingual text. Follow-up requests check the remaining positions. A clear defect is attributed to the affected segment, with a correct counterpart shown as comparison evidence. Differences without a proven error are advisories. Confirmed duplicate findings are counted once.
- **Output scope**. The tool reports issues and evidence, including supported placeholder checks, transfer direction and unsupported completion of truncated source text. It uses the low-latency setting for the first document pass with OpenAI’s `gpt-6-luna`. The reports contain issue counts and keep advisories separate. The website’s scoring algorithm, weights and production prompts are outside this package.
- **Output**. The tool creates `report.html` and `report.xlsx` in a new directory next to the input. Use `--output-dir` for a different new directory. The input file is read-only.

## Tests

```sh
python -m pip install -e '.[test]'
python -m pytest
```

The tests use synthetic documents and a loopback model API stub.

## License

This standalone CLI is licensed under the [MIT License](LICENSE). See [RELEASE.md](RELEASE.md) for the package checks.
