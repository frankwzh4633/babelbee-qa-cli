# BabelBee QA CLI release checklist

Use this checklist for the standalone package. The candidate version is recorded in `pyproject.toml`.

## Prepare the candidate

- [ ] Confirm the version and review the MIT copyright notice for the standalone CLI.
- [ ] Review every added or changed file. Examples and tests must use synthetic content. Keep customer files, generated reports, API credentials, website code, production prompts, and the website scoring algorithm and weights outside the release tree.
- [ ] Review the standalone Git history before changing repository visibility. Start the public repository from the reviewed standalone tree. Keep the BabelBee monorepo history out of that repository.
- [ ] Confirm the proposed repository and package names before publishing.

## Verify the candidate

From the standalone source tree, build into an empty output directory.

```sh
python -m pip install build twine
python -m build --outdir candidate-dist
python -m twine check --strict candidate-dist/*
python tools/check_dist.py candidate-dist
```

The archive check verifies the exact wheel and source file lists, compares files with the reviewed source, checks MIT metadata, and searches for common credential formats. Review the source diff and package contents for confidential logic or differently formatted secrets as well.

- [ ] Run the standalone GitHub Actions workflow. It builds and checks both archives, tests the installed wheel on supported Python versions, and tests an install from the source archive on Python 3.12. The test model runs on loopback and requires no API key.
- [ ] In a clean virtual environment, run `babelbee-qa --help`, then run the README example with an approved configured model. Confirm the intentional printf type change appears in the HTML and Excel reports and verify their terminology coverage and segment tables.
- [ ] Record the test outcomes and SHA-256 checksums of the exact candidate archives to distribute.

## Publish after approval

- [ ] Confirm the release destination and publish only the verified artifacts after release approval.
