"""Check the exact standalone source and wheel release contents."""
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    '__init__.py', 'cli.py', 'direction.py', 'evidence_rules.py', 'inputs.py',
    'provider.py', 'register.py', 'reports.py', 'review.py', 'terminology.py',
}
SOURCE = {
    'LICENSE', 'README.md', 'RELEASE.md', 'MANIFEST.in', 'pyproject.toml', '.gitignore',
    '.github/workflows/ci.yml', 'examples/sensor-guide.csv', 'examples/sensor-guide.xliff',
    'tests/test_cli.py', 'tools/check_dist.py',
} | {f'src/babelbee_qa/{module}' for module in MODULES}
SOURCE_METADATA = {'PKG-INFO', 'setup.cfg', 'src/babelbee_qa.egg-info/SOURCES.txt'}
WHEEL_METADATA = {'METADATA', 'WHEEL', 'entry_points.txt', 'top_level.txt', 'RECORD', 'licenses/LICENSE'}
SECRET_PATTERNS = (
    rb'-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----',
    rb'\bsk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{24,}\b',
    rb'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b',
    rb'\bgithub_pat_[A-Za-z0-9_]{30,}\b',
    rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_names(names):
    require(len(set(names)) == len(names), 'Duplicate archive member.')
    for name in names:
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and '\\' not in name
                and name == path.as_posix(), f'Invalid archive path: {name}')


def check_names(actual, expected):
    require(actual == expected, f'Unexpected files: {sorted(actual - expected)}; missing: {sorted(expected - actual)}')


def check_content(name, content, source_name=None):
    require(not any(re.search(pattern, content) for pattern in SECRET_PATTERNS),
            f'Credential-like value in {name}.')
    if source_name:
        require(content == (ROOT / source_name).read_bytes(), f'{name} differs from reviewed source.')


def check_metadata(content):
    metadata = BytesParser(policy=policy.default).parsebytes(content)
    require(metadata['Name'] == 'babelbee-qa' and metadata['License-Expression'] == 'MIT'
            and metadata.get_all('License-File') == ['LICENSE'], 'Incorrect package name or license metadata.')
    return metadata['Version']


def check_sdist(path):
    with tarfile.open(path, 'r:gz') as archive:
        entries = archive.getmembers()
        safe_names([entry.name for entry in entries])
        prefix = path.name.removesuffix('.tar.gz') + '/'
        require(all(entry.name == prefix[:-1] or entry.name.startswith(prefix) for entry in entries),
                'Unexpected source archive root.')
        require(all(entry.isdir() or entry.isfile() for entry in entries), 'Unsupported source archive entry.')
        files = {entry.name.removeprefix(prefix): entry for entry in entries if entry.isfile()}
        check_names(set(files), SOURCE | SOURCE_METADATA)
        for name, entry in files.items():
            with archive.extractfile(entry) as stream:
                check_content(name, stream.read(), name if name in SOURCE else None)
        with archive.extractfile(files['PKG-INFO']) as stream:
            return check_metadata(stream.read())


def check_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        safe_names(names)
        dist_info = path.name.split('-', maxsplit=2)
        prefix = f'{dist_info[0]}-{dist_info[1]}.dist-info/'
        modules = {f'babelbee_qa/{name}': f'src/babelbee_qa/{name}' for name in MODULES}
        check_names(set(names), modules.keys() | {prefix + name for name in WHEEL_METADATA})
        for name in names:
            source = 'LICENSE' if name == prefix + 'licenses/LICENSE' else modules.get(name)
            check_content(name, archive.read(name), source)
        return check_metadata(archive.read(prefix + 'METADATA'))


def main(dist):
    wheels = list(dist.glob('*.whl'))
    sources = list(dist.glob('*.tar.gz'))
    require(len(wheels) == len(sources) == 1, 'Expected one wheel and one source distribution.')
    require(check_sdist(sources[0]) == check_wheel(wheels[0]), 'Distribution versions differ.')
    for path in (*wheels, *sources):
        print(f'Passed archive and content checks: {path.name}')


if __name__ == '__main__':
    try:
        main(Path(sys.argv[1]) if len(sys.argv) == 2 else ROOT / 'dist')
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        sys.exit(str(error))
