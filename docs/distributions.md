# Packages and Distributions

NeuroCore は Python package として配布します。GitHub Pages には wheel、source distribution、checksum、SPDX metadata を並べる想定です。

## Build Locally

```bash
python -m pip install -e ".[dev,server]"
python scripts/build_package_artifacts.py
python scripts/build_pages.py
python scripts/validate_pages.py
```

生成物は `dist/` と `docs/packages/` に出ます。`docs/packages/manifest.json` には full SHA256 と byte size が入ります。

## Install From Pages

```bash
pip install https://www.yasufumi.net/NeuroCore/packages/python/neurocore-0.1.0-py3-none-any.whl
```

## Release Boundary

- decoder weights、training checkpoints、private EEG data は配布物に含めません。
- source distribution には runtime、docs、samples、tests strategy を含めます。
- release metadata は `scripts/build_release_metadata.py` で `SHA256SUMS` と SPDX JSON を作ります。
- package artifact は clinical or medical device claim をしません。

## Verification

```bash
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
python scripts/build_release_metadata.py
```
