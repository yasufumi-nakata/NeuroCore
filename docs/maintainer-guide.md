# Maintainer Guide

この guide は issue、alert、release、public docs を処理するときの作業順です。

## Daily Triage

```bash
gh issue list --repo yasufumi-nakata/NeuroCore --state open
gh run list --repo yasufumi-nakata/NeuroCore --limit 20
```

Code scanning と Dependabot alerts は repository security settings が有効な場合に API で確認できます。
private repository で Advanced Security が未契約の場合、CodeQL と Scorecard は workflow 内で skip します。

```bash
gh api repos/yasufumi-nakata/NeuroCore/code-scanning/alerts
gh api repos/yasufumi-nakata/NeuroCore/dependabot/alerts
```

## Fix Policy

- public issue には private EEG data、credentials、agent prompt payload を貼りません。
- safety regression は先に blocked path の test を足します。
- dependency alert は minimum supported Python と frontend build へ影響しない範囲で更新します。
- GitHub Actions failure は local mirror command で再現してから直します。

## Release Checklist

1. `CHANGELOG.md` と `CITATION.cff` の version/date を確認します。
2. `python scripts/build_pages.py` を実行します。
3. `python -m build` を実行します。
4. `python scripts/build_release_metadata.py` を実行します。
5. `python -m twine check dist/*.whl dist/*.tar.gz` を実行します。
6. `vX.Y.Z` tag を push します。
7. Release workflow の artifacts と checksums を確認します。

## Pages Checklist

```bash
python scripts/build_package_artifacts.py
python scripts/build_pages.py
python scripts/validate_pages.py
```

公開先は `https://www.yasufumi.net/NeuroCore/` を想定しています。Pages が有効でない場合は repository settings で GitHub Actions source を有効化してください。
