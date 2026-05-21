# GitHub Actions

NeuroCore は OpenRI と同じ方針で、通常の tests だけでなく repository health、Pages、package metadata、security scan を分けて実行します。

## Workflows

- `CI`: Python tests、CLI self-test、ruff、frontend build、package smoke。
- `Repository Health`: OSS 必須ファイル、version consistency、Pages 生成、link validation。
- `Pages`: tutorial、markdown docs、package artifacts を生成して GitHub Pages に公開。
- `CodeQL`: Python と JavaScript/TypeScript の code scanning。
- `Dependency Review`: pull request の dependency diff を検査。
- `Scorecard`: OSSF Scorecard を SARIF として出力。
- `Release`: tag push 時に wheel、sdist、checksum、SPDX metadata を作成。

## Private Repository Notes

private repository では、CodeQL upload と Dependency Review は GitHub Advanced Security が有効な場合だけ実行します。
Advanced Security がない場合、workflow は明示的に skip します。OSSF Scorecard も default integration token では private repository を読めないため、public repository だけで実行します。
Dependabot alerts と automated security fixes は repository API から有効化できます。

## Alert Handling

1. `gh issue list` で issue を確認します。
2. `gh run list` で失敗 workflow を確認します。
3. code scanning、Dependabot、secret scanning の alert API を確認します。
4. 再現 test または health check を足してから修正します。
5. 修正後に local checks と GitHub Actions の両方を確認します。

## Local Mirror

```bash
python -m pytest
python -m neurocore.cli self-test --json
python -m ruff check backend scripts
python scripts/oss_health_check.py
python scripts/build_package_artifacts.py
python scripts/build_pages.py
python scripts/validate_pages.py
python -m twine check dist/*.whl dist/*.tar.gz
cd frontend && npm ci && npm run build
```

`docs/packages/` は generated artifact です。Pages workflow で毎回作り直します。
