# NeuroCore

[![CI](https://github.com/yasufumi-nakata/NeuroCore/actions/workflows/ci.yml/badge.svg)](https://github.com/yasufumi-nakata/NeuroCore/actions/workflows/ci.yml)
[![Repository Health](https://github.com/yasufumi-nakata/NeuroCore/actions/workflows/oss-health.yml/badge.svg)](https://github.com/yasufumi-nakata/NeuroCore/actions/workflows/oss-health.yml)
[![CodeQL](https://github.com/yasufumi-nakata/NeuroCore/actions/workflows/codeql.yml/badge.svg)](https://github.com/yasufumi-nakata/NeuroCore/actions/workflows/codeql.yml)
[![Tutorial](https://img.shields.io/badge/tutorial-GitHub%20Pages-256f5b)](https://www.yasufumi.net/NeuroCore/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

NeuroCore は、EEG を装着したユーザーがマウス、キーボード、AI エージェントを操作するための研究向けランタイムライブラリです。脳波から intent を推定する重み付けや分類器はこのリポジトリには含めません。外部デコーダが出した intent を、NeuroCore が検証し、安全な操作 envelope に変換します。

## いま入っているもの

- `NeuroFrame`: EEG 時系列、チャンネル、時刻系、イベント、由来、検証結果をまとめる標準コンテナ。
- CPU reference kernels: EEG 検証、リサンプリング、FFT bandpass、平均リファレンス、スペクトル特徴量。
- `Pipeline`: カーネルを順に実行し、ステップごとの backend、時間、警告、出力 shape を report 化。
- `ControlRouter`: 外部デコーダから来た `IntentCommand` を mouse / keyboard / agent action に変換。
- Agent safety envelope: EEG 由来の payload を untrusted として包み、prompt-like payload を遮断。
- Stream buffer: リアルタイム入力を window / step に従って `NeuroFrame` に切り出し。
- Signal quality: flatline や高振幅を検出する初期品質スコア。
- Audit log: 操作を OS に送らず dry-run で JSONL 監査ログ化。
- Dataset inventory loader: EEG-DATA の日本語目録 CSV を、信号波形ではなく再利用候補の metadata として読み込み。
- `self-test`: 合成 EEG と破綻ケースで、NaN、Nyquist 超過、stream、signal quality、低 confidence、emergency stop、agent payload guard を自動検査。
- Settings UI: device / signal / safety / route / agent / self-test を操作するローカル設定画面。
- CI: Python tests、CLI self-test、frontend build を実行。

## インストール

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,server]"
```

## CLI

```bash
neurocore settings --json
neurocore self-test --json
neurocore demo --json
neurocore stream-demo --json
neurocore quality --json
neurocore run-csv samples/synthetic_eeg.csv --sampling-rate 250 --json
neurocore dataset-inventory ../EEG-DATA/eeg_dataset_summary_ja.csv --json
neurocore route-intent select --confidence 0.92 --json
neurocore simulate-intents samples/intent_commands.json --json
```

終了コードは、自己テストや pipeline が通れば `0`、安全上ブロックされた intent や破綻検出時は `1` です。

## API

```bash
PYTHONPATH=backend uvicorn neurocore.api:app --reload --port 8010
```

主な endpoint:

- `GET /api/health`
- `GET /api/settings/default`
- `POST /api/settings/validate`
- `POST /api/self-test`
- `POST /api/pipeline/demo`
- `POST /api/signal/quality`
- `POST /api/stream/demo`
- `POST /api/control/route`
- `POST /api/control/simulate`

## 設定画面

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5174
```

API を `127.0.0.1:8010` で起動しておくと、設定画面から self-test を実行できます。

## Python API

```python
import neurocore as nc
from neurocore.synthetic import synthetic_eeg_frame

frame = synthetic_eeg_frame(seconds=3, sampling_rate=250)

pipeline = nc.Pipeline([
    nc.ValidateEEG(min_channels=2),
    nc.Resample(250),
    nc.Bandpass(1, 40),
    nc.ReReference("average"),
    nc.SpectralFeatures(),
])

result = pipeline.run(frame, backend="cpu")
print(result.to_dict())
```

外部デコーダとの接続例:

```python
from neurocore import ControlRouter, IntentCommand

router = ControlRouter()
action = router.route(IntentCommand("select", confidence=0.91))
print(action.to_dict())
```

ストリーミング窓切り:

```python
from neurocore import Channel, StreamBuffer

buffer = StreamBuffer(
    channels=(Channel("Fz"), Channel("Cz")),
    sampling_rate=250,
    window_seconds=1.0,
    step_seconds=0.25,
)
windows = buffer.append([[0.1, 0.2], [0.3, 0.4]])
```

EEG-DATA 目録の確認:

```python
from neurocore import load_eeg_dataset_inventory

inventory = load_eeg_dataset_inventory("../EEG-DATA/eeg_dataset_summary_ja.csv")
print(inventory.summary())
```

この API はデータセット目録を読むためのものです。EEG の raw waveform は、各データセットを取得した後に専用 loader で `NeuroFrame` へ正規化してください。

## 設計資料

- [DESIGN.md](DESIGN.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/testing-strategy.md](docs/testing-strategy.md)
- [docs/settings-ui.md](docs/settings-ui.md)
- [docs/tutorial/index.html](docs/tutorial/index.html)
- [docs/distributions.md](docs/distributions.md)

公開チュートリアルは GitHub Pages の `https://www.yasufumi.net/NeuroCore/` を想定しています。

## OSS 運用

- 参加方法: [CONTRIBUTING.md](CONTRIBUTING.md)
- セキュリティ報告: [SECURITY.md](SECURITY.md)
- サポート: [SUPPORT.md](SUPPORT.md)
- ガバナンス: [GOVERNANCE.md](GOVERNANCE.md)
- ロードマップ: [ROADMAP.md](ROADMAP.md)

## 重要な境界

NeuroCore は医療診断製品ではありません。OS 操作や AI エージェント実行も、このライブラリ単体では実行しません。実行側は、NeuroCore の action envelope を受け取った後も、別途ユーザー許可、OS 権限、agent policy、監査ログを適用してください。
