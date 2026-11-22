# 録画済み EEG を学習済みモデルと NeuroCore へ通す

`try_recorded_eeg.py` は，完了済みの M1 用 SmolLM2 1.7B + EEG connector に，固定済み AIcoding UniBa v3 の **学習集合から各クラス 1 例，計 3 例**を入力する試用コマンドです．モデルの確率，棄権判定，NeuroCore の dry-run 判定理由を `summary.json` に保存します．OS 操作とクラウドへの agent 呼出しは 0 件です．

この 3 例は操作経路を確認するために学習集合から選びます．汎化精度は別の被験者分離評価を参照してください．`ai_prompt_submit`，`keyboard`，`mouse_click` は記録された操作の分類であり，キー内容・UI 対象・潜在的な意図を含みません．分類確率を，校正済みの「ユーザーの意図が正しい確率」として扱いません．

## 実行環境

正 worktree は `/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training` です．既存の `.venv-xavier` とローカルに取得済みの backbone を使用します．依存関係は `../model/requirements.txt` に従います．ネットワークから新しい backbone を取得する処理は起動せず，`HF_HUB_OFFLINE=1` で既存キャッシュを使用します．

既定の固定入力は次の 2 つです．

- Run: `private/xavier/runs/smollm2_1_7b_uniba_natural_v3_fused_v1`
- Dataset: `private/xavier/prepared/ai_coding_uniba_v3_natural_casecontrol_verified/manifest.jsonl`

`--run-dir` と `--manifest` は同一の固定済みファイルを別の場所で参照する場合に使用できます．公開コード内の SHA-256 と一致する必要があるため，別モデルへの差し替えは拒否します．元の `eeg_llm.py`，checkpoint，元 manifest，波形，閾値は変更しません．MPS，bfloat16，8 GiB，候補バッチ 2，棄権閾値 0.8 の検証済み設定を保持します．

## 一括実行

M1 の 5 条件評価が終了し，その終了 receipt とハッシュ一致する完了レポートがそろった後に，未使用の出力ディレクトリを指定してください．評価中は実推論を起動しません．

```bash
cd /Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training
.venv-xavier/bin/python experiments/pj_xavier/integration/try_recorded_eeg.py \
  --output-dir private/xavier/demos/m1_natural_v3_train_examples_local_01
```

ラッパーが来歴を検証し，新しい 3 例の manifest を作り，既存 predictor の `predict` と既存 `replay_to_neurocore.py` の dry-run を順に 1 回だけ実行します．出力ディレクトリが既に存在する場合は上書きせず停止します．途中失敗時も自動で推論を重複起動しません．

## 検証だけ先に行う

`--prepare-only` は CPU での `torch.load(weights_only=True, map_location="cpu")`，分割照合，ファイルハッシュ検証，試用 manifest 作成までを行います．backbone を読み込まず，MPS 推論も行いません．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/try_recorded_eeg.py \
  --output-dir private/xavier/demos/m1_natural_v3_train_examples_v1 \
  --prepare-only
```

評価の終了確認後，同じ Python 環境で，準備済みの計画を実行できます．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/try_recorded_eeg.py \
  --execute-prepared private/xavier/demos/m1_natural_v3_train_examples_v1
```

準備時のソース，配列，checkpoint，分割，コマンド，ラッパー自身のハッシュを再照合します．既に実行開始記録がある場合は再実行を拒否します．

## 検証するもの

- Run summary と training exit receipt の正常終了，検証集合によるモデル選択，元 manifest の SHA-256．
- 安全な weights-only 読込み，connector tensor の有限性，backbone の学習対象が空であること，固定されたモデル設定．
- train / validation / test の全 ID と training membership の一致，元 manifest 内の全 EEG 配列のサイズと SHA-256．
- train に所属し，記録された操作ラベルを持つ各クラスの辞書順先頭 ID．予測結果に基づく選び直しはしません．
- 5 条件評価の終了 receipt，checkpoint と report のハッシュ，5 条件すべての確定状態．
- 予測 ID・EEG 参照・確率和・棄権閾値，dry-run の監査 ID，実行前後の元ファイルの不変性．

`weights_only=False` へのフォールバックはありません．prepared manifest を test 例へ差し替えることや，plan を再ハッシュして任意コマンドへ置き換えることも拒否します．

## 結果の読み方

| ファイル | 内容 |
| --- | --- |
| `prepared_summary.json` | train 3 例と全配列を検証した準備結果 |
| `plan.json` / `plan.sha256.json` | 元ファイル・選択 ID・固定コマンド・ハッシュ |
| `train_examples.jsonl` | 元レコードの項目を保持し，EEG パスを解決した train 3 例 |
| `predictions.jsonl` | モデルの各クラス確率と棄権判定 |
| `dry_run_audit.jsonl` | NeuroCore に通した際の抑止理由 |
| `summary.json` | 来歴ハッシュ，確率，棄権，dry-run 理由，非実行確認 |
| `failure.json` | 失敗・中断の理由と自動再実行不可の記録 |

この既定の router 呼出しでは mouse-click proxy の明示 mapping も human-armed も設定しません．棄権した例は棄権理由で保留され，棄権しなかった例も具体的な意図への mapping がないため保留されます．予測を監査可能な経路で受け取れることを確認する構成です．

## CPU テスト

```bash
.venv-xavier/bin/python -m unittest discover \
  -s experiments/pj_xavier/integration -p test_try_recorded_eeg.py -v
```

9 件の fixture は，train-only 選択，CPU / weights-only 読込み，配列・分割・コマンドの改変，unsafe checkpoint，既存成果物の保全，評価未完了，二重起動を検証します．GPU 推論は行いません．
