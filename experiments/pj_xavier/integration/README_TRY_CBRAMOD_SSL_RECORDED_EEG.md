# 追加EEG学習済み CBraMod＋SmolLM2 の workstation 試用

**回収済みの追加学習モデルを，M1上で実EEGから推論し，NeuroCore routerまで接続する試用を完了しました．** 同じ学習集合の3例を使う動作確認です．汎化性能，自由形式の意図理解，自律操作の成功率は，この3例から評価しません．

## そのまま実行できるコマンド

正 worktree のルートで，未使用の出力名を指定します．下記のユーザー向け出力先では，作成側は推論を実行していません．

```bash
cd /Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training
.venv-xavier/bin/python experiments/pj_xavier/integration/try_cbramod_ssl_recorded_eeg.py \
  --contract private/xavier/demos/cbramod_ssl_train_demo_contract_v1.json \
  --expected-contract-sha256 bde21873e6c0c2c173658eae47f4ce7f38d5be8575944dcf2ce7cf08365797cb \
  --output-dir private/xavier/demos/cbramod_ssl_train_trial_user_01
```

準備検証と1回の推論・dry-run replayを続けて実行し，新しい出力先の `summary.json` に予測とrouterの保留理由を保存します．同じ出力先の上書き・再実行は拒否するため，別の試用には別の未使用名を使ってください．

学習済みモデル，元のEEG，公式基盤重みは `private/xavier/` にあります．別のPCに公開コードだけをcloneしても，これらの実験成果物がない状態では実行できません．モデルの取得・学習を勝手にやり直すwrapperではありません．

## 実行確認した内容

元の試用は `private/xavier/demos/cbramod_ssl_uniba_v3_train_examples_v1/summary.json` に保存済みです．各classから事前に選んだ同じtrain1例ずつで，3例とも観測ラベルと一致しました．出力候補はAI送信，keyboard，mouse clickの3種類です．限定候補内のsoftmaxは約0.8870／0.8821／0.8066でしたが，校正済みの意図の正解確率として扱いません．

routerは全3件を `proxy_label_has_no_explicit_intent_mapping` として保留しました．これらの観測イベントには「どの文字を入力するか」「どの対象を操作して何を達成するか」という具体的指示が含まれないためです．OS操作とcloud agent呼出しは0件でした．閾値0.8，入力，モデル重みを試用結果に合わせて変更していません．

CPUのEEG encoderとMPS上のSmolLM2を使い，MPS driverメモリの観測最大は4,310,302,720Bでした．allocator quotaは8GiBです．これはサンプリングした観測値で，未測定の瞬間最大値ではありません．3例のmodel-call p50／p95は544.989／2948.505msでしたが，cold開始・共用負荷を含む小標本であり，連続利用の代表的な遅延や端から端までの操作時間は未確定です．

[試用結果と検証記録](../reports/cbramod_ssl_recorded_eeg_demo_20261122_v1/README.md)，[学習・固定テストの実測](../model/CBRAMOD_SSL_COMPLETED_RESULTS.md)を参照してください．

## 実行前後の検証

wrapperは元のmanifest・全32下流配列・学習membership，SSL corpusの固定manifest／allowlist／protocol，公式初期weightsと実際に更新されたencoderのtensor，validation履歴による選択，LLM／connectorの構成を照合します．実際の5条件評価childの正常終了，strict検査，509成果物の回収証拠，契約のSHA-256も必要です．既存の実験，閾値，元wrapper，routerは変更しません．

原波形・checkpoint・推論sourceに不一致があれば実行を保留します．失敗時はその試行の記録を残し，原因確認前に同じ推論を自動再投入しません．

準備検証だけを行う場合は，別の未使用ディレクトリで次を実行できます．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/try_cbramod_ssl_recorded_eeg.py \
  --contract private/xavier/demos/cbramod_ssl_train_demo_contract_v1.json \
  --expected-contract-sha256 bde21873e6c0c2c173658eae47f4ce7f38d5be8575944dcf2ce7cf08365797cb \
  --output-dir private/xavier/demos/cbramod_ssl_user_prepare_01 \
  --prepare-only
```

準備した同じplanを1回実行するコマンドです．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/try_cbramod_ssl_recorded_eeg.py \
  --execute-prepared private/xavier/demos/cbramod_ssl_user_prepare_01
```

CPU契約テストは12件が通過しています．fixtureのテストはモデルの汎化性能とは別に扱います．

```bash
.venv-xavier/bin/python -m pytest -q \
  experiments/pj_xavier/integration/test_try_cbramod_ssl_recorded_eeg.py
```

## 評価の位置づけ

追加学習モデルは，先行モデルの結果を見た後に設計した探索実験です．UniBaの固定testを使った比較と，別protocolのITU外部評価をそれぞれ報告しています．このwrapperはtrainの動作確認だけを行い，held-out EEGを試用例へ流用しません．行動イベントの分類から，自由形式の意図やエージェントの課題達成へは，さらに対応する実記録と評価が必要です．
