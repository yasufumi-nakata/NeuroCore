# PJ-Xavier-EVLM-Qwen3.8-27B-v0.1

**状態：research candidate．** 画像・数値 EEG・テキストを同じ VLM に入力する接続を確認済みです．EEG による意図推定の有効性，汎用 computer use，実環境での自律動作は未実証です．本記録の確認日付は 2026-11-24 JST です．

## 1．構成と動作確認済みの候補

- 基盤は `mlx-community/Qwen3.8-27B-4bit`，revision `b935172e4bb5dcd0543774a34748fb92422e3027` です．設定上の実装クラスは `Qwen3_5ForConditionalGeneration` です．
- 既存の画像処理器・vision tower と言語 decoder を MLX-VLM で読み込み，EEG は固定規則で数値テキストへ変換します．独立した学習型 EEG エンコーダは追加しません．
- 成功済みの `candidate_v5_r2` は，言語 decoder の末尾 4 層に学習した v5 EEG LoRA を接続した候補です．rank 8，scale 20，31 対象 module／62 A・B tensor で，base の重みを変更・融合していません．
- 入力はローカル画像 1 枚，契約に合う EEG JSON，課題テキストです．出力は `click`／`type`／`wait`／`abstain` の JSON 提案で，常に `dry_run: true`，`executed: false` です．実 OS 操作は実装していません．

保存先は `/Users/yasufumi/pj-xavier/smac35/evlm_build_20261124_v1/candidate_v5_r2` です．この成功済み bundle は変更しません．

## 2．準備済み範囲と実際に完了した処理

| 項目 | 確認できた状態 | 解釈と限界 |
|---|---|---|
| v5 EEG／no-EEG 学習 | 各 80 updates 完了．EEG adapter は 2026-11-23 20:45 JST に保存済み | 末尾 4 層の LoRA です．各条件の実提示は 80 例で，全データ学習ではありません |
| v5 画像＋EEG smoke | 2026-11-24 05:29:37–05:30:28 JST，base 2/2，adapter 2/2 正答 | START／STOP を左右反転した合成 2 画面です．同一の合成 EEG と指示を用い，画像から対象位置を選べることを確認しました |
| 画像経路とメモリ | 333 vision tensor を strict load し，adapter 接続前後で同一性を保持．推論 peak 17.292 GB | 画像処理の限定的な動作確認です．全 64 層学習の必要メモリを示す値ではありません |
| 新 numeric corpus | 6 source，6 ch block 形式で 358,260 例を準備 | prepared coverage です．この件数を学習済み件数や完走予定として扱いません |
| 初回 all-64 engineering probe | 1 update 後，peak 58.762 GB が運用上限 54 GB を超え guard 停止．checkpoint なし | max sequence 1,024 の試行です．この 1,024 token 枠の試行は guard 停止し，checkpoint を保存していません |
| 6 ch／max sequence 801 再試験 | 05:51:29–05:56:07，2更新完走，peak 44.016 GB，基本指示6/6，保存・再読込PASS | 1 train 例を2回使った技術試験です．開発用評価はまだ行っていません |
| 新全64層 adapter の画像＋EEG | 05:56:49–05:57:48，base 2/2・adapter 2/2，peak 17.579 GB | 496 module／992 LoRA tensor，333 vision tensor 不変．新 raw32／6ch 入力を確認しました |

v5 の開発用 validation 155 例では，EEG／文脈のみ／EEG shuffle の balanced accuracy はそれぞれ 32.26%／33.33%／27.42% でした．これを EEG 効果の実証とは扱いません．最終 held-out test は開いていません．

## 3．新しい全 64 層学習の設計

対象は言語 decoder 全 64 層の **LoRA** です．base 全重みのフルファインチューニングではありません．画像・EEG・行動の正解対を捏造せず，まず実測 EEG の数値再構成を学習します．同一 channel の連続 4 点をマスクし，対応する整数 4 個を JSON として予測する自己教師タスクです．

6 source の各訓練 partition を，有限 update 数の stage に分けて進める設計です．各 stage は前段の検証済み adapter から warm-start し，Adam の状態は新規作成します．optimizer の厳密な resume ではありません．初期の短い probe が通った後，bounded stage と開発用評価を順次実施します．全 358,260 例の完走や完了日時は，実測速度と評価を確認する前には確約しません．

次 stage へ進む条件は，学習・評価の正常終了，保存 adapter と評価対象の SHA 一致，新 checkpoint から作る VLM bundle の検証，および合成画面での base 2/2・adapter 2/2 正答です．失敗時は自動拡張せず停止します．元の train／validation／test 分割，subject・payload の除外条件，QC 証明を引き継ぎ，最終 test は開きません．

## 4．数値 EEG の契約と coverage

`raw32` は，既に **3–40 Hz で前処理し，200 Hz に整えた信号の時間方向の数値サンプル**を指します．未処理の元記録をそのまま入力する意味ではありません．

- Schema は `eeg-raw32-uv1-channel-major-cycle19-v1` です．µV 単位を 1 µV 刻みの整数へ丸め，clipping は行いません．`[-100, 100]` は受入範囲の検証に使います．
- 3 秒・600 点の窓から 32 点の tile を使用します．末尾 tile は 24 点です．現準備は全 19 cycle 中の **cycle 0** であり，窓内全 600 点の時間 coverage が学習済みとは主張しません．
- 元の全 channel を，元の channel 名・順序を維持して最大 6 ch の block に分割します．channel を捨てて 6 ch に限定する処理ではありません．block 数の増加を独立した EEG 時間数として数えません．
- `geometry` に `shape`，`channel_names`，`sample_rate_hz: 200`，`quantum_uv: 1`，`crop_start_sample` を保持します．訓練 metadata の `max_channels_per_training_block: 6` を bundle・入力検証・合成 fixture へ引き継ぎます．超過入力の切捨てや無言の truncation は行いません．
- 旧 raw32 metadata にこの上限がない場合だけ 8 ch を互換既定値とします．明示値は bool を除く整数 1–8 に限定します．新 6 ch 契約を旧 8 ch 契約へ置き換えません．

v5 成功 bundle は別の旧 EEG 形式（8 ch × 448 点，256 Hz，16 点平均と帯域特徴の固定 serializer）を使用します．新 raw32 adapter と旧 v5 入力形式を混用しないでください．

## 5．Bundle の再現性と配布条件

Bundle は base，adapter，processor／tokenizer assets，runtime code，合成 smoke fixture を manifest で束ねます．base の 3 shard は smac35 上で独立に SHA-256 を計算済みです．全対象 LoRA key と A・B の形状，rank・scale，base revision，serializer を検証し，不一致 tensor を無言に落としません．全 64 層の adapter にも同じ strict 接続を使います．

現在の bundle は base を次のローカル path で参照するため，**bundle directory 単独では配布・再現できません**．同一 base と processor assets，固定環境が必要です．新 adapter には新しい出力 directory を使い，成功済み候補を上書きしません．

```text
/Users/yasufumi/pj-xavier/smac35/qwen38_direct_mlx_lora_20261123_v1/model
```

| 固定対象 | 値 |
|---|---|
| 専用環境 | `evlm_build_20261124_v1/venv`：MLX 0.32.4，MLX-VLM 0.8.0，Transformers 5.24.0 |
| v5 EEG adapter SHA-256 | `65f816defa523a1ad70160169faa4c5ae7c1483b86025b2ec78531487e0f0fac` |
| 成功済み v5_r2 runtime SHA-256 | `34e19ceb345d0c6ae0dcd83a78fa935b59786e96d98474b78bb1cf75f26fbd73` |
| v5_r2 manifest SHA-256 | `18025bffc2a753f76782a627950c6555ee7c2877f6fe79fe0cec277c5c6e4bc9` |
| 新 raw32／6 ch runtime SHA-256 | `3d8cc20c38668ba3127a7ee308aafef1aae7d962b15369399b06c73fbec79826` |
| 新 runtime の CPU 検証 | 2026-11-24 05:44 JST，20/20 tests PASS．新 raw32 adapter の GPU 成功を意味しません |

smoke の証拠は `candidate_v5_r2/smoke_20261123T203028073328.json` と `candidate_v5_r2/candidate_status.json` です．環境 lock は namespace 直下の `requirements.lock.txt`，新 runtime CPU 結果は `cpu_test_receipt_raw32_chcap.json` にあります．

## 6．実行例

以下は smac35 上の CLI です．`prepare` と `audit` は CPU 処理です．`smoke` と `infer` はノードの canonical GPU lock を排他的に取得し，取得できなければ実行しません．既存の学習と同時に GPU 処理を実行しない運用を前提にします．

```sh
EVLM_ROOT="$HOME/pj-xavier/smac35/evlm_build_20261124_v1"
EVLM_PY="$EVLM_ROOT/venv/bin/python"
EVLM_BASE="$HOME/pj-xavier/smac35/qwen38_direct_mlx_lora_20261123_v1/model"
EVLM_BUNDLE="$EVLM_ROOT/new_candidate"

# 新 checkpoint から bundle を作成します．path と SHA は実値へ置換します．
"$EVLM_PY" "$EVLM_ROOT/code/evlm_runtime.py" prepare \
  --base "$EVLM_BASE" \
  --adapter /absolute/path/to/checkpoints/000000N \
  --output "$EVLM_BUNDLE" \
  --expected-adapter-sha256 REPLACE_WITH_64_HEX_DIGEST \
  --base-revision b935172e4bb5dcd0543774a34748fb92422e3027

"$EVLM_PY" "$EVLM_BUNDLE/evlm_runtime.py" audit \
  --bundle "$EVLM_BUNDLE"

"$EVLM_PY" "$EVLM_BUNDLE/evlm_runtime.py" smoke \
  --bundle "$EVLM_BUNDLE" \
  --gpu-lock "$HOME/pj-xavier/smac35/train_gpu.lock" \
  --memory-gb 54 --max-tokens 128

"$EVLM_PY" "$EVLM_BUNDLE/evlm_runtime.py" infer \
  --bundle "$EVLM_BUNDLE" \
  --gpu-lock "$HOME/pj-xavier/smac35/train_gpu.lock" \
  --image /absolute/path/screenshot.png \
  --eeg-json /absolute/path/eeg.json \
  --task "Propose a click at the center of the button labelled START." \
  --memory-gb 54 --max-tokens 128
```

`prepare` は adapter metadata から入力形式を選び，その契約に合う合成 fixture を保存します．以降は bundle 内の固定 runtime を使います．成功済み候補を再確認する場合は `EVLM_BUNDLE` を `candidate_v5_r2` に設定し，`prepare` を繰り返さず `audit` 以降を実行します．controller は画像 gate 全体に最大 900 秒の時間上限も設けます．

## 7．全64層の再試験結果と残る未確認事項

**v5 候補と新全64層の技術試験候補の双方で，画像経路と EEG adapter を同時に読み込み，合成画面の位置選択ができることを確認しました．** EEG による意図推定，実 EEG 条件での行動改善，長期の agentic タスク成功，GUI 操作の安全な自律実行は未確認です．画像 smoke の正答だけでこれらが完成したとは扱いません．

全64層801 token枠の技術試験は，2更新・基本指示6/6・保存adapterの数値一致・再読込後の生成一致を確認し，exit 0で終了しました．peakは44.016 GBです．126.614秒と129.364秒の2更新は同じ最長train例への反復であり，独立訓練例数は1です．検証用データの生成は0です．

そのadapterを新しいプロセスでVLMへ接続した合成画像試験もbase 2/2・adapter 2/2でした．保存先は `evlm_build_20261124_v1/engineering_vlm_probe_6ch_v2/bundles/all64_probe_6ch_v2` です．adapter SHAは `34e3e22b31d6fd2a3c8807537c56ef5badf3ebe54b2dca10c500e99020cb8af0`，画像receipt SHAは `75e54092be08ecf335681b7603ac3d46bb8f081b9a4e35fe5688a397f5a8fe5d` です．この新候補も技術試験用で，通常学習済みモデルとは区別します．

通常学習はfresh baseから別namespaceで開始し，20／累積100／累積400更新を評価点とする方針です．数値悪化・2stageの停滞時は次stageへ進めず，初回予算400更新で再評価します．実行状況と実測時間は[構築報告](../reports/EVLM_BUILD_20261124.md)，データ範囲は[変換報告](../reports/EVLM_NUMERIC_DATA_20261124.md)を参照してください．EEGの意図推定への寄与，実画面と対応したEEGでの操作改善，長期タスク成功は引き続き未確認です．
