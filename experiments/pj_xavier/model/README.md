# PJ-Xavier EEG–LLM 学習パイロット

EEG を学習可能な連続埋め込みに変換し，既存の causal LLM の `inputs_embeds` に挿入します．回答トークンの生成損失を，凍結した LLM 内部を通して EEG encoder／connector へ逆伝播します．分類器の出力を LLM へ文章で渡す方式ではありません．

この実装で確認するのは，B 案の学習経路，保存，評価，実機メモリと実行時間です．EEGlass の行動ラベルだけから，ユーザーの意図，画面上の対象，汎用的なエージェント操作が学習できたとは扱いません．

## 実装範囲

| 項目 | 実装 |
|---|---|
| EEG 入力 | 共通チャネル順・共通長の `[channels,time]` 波形 |
| EEG encoder | 学習可能な時間畳み込みと pooling |
| connector | LayerNorm と線形射影，既定 8 prefix tokens |
| LLM | Hugging Face `AutoModelForCausalLM`，リモートコード実行なし |
| 学習 | encoder／connector，明示指定時のみ追加 LoRA |
| 損失 | 回答文字列と EOS のみ，EEG prefix・質問・padding は除外 |
| モデル選択 | 各 epoch の validation balanced accuracy，最初の同点モデルを保持 |
| 最終評価 | `--finalize-test` 指定時に最良 checkpoint を test で一度評価 |
| 対照 | 同じモデルへ異なるサンプルの EEG を入れる固定 permutation |
| 追加モダリティ | 実際の時刻付き `environment_text` がある場合のみ fused／context_only |
| 出力 | 候補ラベル，候補間で正規化した尤度，保留，時刻，checkpoint |
| 実行境界 | このスクリプトは OS 操作を実行しません |

既定 135M backbone は接続診断用です．最終モデルの大きさを固定するものではありません．同じコードでモデル名，重み精度，バッチ数を変え，利用する実機で実測して選びます．1.7B や 7B の可否を，小型モデルの速度から推定して合格にしません．

## 実行環境

`requirements.txt` は実機確認した Transformers 5.22.1／5.24 と PyTorch 2.14.1／2.15 を含む依存範囲です．Colab の既存 CUDA 対応 wheel は保持します．Colab への接続と実行は workstation の CLI 経由で行います．新しい Mac 環境では，例えば次の手順を使います．

```bash
uv venv --python 3.13 .venv-xavier
uv pip install --python .venv-xavier/bin/python -r requirements.txt
.venv-xavier/bin/python -m unittest -v test_eeg_llm.py test_ai_coding.py
```

2026-11-22 に Linux CPU，Python 3.12.14，PyTorch 2.15.0+cpu，Transformers 4.57.6 で基本機能を検証し，後から実 Colab の Transformers 5.22.1 と M1 の Transformers 5.24 で互換性を確認しました．現在の試験はモデル側 24 件，AI-coding 同期・実 schema 側 12 件です．4 系は初期診断に使用した版で，新規導入の推奨範囲には含めていません．

- 回答損失から EEG encoder へ非零の勾配が届くこと．
- 凍結 LLM に勾配や重み変更が発生しないこと．
- prefix，質問，padding を学習対象から除外すること．
- checkpoint 保存・復元で予測が一致すること．
- context_only が異なる EEG 値を参照しないこと．
- AI 擬似ラベルを評価の正解として拒否すること．
- 元の記録時間が重なる train／val／test を拒否すること．
- EEG より未来の環境スナップショットを拒否すること．
- 被験者が分離した有効なデータを受け入れること．
- 既知の混同行列，ファイル内配列選択，非有限波形の検査．
- Transformers が revision を返さない場合も，同一 Hub snapshot を config，weights，tokenizer に使用すること．
- immutable revision を解決できない場合，重みを読み込む前に停止すること．
- メモリ予算超過後には validation／test を実行せず，checkpoint を保持すること．
- epoch 途中の中断・再開と，中断しない学習で全 connector 重みが完全一致すること．
- manifest の文章が同じでも，EEG 配列の実ファイル差し替えを検出すること．
- MPS allocator の上限値を機器の推奨メモリから計算し，無制限設定を使わないこと．
- 推論時の runtime 設定変更が checkpoint ファイルを書き換えないこと．
- MPS の非整数分割 pooling が参照実装と値・勾配で一致し，既存 checkpoint の key を変更しないこと．
- dataset version 名や被験者名の変更で，同じ人や同一原波形を split 間へ隠せないこと．
- EEG 置換が同じ被験者・session・recording 内の非自己対応 permutation になること．

さらにランダム初期化した極小 GPT-2 と明示的な合成 fixture で，学習，epoch 保存，best 復元，通常・EEG 置換評価，別プロセスでの推論まで実行しました．この結果は接続診断であり，実 EEG の精度や pretrained LLM の性能結果ではありません．合成 fixture は通常の pretrained model 学習では拒否します．

## EEGlass の準備

```bash
python prepare_eeglass.py \
  --set-file /data/eeglass/eeglass_pctasks.set \
  --output /data/prepared/eeglass_v1
```

同じディレクトリの `.fdt` と `.set` を使用します．記録された操作の 2 秒前から 0.25 秒前までを切り出し，その区間に別の記録イベントがある候補を除きます．入力窓同士の重複も除きます．このフィルターだけで筋電・動きの混入が消えるわけではありません．

記録全体の 60%・80% の時刻を境界に，train／val／test の順で分割します．境界の前後 10 秒を除外します．同一被験者の時系列パイロットであり，被験者をまたぐ一般化評価ではありません．境界をラベル分布に合わせて動かしません．

保存するものは `eeg_windows.npy`，`manifest.jsonl`，`preparation.json` です．元の波形とイベントファイルの URL／SHA-256，元の記録時刻，実際の操作時刻，入力を操作より前で止める時間差を保持します．チャネル名はファイル内の値をそのまま記録し，確認できていない電極位置は補完しません．

## 学習

```bash
python eeg_llm.py train \
  --manifest /data/prepared/eeglass_v1/manifest.jsonl \
  --output /data/models/eeglass_135m_run1 \
  --model HuggingFaceTB/SmolLM2-135M \
  --samples 438 \
  --split-policy within_subject_time \
  --device cuda --dtype bfloat16 \
  --epochs 3 --batch-size 4 \
  --memory-budget-gib 20 \
  --max-seconds 1800
```

`--samples` は `preparation.json` の実値を使います．異なる長さを学習器内で勝手に補間しません．異なる標本化周波数の入力は準備段階で処理します．各窓の標準化にはその窓内の値だけを用います．

Mac は `--device mps --dtype bfloat16` と明示した予算で実測します．M1 16 GiB では 1.7B の保存済みモデルを実際に読み込み，3 窓の推論が完了しています．この小数例の容量確認から，8B が同じ機器で動くとは推定しません．GPU を指定したのに使えない場合，CPU の結果を GPU の成功として返さず停止します．MPS の非整数分割 `AdaptiveAvgPool1d` 制約には同一 floor／ceil 区間の平均で対応し，CPU／CUDA の処理と checkpoint の key は維持します．

7B 級を実測する場合は，モデル名を対象の causal LLM に変えた上で，例えば次の制限を明示します．

```text
--max-base-params 9000000000
--device cuda --dtype bfloat16
--batch-size 1 --candidate-batch-size 1
--memory-budget-gib 21
```

これは搭載可能という事前保証ではありません．モデルの重みだけでメモリ予算の 80% を超える設定は読み込む前に停止します．実行時に観測したメモリが予算を超えた場合も記録して停止します．CUDA は allocator peak，MPS はサンプリング値とプロセス RSS を保存するため，MPS の真のピークを測定したとは記載しません．

`--max-seconds` は最適化ループの停止条件で，validation や最終評価の時間は別に必要です．メモリ予算超過時は完了済み学習を保存し，その後の validation／test を止めます．CLI の終了コードは 2 です．評価中の候補 scoring でも予算を検査します．CUDA が out-of-memory になった場合は `failure.json` に記録されます．自動で CPU に変更して続行しません．

### LoRA

`peft` がある環境では `--lora-rank 8` などで追加できます．既定は LLM 凍結です．凍結していても入力埋め込みへ勾配を戻すため，LLM の forward と入力への backward は必要です．小さな connector のみを計算する費用にはなりません．

### 再開

```bash
python eeg_llm.py train \
  --manifest /data/prepared/eeglass_v1/manifest.jsonl \
  --output /data/models/eeglass_135m_run1 \
  --resume /data/models/eeglass_135m_run1/epoch_003_step_0000240.pt \
  --model HuggingFaceTB/SmolLM2-135M \
  --samples 438 --split-policy within_subject_time \
  --device cuda --dtype bfloat16 --memory-budget-gib 20 \
  --epochs 5 --finalize-test
```

上の checkpoint 名は例です．実際には `run_summary.json` の `last_checkpoint` を指定します．manifest のハッシュ，実 EEG 配列の SHA-256，ラベル集合，architecture，学習モードが一致する場合に再開します．epoch 内の次の batch 位置，optimizer，乱数状態を保存しているため，途中で止めても未処理の batch を飛ばしません．backbone は保存済み revision を再使用します．

初期診断で保存した旧 checkpoint は推論に使用できますが，その時点の実 EEG 配列のハッシュが入っていないものを厳密な再開と呼ぶことはできません．新コードはその再開を拒否します．旧成果物を書き換えて検証済みと扱いません．

`test_metrics.json` が存在する実験ディレクトリへ再学習することは拒否します．外部で test 結果を見て別名の実験を調整したかどうかまでは，コードから確認できません．最終評価を複数モデルへ行う場合は，比較する実験群を先に固定します．

## 結果と推論

| ファイル | 内容 |
|---|---|
| `run_config.json` | backbone の解決済み revision と全学習条件 |
| `hardware.json` | 実際の device，依存版，モデルの parameter 数 |
| `training_membership.json` | train／val／test ID と hash，validation によるモデル選択 |
| `data_validation.json` | 分割とクラス分布，未検証の意味上の限界 |
| `eeg_artifacts.json` | manifest が参照する実 EEG 配列の SHA-256 と bytes |
| `epoch_*.pt` | connector，任意の LoRA，optimizer と再開情報 |
| `epoch_*.metadata.json` | checkpoint，元 manifest，train ID の SHA-256 |
| `val_epoch_*.jsonl` | 各 checkpoint の予測 |
| `history.jsonl` | 学習損失と validation 評価 |
| `best.pt` | validation で選択したモデル |
| `test_predictions.jsonl` | 明示的な最終評価の予測 |
| `test_shuffled_predictions.jsonl` | 他試行の EEG を入れた対照 |
| `run_summary.json` | 時間，実測メモリ，結果，未検証の能力 |

```bash
python eeg_llm.py predict \
  --checkpoint /data/models/eeglass_135m_run1/best.pt \
  --manifest /data/input/live_windows.jsonl \
  --output /data/output/predictions.jsonl \
  --device mps \
  --memory-budget-gib 8 --candidate-batch-size 1 \
  --dtype bfloat16 --max-examples 3
```

推論用 JSONL に正解ラベルは不要です．波形，元の ID と時刻は必要です．fused で学習したモデルには，学習時と同じ構造の実際の環境情報が必要です．

この例は最初の 3 窓を使う容量診断です．全件を処理するときは `--max-examples` を外します．`--memory-budget-gib`，`--candidate-batch-size`，`--dtype` は runtime 設定だけを変更し，保存した学習重みや学習条件は書き換えません．例えば BF16 がその実機で未対応なら，その失敗を残した上で FP16 の別条件を試します．

推論時には MPS／CUDA の framework allocator にも上限を設定します．MPS は `recommended_max_memory()` に対する割合を設定し，推奨量を超える設定や無制限設定を使用しません．これは GPU allocator の上限であり，CPU のロード時メモリや他アプリのメモリを含む全機の上限ではありません．実機情報，実測メモリ，採用した precision と予算，checkpoint hash は予測ファイルに隣接する `.metadata.json` へ保存します．

候補確率は，候補回答の条件付き尤度を候補間で正規化した値です．校正済みの確率ではありません．既定の 0.8 は保留の初期設定であり，ユーザーの最終用途で達成できる精度を保証する値ではありません．

## AI-assisted coding データの同期と準備

```bash
python prepare_ai_coding.py \
  --bids-root /data/ai_coding_extracted \
  --site uniba \
  --output /data/prepared/ai_coding_uniba
```

最初は監査だけを行います．`--prepare` を付けると，EEG と behavior の両 archive が完全取得・配布 checksum 検証済みであることを示す `source_verification.json` を要求し，同期検査を通った記録だけを準備します．部分取得 archive は学習用に使用しません．実取得 UniBa では，34 人・34 記録の 30.6154 時間中，32 人・28.5895 時間が同期検査に合格しました．2 人は大きな clock drift／残差のため除外しました．GUI ログ側の最初の実験イベントと EEG の最初のサンプルには異なる時間原点があり，実際に約 236.926 秒の差がある記録もありました．同じ `onset` をそのまま使わず，EEG 内の `otree_phase` と行動側の実験段階を対応させます．監査の集計は `../reports/ai_coding_uniba_audit.aggregate.json` に保存しています．

各コードが両側に一度だけ出現する段階を少なくとも 4 個使い，時刻の offset と drift を推定します．全 anchor の残差と，1 個ずつ除外して予測した残差を検査します．合格させるための恣意的な外れ値削除は行いません．閾値はこの工程の設定であり，研究分野の普遍的な合格基準ではありません．

`otree_phase` と `STI_artificial` は正解や実験状態を直接含むため，モデルの EEG 入力から除外します．UniBa／ITU に共通する 8 チャネル `CP3,C3,F5,PO3,PO4,F6,C4,CP4` だけをこの loader の入力に使います．原ファイルは全チャネルのまま保持します．reader が同期チャネルへ単位変換を行った場合は，実ファイルを確認した上で `--phase-scale` を明示します．

対象ラベルは記録された key press，mouse click，GPT prompt submission です．実公開 TSV の `action=down` を key press，`up` を release として読み，release は入力窓の動作除外に使用します．実ファイルの時刻・段階とキー内容を除いた最小 fixture は `fixtures/ai_coding_v2_minimal_recorded_schema.json` です．近接して同時に発生した UI イベントを同一操作として処理し，同じ EEG 窓を複数の正解へ重複登録しません．操作前の窓に別の記録操作があるものを除外します．既定では mouse movement は除外しないため，運動・筋電の交絡は残ります．`--exclude-mouse-movement` を指定した比較も可能です．

環境テキストは，EEG 入力前に記録された最後の実験ページ／イベントだけから作ります．これは実際の過去ログですが，画面全体の再現ではありません．未来の GPT 回答，その後のキーやクリック先を環境説明へ含めません．被験者 ID は施設を含めて区別し，被験者単位で train／val／test に固定します．

元データの `beginRecording` に巨大な負の `onset` があるのは，公式 importer が追加した `t=0` の sentinel から LSL の基準時刻を引いた結果です．実際の生理信号の時刻異常とは区別し，負の時刻と sentinel は使用しません．5 秒ごとの EEG `events.tsv` は人工 epoch であり，行動の正解には使用しません．

公開 BrainVision は既に offline band-pass／notch filtering が施されています．原 XDF は配布されていないため，同期が合格しても online causal preprocessing が確認できたとは扱いません．`alignment_audit.json` は，元記録時間，準備済み窓の時間，split 別の使用時間と bytes，合格／除外理由を分けて保存します．

`audit_filter_support.py` は，公開 exporter の処理を MNE 1.10.2 で再構成します．band-pass が 423 taps，notch が 1,691 taps，合成 FIR が 2,113 taps となり，256 Hz では未来方向へ 4.125 秒の support を持ちます．これは公開 binary が記載の recipe 通りに生成されたという条件付きの計算です．単に 0.25 秒前で窓を止めても，元サンプルの未来参照を除いたことにはなりません．原未処理 EEG を確保するか，filter の遅延を実際の特徴量利用可能時刻に含める評価設計が必要です．計算根拠は `filter_support_evidence.json` に保存します．

今回の準備は `--lead-s 4.5 --filter-lookahead-s 4.125` を既定とし，1.75 秒の EEG 窓を操作時刻の少なくとも 4.5 秒前で止めます．`decision_time_s` は窓の終端に仮定上の filter delay 4.125 秒を加えた値で，イベントまでの `minimum_lead_s` は 0.375 秒です．元の 4.5 秒の間隔は `eeg_window_minimum_lead_s` に別保存します．実際の推論時間はこの delay に追加されます．binary と recipe の同一性を未検証のため，`feature_availability_verified` と `online_causality_verified` は false を維持します．この設計は未来情報混入を減らす診断であり，オンライン意図推定の達成判定には使いません．

実準備結果は 7,819 窓，入力区間の合計 3.80090 時間，float32 配列 112,097,280 bytes です．18 人の train に 4,357 窓，6 人の validation に 1,506 窓，8 人の test に 1,956 窓を割り当てました．入力窓内の別動作と窓重複を除いたため，`ai_prompt_submit` は train 2，validation 2，test 1 しか残りません．同期合格した元ログには 150 件の `GPTSentPrompt` が存在しますが，この 5 例から agent 委任の性能を判定することはできません．

`profile_training.py` は train の最長 token 列を選び，1 batch サイズにつき新しい process で短い学習を行い，重みを破棄します．validation／test の予測は行いません．Qwen3-8B 固定 revision と L4 では batch 1・2・4・8 が 21 GiB 予算内で完了し，試した中で最大の batch 8 を採用しました．batch 8 の CUDA peak allocated は 20,422,237,696 bytes，peak reserved は 21,162,360,832 bytes，warm throughput は約 16.76 examples/s でした．3 step の診断値で，長時間の実学習・評価まで保証する値ではありません．具体的な本学習設定と評価条件は `ai_coding_training_plan.json` に固定しています．

## UniMind との関係

公式 UniMind の公開先は `kaleidoyao/neuroverse` へリダイレクトします．公開コードは LaBraM と spatial／temporal の query モジュール，LLM への埋め込み挿入を含み，公開 README の学習例は InternVL2-8B です．今回の実装は，持ち運びやすい小型 encoder を用い，まず同じ種類の連続埋め込み経路と学習の成立を確認するものです．LaBraM，task-aware query selection，論文の多タスク成績を再現したとは扱いません．

| 比較 | 公式公開実装 | 今回のパイロット |
|---|---|---|
| EEG encoder | LaBraM | 時間畳み込み encoder |
| EEG と LLM の接続 | EEG 表現を LLM の入力埋め込みへ挿入 | 学習可能な EEG prefix を入力埋め込みへ挿入 |
| 複数タスクへの適応 | spatial／temporal query と task-aware selection | 共通の短い prefix，task-aware router は未実装 |
| 学習用 LLM | 公開例は InternVL2-8B | 通常の causal LLM を交換可能 |
| データ整合 | 各既存 EEG タスクのラベル | 原イベント，時刻，出典を保持した pre-action 窓 |
| 目標に対する評価 | 既存の EEG decoding tasks | まず観測行動パイロット，環境付き意図・agent 評価は別 gate |

一次資料は以下です．

- [UniMind 公式プロジェクト](https://kaleidoyao.github.io/UniMind-project/)
- [公式実装 README](https://github.com/kaleidoyao/neuroverse)
- [公式モデルコード](https://github.com/kaleidoyao/neuroverse/blob/main/InternVL-EEG/internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py)
- [公式の開発資料](https://github.com/kaleidoyao/neuroverse/blob/main/docs/development.md)
- [Transformers causal model API](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [診断用 SmolLM2-135M の公式モデルカード](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [EEGlass の元データ](https://zenodo.org/records/4453396)
- [環境付きデータの候補 AI-assisted coding v2](https://zenodo.org/records/20801101)
