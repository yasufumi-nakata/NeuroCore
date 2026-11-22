# クラスター用のAIラベル整備worker

このworkerは，公開データに付随する**観測イベント，実験手順，事前の環境記録，機器仕様，既存annotationを，出典付きの弱教師ラベルへ正規化**します．生EEGをLLMに渡して，未学習のLLMが脳内の意図を読めるかのようにラベルを付ける機能はありません．

モデルの出力がschemaに合っていても，人による独立した確認を受けていないラベルは常に `ai_pseudo` です．独立EEG記録の時間・人数が増えたことにも，評価用の正解が増えたことにも数えません．同じEEGに説明文を多数作成して，独立サンプル数として加算しない運用です．

## 実装済みの範囲

| 項目 | 動作 |
| --- | --- |
| 学習分割 | `split: train` の申告と，凍結した元manifestのbytes hash・所属を照合します．validation／test／holdoutはHTTP推論を行わず拒否します |
| 既存gold | 入力を変更せず，出力に別snapshotとhashを保存します |
| AIラベル | 別の `pseudo_labels` に保存し，`ground_truth: false` と明記します |
| 正規化先 | jobが宣言したnamespaceと許可値だけを選べます |
| 出典検証 | 全evidence IDを照合し，原文中に実在する完全一致引用を必須にします |
| 不確実性 | `unknown`／`abstain`を保存できます．ラベルを無理に作りません |
| 不正出力 | 出典捏造，非原文引用，未知の列，未許可の値は`failed`とし，ラベルを採用しません |
| クラスター分担 | job IDのSHA-256を用いて，独立したshardへ決定的に割り当てます |
| 再開 | job ID＋出典payload hash＋teacher digest＋prompt／schema／生成設定が同じ完了結果をskipします |
| 資源制約 | max jobs，最大試行回数，HTTP timeout，出力token上限，短いkeep-alive，空きdisk reserveを設定します |

namespaceと出典の対応も検証します．「この課題ではキーを押してください」という手順だけから，実際にキーを押したラベルを作ることはできません．

| namespace | 必要なevidence kind | 意味 |
| --- | --- | --- |
| `observed_action` | `recorded_event` | 記録された操作 |
| `task_protocol` | `task_protocol` | 実験・課題の手順や条件 |
| `environment_context` | `predecision_context` | 判断前に利用可能な環境情報 |
| `recording_metadata` | `recording_metadata` | 機器・データ形式などの記録情報 |
| `source_label` | `original_annotation` | 原典に明示されたannotationの正規化 |

出典IDと原文引用の一致は，意味の正しさを完全に証明するものではありません．意味を取り違えた正規化は残り得るため，これらをgoldへ昇格させません．teacherの自己申告confidenceも正答確率として使用しません．

## 入力JSONL

`job.schema.json`が1行のschemaです．次は**形式例**で，実データの結果ではありません．SHA-256は実際の分割manifest・元ファイルから取得してください．`source_manifest_sha256`は，全splitを含む凍結した元レコードJSONLのbytes hashです．必須引数 `--source-records` の実ファイルと照合し，`record_id` がtrainに所属する場合だけ推論します．任意の `source_record_ids` を付ける場合は代表の `record_id` を含め，全IDがtrainに属する必要があります．元ファイルのsubject，session，splitの変更もbytes hash不一致として拒否します．

```json
{
  "job_id": "dataset/participant/session/event-001",
  "record_id": "dataset/participant/session/window-001",
  "dataset_id": "dataset-version",
  "split": "train",
  "source_manifest_sha256": "ACTUAL_SHA256_OF_FROZEN_SPLIT_MANIFEST",
  "source_assets": [
    {"uri": "local-or-provider-event-file-uri", "sha256": "ACTUAL_SOURCE_FILE_SHA256"}
  ],
  "evidence": [
    {
      "id": "event-001",
      "kind": "recorded_event",
      "text": "Keyboard key A pressed",
      "source_uri": "source-event-file-uri#event-001",
      "timestamp_s": 10.0
    }
  ],
  "allowed_values": {"observed_action": ["keyboard", "mouse_click", "mouse_scroll"]},
  "original_gold": [{"value": "source-event-type", "origin": "original_annotation"}]
}
```

source assetsには原ファイルのURI/hashを残し，evidenceには採用した部分のID・原文・位置を記録します．大きなREADMEやログは出典を保った小さなjobに分けます．workerは長い入力を黙って切り捨てません．token budgetを超えるものは拒否し，学習に使える範囲を明示します．

`predecision_context`には `timestamp_s`，jobには `decision_time_s` が必要です．環境記録が判断時刻より未来なら拒否します．観測eventの正解ラベルは判断後に発生した操作を示して構いませんが，そのevent textをモデルの判断前入力へ戻してはいけません．このworkerの出力を学習入力へ接続する工程でも，元のcutoff制約を維持してください．

入力に波形配列，EEG embedding，画像，実行する指示は受け付けません．今回のworkerはtextによる整備専用です．VLMが登録済みでも，この実装では画像を送らずtextの正規化にのみ使用します．画像由来のラベルは，別途cutoffと出典が検証された入力経路が必要です．

## 実行

workstationまたは各Macで，そのノードのOllamaが起動し，指定したモデルが登録済みであることを前提にします．モデルを勝手に追加downloadする処理はありません．

```bash
python ollama_label_worker.py \
  --jobs /absolute/path/train_label_jobs.jsonl \
  --source-records /absolute/path/frozen_all_records.jsonl \
  --output-dir /absolute/path/weak_labels \
  --model qwen3.5:2b \
  --shard-index 0 \
  --num-shards 1 \
  --max-jobs 100 \
  --timeout-s 90 \
  --num-predict 768 \
  --num-ctx 8192 \
  --keep-alive 30s \
  --min-free-gib 1 \
  --node-id workstation
```

例えば38ノードを**実際に利用可能と確認した後**に分担する場合は，同じ凍結jobファイルを各ノードへ配り，`--num-shards 38`と，重複しない`--shard-index 0`〜`37`を設定します．38台が現在利用可能だと確認した実績を，この実装だけで主張しません．`--node-id`は実際の実行ノード名です．

各ノードは自分のlocalhostのOllamaを使います．外部URL，認証情報付きURL，redirect，環境変数のproxy経由の転送は受け付けません．ローカルendpointが異なる場合は `--endpoint http://127.0.0.1:PORT` を指定します．

`--max-jobs`に到達したら，同じコマンドで再開できます．完了・unknown・abstain・拒否済みの同一jobはskipし，失敗は既定で合計2試行までです．source，teacher，prompt，生成設定が変わったjobは別結果になります．異なるteacherの出力を同じ独立goldとして数えないでください．

## 保存結果

各shardに次のファイルを保存します．

```text
shard-000-of-038.jsonl
shard-000-of-038.manifest.json
shard-000-of-038.lock
```

JSONLはappend＋flush＋fsyncで保存します．途中で壊れた最終行や不正な既存JSONを見つけた場合は，勝手に切り捨てず再開を止めます．同じ出力shardへ複数workerが同時書き込みすることをfile lockで防ぎます．manifestは一時fileから原子的に置換します．

結果にはteacher名・登録済みdigest，system/prompt/schema/request hash，出典hash，元gold snapshot/hash，evidenceごとのhash，UTC時刻，実行node，shard，試行数，HTTP時間，Ollamaが返したtoken数を保存します．

`pseudo_labels_created`と`source_gold_labels_preserved_in_results`は別のカウントです．後者は結果に保存したsnapshot数で，独立したgoldレコードの総数ではありません．workerの再開やteacher変更後の結果を単純加算して，独立ラベル数としないでください．

manifestの`input_exhausted`は「そのshardで入力を最後まで確認した」という意味です．すべてがラベル化された，学習済みモデルが完成した，意味の正しさが検証された，という意味ではありません．`counts_this_run`のlabeled／unknown／abstain／failed／rejectedを併せて確認します．

## 同じ観測eventへの辞書再利用

`propagate_weak_labels.py` は，既存teacher結果を同じ記録eventの説明へ再利用し，代表jobと全trainレコードの逆mappingを保存します．**追加のLLM呼出しやgold作成は行いません**．元manifest，training_membership，全job，代表job，teacher結果とmanifestのhashを照合します．説明文・元asset・ontologyの完全一致が必要で，heldout参照や新しい意図への補完は拒否します．

```bash
python propagate_weak_labels.py \
  --source-records /absolute/path/frozen_all_records.jsonl \
  --training-membership /absolute/path/run/training_membership.json \
  --all-jobs /absolute/path/train_label_jobs.jsonl \
  --dictionary-jobs /absolute/path/unique_train_label_jobs.jsonl \
  --teacher-results /absolute/path/weak_labels_dictionary/shard-000-of-001.jsonl \
  --results-manifest /absolute/path/weak_labels_dictionary/shard-000-of-001.manifest.json \
  --output-dir /absolute/path/new_propagation_audit
```

新しい空の出力先へ `propagated_weak_labels.jsonl`，`dictionary_membership.jsonl`，`propagation_audit.json` を書きます．既存出力を上書きしません．teacherが実際に読んだ代表jobの引用IDを保ち，再利用先のevent出典を別列に保存します．例えば15件のteacher推論を225窓へ再利用した場合，独立推論数は15，trainレコードへの参照数は225です．既存eventラベルとの一致率は正規化の整合性であり，EEGから意図を読んだ精度ではありません．

## 検証と公式API

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

検証はlocalhostのfake HTTP serverだけで行います．実際のOllamaモデル・EEGデータの推論実績とは数えません．held-out拒否，出典捏造，原文引用，taskと実操作の区別，再開，shardの完全分割，gold維持，失敗回数，disk reserveを確認します．

実装に使用するAPIは，Ollamaの[Chat API](https://docs.ollama.com/api/chat)，[Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)，[List Models](https://docs.ollama.com/api/tags)に基づきます．schemaをformatへ渡したうえで，返答をローカルでも再検証します．
