# PJ-Xavier ノード割当と実行

**実測済みのノードだけへ，空き容量とreserveを守って公開EEG候補を割り当てる，小さいstdlib実装です．** 38台の過去の合計RAMを一つのモデルが使えるメモリとは扱いません．SSHの認証情報は保持・入力せず，認証済み接続または公開鍵等でBatchModeが通る場合だけ先へ進めます．

## 実装

- `fleet.py`：inventory作成，読み取りのみのSSH preflight，決定的な容量付き割当，ノード別の配備bundle・起動command生成です．
- `node_worker.py`：`download`，`label`，`train`の種類ごとにnode-local排他lockを取得し，設定済みのコマンドだけを実行して状態を記録します．
- `test_fleet.py`：13件のテストです．容量，未確定EEGの扱い，必須companionの同居，割当の再現性，SSH設定，二重起動防止，終了コードと完了状態の違い，quarantineがあるときの未完了判定を確認しています．

実装したことは，クラスターへの配備成功や，モデル学習成功を意味しません．実機へのdispatchはroot controllerが担当します．

## 手順

```bash
python fleet.py template --output /data/xavier/nodes.json

python fleet.py preflight \
  --inventory /data/xavier/nodes.json \
  --output /data/xavier/nodes_observed.json

python fleet.py plan \
  --manifest /data/xavier/catalog_queue.jsonl \
  --inventory /data/xavier/nodes_observed.json \
  --output /data/xavier/fleet_plan.json

python fleet.py bundle \
  --plan /data/xavier/fleet_plan.json \
  --output /data/xavier/fleet_bundle \
  --acquire-code /path/to/acquisition/acquire.py
```

`dispatch_commands.json`はargvと読みやすいshell表記を含みます．自動実行は行いません．同ファイルの順番で，専用ディレクトリ作成，割当manifestとコードのupload，worker起動を実行できます．SSHコマンドには必ず次を指定します．

```text
BatchMode=yes
StrictHostKeyChecking=yes
ConnectTimeout=10
ConnectionAttempts=1
```

テンプレートに含めるのは既知の`smac31`，`smac01`，`smac06`，`smac11`だけです．全て初期状態は`unverified`で，容量は割り当てません．`smac31`のSSH targetは既存の`smac` aliasを使います．実測情報の有効期限は既定24時間です．過去の38台という情報から稼働ノードを補完しません．

preflightはCPU，RAM，Python版と，専用ディレクトリを置くファイルシステムの空き容量を読み取ります．GPUの学習速度や適合性を検証したとは記録しません．`gpu_training_verified`はfalseのままです．

## 割当の基準

`include_provisional`，`component_only_provisional`，および対応する日本語監査の「頭皮EEG候補」「EEG成分のみ候補」を対象にします．`unresolved`はファイル名が`.edf`等でもreview queueへ残します．非ヒト・頭蓋内等の対象外と，重複行は取得割当から外します．取得対象であっても，利用許諾，波形の妥当性，ラベルや被験者の重複を検査し終えた学習例とは数えません．

割当可能容量は各ノードで次のように計算します．

```text
free_bytes − reserve_bytes − reserved_pending_bytes − preprocessing_headroom_bytes
```

全ファイルのサイズが分かる場合は，まずdataset単位で同じノードへ置きます．収まらない場合はファイル群に分けますが，BrainVisionの`.vhdr/.eeg/.vmrk`，EEGLABの`.set/.fdt`，分割archiveは同一ノードに置きます．独自ファイル名のcompanion参照はヘッダー検査が別途必要です．manifestのbyte数が不明なら，HEAD等で容量を測定するまでblockedです．1つのファイル群がどのノードにも収まらなければ，容量を合算して入ったことにはせずblockedにします．

同じinputとinventoryなら同じ割当になります．ノード順・カタログ行順を変えても割当digestは同一です．`priority`を行に指定すると，小さい値のデータを先に割り当てます．同順位では大きいdatasetから処理し，IDで同順位を解消します．

## 実行状態

各ノードの専用ディレクトリには次を置きます．

```text
assigned.jsonl
acquire.py
node_worker.py
downloads.sqlite
download.json / label.json / train.json
download_state.json / label_state.json / train_state.json
download.log / label.log / train.log
locks/download.lock / label.lock / train.lock
```

SQLiteはnode-localです．同じSQLiteへ複数workerを並行投入する運用に依存しません．同じ種類のworkerの二重起動はOSのfile lockで防ぎます．別ノードのSQLiteをNFSで共有しないでください．

| 状態 | 意味 |
|---|---|
| `ready` | コマンドと前提を指定済みです． |
| `running` | このrunnerが子プロセスを起動しています． |
| `paused` | プロセスは正常終了しましたが，完了を示す証拠がそろっていません． |
| `complete` | ファイル台帳または指定artifactが完了条件を満たしています． |
| `capacity_blocked` | reserveを維持できないため，開始または継続を止めました． |
| `dependency_blocked` | 必要な先行jobが完了していません． |
| `blocked` | workerやモデル等の設定が不足しています． |
| `failed` | 実行したコマンドが失敗しました． |
| `already_running` | 同じkindの排他lockが別workerに保持されています． |

`download`でexit codeが0でも，SQLiteに`paused`や未取得行があれば`complete`にはなりません．取得ジョブは既定600秒の区間で実行します．root controllerが状態と空き容量を再確認し，同じjobを再起動することで続行できます．

`label`と`train`は，実際のモデル・入出力・完了条件がない段階で，動いているふりをしません．生成bundleではblockedです．AIラベルworkerを接続するときは，元の正解とAI補助ラベルの区別，教師モデルと版，ラベル根拠，confidence，splitを明記する実装を指定してください．同じノードにGPUラベル処理と学習を重ねるかは，root controllerがメモリ実測に基づいて判断します．

ラベル worker の現在の CLI では，`--source-records` が必須です．次は接続時の形式例で，生成 bundle の blocked 状態を自動解除するものではありません．

```bash
python ollama_label_worker.py \
  --jobs /absolute/train_label_jobs.jsonl \
  --source-records /absolute/frozen_all_records.jsonl \
  --output-dir /absolute/weak_labels \
  --model qwen3.5:2b --shard-index 0 --num-shards 1 \
  --max-jobs 100 --timeout-s 90 --min-free-gib 1 --node-id workstation
```

`source-records` は train／val／test をすべて含む元の flat manifest JSONL です．`id`，`dataset`，`split` を必須とし，subject／session／EEG reference 等も原列のまま保存します．job 内の `source_manifest_sha256` はこの全ファイルの bytes hash と一致する必要があります．job の `record_id` と，任意の `source_record_ids` 全件が元 manifest の train に属することをラベル worker が確認します．汎用 node runner の非空ファイル検査だけを label／train の完了根拠にせず，job ID，出力・checkpoint の hash，status の整合を別途照合してください．

## Colabと継続実行

Colab inventoryのtransportは`workstation_colab_cli`とし，`runtime_id`とworkstation CLIで確認した空き容量を記録します．SSH preflightや直接のColab connectorは呼びません．生成bundleのupload・起動・checkpoint回収はworkstation CLIで行います．

この実装はProセッションの継続やColabランタイムの存続を保証しません．スケジュールの再開時には，保存された台帳とartifactを読み，停止・取得制限・容量不足・期限切れ認証を区別して再開します．
