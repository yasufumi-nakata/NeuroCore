# PJ-Xavier workstation controller

既存の Colab セッションを workstation から監視し，完了した checkpoint・評価レポート・EEG 本体を workstation に回収する stdlib 実装です．本実装の検証は fixture によるもので，この subtask では既存ジョブの起動・停止・再分類を実行していません．実機への配備・起動は root 担当です．

## 固定した接続経路

| 対象 | 設定 |
| --- | --- |
| 実行ホスト | workstation Mac |
| 使用 CLI | `/Users/yasufumi/.local/bin/colab` |
| セッション | `xavier-eeg-20261122` |
| Colab 内ディレクトリ | `/content/xavier` |
| workstation 作業ツリー | `/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training` |
| 状態・snapshot 保存先 | 同作業ツリーの `private/xavier/controller` |
| raw 保存先 | 同作業ツリーの `private/xavier/colab_raw_archive` |

2026年11月22日に workstation の CLI v0.7.4 の read-only help で，`exec --session … --file … --timeout …`，`download REMOTE LOCAL --session …`，`usage`，`status --session …`，`stop --session …` の形式を確認しました．直接の Colab connector は使いません．`new`，課金，追加単位購入，CLI 更新，別セッションへの操作は実装していません．

## 配備と実行

`controller.py` と `remote_core.py` は同じディレクトリに置きます．以下は workstation で root が実行する例です．パスを空白で結合する shell 組立てに依存しません．

```bash
python3 experiments/pj_xavier/controller/controller.py config \
  --output private/xavier/controller/config.json

# 読み取り観測のみ．既存ジョブに変更を加えません．
python3 experiments/pj_xavier/controller/controller.py tick \
  --config private/xavier/controller/config.json \
  --state private/xavier/controller/state.json

# 許可済み作業を継続：reconcile，回収，条件を満たす download の再開．
python3 experiments/pj_xavier/controller/controller.py run \
  --config private/xavier/controller/config.json \
  --state private/xavier/controller/state.json \
  --apply --interval-seconds 300 --max-seconds 18000
```

既存 config を上書きする必要はありません．`config.example.json` は同じ既定値を保存した参照用です．`tick` と `run` は JSON の状態を原子的に更新し，file lock により同じ state の同時操作を防ぎます．`run` は daemon lock と PID ファイルも持ちます．1回の `run` は最大5時間の観測ループです．再起動する場合は元 launch receipt の起動時刻＋上限秒数を `controller_deadline_epoch` に固定し，`--max-seconds` にもその残り時間だけを渡してください．この絶対 deadline は CLI 操作・chunk 回収にも適用します．既存 worker の元 deadline を後ろに延長しません．

## 監視と再開の判定

Colab の PID が `/proc` に存在するだけでは実行中と判定しません．`Z`／`X` は終了済みです．`start_ticks`，起動時刻，argv の SHA-256 を保存し，同じ PID の再利用や別コマンドへの入れ替わりを検出します．最初の観測で，期待したコマンドが実際に動いていた場合のみ元の identity と deadline を採用します．すでに消えたプロセスについて起動証跡を推測して再開することはありません．runtime の boot ID が変われば，セッション名が同じでも変更操作を停止します．

download の再開には次を要求します．

- 元の process identity があり，該当プロセスが確実に終了しています．不明状態や別プロセスによる SQLite 使用中は再開しません．
- job の元4時間 deadline と，最初の観測で固定した session の5時間上限の両方に時間が残っています．
- Colab の実空き容量が30 GiBの reserve と転送用の余裕を上回ります．
- CLI `usage` による現在残高と消費率が確認でき，残り時間に必要な単位と reserve 2 units を既存残高で満たします．
- SQLite 上に未完了の転送候補があります．完了 job を再開しません．

現行 acquisition の既存 SQLite は累積受信 traffic を保存していません．そのため，元コマンドに `--max-bytes` があり，検証済みの累積 traffic receipt がない場合は自動再開を保留します．保存済みファイルのサイズを過去の再転送量の代用にしません．traffic receipt を使う場合は現在の process identity と一致させ，一度の再開で消費済みにします．deadline と reserve は引き続き必須です．

train／label の再起動はこの controller の対象にしていません．同じ run を再学習して完了済み checkpoint や test 結果を上書きしません．終了した run の `run_summary.json`，`failure.json`，または中断時に残った `.pt` がある場合に成果物回収へ進み，summary／failure／interrupted checkpoint を区別します．

UniBa の追加学習 supervisor も `train_uniba.pid` で監視します．期待するスクリプトは `/content/xavier/run_uniba_training.py` です．Python 実行ファイルを推測せず，`train_uniba_launch.json` の `supervisor_command` を実際の argv・起動時刻と照合します．この起動 receipt の元 deadline と4800秒の job 上限を保ちます．

workstation の bulk worker は `bulk_launch_v3.json` を優先し，なければ v2 を監視します．v3 の PID 8814 は今回観測した値で，controller は固定 PID より receipt を使用します．v3 の deadline は v2 の `1795322496.632661` が維持されていることを read-only で確認しました．`metadata_refresh_launch.json` も監視し，4時間の元 deadline を表示します．これらの workstation ジョブは自動再起動しません．

## priority 台帳の照合

`downloads.sqlite` の `delegated_priority` を `priority_eeg.sqlite` の `complete` と照合します．次の全条件を満たす行だけ main 側を complete に更新します．

1. job ID，dataset key，relative path，URL，公開サイズ，checksum 記録が一致しています．
2. 実ファイルが指定された raw ディレクトリ内にあり，サイズと priority 側の保存済み SHA-256 が一致しています．
3. MD5 等の公開 checksum がある場合は，本体を再計算して一致を確認します．
4. 検証中に inode・サイズ・更新時刻が変わっていません．

照合結果は `priority_reconciliation_ledger` と controller state に残します．一つでも不一致なら held として保留し，ファイルを捨てず，元の status を変更しません．本 subtask では fixture のみで実行済みで，本番 UniBa の reconcile は root が controller を `--apply` で動かした時に行われます．

## 大きな raw ファイルの保存と停止条件

小型 ZIP snapshot だけでは数 GB の raw が取り残されるため，raw を独立して回収します．二つの download SQLite の **全 complete 行**から一覧を作り，同じ物理ファイルは一回だけ数えます．実ファイル欠落・サイズ未確認・SHA-256 欠落・台帳間不整合は無視せず held に残します．

原則として `colab_raw_archive/<dataset_key>/<relative_path>` へ保存します．root が先に回収した一意 basename のファイルも，実サイズと SHA-256 を確認して利用します．既存の不完全・不一致ファイルを勝手に上書きしません．新規転送は専用 `.controller.download.part` へ行い，全体 SHA-256 の照合後に，既存ファイルを上書きしない atomic hardlink で完成パスへ昇格させます．CLI 子プロセスはローカル reserve を監視し，超過しそうならその転送だけを止めます．

installed Colab CLI v0.7.4 の `download` は JSON/base64 応答をすべて受信してから保存するため，数 GB の原本を直接渡すと RAM と回収時間が問題になります．64 MiB を超える raw は，controller 専用 protocol で **最大64 MiBの派生 chunk** を remote に作り，その chunk だけを workstation CLI で回収します．installed CLI のコードは変更しません．原本は read-only のままです．

各 chunk は source SHA-256・source size・source identity・offset・size・chunk SHA-256 を照合します．workstation 側では 1 MiB buffer で追記し，`.chunks.json` に確認済み offset と pending chunk を原子的に保存します．append 後に receipt 保存前で中断しても，保存済み tail を chunk と照合して残りだけを追記します．確認済み prefix が変わった場合や，receipt のない未知の partial は上書きせず保留します．chunk の保存と receipt commit 後に，対応する派生 chunk だけを remote から削除します．最後に原本全体の SHA-256 が一致して初めて回収完了です．

chunk の進捗は state の `raw_recovery_progress` にも保存します．原本の回収中でも，確認済み byte 数と再開 receipt の場所を確認できます．v3 追加学習には `uniba_natural_v3_five_variant_evaluation` gate を追加し，`runs/qwen3_8b_aicoding_uniba_natural_v3_fused_30e_v1/variant_evaluation_completion.json` と同 run の `best.pt` を照合します．daemon は config を起動時に読み込むため，gate を追加したら元絶対 deadline を保持した再起動が必要です．

1 cycle の新規回収は既定2ファイルで，次の cycle が継続します．32 GiB の cycle 目安を超える一個のファイルも，実ディスク容量が満たせば取り残さないよう単独で回収できます．未回収ファイルがある限り，Colab の自動停止は許可しません．

`allow_idle_stop` は既定 false です．root がこれを true に設定した場合にも，次の全条件が必要です．

- 自分の固定セッションで，boot ID が変わっていません．
- 全既知 job と `/content/xavier` の追加プロセスが終了しています．GPU 使用率ゼロでも download やその他の CPU worker が動いていれば停止しません．
- GPU の idle 状態を確認でき，120秒の猶予を経過しています．
- checkpoint／report snapshot をローカルで展開検証済みで，その後変更・欠落していません．
- **最新の complete raw inventory 全件**のサイズと SHA-256 が workstation 回収物と一致しています．回収後に新しいファイルが complete になれば，新 inventory との不一致により停止を保留します．
- 全終了 run の fingerprint がローカル回収済みの fingerprint と一致しています．
- UniBa の5 variants評価の完了 receipt と実ファイルを照合し，その checkpoint と評価レポートをローカルへ回収済みです．
- 未回収 `.part` が残る場合も既定で停止を保留します．部分転送を破棄してよい方針を既に明示している場合だけ，root が `allow_stop_with_unpreserved_partials` を変更できます．complete raw の必須条件は解除できません．

一度停止した receipt があれば，次の cycle はその runtime に再接続せず終了します．Colab 側からサービス都合で runtime が終了することや，workstation のスリープ・終了，ネットワーク断，ChatGPT Pro の将来セッション起動を controller が保証するものではありません．未完了・保留理由を JSON で残し，次の担当が同じ状態から継続できるようにします．

### UniBa 評価完了 receipt

`runs/qwen3_8b_aicoding_uniba_fused_v1/variant_evaluation_completion.json` について，`status='completed'`，`exit_code=0`，5条件の `variant_names`，`checkpoint_sha256`，`report_path`，`report_sha256` を検証します．`best.pt` と report 本体の SHA-256 が一致すること，report schema が `pj-xavier-variant-evaluation-v1` で，同じ checkpoint を指すことを要求します．

5条件は `fused`，`context_only`，`eeg_only`，`eeg_shuffled`，`eeg_time_shifted` です．report の `conditions` に全条件があり，各結果が `measured` または理由を明記した `blocked` として確定していれば，研究評価の終了として扱います．これは autonomous agent の成功を認める判定ではありません．性能不足や評価上の限界が判明した場合にも，それだけを理由に GPU を保持し続けません．receipt 欠落・未完了・5条件欠落・hash 不一致の場合は自動停止を保留します．

## 検証

```bash
python -m unittest discover -s experiments/pj_xavier/controller -p 'test_*.py' -v
```

36 tests が成功しました．既存26件では zombie と PID 再利用，deadline・capacity・compute credit gate，priority 照合成功／SHA 不一致，raw の重複排除・欠落・正しい回収・不正 checksum・inventory 更新後の停止禁止，snapshot member と SHA 検証，CLI scope，既存停止 receipt の冪等性，fake CLI による一連の回収と停止，5 variants評価の完了・欠落・checkpoint変更を検証しています．追加10件では chunk 転送の途中再開，append と receipt commit 間の中断，local／remote 内容変更，chunk checksum と最終 source checksum，offset，64 MiB上限，元絶対 deadline，ack の同定，binary 削除後の ACK 中断からの復旧を確認しています．fixture のために新しい paid runtime を起動していません．実際の回収状況は private の実行 receipt と公開集計で区別して報告します．
