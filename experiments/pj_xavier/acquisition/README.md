# PJ-Xavier EEG取得ジョブ

既存NeuroCoreのprovider resolverを再利用し，公開EEG候補と付随する画面・行動・イベント等を，途中再開できる取得ジョブへ渡します．実際の取得，波形の妥当性，学習への利用許諾，独立したEEG記録時間は別々に記録します．

## 実装した範囲

- `acquire.py`：全カタログ行の正規化，Zenodo・Figshare・Dataverse・PhysioNet・OpenNeuro・OSFの公式公開情報による解決，Range付きストリーミング転送，MD5等の公開checksumと取得後SHA-256，SQLite再開台帳，容量・時間・転送量の上限です．
- `neurocore_resolver_bridge.py`：`neurocore.materialization.resolve_remote_files`を利用し，さらに広いproviderの公開メタデータを解決します．既存の全ファイルをメモリに載せるmaterializerは大量転送に使いません．
- `build_cached_queue.py`：過去の公式API応答のSHA-256を照合し，保存済み監査キャッシュから取得候補を再構築します．現在の取得成功と区別します．
- `test_acquire.py`：30件のテストで，ゲートの除外，再開時のHTTP 200/206の違い，checksum不一致，容量停止，転送量上限，HTMLログイン応答拒否，同名異fileの保持，台帳の冪等性，manifest変更の拒否，short EOF／中断応答の再開と受信 traffic 計上，明示的な Guestbook 要件の保留を確認しています．
- `hold_access_requirement.py`：公式応答の証跡を照合し，Guestbook が必要な release の未着手・再試行ファイルだけを保留します．2件のテストで証跡の整合と既存の完了・実行中行の保持を検証しています．
- `append_public_queue.py`：更新された全カタログを immutable snapshot として保存し，既存の EEG 対象候補だけを台帳へ追加します．10件のテストで既存行の保持，版・checksum の衝突保留，名前空間，再実行，取得条件，未解決 scope の扱いを検証しています．

本ディレクトリのコードはデータや外部スクリプトを実行せず，pickle等も読み込みません．認証情報をファイルやコマンド引数に持ちません．401・403は取得制限として記録し，429は再試行時刻を記録してその実行を止めます．

Dataverse が HTTP 400 とともに「所定の Guestbook 応答が必要」と明示した場合は `access_action_required` とし，自動再試行を止めます．任意の HTTP 400 をアクセス制限へ変更する処理ではありません．既存台帳の同一 release に適用する場合は，公式 dataset API の PID・版・guestbook ID・file ID 群と取得時の応答を照合します．個人情報の捏造や Guestbook 要件の回避は行いません．[公式 Data Access API](https://guides.dataverse.org/en/latest/api/dataaccess.html)

## 起動例

以下は例です．実際の保存先と制限は各ノードの空き容量に合わせて指定してください．ノードの予約はこのツールの範囲外です．

```bash
python acquire.py seed \
  --zenodo 001430=4453396 \
  --zenodo 010914=20801101 \
  --output /data/xavier/priority.jsonl

python acquire.py download \
  --manifest /data/xavier/priority.jsonl \
  --db /data/xavier/downloads.sqlite \
  --root /data/xavier/raw \
  --reserve-gib 30 \
  --seconds 300 \
  --max-bytes 12000000000

python acquire.py report --db /data/xavier/downloads.sqlite
```

同じコマンドを再実行すると，台帳の完了ファイルを再取得せず，未完了ファイルから再開します．完了はサイズと取得後SHA-256を記録し，公式checksumがある場合は一致を必須とします．サーバーがRangeを無視した場合は，先頭から書き直します．ETag等と公式checksumのどちらもない部分ファイルは，異なる版のバイト列を継ぎ足すことを避けるため最初から取得します．

HTTP 206の応答に旧ETag／Last-Modifiedと異なる値があれば，追記を拒否します．公式checksumがない場合には，再開応答のvalidator欠落も拒否します．サイズ不明のファイルはbulk ingestで`metadata_blocked`とし，測定してmanifestを更新するまで転送対象にしません．

## 同名ファイルを保持するmanifest移行

Figshare等には，同じrelease内で表示名が同じでも，別file ID・別内容のファイルがあります．表示名を一意キーとせず，`provider_file_id`，URL，`original_path`を保持し，同名異fileを`provider_files/<provider_ID等>/...`へ分けて保存します．Macの大文字小文字やUnicode正規化による衝突も考慮します．

```bash
python acquire.py migrate-manifest \
  --input /data/xavier/catalog_queue.jsonl \
  --output /data/xavier/catalog_queue_v2.jsonl \
  --ledger /data/xavier/manifest_migration.json
```

入力と同じ出力pathは禁止し，元manifestを残します．ledgerには元・新manifestのSHA-256，旧pathと新path，remote file ID，URL，サイズ，checksumを記録します．manifestのfile entryは1件も削除しません．取得済みrawファイルも移動・削除しません．旧保存pathが台帳に存在する場合はalias履歴を保存し，完了済みrawの存在を維持します．

同名衝突があったreleaseでは`companion_resolution_required=true`とし，元のディレクトリ関係やsidecarの組合せを確認するまで読み込み・学習を保留します．archiveだから同名の別fileを捨ててよいとは判断しません．一方，同じpinned provider file IDに対するサイズやchecksumの変化は別fileとしてごまかさず，releaseをquarantineします．ほかのreleaseは継続して台帳へ投入します．

2026年11月22日のローカル台帳検証では，5,165行・50,696 file entryを保持し，2,201 entry（32 release）の保存pathを区別しました．一時SQLiteへの投入は50,696 jobs，storage path重複0，quarantine0，再投入追加0でした．これは取得台帳の検証で，50,696ファイルの本体を取得した意味ではありません．既存EEGlass＋AI-assisted codingの23ファイルはpath変更0，job IDも同一です．

すでに取得した公式Zenodo API JSONも入力できます．

```bash
python acquire.py seed \
  --zenodo-json 001430=/data/xavier/api/4453396.json \
  --zenodo-json 010914=/data/xavier/api/20801101.json \
  --output /data/xavier/priority.jsonl
```

全件の処理は次のように開始します．日本語監査CSVと英語監査CSVの両方に対応しています．行の重複判定はproviderと版の同一性に限られ，REVE・EEGDash・元データ間の記録重複を解決したものではありません．

```bash
python acquire.py normalize \
  --catalog /data/xavier/catalog_audit_snapshot.csv \
  --output /data/xavier/queue.jsonl

python neurocore_resolver_bridge.py \
  --repo /path/to/NeuroCore \
  --inventory /data/xavier/catalog_latest.csv \
  --queue /data/xavier/queue.jsonl \
  --output /data/xavier/queue.jsonl \
  --limit 50 --seconds 600 --timeout 20 --max-pages 100

python acquire.py download \
  --manifest /data/xavier/queue.jsonl \
  --db /data/xavier/downloads.sqlite \
  --root /data/xavier/raw --reserve-gib 30 --seconds 300
```

Bridgeは処理済み行を再度照合せず次の未処理行へ進みます．再試行するときだけ`--retry`を付けてください．既存resolverは全ページ完了の証明を返さないため，Bridgeのmanifestは`manifest_complete=false`とします．全件確認が終わったと数えるにはproviderごとの完全性確認がさらに必要です．

## 更新された公開候補の追加

既存 worker を動かしたまま使う場合は，全 manifest の再投入ではなく次の追加専用 CLI を使います．`plan` は更新元ファイルを変更せずに SHA-256 名の snapshot を作り，現在のカタログ CSV と ID 集合を照合します．`apply` は snapshot・plan・各添付台帳の hash を再照合し，`downloads` に対する UPDATE／DELETE を SQLite authorizer で禁止した接続から INSERT だけを実行します．

```bash
python append_public_queue.py plan \
  --source /data/xavier/catalog_fresh_resolution.jsonl \
  --snapshots /data/xavier/fresh_queue_snapshots \
  --catalog /data/xavier/catalog_latest.csv \
  --db /data/xavier/bulk_downloads.sqlite \
  --output /data/xavier/fresh_queue_plan_01 \
  --reserve-gib 35

python append_public_queue.py apply \
  --plan-dir /data/xavier/fresh_queue_plan_01 \
  --db /data/xavier/bulk_downloads.sqlite \
  --expected-plan-sha256 <plan.jsonのSHA-256> \
  --receipt /data/xavier/fresh_queue_append_receipt_01.json \
  --reserve-gib 35
```

追加対象は `resolved` かつ既存 scope が `include_provisional` または `component_only_provisional` の公開ファイルです．別台帳で管理する EEGlass・AI-assisted coding の2 seed は除外します．同一 provider file ID の公開 size・checksum・URL が変化していれば release 単位で保留し，ほかの release は継続します．既存の保存 path は変更せず，新規の同名異 file は provider ID による名前空間を使います．未知のサイズは `metadata_blocked`，公式取得要件が台帳に記録された release は `access_action_required` として追加します．

`catalog_audit.jsonl` はカタログの全行を保持します．対象範囲が未解決の行は `scope_review.jsonl` に残し，公式応答の説明に EEG データの明記がある候補，題名だけの候補，矛盾する説明がある候補を区別します．拡張子だけで EEG 波形と断定せず，この review queue から自動的に取得範囲を広げたり学習を承認したりしません．

この CLI は worker の起動・停止，deadline・転送量 budget の更新，raw ファイルの移動・削除を行いません．候補 byte 数が利用可能なストレージを超えていても，その分を取得済みとは数えません．実転送は既存 worker の時間・容量・転送量上限に従います．snapshot，詳細ファイル一覧，実行 receipt は `private/` に保存し，公開レポートへは集計値だけを出してください．

## 保存と並列化

SQLite・`.part`・rawデータ・カタログ取得台帳はリポジトリの`private/`または別のデータ領域へ置いてください．Gitへraw波形や取得台帳を追加する必要はありません．

SQLiteは各ノードのローカルディスクへ置き，NFS越しに共有しないでください．同一SQLiteを同じノード上で使う場合の二重claimはleaseで防ぎます．単一workerの同時転送は1ファイルです．複数ノードを使う場合は上位schedulerがdataset単位の割当表とノード別manifestを作り，同一providerへの集中を抑えます．このコード自体が38台へ配備されたことや，クラスター全体で同時接続数を管理できたことは意味しません．

Colabの一時ディスク上に置いたデータとcheckpointは，ランタイム終了前にworkstation等へ回収する必要があります．Colabとの接続はユーザー指定どおりworkstationのCLI経由で行います．

## 台帳の読み方

| 状態 | 意味 |
|---|---|
| `resolved` | 公開ファイルのURL等を解決済みです．波形を取得・検査した意味ではありません． |
| `restricted` | 公開条件やHTTP応答に取得制限があります． |
| `metadata_only` | メタデータのみを確認できています． |
| `unknown` | providerの未対応，応答失敗，未処理などです． |
| `excluded` | 非ヒト，頭蓋内，波形なし等の既存監査で対象外候補です． |
| `duplicate` | 同じprovider releaseの別カタログ行です．異なるコーパス間の波形重複ではありません． |

ファイル台帳ではさらに`complete`，`paused`，`retry`，`capacity_blocked`，`integrity_failed`，`metadata_blocked`，`access_action_required`を区別します．取得済みbyte数はファイル全体の量であり，純粋なEEG信号量ではありません．時間はヘッダー・チャンネル型・raw/加工版・被験者/セッションの重複を照合した後に計算します．AIが生成したラベル数をEEG記録時間へ加算しません．

## 優先データと版の注意

1. EEGlass：カタログ`001430`，Zenodo `4453396`．4ファイル，33,893,656 bytesです．raw版とfiltered版が含まれるため，2倍の独立記録時間にはなりません．公式説明のチャンネル位置は1:Nz，2:TP9，3:TP10です．この位置記載はfilteredデータの前処理欄にあり，rawファイルへ適用する際は順序も照合します．[公式API](https://zenodo.org/api/records/4453396)
2. AI-assisted coding：カタログ`010914`，concept `20120211`．保存済み公式応答が解決した版は`20801101`で，19ファイル，11,187,259,866 bytesです．UniBaとITUのEEGアーカイブ合計は5,532,757,501 bytesであり，展開後のEEG signal bytesではありません．旧版`20120212`の3.93GB・14ファイルを全体量として扱わないでください．[公式API](https://zenodo.org/api/records/20801101)
3. CHB-MIT：カタログ`007898`，版`1.0.0`．監査済み686 EDFの保存ファイル合計は45,756,994,304 bytesで，保存標本時間の合計は982.9352778時間です．取得用seedはEDFファイルのみの部分manifestです．seizure summaryや注釈もpublic treeから取得する必要があります．この数値は既存ヘッダー監査から計算したもので，今回すべてを取得したという意味ではありません．[公式配布](https://physionet.org/content/chbmit/1.0.0/)

`priority_seed_manifest.jsonl`は2026年11月20日の保存API応答に基づく歴史的seedです．実行時には上の`seed`コマンドで公式応答を再確認できます．`catalog_resolved_queue.jsonl`等の大きい台帳は再生成可能な実行成果物です．
