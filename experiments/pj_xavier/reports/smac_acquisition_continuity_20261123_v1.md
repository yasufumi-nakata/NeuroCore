# S Mac 全量取得の継続管理と既知 retry 回復

## 結果

2026-11-23 05:18:11 JST に，S Mac 内の取得 guardian が本番モードで稼働し，37 台すべての既存取得 window を監視していることを確認しました．追加 worker 起動は 0 件で，既存 DB・部分ファイル・各ノード 1 stream を維持しています．候補全体の取得・学習は継続中です．

既知の空ファイル処理で停止していた 177 件は，通常 worker による各 1 回の追加試行で **177 / 177 件が complete に回復**しました．実ファイルが通常ファイルであること，0 byte であること，空内容の SHA-256 が一致することを検証しました．期待サイズ，source ID，status，試行回数を手作業で書き換えたり，raw を手作業で作成したりしていません．

## 保存量の観測

以下は **2026-11-23 04:54:24.724636 JST** のファイル存在・サイズ照合付き snapshot です．guardian heartbeat とは観測時刻が異なります．TB は 10^12 byte です．

| 項目 | 観測値 |
|---|---:|
| 計画ファイル数 | 598,556 |
| 計画の既知バイト数 | 23,071,349,438,764 |
| complete ファイル数 | 217,567 |
| complete 実バイト数 | 2,256,602,081,121 |
| partial 実バイト数 | 660,304,934,929 |
| complete + partial | 2,916,907,016,050（約 2.917 TB） |
| 稼働 supervisor / データ保存ノード | 37 / 37 |
| complete のファイル欠損 | 0 |
| complete のサイズ不一致 | 0 |
| 最小ノード空き容量 | 3,423,100,162,048 byte |

complete の照合は存在と期待サイズの検証です．この snapshot では全 payload の再ハッシュは行っていません．partial は取得途中であり，学習投入可能量には数えません．

## guardian の起動と 48 時間 window

guardian は 2026-11-23 **05:15:17.071599 JST** に S Mac 内で切り離した process として起動しました．初回 heartbeat は **05:15:20.257302 JST**，05:18:11 JST の再確認で process 生存と singleton lock 保持を確認しました．本番状態は `apply=true`，`phase=monitoring`，`window_active=37`，既存 window 採用 journal は 37 件，追加起動は 0 件です．

- 監視間隔は 300 秒，各 worker の実行枠は従来どおり 48 時間です．guardian 自身の監視予算は 30 日です．
- 枠終了後に，現行 host・command・storage policy・DB 行数・割当 manifest SHA-256・許可済み worker code SHA-256 を再確認します．
- 新しい枠を作る前に，bulk / download lock，STOP marker，未取得件数，retry の eligible 時刻，HTTP 429 cooldown，100 GiB + 64 MiB 以上の空き容量を確認します．
- 稼働中の取得は採用し，二重起動しません．DB と部分ファイルを引き継ぎ，既存 retry / lease / provider backoff を維持します．
- 技術的失敗・configuration drift・STOP は理由付きで停止します．アクセス条件付き source や検証失敗を complete とみなしません．各新規 window の根拠と期限を journal に記録します．
- workstation の SSH・VPN 接続に依存せず，S Mac 側で監視を継続します．30 日の監視予算満了時に未処理があれば，その状態を明示して終了します．

本番初回 heartbeat では新しい 48 時間枠を必要とするノードはありませんでした．将来の枠更新は policy tests と live dry-run で経路を検証しており，48 時間経過後の本番更新が既に発生したという意味ではありません．

## 既知 retry の修正と検証

対象は `experiments/pj_xavier/acquisition/acquire.py` と `test_acquire.py` です．既存の空ファイル HTTP 検証を使い，既に retry の対象時刻を迎えた `expected_bytes=0` かつ既知 `FileNotFoundError` の行だけを選択順の先頭に置きます．WHERE 条件，retry_at，status，attempts，source identity，期待サイズは変更していません．

修正済み code は 37 台すべてへ atomic 配置し，旧 SHA と新 SHA を確認しました．全台の worker を強制中断せず，通常の次 cycle で反映しました．回復確認は **2026-11-23 05:15:13.527440 JST** で，各 177 件の attempts 増分はちょうど 1，期待サイズは全件不変でした．空ファイルなので，この回復によるデータ量の増分は 0 byte です．

実施した focused checks：

```sh
python3 -m unittest discover -s experiments/pj_xavier/acquisition -p test_acquire.py -v
python3 -m unittest discover -s private/xavier/smac_execution_20261122/acquisition -p test_acquisition_guardian_v1.py -v
git diff --check
```

acquisition 35 tests，guardian policy 10 tests が PASS しました．guardian は既存 lock，STOP，容量不足，429，未来の retry / lease，技術失敗，DB 件数・manifest・code の不一致，complete と外部条件の区別を検証しています．本番起動前の live dry-run も 37 台を既存 window と認識し，追加起動 0 件でした．

## seal と根拠

| 対象 | SHA-256 |
|---|---|
| 承認済み全量 plan | `155a332f2336913abab7b37052cf0148b6a58a563454147fef4d303e1f5a29f3` |
| guardian configuration | `9146c30fdeb4edcf7dcdc696fab306e965c9e371ef9aac54d4c544ef55ab97bd` |
| 各ノード割当 manifest / 期待件数の固定値 | `8506077990ba950a5ae80af69feed9b6ac7efad03315e1296ae65549a42d0e56` |
| acquire.py | `44a69cead0290622023de00f894a574890f15e0a3c743c4c57d712b2f5ed29e8` |
| test_acquire.py | `dbe8297b7537263f43fb5a14aa455460c85f7e98d5a5101e705730cdca3a5180` |
| guardian private implementation | `6d965d569c8131fc8cb530fe47cc2e7b1e5e1dfed71cde3f6c3dafa8cdb24fda` |
| resume helper | `86554109eb9b7216ffd5ea5ddd1334b2535f341b9dc1fe808f7bdbafc2121fcc` |
| guardian policy tests | `8c0178a7c1cb3fabce289eeaea24223055f659b88f1bbfd23d4cfec1233b7f93` |
| 177 件の回復確認 receipt | `044b9184ff0517385ba8f7fcf30307086b34df271fba54687ae61bda5a2d076f` |
| 起動後 heartbeat receipt | `bcb70915b7489c1775cf5720689c4b169791cf9a4f078323c99dc840c14588a5` |

個別 ID・URL・認証情報・参加者情報を含む詳細は private の運用 receipt に保持しています．guardian 自体と実行ノード対応表も private 管理です．既知 retry の回復後も，provider のサイズ不整合，404，アクセス条件，TLS 検証などの別件は残っています．これらは一次 metadata と照合し，取得可能なものと具体的な外部条件を区別する対象です．TLS や公開条件を緩めていません．
