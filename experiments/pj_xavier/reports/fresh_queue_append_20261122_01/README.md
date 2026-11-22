# 公開 EEG 候補の取得台帳追加結果

**2026年11月22日 00:55 UTC の検証時点で，206 release・438,834 file jobs を既存の取得台帳へ追加しました．** 既存の 29,206 file jobs は pinned metadata を全件照合し，一致を確認しました．取得 worker の再起動，期限・転送量予算の変更，既存 raw ファイルの移動・削除は行っていません．

## 確認した数量

| 項目 | 結果 |
|---|---:|
| 全カタログ行 | 5,165 |
| カタログ ID の欠落／余分 | 0／0 |
| 公開ファイル URL を解決済みの行 | 1,515 |
| うち既存 EEG scope が provisional の行 | 612 |
| 別台帳で管理する seed | 2 |
| 既存 bulk release | 403 |
| 今回追加した release | 206 |
| 保存キーの長さにより保留した release | 1 |
| 今回追加した file jobs | 438,834 |
| 追加分の公開サイズ既知の合計 | 11,158,312,031,743 bytes |
| 追加分のサイズ未確定ファイル | 38 |
| 既存 bulk file jobs | 29,206 |
| 既存 bulk の公開サイズ既知の合計 | 4,662,735,278,723 bytes |
| 追加後の bulk file jobs | 468,040 |
| 追加後の公開サイズ既知の合計 | 15,821,047,310,466 bytes |
| 独立した EEG 記録時間 | 未確定 |

**ここでの byte 数は取得候補ファイルのメタデータ上の合計です．** 取得済み容量でも，EEG 信号だけの容量でもありません．multimodal の付属データ，複数版，異なる公開経路にある同一内容を含み得ます．provider のページ一覧の完全性，波形の同一性，被験者・セッション・raw／加工版の重複は，この追加処理では確定していません．EEGlass と AI-assisted coding の2 seed は別台帳のため上表の bulk 合計へ含めていません．

## 追加分の provider 別内訳

| 台帳の provider 分類 | release | file jobs | 公開サイズ既知の合計 bytes | サイズ未確定 |
|---|---:|---:|---:|---:|
| OpenNeuro | 125 | 430,548 | 10,637,994,071,977 | 0 |
| landing resolver 経由 | 56 | 4,309 | 162,429,195,194 | 38 |
| OSF | 19 | 3,251 | 201,786,679,406 | 0 |
| Kaggle | 4 | 684 | 48,103,740,505 | 0 |
| DANDI | 1 | 37 | 107,315,450,285 | 0 |
| Figshare collection | 1 | 5 | 682,894,376 | 0 |

landing は元カタログ側の分類です．実ファイルの配信元は release 内の file metadata に保持しており，この表を単一の配信サービスの数とは解釈しません．既存 scope の provisional 判定を引き継いだ取得候補であり，この分類によって全ファイルを EEG 波形や学習利用可能データと認定していません．

## 既存行・取得制限・容量の保持

追加専用 CLI は，固定した plan の hash を再確認し，release ごとの SQLite transaction で既存 size・checksum・URL・provider file identity を検査します．既存 downloads 行の UPDATE／DELETE は SQL authorizer で拒否しています．取得 worker が変更する進捗・lease・状態を元へ戻す処理はありません．

実機検証では，既存 29,206 行と追加 438,834 行の pinned metadata がすべて plan と一致しました．既存 bulk worker PID 8814 の識別と launch receipt の hash も維持されています．当初の deadline は epoch 1795322496.632661，max-bytes は 64,062,645,375 です．新しい worker を起動して取得枠を増やしていません．

検証時の空き容量は 81,993,601,024 bytes，reserve は 37,580,963,840 bytes（35 GiB）でした．候補全体はこの容量を大きく超えるため，今回の環境で全件本体の保存が完了したとはいえません．実転送は既存 worker の容量・時間・転送量制限内で継続します．

Dataverse の DOI `10.7910/DVN/K5EDB4` は，公式 dataset API と取得時の応答から Guestbook 217 の回答要件を確認し，342 ファイルを `access_action_required` に保留しました．個人情報を作成して送信したり要件を回避したりしていません．HTTP 400 一般を取得禁止へ変更する処理でもありません．[公式 Data Access API](https://guides.dataverse.org/en/latest/api/dataaccess.html)

## 未解決範囲の再分類候補

対象範囲が未解決の 2,925 行を review queue に保持しました．全カタログのうち `scope_decision=unresolved` の行であり，公開 URL の解決済み行だけを数えた値ではありません．

| 公式説明・題名からの review 優先度 | 行数 |
|---|---:|
| 保存済み公式説明が EEG データを明示 | 231 |
| 題名が EEG／BCI を示唆するが公式説明の精査が必要 | 1,902 |
| 情報不足 | 778 |
| 除外候補や異種データを示す語との矛盾あり | 14 |

たとえば catalog `000001` は，[Zenodo 19440997](https://zenodo.org/records/19440997) の公式説明が小児 EEG の SSVEP 実験データを明記しているため，review の優先候補へ残しました．元の scope は未解決のまま保持しており，自動で学習を承認していません．拡張子だけで EEG 波形とは判定していません．

catalog `000352` は元データを捨てず，保存キーが portable filesystem component の上限を超えるという具体的な理由付きで保留しました．全カタログの audit 行と元の immutable snapshot に残っています．

## 再現・検証資料

- 実装：[append_public_queue.py](../../acquisition/append_public_queue.py)
- 実行手順：[acquisition README](../../acquisition/README.md)
- 公開集計：[summary.json](summary.json)
- 追加処理の fixture：10件 PASS（scratch と workstation の両方）．
- 取得エンジンの fixture：30件 PASS．
- 公式取得条件の release 保留 fixture：2件 PASS．
- 別 agent が，単一 appender と既存 worker の並行運用に関する transaction・metadata guard を read-only で確認しました．別の importer を同時稼働させる運用は今回行っていません．

元カタログ CSV SHA-256：

`142a7e8da1ca3c8439082cf3af768cab9ffcdb0f967c003f86dd5ee71f1ee5f8`

固定 snapshot SHA-256：

`ef5aa95b592bc88c45588c9fe6ea9d39bf776bc724ceaa91150fc9e55d547a84`

plan SHA-256：

`8c75d6814fb24e57a9fa5f8a21c69771bd1fa38a6a95e22852b3788d5b2a46df`

apply receipt SHA-256：

`3393da1c68a09a4ce1e3af27200970dd839fef47b1da444cebacc83102454754`

詳細 snapshot・ファイル URL 一覧・review 全行・実行 receipt は workstation の private 領域に保存されています．公開集計は取得候補と実完了を区別しており，今回の追加による学習承認や独立 EEG 時間の推定は行っていません．
