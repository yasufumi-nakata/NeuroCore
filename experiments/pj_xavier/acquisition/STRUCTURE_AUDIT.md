# 公開取得キューの構造監査

`audit_queue_structure.py` は，取得台帳の全 file job を **パスと拡張子の証拠だけ**で排他的に分類します．EEG を含む release から MRI・MEG・動画・刺激画像・付随ファイルまで取り込まれている量を，EEG 波形候補と分けて確認するための読み取り専用ツールです．取得済みデータ量や記録時間の計測には使用しません．

## 実行

Python 標準ライブラリだけを使用します．出力ディレクトリは新規である必要があります．

```bash
.venv-xavier/bin/python experiments/pj_xavier/acquisition/audit_queue_structure.py \
  --db private/xavier/bulk_downloads.sqlite \
  --output-dir private/xavier/structure_audit_new/aggregate \
  --private-detail private/xavier/structure_audit_new/path_classifications.jsonl.gz
```

SQLite を `mode=ro`，`PRAGMA query_only=ON`，1 つの読み取り transaction で参照します．全量を RAM に展開せず，取得処理が更新する status・lease・URL を SELECT しません．比較する列は job ID，catalog ID，release key，取得先相対パス，元パス，公開サイズです．job ID 順の canonical JSON stream の SHA-256 を保存します．

全件走査前に `structure_audit_rules_v1.json` とコードの SHA-256 を `frozen_protocol.json` に固定します．分類を作った後に都合よく対象件数を調整しません．ルールを変える場合は，新しい版として別出力へ監査し，旧結果と区別してください．

## 分類と解釈

- 元パス `original_relative_path` を優先し，ない場合だけ取得用 `relative_path` を参照します．同名ファイルの provider namespace が証拠を隠すのを避けます．
- Archive は名前に EEG が含まれていても未展開のまま分離します．`.edf.gz` のような識別可能な単一圧縮形式と，`.tar.gz` のような容器を区別します．
- 動画・音声・画像・neuroimage 拡張子を先に分離します．`eeg/` にある動画を EEG 波形として集計しません．
- `eeg`／`ieeg`／`meg` の明示パスと命名を区別し，信号形式，数値・表容器，sidecar，その他へ分けます．
- `.vhdr`，`.vmrk`，JSON，channels／events TSV 等は，波形本体とは別の companion 候補として保持します．
- モダリティを示すパスがない EDF／BDF／SET／FIF／NWB 等は，未確認の汎用形式として扱います．
- MEG の容器には同時計測した EEG が含まれる場合があります．`meg` パスを理由に EEG 不在とは判定しません．`EEGChannelCount` と channel types の確認が次の手順です．
- iEEG は頭皮 EEG と別区分です．転移学習への使用には記録方法と学習目標の適合確認が残ります．

分類は header 検証を代替しません．EEG と命名されたファイルでも，チャンネル種類，サンプリング周波数，単位，raw／派生物，必要な companion の存在は未確認です．`.set` と `.fdt` のような関連ファイルは file job として各 1 回集計しますが，同じ記録を二重に時間へ換算しません．

基準とした一次資料は Brain Imaging Data Structure 1.11.2 の EEG，iEEG，MEG，MRI 各章です．MEG 章の同時 EEG 記録の扱いも参照しています．参照日は 2026-11-22 です．公開 report にダウンロード URL や個別の波形パスは掲載しません．

## 成果物

| ファイル | 内容 |
| --- | --- |
| `summary.json` | 排他的な分類別・provider 別・拡張子別件数／known bytes／unknown size／release 数 |
| `release_review.csv` | 各 release に 1 つの確認段階を割り当てた，次の検証候補一覧 |
| `frozen_protocol.json` | 実行前に固定した分類ルールとコードハッシュ |
| private detail gzip | 全 job の分類・元パス・根拠．公開しない証拠台帳 |

分類別の **file 数と known bytes は合算可能**です．分類別 release 数は，同じ release が複数形式を持つため合算できません．別途 `release_review_groups` では release 自体を排他的に振り分け，その release 数も総数に一致させます．

公開サイズが NULL の job は unknown size として数え，0 byte は既知サイズ 0 として別に記録します．0 byte の信号形式を使用可能な記録とは扱いません．

`derivatives` パスの量は非加算の補助フラグです．raw と同じイベントを含む可能性があり，独立した学習時間とは見なしません．release の異なる版や mirror の内容重複も，この監査だけでは除去しません．

## 検証

```bash
.venv-xavier/bin/python -m unittest discover \
  -s experiments/pj_xavier/acquisition -p test_audit_queue_structure.py -v
```

8 件の fixture により，EEG sidecar と信号の区別，汎用形式の保留，MEG 同時 EEG の留保，archive と単一圧縮の区別，モダリティ競合，全 job の排他性と総和，URL 非公開，元 DB の不変性，既存出力の保全，不正サイズの拒否を検証します．実台帳でも SQL の全 job 数・known bytes・unknown size 数との一致を必須にしています．

監査によって既存キューの順序・状態・対象範囲を変更せず，新規取得，データ削除，学習承認も行いません．
