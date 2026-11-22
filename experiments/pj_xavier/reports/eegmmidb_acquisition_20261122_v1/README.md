# EEGMMIDB 1.0.0 の追加取得・全記録監査

GitHub EEG-dataset の catalog ID 009861 を公式公開配布から取得しました．**1,526 記録・174,724 秒（48.534444 時間）**の EEG 保存時間を，実 EDF header と payload 長から確認しました．109 は公開元の subject group 数です．他データセットとの参加者同一性を検証した人数ではありません．

| 量の定義 | 実測値 |
| --- | ---: |
| ZIP 配布ファイル | 1,997,435,063 B |
| ZIP 内の全 regular member の展開サイズ | 3,614,967,988 B |
| EDF ファイル全体 | 3,613,708,800 B |
| EEG の二-byte sample 部分 | 3,560,112,128 B |
| EDF annotation payload | 27,813,376 B |
| EDF header | 25,783,296 B |
| EEG のスカラー sample 数 | 1,780,056,064 |
| 完全同一 EEG payload による重複記録 | 0 |

配布サイズ，展開サイズ，EEG sample bytes は同じ内容の異なる表現なので，合計して追加データ量とはしません．記録時間には64チャンネルを掛けません．他サイトの別名コピーは今回取得しておらず，元の同じ recording から作った学習窓や epoch 回数も加算しません．

## 検証範囲

公式 SHA256SUMS の全 3,058 member と，ZIP 内 SHA256SUMS 自体の公式取得物との一致を確認しました．RECORDS の1,526 EDFがすべて含まれ，ZIP の全3,059 regular memberを読み切って CRC／SHA を検証しました．ZIP 全体の SHA-256 は **0ff5f7f38d1d14b8e485581ad4cf5b327e6494bc2ac7c7d7d4f7e9518a950cbe** です．ZIPを全量展開せず，1 data recordずつ読み，元データを変更していません．

全EDFの普通信号チャンネルは64で，同じ電極名・順序，header上の単位はuVでした．この64信号をEEGと数えるのは，公式の64 scalp EEG＋annotationという release 固有の説明と電極 header を根拠にしています．任意のEDFの普通信号をEEGとみなす規則ではなく，装置の物理校正まで独立検証したものでもありません．

公式説明の標準160Hzに対して，実 header は **160Hzが1,490記録，128Hzが36記録**です．計算には各記録の実際の sample数／record時間を用い，全件を160Hzに置き換えていません．全1,526件がEDF+Cで，保存時間は37〜181秒，record数を不明値から推測したファイルは0でした．

## ラベルの利用判断は別途必要です

| run に対応する課題 | 記録数 | 保存時間（h） | annotation review対象 |
| --- | ---: | ---: | ---: |
| baseline_eyes_closed | 109 | 1.846667 | 1 |
| baseline_eyes_open | 109 | 1.845833 | 1 |
| motor_execution_both_fists_feet | 327 | 11.188056 | 12 |
| motor_execution_left_right_fist | 327 | 11.228611 | 12 |
| motor_imagery_both_fists_feet | 327 | 11.214444 | 12 |
| motor_imagery_left_right_fist | 327 | 11.210833 | 12 |

T0/T1/T2 は各runの課題で意味が変わります．運動の実行と運動想起，片手と両手／両足を区別し，T1を常に同じ行動と扱っていません．自由な意図や，エージェントに実行してほしい具体的な操作のラベルではありません．

厳密な注釈区間チェックでは48記録にフラグがあり，重なり547件は **5〜50ms**，信号区間の外にはみ出す12件は各 **25ms** でした．小さな重なりは時刻表記の丸めとも整合し得ますが，原因や修正方法は断定していません．さらにbaseline2記録にT1があり，run固有の意味が割り当てられませんでした．合計 **50記録をannotation review対象**として私的台帳に残しました．残り1,476記録も限定したmetadataチェックを通っただけで，信号品質や将来の学習適格性を保証しません．

すべての原記録を容量・時間の集計に保持し，ラベル修正，除外，学習への追加は行っていません．独立MNE readerによる代表確認は別の監査結果として保存します．

## 実行量と再現性

取得の実所要時間は 477.511 秒，全member＋EDF監査は 23.464 秒でした．workstation上で35GiBの空き容量を残す設定と2.1GBの取得予算を使い，既存の絶対期限を延長していません．この速度は今回の観測値であり，他の配布元や将来の所要時間の保証ではありません．

監査コードは ../../acquisition/audit_eegmmidb_zip.py，fixtureは9件がlocal/workstationで通過しました．summary.json と header_label_review.json に範囲と数値，per_task.csvに排他的な課題集計，validation.jsonに実行証拠を記録しています．個別EDF台帳と注釈はprivateに保存しています．既存のSSL／alignment／外部testへは追加しておらず，新たな学習には別の固定protocolが必要です．

## 一次資料

- [PhysioNet公式データ説明・ライセンス・配布](https://physionet.org/content/eegmmidb/1.0.0/)．公開元のライセンス表記はOpen Data Commons Attribution License v1.0です．
- [公式SHA256SUMS](https://physionet.org/files/eegmmidb/1.0.0/SHA256SUMS.txt)と[公式RECORDS](https://physionet.org/files/eegmmidb/1.0.0/RECORDS)．
- [MNE公式のrunと課題の対応表](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html)．
- [EDF仕様](https://www.edfplus.info/specs/edf.html)と[EDF+ annotation仕様](https://www.edfplus.info/specs/edfplus.html)．
