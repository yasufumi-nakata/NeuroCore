# Figshare 9970619：metadata-only splitの固定とtrain専用数値診断（2026-11-23 03:18 JST）

## 結果

S Mac `smac02` にある取得済み44件のEEGLAB `.set`/`.fdt`ペアを用い，**元ファイル名の11グループtokenを信号読込前に7 train／2 val／2 testへ固定**しました．内訳はtrain28記録／18,320秒，val8記録／5,273秒，test8記録／5,224秒です．各時間は200Hz・73chのFDT長に基づく**名目記録時間**であり，重複排除後の学習適格時間ではありません．11 tokenは実人数が独立に認証された11人という意味ではありません．元の`.set`ヘッダー44件はSHA256を計算しましたが，validation/testの`.fdt`信号は読み込んでいません．

凍結manifest SHA256 `5a896a2d8031846743dfd8f80379672ecf8879f77f5226d7d8ebf195d40f9f70` を入力として，再現性のあるtrain専用診断をS Mac上で実行しました．**train28 `.fdt`について全ファイルSHA256を計算し，train内で同一内容の重複は検出されませんでした．** 全28記録で有限の数値sampleを確認しました．名義をまたぐval/testとのpayload重複は，未読込のため未解決として残しています．

| 診断項目 | 実測 | 意味 |
| --- | ---: | --- |
| train28記録の元数値絶対値中央値（記録間中央値） | 4,626.26 | 物理電圧ではない未校正数値 |
| 元数値絶対値99％点（記録間中央値） | 17,872.43 | 物理電圧への換算不可 |
| チャネル内中央値差引後の絶対値99％点（記録間中央値） | 1,336.02 | 計算上の比較であり前処理採用ではない |
| common-average差引後の絶対値99％点（記録間中央値） | 14,621.51 | 計算上の比較であり参照設定認証ではない |
| 新規gradient／fit | **0窓・0steps** | 単位・cross-ID must-link未解決のため保留 |
| validation/test信号読込 | **0／0件** | train専用の診断だけを実施 |

**重要：** これらはFDT保存値のdimensionless数値です．µVともVとも断定していません．数値の倍率・baseline・clipping・閾値を恣意的に変更して学習投入していません．原著（Minami・Azuma・Nakauchi，PLOS ONE，2020）はBioSemi ActiveTwo 64ch，512Hzでの収録，EEGLABで200Hzへ連続データのdownsample，80µVのartifact rejectionを記載していますが，配布FDTの保存値と物理Vの**検証済み変換式**はこの情報から確定できません．また，元配布のoffline resamplingは後段の因果filterで修復できず，online harnessへの適格性はありません．元のevent欄は空であり，cueや自由意図goldを発明しません．

## S Mac専用の実行証跡

- 元取得：`smac02:~/pj-xavier/smac02/raw/figshare-9970619-1`，88ファイル取得済み．
- 凍結manifest（個別名義IDを含むため非公開）：`smac02:~/pj-xavier/smac02/inspection/ssvep9970619_20261123_group_split_v1/metadata_split_v1.json`，ファイルSHA256 `028ba99acdf9ccdf230278e611d52bb3046294c1de5c2644e89623575eac4c6e`．
- train専用数値診断（非公開）：同ディレクトリ `train_only_numeric_v2.json`，ファイルSHA256 `5a36631c2cfcdf4569d052ff474c4429decba94e40abbeea7b0bd248a4da71bd`．実コードSHA256 `a0d98c888c81a8396d94806a3b3d884c4dc56ede5ae4c3c8e79d440e5591c699`．
- public source code：`next_multichannel/freeze_smac_ssvep9970619_metadata.py`，`diagnose_smac_ssvep9970619_train.py`．metadata SHAの不一致，test/valの誤参照，rawパス逸脱，ホストsmac19を拒否します．CPU回帰12件PASS．

## 並行取得の実測

**2026-11-23 03:17:35 JST**，smac31を経由して既存37ノードのDBと実ファイルサイズを照合しました．完了192,660ファイル／**1,920,312,507,448B**，partial **553,929,410,805B**，生存supervisor **37/37**，complete欠損／サイズ不一致／snapshotエラー **0**．これは全modalityのbyte数で，EEGのみの容量でも学習時間でもありません．

CBraMod系統の既存train **27,098窓／22.581667時間**を保持し，今回の凍結監査では新規学習0です．smac19を除外し，workstationへrawを転送・回収していません．

## 次の再開条件

配布者による物理電圧校正の根拠，信号・参照の来歴とcross-ID must-link判定，独立holdoutとの重複防止を確定してから，新しいimmutable offline SSL protocolを発行しtrainだけ追加します．これらがない状態で単位を推定してgradientへ投入しません．online harnessは別の因果性を満たす生波形・時刻証明が必要です．

- 一次論文：https://doi.org/10.1371/journal.pone.0235309（2020-07-02）
- データ：https://doi.org/10.6084/m9.figshare.9970619.v1
