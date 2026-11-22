# 最終追加releaseの取得・監査

起動済みの追加取得は完了し，元台帳へ対象行だけを照合して戻しています．新規ダウンロードの拡大は停止しました．最新の取得容量と波形時間は `../reports/acquired_progress_006.json` と `../reports/verified_eeg_quantities_20261122_006.json` に分けて保存しています．新しい取得物を固定済みモデルへ混ぜた時間は0です．

## 完了した独立監査

|実装|固定したsource|主な確認|
|---|---|---|
|`audit_brainstorm_vhdr_zip.py`|Figshare5756277 v2|全archive member，10原triplet，62 EEG候補と2補助未確定信号，MNE独立校正probe，同梱ソフト例示EEGの除外|
|`audit_inmid_edf.py`|Mendeley rjx76wd5v6 v2|全237ファイル，92 EDF，14 EEGと44補助信号，MNE独立校正probe，MAT派生と動画等の分離|
|`audit_small_bci_releases.py`|Figshare26893396 v2／Zenodo2392979／Zenodo1226586|全114ファイル，SETとMATの全値一致，FIFバッファ会計，原イベント保持，5列TXTの意味未確定を明示|
|`derive_fif_duplicate_constraints.py`|上記FIFの封印済み記録台帳|7組の異なるID間の完全一致EEGと，分割前のmust-link制約を導出|
|`resume_delegated_acquisition.py`|停止済み委譲workerの同じscope|元期限・総受領上限・既存partial・単独lockを維持し，完了済み別sourceの予約だけを解放|

詳細は次の公開集計にあります．

- `../reports/Brainstorm5756277_acquisition_20261122_v1/`
- `../reports/InMID_rjx76wd5v6_acquisition_20261122_v1/`
- `../reports/small_BCI_releases_acquisition_20261122_v1/`
- `../reports/final_acquisition_validation_20261122_v1.json`

## 数量の意味

原形式の連続記録，処理済みcontinuous export，保存epoch，列対応未確定のtableを分けています．全配布ファイルの容量には補助信号・動画・metadataも含まれるため，EEG scalar bytesとは一致しません．ヘッダの物理単位変換がMNEと一致しても，生理学的品質やアンプの独立校正を確認したことにはなりません．

MI-BMPIはSETとMATを一度だけ数え，epoch絶対区間・重なりが不明なので原時間のunionへ加算しません．Starlabは3 EEGの記述に対し実際のTXTが5列で，列割当と整数尺度が未確定です．データが存在しない扱いにはせず，公称rateによるtable時間を別欄に残します．

Zenodo2392979では36配布FIFから29種類のcalibrated EEG payloadを確認しました．7組が異なるsource IDを跨ぎ，6組はwhole-file SHAも一致します．7 IDに対する3つのmust-link連結成分は，その全record/sessionを同じfoldへ置くか，曖昧な成分全体を保留するための制約です．全体の8分割制約群を8名の実参加者と呼びません．private constraint ledgerを用いずに通常のID分割を行ってはいけません．

追加releaseのラベルは，原コード・配布metadataと意味確認済みのgoldを区別します．モデルによる擬似ラベル，刺激時刻の補正，振幅の修正，既存評価入力への追加は行っていません．MI-BMPIのAPIライセンスはCC BY4.0で，説明文の商用許可noticeも併記しています．今回の研究用取得・監査と，将来の商用利用適格性を分けています．

## 環境と検証

実行したworkstation環境はPython3.12，NumPy2.5.5，SciPy1.18.1，MNE1.13.2です．MATLAB v7.3用にh5py3.16.0のみを追加しました．NumPy／SciPy／MNEとTorch2.15.0のversionが以前の封印receiptと同一であることを確認し，installerが追加したpackageもh5pyだけでした．環境全体のbyte hash不変を証明したという意味ではありません．新規環境では必要な科学計算依存を用意し，MNEの実buffer構造が固定実行版と一致するかを確認してください．

既存環境にMAT v7.3の追加が必要な場合の依存は `requirements_mat73_addon.txt` です．既存主要packageを更新する指示ではありません．

```sh
python -m unittest test_resume_delegated_acquisition test_audit_inmid_edf test_audit_brainstorm_vhdr_zip test_audit_small_bci_releases test_derive_fif_duplicate_constraints
```

上記はこのディレクトリを作業ディレクトリとして実行します．16本と重複制約2本の計18 focused testsを実workstationで確認し，元ログとsource SHAを保存しました．機能上の検証は，元deadlineと総trafficの保持，別worker予約解除，shapeと軸の違い，MAT/SET値の不一致，原ラベル数不一致，FIFの欠落bufferと実byte数，列意味未確定の保持，重複の推移的must-linkなどです．

各監査は既存complete台帳，source固定resource，原データroot，完了receipt，新規output directoryを必要とします．過去の期限切れresourceでdownloadを再開したり，既存outputを上書きしたりしません．最終006観測ではworkstationの空きが35GiBを下回ったため，追加取得と大きな一時展開を停止した状態を記録しています．原因はこの監査から推定していません．
