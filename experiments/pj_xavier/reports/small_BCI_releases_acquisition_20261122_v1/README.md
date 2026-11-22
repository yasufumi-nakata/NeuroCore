# 小型3公開releaseの全ファイル・波形監査

**114ファイル，216,099,696 bytesを完全取得しました．** 元の容量予約による一時停止を保全し，他の起動済み取得が完了した後にその予約のみを解放しました．元の06:25:26 UTC期限と総受領上限を延ばさず再開し，全公開checksum・独自SHA256・前後不変性を照合しています．新規モデル学習・推論は0です．

|source|配布全体|現物の確認結果|時間の意味|
|---|---:|---|---|
|MI-BMPI Figshare26893396 v2|48 files，75,523,098 bytes|16記録対，8 ID群，13 EEG，128Hz，1,688 epochs|保存epoch合計1.641111111時間．原連続時間や区間unionではありません|
|SSVEP-exoskeleton Zenodo2392979|36 FIF，110,071,616 bytes|12 ID群，8 EEG＋1 stim，256Hz|配布FIFの連続波形2.521284722時間．元GDFは未取得です|
|Starlab LED Zenodo1226586|30 TXT，30,504,982 bytes|5 ID群，5列の整数table|公称250Hz換算1.295405556時間．列対応未確定のため確定EEG時間へ足しません|

## MI-BMPI：二重形式を一度だけ数えます

全16 HDF5 SETとMATLAB EEGの全要素を，宣言shape・軸順・float32精度込みで比較して一致を確認しました．16 labels MATの数は各epoch数と一致し，原コード1/2を保持しています．SETとMATを別のEEGとして二重計上しません．1epochは448/128＝3.5秒で，最後のサンプル時刻xmaxと混同していません．

eventとepoch provenanceは全件空です．原連続記録の絶対区間・epoch重なりが不明なので，保存epoch時間を独立した原時間と呼びません．明示振幅単位と数値1/2の意味対応も未確認です．[API](https://api.figshare.com/v2/articles/26893396/versions/2)はCC BY4.0ですが，説明文には商用研究の許可を求める記載もあるため，両方を保全し，将来の商用利用適格性を未解決として分けています．今回行ったのは研究用取得・監査です．[原著](https://doi.org/10.1007/s00521-024-10917-5)．

## SSVEP-exoskeleton：配布FIFと元コードを区別します

全36件で実FIFバッファをサンプル境界と照合し，EEG scalar 97,760,256 bytes，stim 12,220,032 bytes，tag/metadata 91,328 bytesへ分離しました．MNEのヘッダ宣言校正によるV変換は物理的妥当性や信号QC合格を意味しません．全値のfinite確認と宣言変換による振幅範囲を private ledgerへ保存しています．

原資料はGDF/OpenViBEのコードを記載しますが，現在の配布物はFIFで，原stimには1〜4が見られます．その変換対応を推測して周波数ラベルへ割り当てません．著者が注意する5記録のprotocol不良を将来適格性として保持し，取得量から自動削除していません．日時候補の一致件数は0で，timezoneの同一性は独立に証明していません．原記録中に外骨格が操作されたデータとは扱いません．[Zenodo](https://zenodo.org/records/2392979)，[原著](https://arxiv.org/abs/1501.03227)．

36記録のうち異なるsubject ID間に7組の完全一致EEGがあり，量は29個の独立した配布波形へ重複排除しています．6組はファイル全体SHAも一致し，1組はヘッダ等が異なります．全7組で宣言校正・実stim配列・日時も一致しました．重複排除後のEEG scalarは74,355,712 bytesです．7 IDを3つのmust-link連結成分へ束ねる制約をprivateに封印し，今後のtrain/testに同じ波形を跨がせません．全体で8つとなる分割制約群は，8名の実参加者を確認した意味ではありません．

## Starlab LED：原列の意味を推測しません

全30 tableを読み，行数・5列・整数・元token bytesと区切り文字を照合しました．[原著](https://doi.org/10.1371/journal.pone.0218771)はO1/Oz/O2の3 EEG，250Hz，右耳朶参照，AsTeRICS記録を説明しますが，この説明だけでは5列の対応と整数尺度を特定できません．最初の3列をEEG，残りを参照／マーカーと決めつけず，EEG固有のsample bytesをnullにしています．保存された波形候補tableは存在しますが，その公称rate換算時間は別欄です．原列・イベント・振幅を変更していません．[Zenodo](https://zenodo.org/records/1226586)．

## 再現性

summary/protocolは実行時bytesです．independent_accountingはprivate記録台帳からサンプル数と分数を別計算しています．h5py3.16.0だけを追加し，既存NumPy2.5.5／Torch2.15.0／SciPy1.18.1／MNE1.13.2のversion不変を前の封印receiptと照合しました．原データ，既存モデル，固定cohortへの変更はありません．個別ID・生波形は公開集計に含めません．
