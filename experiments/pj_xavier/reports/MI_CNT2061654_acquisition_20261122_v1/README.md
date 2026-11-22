# 3クラス運動イメージCNT取得・数量監査

**CC0の完全release 156,181,084 bytes を取得し，26 CNT／4 ID群の6.863233333時間を確認しました．** 14 scalp EEGと2 EOGを分け，EEGは172,953,480 bytes，EOGは24,707,640 bytesです．ZIPのMD5・SHA256，全memberのCRC・SHA256を検証しました．未圧縮全体202,503,631 bytesをEEG・EOG・ヘッダ・イベント・末尾メタデータへ排他的に会計しています．

## 形式の曖昧さをどう解決したか

全26記録で旧NumSamples欄が実バイト数と一致せず，continuous_seconds欄は0でした．MNE自動形式判定の失敗を保存し，16/32bitを都合のよい時間から選んでいません．[公式EEGLAB Neuroscan readerの固定commit](https://raw.githubusercontent.com/sccn/neuroscanio/91f74b2878d847b91f12b1fc51042a4b87f71a90/loadcnt.m)が参照する原ファイルnextfile+52の精度タグを全件確認し，すべて0＝int16でした．タグ位置とイベント終端も一致しています．

この原タグと波形領域のバイト数からサンプル数を算出し，250Hzで時間を求めています．さらにMNE 1.13.2へ精度を明示し，先頭・中央・末尾の校正済みサンプルを全26件で独立に照合しました．元ヘッダを書き換えたり，イベントを切り詰めたりしていません．ヘッダの校正値との一致は生理学的な信号品質の保証ではありません．

## 学習への適格性

元イベント2,383件をそのまま監査しています．コード1/2/3等を左手・右手・足へ対応させる根拠は未確定です．タイトルから意味を作らず，現時点の教師あり学習適格性はfalseです．SSLについても新たな14ch単位・QC・参加者分割の契約が必要です．今回の取得で既存モデルへ投入した時間は0です．

[配布v1](https://doi.org/10.6084/m9.figshare.2061654.v1)，[公式API](https://api.figshare.com/v2/articles/2061654/versions/1)，[Neuroscan precisionの注意](https://www.fieldtriptoolbox.org/faq/preproc/dataformat/neuroscan/)．summary/protocolは実行時のbytes，independent_accountingは原台帳からの再計算です．
