# P300 原DAT取得・数量監査

**216ファイル，1,563,795,662 bytes を完全取得し，105原DATから7.488324653時間を確認しました．** 公開チェックサムと独自SHA256を照合し，元取得台帳の委譲216件のみを完了へ戻しています．新規学習・モデル変更は0です．

|区分|実測値|
|---|---:|
|原DAT|105記録，32 EEG，256Hz，1,121,226,995 bytes|
|原EEGサンプル|883,358,720 bytes|
|DATのアプリ状態ベクトル|234,642,160 bytes|
|DATヘッダ|3,226,115 bytes|
|派生EDF|105記録，442,566,400 bytes，追加EEG時間0|
|その他メタデータ|2,267 bytes|

時間は原DATのフレーム数÷256Hzです．全105派生EDFは原DATとサンプル数が一致しますが，宣言周波数が異なり，物理単位欄も空です．これらを独立記録として追加しません．同名対とサンプル数を確認したもので，EDF変換の全波形忠実性を証明したものではありません．原DATのヘッダ校正式による単位解釈は可能ですが，ハードウェアの独立校正や信号QCの合格とは異なります．

保存されたStimulusBeginの立上りは116,515件，SelectedTargetの非ゼロ変化は2,333件です．原コードを保持し，自由な意図や画面の実刺激時刻とは扱いません．BCI2000の通常stateは信号に対し1ブロック遅れる可能性があり，全記録の8サンプルは31.25msに相当します．補正・再ラベルは実施していません．

取得物には11のID群がありますが，[査読論文](https://doi.org/10.1371/journal.pone.0303390)のMethodsは10名です．IDと実参加者の対応確認を新しい分割の前提にし，11名の独立参加者と断定しません．READMEの著者解析除外1群は課題パラダイム不足の注意として保持します．配布上のses-train/testはPJ-Xavierの学習・評価分割ではありません．論文の言語モデルはBrown corpus由来の確率的仕組みで，今回のTransformer LLM融合と同じモデルではありません．

全ファイルはpublic S3のversionIdと内容チェックサムを固定しています．DOI v1.0.0の記述と観測時listingの同一性は断定しません．ライセンスはCC0です．

## 根拠と再現性

[データ](https://openneuro.org/datasets/ds005028)，[原形式](https://www.bci2000.org/mediawiki/index.php/Technical_Reference:BCI2000_File_Format)，[校正](https://www.bci2000.org/mediawiki/index.php/User_Reference:DataIOFilter)，[stateの定義](https://www.bci2000.org/mediawiki/index.php/Technical_Reference:State_Definition)．summary/protocolは実行時bytesを保存し，independent_accountingは記録台帳から独立に再計算しています．個別ID・元波形・入力文字列は公開集計に含めません．
