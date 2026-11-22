# Zenodo8089820：S Macでの完全取得・固定分割・train信号QC実測（2026-11-23）

## 完了した実作業

- smac05が対象ZIPの**27,475,466,178Bを全量取得**しました．実ファイル・取得DBのcomplete状態を一致照合し，**ZIP全体のSHA256 `7c611b08995f10c2d313d194e3e75c88f7792403455994d77ca37e16104fcacf`**をsmac05上で新たに再計算して元配布の固定値と照合しました．ZIPはファイル1,071＋ディレクトリ96＝計1,167エントリで，GDF694記録を含みます．ZIPのサイズと構造確認は完了していますが，今回全メンバーを再CRCしたわけではありません．過去の元配布全メンバー監査レポート（`reports/bci_8089820_acquisition_20261122_v1/`）は別の過去実行結果です．
- 以前に凍結済みの**694件のGDFソース台帳**と元著者の7群のQC注意・クロス名義完全一致1組を根拠に，信号読込前に87元グループを独立分割しました．**train56群／446記録，validation12群／96記録，test12群／96記録，review7群／56記録**です．別名義の完全一致元グループは同じfoldとし，7注意群はreviewへ隔離しました．group IDは実人数の保証ではありません．凍結manifest SHA256は`91d9c764dcf27ac7d85be5cfd90c1484d5c7d41e4d5e0c789c49049b04b461bb`．
- smac05上でtrain指定の**元GDF1件だけ**をZIPから取り出し元ソースSHAと照合，S Mac間（smac05→smac35）だけで24,527,154Bを直接ストリーム転送し，受信側実ファイルのSHAも一致確認しました．workstationやsmac31にはrawを保存していません．
- smac35で27 EEGチャンネル，native512Hz，元物理単位のheaderはµV相当と確認しました．元波形の約187.0625秒，**元信号ピーク0.010583V**は固定native上限0.1V以内．指定された既存の**因果的3–40Hz SOSフィルタと2秒warmup，200Hzへの遅延線形補間**を用いた元GDF1件の3秒窓は**61窓中61窓が固定100µVピーク・0.01µV/ch std条件に合格**しました．元信号のDC/低周波オフセットの影響によりフィルタ前ピークは約10.6mVであり，フィルタ後の合格がその元信号のオンライン性能を認証するものではありません．
- ローカルCPUの固定分割・fail-closed ZIP検証回帰テストは**11件PASS**．さらにZIPの全体SHA，メタデータsplit，S Mac→S Mac SHA転送，実GDF有限値・単位・数値QCをそれぞれ検証しました．

## 厳密な区別と保留

- 過去の元データ監査で deduplicated original EEG は**72.256858時間／693元記録**でした．これは原ソースのEEGの長さで，今回の学習で新たに見た時間ではありません．原GDFイベント時刻のレート欄が全694件で`0`なので**MI cueの正確なイベント時刻は不確定**です．教師学習のgoldとして登録しません．
- 今回S Macで確認したのは**1件のtrain GDF**だけです．残る445 train GDFの独立物理校正・raw payload・固定train QC・他sourceとの重複／deny継承・全新形状MPS backwardは未完了です．**新gradient 0時間／0steps**，CBraMod累積train**22.581667時間／27,098窓**を変更しません．
- `preflight_smac_bci8089820_zip.py`の`ready_for_train_only_physical_qc`は**学習許可でなく**，原ZIPとsource/memberの存在だけを確認した状態です．manifest側`offline_ssl_training_allowed_now=false`を保持し，安全性条件を満たすまで既存CBraModを更新しません．validation/test/reviewの信号は今回一切読み込んでいません．
- 自由意図，EEG＋判定前環境からの直接X/Y，LLM行動や実OS成功の新しい評価はありません．元保存・元収録におけるoffline非因果処理をあとからfilterで修復できません．

## 運用と保存

原波形・個別元記録の予測・モデル重みはS Macに限って保持．smac19は除外，Colabおよびworkstationでの新規raw取得／学習なし．新ソースは保留のまま，train-only物理QCとcross-release deny完了後に新protocolの別runで学習します．`freeze_smac_bci8089820_metadata.py`，`preflight_smac_bci8089820_zip.py`を公開コードとして追加．private manifest・個人別信号はGit管理対象外です．

Source： https://zenodo.org/records/8089820 ， https://www.nature.com/articles/s41597-023-02445-z
