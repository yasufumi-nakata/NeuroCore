# P300未学習被験者の固定validation（2026-11-22 21:28 JST）

## 結果と対象

S Macのsmac05からsmac35へ，OpenNeuro `ds005028` の**既存・凍結済みvalidation群に属するオリジナルBCI2000 DAT 9記録（85,040,079B）だけ**をクラスタ内で転送しました．転送前後の元ファイルSHA256を照合し，元配布のtrain用65記録とは独立した**validation被験者1群，32ch，609個の3秒窓**について，事前に固定した2つのcheckpointを比較しました．新たな学習やcheckpointの選択は行っていません．テストに指定した2被験者群のDATと著者除外群は開いていません．

| P300のvalidation（1被験者群，609窓） | 固定mask再構成MSE |
| --- | ---: |
| 追加学習直前のInMID後checkpoint | 0.001771007939769423 |
| P300原DAT追加学習後checkpoint | **0.001313861299685412** |
| 差（後－前） | **−0.00045714664008401104** |
| 変化率（同じvalidationでの相対減少） | **25.8127945%** |

**解釈上の制限**：これはoffline masked EEG波形再構成の，1被験者群に限定した事後固定validationです．P300の別被験者へ広く一般化する証拠，未使用testでの性能，統計的有意性，刺激時刻の正確性，オンライン因果性，EEG→LLM行動性能や実OS操作成功の証拠ではありません．今回確認したvalidationを後続モデル選択やtrain用に再利用しません．過去の806023 validation（327窓）とは別のデータです．

## 作業実績と検証

- `next_multichannel/evaluate_smac_p300_validation.py` を新規実装しました．凍結済みraw ledger，metadata-only split，学習済みcorpus／checkpoint／前処理sourceのSHAを検査し，P300 validationだけで固定20261122 seedのmasked再構成を集計します．smac19やworkstationでの実行を拒否します．EEGだけを読込対象とし，oracleや未来contextをモデルへ渡しません．
- smac31経由でsmac05→smac35を直接転送しました．受信データと元9 DATの全SHAが一致し，転送receiptを保存しました．workstationでrawを中継・保存していません．
- smac35の既存コードと学習済みcheckpointは変更せず，独立した`training_code_p300_eval_v1`に新コードを設置しました．新しいimmutable評価runを作成し，実Apple MPSで2つの保存済みcheckpointを評価しました．
- 初回起動は誤ったmetadata-admissionパスで，**信号読込前にexit1**となりました．失敗証拠は保存し，元学習で用いた封印済みmetadata（SHA256 `555b3247b6a9eadcf48bed22f29ef381b06a601e87909241aedcc60900c70e1d`）を指定した`config_attempt2`だけ再起動しました．2回目はexit0で完了し，元の科学ソースやQC閾値は変更していません．
- 別のread-only実装で全9個の生成配列（46,772,352B，計609窓）のSHA／形状／有限値／固定振幅範囲とcheckpoint SHAを検証しました．テストDAT読込0，追加fit 0 stepsです．
- workstation上の新規・関連CPUテストは**31件成功（13 subtests成功）**しました．個別のEEG配列，受信DAT，重み，個別予測はS Macにのみ保存しています．

### 封印済み参照先

- 新評価前checkpoint SHA256：`600f1e25e89d5229ea70e5d8a4f873b8653dfc445503850d77ff4739bc184ad5`
- P300学習後checkpoint SHA256：`9448e34ea1549d406a14b1a3e3eae5603a2ff7882008156d0937e4edb482cc8c`
- 新評価protocol SHA256：`617571bfeead3ef04a61fa40e93bfe8b4822111ef15a729f61ac08c88398b9b2`
- 新評価results SHA256：`53ee1ac4f781567089371537e72bb948486b869b3cae6489f0103a7ea987f81a`
- 新評価ソース SHA256：`82260f3f0117d243a1e6437d82f478013d652364a6a974349cca4331fa58266f`
- smac35上のprivate評価出力：`~/pj-xavier/smac35/evaluations/p300_heldout_validation_20261122_v1/eval_attempt2`

## 37台の取得も継続

smac31が2026-11-22 **21:20:40 JST**に取得DBとファイル実体を再照合したところ，smac19を除く37台のsupervisorが稼働し，**取得完了78,614ファイル／604,580,599,121B，途中129,871,796,126B，完了ファイル欠損0，サイズ不一致0，監査エラー0**でした．これは全modalityの実保存量であり，EEGだけ・unique被験者時間・学習済み時間とは区別します．BCI8089820の27.48GB archiveは当時retry中のpartialで，取得完了へ加えません．SSVEP9970619の単位未検証も解消していません．今回の学習済み固有量は以前の**21.563333hのまま**で，新しいgradient inputは0hです．

次は未学習の新sourceについて，取得完了と原物理単位，前処理因果性，重複must-linkと固定heldoutを監査してから，新corpus／protocolを固定し，MPS capacity probe→train→validation→独立testと負対照→安全な実機試用へ進めます．封印済みのphase1～3と既存P300学習runは再実行しません．
