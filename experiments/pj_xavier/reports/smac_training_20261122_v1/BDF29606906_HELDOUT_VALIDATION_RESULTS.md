# Figshare 29606906：BDF未学習5群の固定validation（2026-11-23）

## 凍結済みモデルの比較

**S Mac smac35上で，別の新規学習を行わず，BDF未学習validation5群の327個の3秒窓を用いて2つの固定checkpointを評価しました．** 結果は以下の通りです．

| 同じBDF validation 5群（327窓） | 固定mask波形再構成MSE |
| --- | ---: |
| BDF追加学習前：P300までのcheckpoint | 0.0012622528629746984 |
| BDF追加学習後：新checkpoint | **0.0009759836634570468** |
| 相対変化 | **22.6792％減少** |

BDF元データの20記録は信号読込前に train14／validation5／test1 に凍結してあります．今回は**validation5記録だけ**をS Mac上で展開し，元BDFヘッダーµV→V校正，実MNE読込，因果的3–40Hz処理と固定100µV／0.01µV QCを学習用と同じコードで実行しました．出力配列5件，327窓，実MPSで2つの固定モデルを評価しました．旧parentのtrainに同一元ファイル／物理波形／被験者群が入っていないことを事前チェックしました．test1記録は未読込です．

独立した読取検証で，validation5群，327窓，source BDF5件，test未読込，fit_steps0，元runのcheckpoint SHAと成果物SHAの一致を確認しました．

- 評価前checkpoint SHA256：`9448e34ea1549d406a14b1a3e3eae5603a2ff7882008156d0937e4edb482cc8c`
- 評価後checkpoint SHA256：`f2864fc4dba584f6ccda232a43b000f51d15c5124d7cb22a6ca9f95a0b6e8503`
- 固定評価config SHA256：`c6182cbc6c884b62cd6c668117e0bf66dc99e11698cd75484e22ba8045396b1b`
- 評価ソースSHA256：`437529321279b8e0eff801f9f295bee3feb66132a531c09242904d1f50133418`
- S Macの評価結果SHA256：`2903c1241a3d9d9ba3b167e713c0d80e17d391c8536d22c267f98665fc8bf976`
- 実行先：`smac35:~/pj-xavier/smac35/evaluations/bdf29606906_fiveval_20261123_v1`

**解釈上の制限**：これは元の学習に入れていないBDFのvalidation5群でのoffline masked EEG再構成の診断です．同じ5群を今後の学習用やcheckpoint選択に再利用しません．独立test，集団汎化，脳波からの自由意図，EEG＋判定前環境からのLLM/行動出力，実OS操作成功，オンライン因果性や統計的有意性は示していません．既存のP300 validationとは別のBDF validation結果です．

学習元の[BDF追加run](BDF29606906_ADDITIONAL_SSL_RESULTS.md)と，今回の[固定validationソース](../../next_multichannel/evaluate_smac_bdf29606906_validation.py)を参照してください．raw，予測，個人識別情報，checkpoint本体はS Macからworkstation/GitHubへ移していません．
