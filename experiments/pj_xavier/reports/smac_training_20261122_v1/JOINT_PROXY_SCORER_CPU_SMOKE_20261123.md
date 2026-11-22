# EEG＋判定前環境→X/Yの固定proxy評価器（2026-11-23）

## 完了範囲

`next_multichannel/score_joint_alignment.py` を新設し，実際に観測されたA/B cueから，事前固定された4種類の環境配置を通して合成X/Yを定める**proxy**評価器を実装しました．この作業は新しいEEG学習・実OS操作ではなく，独立未使用holdoutの予測が生成された際に比較を検証する評価・集計基盤です．

- ソース側で事前凍結したEEG trial SHA・must-link component・val/test割当・元A/Bラベルを持つ独立registryを必須にし，registryの外部固定SHA256との一致がなければ拒否します．予測ファイル自身のgold，splitを信用しません．
- 同一EEG trialに4種類の環境配置を各1回のみ認め，検証には6シナリオ（joint，環境のみ，EEG除去，record内shuffle，実EEG −10秒，正解クラスrouting上限）で完全に同じtrial/layout群を要求します．各介入に別々の記録SHAが必要です．
- 環境配置の幾何が固定配置と一致すること，未来情報・oracleが入力へ混入しないことを検査します．自由意図gold，実OS実行は明示的にfalse／nullでなければ拒否します．
- must-link component単位のbalanced accuracy，NLL，Brier，confidence 0.8のcoverage/accuracy，context-onlyとのpaired group bootstrap（事前seed）を実装しました．4環境表示を4独立EEG trialと数えません．
- 介入receiptの**実行内容・真正性**，sourceのonline因果性，原cueとEEG時刻の整合はスコアラーでは証明できません．必ず別の凍結実行ハーネス／元データ証拠で監査します．T0/自然idle/OS操作はこのproxy指標に含めず，別protocolで検証します．

## 実行・検証

- workstation：新規scorer単体15テストPASS，既存next_multichannel/P300関連23テストPASS（全38）．workstationでは実データの取得・学習なし．
- S Mac：smac35の隔離コードディレクトリ `~/pj-xavier/smac35/benchmarks/joint_proxy_scoring_20261123_v2` にコードだけ転送し，SHA256一致とtask venvの15テストPASSを検証．S Mac側に `test_receipt.json` を保存．
- 合成テストfixtureのmodel BA=1／context-only BA=0.5は仕様検証用に人工的に指定した数値であり，実際のEEGモデルの性能として報告しません．

## データ取得・学習の並行観測

2026-11-23 02:14 JST頃にsmac31を通して37台のDBとファイルの存在・サイズを再照合しました．完了 **179,003ファイル／1,677,468,786,313 B**，部分 **483,485,762,153 B**，supervisor **37/37**，complete欠損／サイズ不一致／snapshot errors **0**．ここで全ファイルSHAを再検証した意味ではありません．候補の約23.07TBは完了量でもEEG-only量でも学習済み時間でもありません．

今回の**新規gradient=0窓／0steps**です．既存CBraMod系統の学習済み固有量は直近のBDF追加後で**27,098窓／22.581667時間**です．固定806023 validation MSE最良0.001426294970とBDF別群validation（固定5群・327窓）での変化22.68％は過去runの評価であり，本scorerの新しい行動性能ではありません．

smac05のZenodo8089820アーカイブは今回も部分取得中（最初の照合時点17,879,271,520 B／27,475,466,178 B）で，新しいtraining corpusへは投入していません．SSVEP9970619はphysical unit・original acquisition causalityの根拠不足，Zenodo2392979は固定QC不適格のため保留です．rawはS Macのみで，smac19とworkstation/Colabへのraw取得・新規学習は除外しています．

## 残件

実モデルからの検証済みX/Y予測と介入receiptを生成し，完全未使用holdoutでこのスコアラーを実行する必要があります．その際，元cue観測・時刻のgoldと自由意図goldを混同せず，実MPS train，negative controls，T0，安全な実機試用を別々に記録します．
