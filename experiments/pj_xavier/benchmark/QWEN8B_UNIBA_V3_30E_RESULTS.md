# Qwen8B＋EEG：UniBa v3 自然操作 cohort

**固定した 30 epochs／最大 3,600 秒の学習予算で，validation による best 選択と，同一 checkpoint の 5 条件最終評価を実行しました．自律エージェントという目標は未実証です．**

実際の終了 status は completed_epochs，完了 epoch は 30，optimization step は 3000，所要時間は 1340.03 秒です．事前上限と実際の完了回数を分けて記録します．

## 固定した test 203 件／8 人

| 条件 | forced accuracy | forced balanced accuracy | forced macro F1 | confidence 0.8 採用率 |
| --- | ---: | ---: | ---: | ---: |
| fused | 43.35% | 35.77% | 0.3107 | 0.99% |
| context_only | 19.70% | 33.74% | 0.1159 | 0.00% |
| eeg_only | 39.41% | 32.52% | 0.2819 | 0.99% |
| eeg_shuffled | 42.86% | 35.37% | 0.3015 | 0.49% |
| eeg_time_shifted | 44.33% | 36.59% | 0.3113 | 0.99% |

5 条件は同じ fused checkpoint に対する入力 ablation／negative control です．別々に学習した unimodal baseline ではありません．confidence gate を保持した strict 指標と，全件で最大確率ラベルを採る forced-choice 指標を分けています．

## AI 送信対他操作：履歴 baseline との比較

以下は事前指定の AI probability 0.5 による binary 主評価です．class 選択後に confidence 0.8 を適用する操作採用 gate とは別の記述 endpoint です．

| 条件 | balanced accuracy | recall | specificity | AUROC | average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| ai_vs_other:history_pre_eeg | 0.7041 | 0.7436 | 0.6646 | 0.7844 | 0.5009 |
| ai_vs_other:history_pre_eeg_plus_context | 0.7298 | 0.7949 | 0.6646 | 0.7933 | 0.5100 |
| ai_vs_other:history_pre_decision_guarded | 0.8311 | 0.9487 | 0.7134 | 0.8885 | 0.5589 |
| ai_vs_other:history_pre_decision_guarded_plus_context | 0.8659 | 1.0000 | 0.7317 | 0.8870 | 0.5072 |
| fused | 0.5000 | 0.0000 | 1.0000 | 0.6215 | 0.3517 |
| context_only | 0.5823 | 1.0000 | 0.1646 | 0.5823 | 0.2216 |
| eeg_only | 0.5000 | 0.0000 | 1.0000 | 0.6080 | 0.3030 |
| eeg_shuffled | 0.5000 | 0.0000 | 1.0000 | 0.5782 | 0.2704 |
| eeg_time_shifted | 0.5000 | 0.0000 | 1.0000 | 0.5355 | 0.2132 |

1,000 回の paired participant-cluster bootstrap と全条件間差を公開 aggregate に保存しました．8 人の固定モデル・固定抽出集合に条件づけた区間であり，学習反復や新規ユーザー全体の保証ではありません．同じ test 予測を再集計し，追加 inference や test による閾値調整をしていません．

## データと解釈

- train 400 窓／18 人，validation 155 窓／6 人，test 203 窓／8 人です．観測された AI 送信はそれぞれ 80／31／39 件です．
- v3 は同じ原記録から AI 送信を優先保持した enriched case-control 集合です．758 個の非重複 1.75 秒窓は 0.368472 時間で，v2 と source hours を加算しません．
- 過去のキー／マウス操作を自然なまま保持したため，履歴だけで予測可能な部分，motor/EMG の交絡，EEG の追加寄与を分ける必要があります．
- test の AI 比率 19.21% は日常発生率ではありません．precision，average precision，calibration はこの集合に限られ，false actions/minute は未測定です．
- 元 EEG の offline filtering と 4.125 秒の将来支持仮定は実 XDF から検証できていません．オンラインの因果性，潜在意図，実ハーネスのタスク完了は未実証です．
- Smol1.7B の 3 epochs と今回の最大 30 epochs は学習予算が異なります．性能差を parameter 数だけに帰属しません．

## 保存

checkpoint SHA-256: b7c8e8f37a1f9276cc58b35609a6e21191faacc138a6f37e37296976be2ed924．protocol SHA-256: 3c0c702d66024df9dee302536f6a643ad6397a1584936f63b532d29a5e87b95b．公開結果は qwen3_8b_uniba_v3_30e_aggregate.json に保存しています．全予測，checkpoint，source，raw-shift 配列と strict 証拠は hash 検証済みの workstation private archive に回収しました．

データ公開元: https://zenodo.org/records/20801101 ．
