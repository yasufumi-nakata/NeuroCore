# SmolLM2 1.7B：v3 の履歴対照との比較

**M1 で固定した best checkpoint の test 予測を，EEG 不使用の履歴 baseline と比較しました．追加の inference・test による再学習・閾値調整は行っていません．**

共通の test は 203 窓／8 人で，観測 AI 送信 39 件と他操作 164 件です．以下は事前指定の AI probability 0.5 による binary 主評価で，model の confidence 0.8 の行動採用 gate とは別です．

| 条件 | balanced accuracy | recall | specificity | AUROC | average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| ai_vs_other:history_pre_eeg | 0.7041 | 0.7436 | 0.6646 | 0.7844 | 0.5009 |
| ai_vs_other:history_pre_eeg_plus_context | 0.7298 | 0.7949 | 0.6646 | 0.7933 | 0.5100 |
| ai_vs_other:history_pre_decision_guarded | 0.8311 | 0.9487 | 0.7134 | 0.8885 | 0.5589 |
| ai_vs_other:history_pre_decision_guarded_plus_context | 0.8659 | 1.0000 | 0.7317 | 0.8870 | 0.5072 |
| fused | 0.5000 | 0.0000 | 1.0000 | 0.6165 | 0.2932 |
| context_only | 0.5000 | 0.0000 | 1.0000 | 0.5823 | 0.2216 |
| eeg_only | 0.5000 | 0.0000 | 1.0000 | 0.5231 | 0.2240 |
| eeg_shuffled | 0.5000 | 0.0000 | 1.0000 | 0.6198 | 0.2590 |
| eeg_time_shifted | 0.5000 | 0.0000 | 1.0000 | 0.6137 | 0.2682 |

1,000 participant-cluster bootstrap の各区間と paired difference は smollm2_1_7b_uniba_v3_history_aggregate.json に保存しました．8 人の固定抽出集合・固定モデルに対する記述的な不確実性で，学習の再実行ばらつきや新規ユーザー全体の保証ではありません．

v3 は AI 送信を優先保持した enriched case-control 集合です．precision，average precision，calibration を日常発生率へ変換しません．no-action の独立注釈がないため false actions/minute は未測定です．潜在意図の gold，実ハーネス完了，オンライン前処理の因果性も未実証です．

Smol1.7B の学習は 3 epochs で，別 Qwen8B 実験の最大 30 epochs とは予算が異なります．差を parameter 数だけの効果としないでください．M1 の実行成功と，true instantaneous MPS memory peak の未測定は区別します．

checkpoint SHA-256: 281716da86293d2b700ec003c967ca2c141515cc565b136a12137c34b73a3475．元 report SHA-256: e1d32bfbc462f6fcb21c7118f73a0db1e78940f320da2eee7580bfb6d88d4026．データ公開元: https://zenodo.org/records/20801101 ．
