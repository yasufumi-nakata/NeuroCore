# v3：固定した 2 モデルと操作履歴の比較

**既存の 2 モデルは，固定した AI probability 0.5 で 39 件の AI 送信を 1 件も検出できませんでした．操作履歴の baseline が上回り，現時点では EEG を追加した価値を確認できていません．**

UniBa の同じ test 203 窓／8 人について，保存済み予測だけを比較しました．モデル推論，学習，checkpoint 選択，閾値調整の追加実行はありません．10 種類のモデル条件と 4 種類の独立して学習した履歴 baseline を含む全 14 条件を集計しています．

## AI 送信対他操作の主評価

| 方法 | balanced accuracy | 検出した AI 送信 | specificity | AUROC | average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| SmolLM2 1.7B＋EEG＋環境／3 epochs | 0.5000 | 0/39 | 1.0000 | 0.6165 | 0.2932 |
| Qwen3 8B＋EEG＋環境／30 epochs | 0.5000 | 0/39 | 1.0000 | 0.6215 | 0.3517 |
| 脳波窓開始前の履歴＋環境 | 0.7298 | 31/39 | 0.6646 | 0.7933 | 0.5100 |
| 判定直前の guard 済み履歴＋環境 | 0.8659 | 39/39 | 0.7317 | 0.8870 | 0.5072 |

ここでの 0.5 は事前指定した binary 主評価の閾値です．各 LLM の confidence 0.8 による操作採用 gate とは別です．39/39 を検出した履歴条件にも，他操作 164 件中 44 件の誤検出があります．この標本は送信イベントを多く含むよう抽出されており，日常使用時の適中率・誤操作数／分は推定できません．

## 参加者単位で対応を保った不確実性

1,000 回の paired participant-cluster bootstrap を用いました．8 人を復元抽出し，同じ参加者・窓を全条件で対応させています．固定した学習済みモデルと標本に対する記述的な区間で，多重比較の補正や学習をやり直す不確実性は含みません．

| 比較（前者 − 後者） | AI AUROC 差 | 95% 区間 |
| --- | ---: | ---: |
| Qwen fused − Smol fused | 0.0050 | [-0.0516, 0.0614] |
| Qwen fused − 同記録内 EEG shuffle | 0.0433 | [-0.0499, 0.1327] |
| Qwen fused − 10 秒前の実 EEG | 0.0860 | [-0.0113, 0.1632] |

全ての上記区間は 0 を含みます．EEG の対応を崩す対照との頑健な差は確認できません．average precision など他の endpoint は JSON に全件保存していますが，探索的な多数の比較から都合のよい指標だけを合格根拠にはしていません．

## 解釈と実行範囲

- Smol は 3 epochs，Qwen は 30 epochs で，backbone，tokenizer，学習予算が異なります．性能差を parameter 数だけの効果と解釈しません．
- LLM の context_only／eeg_only は同じ fused checkpoint からの入力除去です．独立して学習した unimodal baseline とは区別します．
- 履歴 baseline は train のみで特徴量変換と係数を学習し，validation で選択しました．test で係数や閾値を調整していません．
- source EEG は公開時点で offline filter 処理済みです．オンライン因果性，潜在意図の正解，idle 時間の注釈，実ハーネスでの自律タスク完了は未検証です．
- 原ラベル・test ID・確率は不変で，モデル間の条件名の衝突を避けるため派生予測の名前だけに prefix を付けました．原 file SHA と派生 provenance は private に保管し，公開 JSON に出典 hash を記録しています．

公開 aggregate: v3_fixed_models_history_aggregate.json．データ公開元: https://zenodo.org/records/20801101 ．
