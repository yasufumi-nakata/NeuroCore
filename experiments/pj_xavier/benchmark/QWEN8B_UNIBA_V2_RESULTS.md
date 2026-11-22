# Qwen8B＋EEG：UniBa v2 の実測結果

**3 epochs の学習と，同じ best checkpoint による 5 条件の評価・strict 検証・workstation 回収が完了しました．自律エージェントとしての目標は未実証です．**

## 5 条件の比較

全て同じ held-out 1,956 件／8 人です．下表は confidence によらず最大確率のラベルを採る forced-choice の記述成績です．同じ fused checkpoint の入力 ablation であり，それぞれ別に学習した baseline ではありません．

| 入力条件 | Accuracy | Balanced accuracy | Macro F1 | confidence 0.8 の採用率 |
| --- | ---: | ---: | ---: | ---: |
| EEG＋環境 | 57.21% | 39.54% | 0.3737 | 0.00% |
| EEG を除く入力 ablation | 55.88% | 39.31% | 0.3502 | 0.00% |
| 環境を除く入力 ablation | 45.25% | 33.33% | 0.2077 | 3.58% |
| 同一記録内の EEG shuffle | 57.00% | 39.42% | 0.3720 | 0.00% |
| 原記録から実抽出した 10 秒前の EEG | 55.78% | 38.68% | 0.3622 | 0.00% |

fused の採用率は 0% です．固定された abstention を誤りとして数える strict balanced accuracy／macro F1 は 0 で，実行すべき操作を信頼して選べることは確認できていません．forced-choice 成績を採用済み操作の成功率と混同しないでください．

## EEG の追加情報

strict 指標とその差は固定の abstention を保持して JSON に保存しています．別途，固定 confidence gate を外した記述的な participant-cluster bootstrap を計算しました．この補助解析は partial score 閲覧後に追加したもので，選択や学習の再実行には使っていません．

| fused minus 対照 | Macro F1 の差 | 95% participant bootstrap 区間 |
| --- | ---: | ---: |
| EEG を除く入力 ablation | +0.02354 | [+0.00848, +0.03821] |
| 同一記録内の EEG shuffle | +0.00171 | [-0.00700, +0.00946] |
| 原記録から実抽出した 10 秒前の EEG | +0.01145 | [+0.00616, +0.01781] |

shuffle は同一 participant／session／recording 内の固定点なし置換です．time shift は，取得済み BrainVision 原記録の実際の 10 秒前を抽出し，source hash，時刻，sample interval，実 array bytes を独立照合しました．

## 学習量とリソース

- 入力は 8 電極×256 Hz×448 samples（1.75 秒）です．train 4,357，validation 1,506，test 1,956 窓で，重複しない選択窓の合計は 3.800903 時間です．元の記録時間と，実際に使った窓時間を分けます．
- 3 epochs／1,635 optimization steps を完了しました．停止履歴を保持した attempt 2 の訓練時間は 1332.96 秒です．best は各 epoch の validation だけで選択しました．
- 事前学習済み Qwen3-8B を凍結して EEG encoder／prefix connector を学習しています．出力候補は記録された 3 操作で，任意の OS 操作を自律実行できることを検証したモデルではありません．
- 実機 Colab の resource gate は passed です．GPU peak と別 CPU RSS を aggregate に保存しています．同じ 8B モデルの M1 実行可能性を示しません．
- 推論時間は candidate scoring と accelerator synchronization の範囲で，EEG 前処理やハーネス実行を含む end-to-end latency は未測定です．

## 未解決の条件

- AI 送信の正例は train 2／validation 2／test 1 件で，この class の成功率・汎化性能は推定できません．
- 公開 EEG は offline filter 済みです．4.125 秒の将来支持という仮定と実 binary の対応，元 XDF の因果性は未検証です．
- no-action／idle の独立注釈がないため false actions/minute は未測定です．
- 実ハーネスのタスク完了，潜在意図，連続運用，利用者本人への汎化は未検証です．
- coarse environment は直前の実験 page と AI availability です．画面動画を vision encoder へ入れて学習した実験ではありません．

## 保存と再現性

checkpoint SHA-256: 29f349c199bd99232a2431f83179085e136a1509beb9c84d5b25dd25edba21f7．manifest SHA-256: af878638c2fff6d6c7f619cd22005a057e797f1152553517bf05ce795aec2eac．公開集約は qwen3_8b_uniba_v2_aggregate.json です．全予測，checkpoint，raw-shift array，strict report，source code，実行 receipt は検証済み private archive に保存しています．事前学習済み 8B backbone weights は connector checkpoint に含まれず，固定 revision を別途読み込みます．

データ公開元: https://zenodo.org/records/20801101 ．集計元 report と recovery archive の SHA は公開 JSON に記録しています．
