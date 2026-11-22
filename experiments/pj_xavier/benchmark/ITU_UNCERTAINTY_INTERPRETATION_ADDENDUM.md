# ITU 外部評価の不確実性表示に関する補足

**ITU 外部評価の participant bootstrap は，25 個の site-qualified observation groups を再標本化しています．共有関数の説明文に残った「8 subjects」は UniBa 用の定型文であり，ITU の計算に使った群数ではありません．**

対象は，既存 3 モデルと履歴基準の 19 条件・各 630 件の外部評価です．計算結果と原 JSON は変更せず，この補足から参照します．

| 根拠 | 内容 |
|---|---|
| [`itu_external_three_models_aggregate.json`](itu_external_three_models_aggregate.json) | SHA-256 `058d8a6255b57057520262206ffae1663cf2aba9baa978195249defee4258e86` の保存済み集計です．実計算の `n_subjects` は 25 です． |
| [`itu_external_completion_audit.json`](itu_external_completion_audit.json) | SHA-256 `524a00074112fd696b1ea9e8a2a405d16f4f562b26135d6126495c81831aa94d` の独立 QA です．630 件・25 observation subjects を確認しています． |
| [`history_baselines.py`](history_baselines.py) | `clustered_comparison` が受け取った group ID から再標本化し，`n_subjects=len(set(subject_ids))` を記録します．末尾の `limits` 文字列だけに UniBa 用の「8 subjects」が残っています．凍結済み source は変更していません． |
| [`score_external_predictions.py`](score_external_predictions.py) | ITU manifest の site と subject を結合した group ID を渡し，固定 seed の 1,000 回 bootstrap を行います．8 群への間引きは行いません． |

この 25 群は，ITU の記録上の participant grouping です．UniBa と ITU の匿名 participant label に共通部分があるため，両 site の実在する人物が完全に別人であるとは証明していません．1 つの外部 site だけを評価しているため，未知の site 間のばらつきを推定する gate は `blocked` のままです．

信頼区間は，固定済みのモデルと，AI 送信を多めに含む case-control sample に条件づけた値です．再学習のばらつき，日常利用時の出現率，連続使用時の誤作動／分，EEG による課題達成を表すものではありません．4 番目の SSL モデルの追加集計でも，実際の 25 群とこの定型文の違いを明示します．既存 3 モデルの推論・閾値・数値は再計算によって置き換えません．
