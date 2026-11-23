# PJ-Xavier：Qwen3.8-27Bに対するtrain-only文脈ベースライン（2026-11-23）

## 結論

**定型的な事前イベント文脈の単純な学習済み多数決（EEGなし，LLMなし）は，同じ155件の固定validationでQwen3.8-27BのEEGあり／なし両モデルより記述的な正答率が高くなりました．** ただし被験者はvalidationで6人に限られ，検定で差は確定していません．この実験は次の記録操作カテゴリを予測する限定試験であり，潜在意図の解読，画面条件付きVLM，実OS操作を実証するものではありません．

| 条件 | Accuracy | Macro-balanced accuracy | `ai_prompt_submit` recall |
|---|---:|---:|---:|
| **事前文脈別train多数決（今回）** | **71/155＝45.81%** | **38.17%** | 0/31 |
| Train最多クラス固定（今回） | 62/155＝40.00% | 33.33% | 0/31 |
| Qwen3.8-27B，固定EEGテキスト＋文脈（既完了） | 57/155＝36.77% | 30.65% | 0/31 |
| Qwen3.8-27B，文脈のみ（既完了） | 62/155＝40.00% | 33.33% | 0/31 |
| Qwen3.8-27B，EEG被験者内循環置換（既完了の負対照） | 64/155＝41.29% | 34.41% | 未独立再計算 |

今回のCPU baselineは新しいtestへの外挿ではありません．予め固定された自然行動データの `train400`（18被験者）だけで，各文脈テキストに対する最多操作ラベルを学習します．未知文脈ではtrain全体の多数ラベルを返します．同じ固定 `val155`（6被験者）を一度評価し，test203は読みません．Qwenと同じ，EEGを除いた事前情報のJSONLを利用します．入力のEEG量子化文字列・特徴量は受け付けません．

Trainとvalidationで観測された定型文脈は各5種類で，validationに未知文脈はありませんでした．正解ラベルの内訳はtrainが `mouse_click=160, keyboard=160, ai_prompt_submit=80`，valが `62,62,31` です．今回のベースラインの予測分布は `keyboard=127, mouse_click=28, ai_prompt_submit=0` で，依然として重要なクラスを識別できていません．単純に多数クラスを返すQwen noEEGの40.00%と比較し，ベースラインの改善は9/155件です．

## 対応付き比較と不確実性

- 文脈別多数決 − EEGありQwen：+9.03ポイント．6被験者cluster bootstrap（10,000回）の記述的95%区間は+4.76〜+13.34ポイントですが，少数群と評価済みvalidationに由来する後付けの解析です．独立6群の全64通り符号反転に基づく両側の探索的p=0.125．
- 文脈別多数決 − noEEG Qwen：+5.81ポイント．同参考区間+2.58〜+10.83ポイント，探索的p=0.125．
- これらの信頼区間は6群の有限再標本化による条件付き区間で，集団一般化・有意差の確証ではありません．今回のベースライン追加自体が既観測validation後の診断であり，論文の独立最終評価には使用しません．

## 実装と監査

- 公開可能な再実行コード：`baselines/context_event_lookup.py`（stdlib only）．4つの人工fixture test：`baselines/test_context_event_lookup.py`．source/ラベル分離，群分割の独立性，EEG特徴入力拒否，未知文脈のfallbackを検証します．
- 今回の数値はS Mac smac35で実計算（CPU，GPU学習0）し，公開版コードを同じS Macに読取り専用投入して結果 `71/155`・`38.17%` が一致することを検証しました．原データ，個票，参加者ID，予測文字列はcluster外へ出しません．
- S Mac側の私有集計：`~/pj-xavier/smac35/context_only_event_baseline_20261123_1700_v1/aggregate.json`．SHA256 `253871e23cdbe6609e5e247dd02f22595c97c9eb93e6d8d483174fa726f3b247`．分割400/155・学習者18/検証者6・train/val文脈5/5・未使用testが記録されています．
- 本実験の入力は既存のQwen noEEGセット（train sha `ec71ac2deebcbdc877a25504aa6d8fa171c272d500dff0f4eeacdfbf125fe943`，val sha `7c8c07c68d2df14ed700fed1df471b42b519e37d8d426bfdea95fc256b9574f3`）です．新たなvalidationによる学習・ハイパーパラメータ選択・train+val refitはしていません．

## 次の研究課題

重要なのはLLMサイズではなく，**正当な同期EEG・画面／履歴と適切な意図／行動のground truth，および被験者・セッション単位の保持と強い非EEG対照**です．既存Qwenは単一クラス偏重，少数の文脈しか持たず，予測器として弱いです．新しいEEG利用手法の比較は，train-onlyで固定した強いcontext baseline，predecision EEGのshuffle/time-shift，独立した未使用評価集団が揃うまで改善と見なしません．

従来の `baselines/cpu_baselines.py` は別のsplit／cohortと一部testの結果であり，**この400/155小規模Qwen比較に混算しません**．
