# v3 履歴対照と AI 送信判定

## 結果

EEG を使わない履歴対照の学習・validation 選択・固定 test 評価を実行済みです．400 train 窓／18 人，155 validation 窓／6 人，203 test 窓／8 人を使いました．test には観測された AI 送信 39 件と，キーボード／マウスの他操作 164 件が含まれます．CPU 2 threads，処理時間 10.35 秒です．

| binary 主評価の入力 | balanced accuracy | participant bootstrap 95% 区間 | AUROC | average precision | recall | specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EEG 窓より前の履歴 | 70.41% | 63.19–75.49% | 0.7844 | 0.5009 | 74.36% | 66.46% |
| 同履歴＋coarse context | 72.98% | 66.26–78.68% | 0.7933 | 0.5100 | 79.49% | 66.46% |
| guarded decision より前の履歴 | 83.11% | 77.02–88.39% | 0.8885 | 0.5589 | 94.87% | 71.34% |
| 同履歴＋coarse context | 86.59% | 81.13–90.54% | 0.8870 | 0.5072 | 100.00% | 73.17% |

以上の 4 条件は事前に固定した対照で，test 成績による再学習・閾値調整はしていません．binary 判定閾値は 0.5 です．最後の条件は 39 件の送信を全件検出する一方，他操作 164 件中 44 件も送信と判定しました．このサンプル内の precision は 46.99% です．

**この結果は脳波から意図を読めたことを示しません．** 過去のキー／マウス操作だけでも次の観測操作をかなり予測できるため，EEG を加えたモデルには同じ test 集合でこの対照を超える追加情報の検証が必要です．EEG なしの履歴モデルの結果を EEG モデルの性能として記載しないでください．

## 範囲と不確実性

これは AI 送信を優先的に採った enriched case-control 集合です．test 内の送信比率 19.21% は日常利用の発生率ではありません．precision，average precision，Brier score はこの抽出集合に限られ，日常利用の precision，false actions/minute，タスク完了率には変換できません．他操作は idle のラベルでもありません．

区間は 8 人の held-out subjects を単位とする 1,000 回の paired cluster bootstrap です．同じ被験者の窓をひとまとまりに再標本化し，条件間差も同じ draw で算出しました．学習の再実行ばらつきは含まず，多重比較補正をした優越性の証明でもありません．39/39 という recall の bootstrap 区間が 100–100% になるのは観測済みの誤りがないためです．未知の被験者でも感度が 100% だと証明する区間ではありません．

v3 の EEG は v2 と同じ原記録の再抽出です．v2 と v3 の元 EEG 時間を加算してデータが増えたとは扱いません．公開 EEG の offline filtering，仮定した 4.125 秒の将来支持，motor/EMG 混入，オンラインの因果的利用可能性の未検証は残ります．ハーネス実タスクの完了，潜在意図，連続利用の誤作動は未測定です．

## 固定した分析

- `protocol.history_uniba_v3.json` を test 推論前に固定しました．元 cohort protocol `d4e6c5f6…` と manifest `030e8550…` を照合します．
- 共通の `baselines/cpu_baselines.py` は変更せず，SHA `1b74445e…` を指定して import します．training membership と全 32 prepared EEG array の bytes/hash を M1 run の ledger に照合します．
- L2 logistic regression を binary 主評価と 3-class 副評価に別々に fit します．C は 0.001，0.01，0.1，1，10，100 から validation balanced accuracy，次に macro F1，最後に小さい C で選びます．train＋validation での再 fit はしません．
- 入力は 1／5／15／60 秒以内の過去 event 数，直近 event からの時間，source/missing indicator だけです．coarse context は前の実験 page と AI availability，その組合せです．subject／session／record ID，絶対時刻，target，GPT events，キー文字，座標，duration は学習入力にしません．
- 不明な直近時間は固定上限 60 秒と missing indicator で表し，scaling と category vocabulary は train のみで fit します．
- 各モデルは固定 test を 1 回だけ `predict_proba` します．出力先の再使用は拒否し，selection artifact を書いた後に test-consumed marker を作ります．
- 履歴の厳密 cutoff，future onset，追加された禁止 feature，矛盾する missing/count，GPT 出典を拒否します．この baseline 自体は元 TSV から履歴を再構築せず，検証済み prepared manifest の SHA に結び付けます．

## コマンド

```sh
python history_baselines.py run \
  --protocol protocol.history_uniba_v3.json \
  --expected-protocol-sha256 0324dbbbc10706738723265b36d3478c0982714fdf3ab4597feb20eefcaa7cf5 \
  --expected-source-sha256 5be7b7c4e91f6776895e2454adbec2ba12bfba1548e572e69b61ce7b9704ea13 \
  --common-module ../baselines/cpu_baselines.py \
  --manifest PREPARED_V3/manifest.jsonl \
  --reference-run FROZEN_M1_V3_RUN \
  --output NEW_PRIVATE_OUTPUT
```

LLM の test 予測が確定した後，追加推論せず既存予測と履歴対照を同じ集合で集計できます．

```sh
python history_baselines.py score-predictions \
  --manifest PREPARED_V3/manifest.jsonl \
  --expected-manifest-sha256 030e8550bd6f4c79005722eb617e9b64f5f80e87b77866b7f17bec4e7e2a1093 \
  --predictions PRIVATE_HISTORY_RUN/test_predictions.jsonl \
  --predictions PRIVATE_FROZEN_LLM_VARIANTS/predictions.jsonl \
  --output NEW_PRIVATE_AI_COMPARISON.json
```

## 保存と検証

公開可能な集約結果は `history_baseline_v3_aggregate.json` です．report SHA-256 は `8444307baa019ca4de01987962d7473b095ec4b943be8620bf3759af711fd795` です．個別予測と model coefficient artifact は private run に保存しています．

`test_history_baselines.py` の 10 tests が scratch と実 workstation の両方で成功しました．境界時刻の混入拒否，禁止 feature，数え上げの矛盾，binary metric の正解値，subject 単位の再標本化，同一モデル差がゼロになること，欠損 test ID の拒否を確認しています．
