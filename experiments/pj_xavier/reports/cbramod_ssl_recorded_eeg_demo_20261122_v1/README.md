# 追加 SSL モデルの M1 録画 EEG 試用

追加の自己教師あり学習で更新した CBraMod と，その出力に合わせて学習した SmolLM2 connector を，M1 の同じ train 3 例で 1 回実行しました．predict と dry-run replay はともに終了し，全 3 件を router が保留しました．OS 操作 0，cloud agent 呼出し 0 です．

| 元の観測行動ラベル | 予測ラベル | 最大 restricted softmax | router |
| --- | --- | ---: | --- |
| ai_prompt_submit | ai_prompt_submit | 0.8870 | 保留 |
| keyboard | keyboard | 0.8821 | 保留 |
| mouse_click | mouse_click | 0.8066 | 保留 |

この 3 例は既に学習に使われたデータです．上の一致は実行経路の確認であり，汎化性能ではありません．softmax は限定した A/B/C 候補間の未校正の出力です．行動ラベルだけでは実行したい具体的操作を定められないため，proxy_label_has_no_explicit_intent_mapping で保留しています．閾値 0.8 と router は変更していません．

## モデルとデータの固定

alignment checkpoint は d2ef4ba0fa7e2d01aa30b73faf977ab0edca0baa9ee12470ce6065cf5bdcfa16，追加学習済み EEG weights は 0132c9372cbcff28ad36965de043f7b8190233e9faba8ce3ec02c2ae8c93f3e7，選択 seal は b5367f60bde3a3156d3edb47565fed29c5009edbdfcd515d2dc4c698df63f171 です．SSL の元 checkpoint と export の全 tensor，公式初期値からの更新，固定 validation MSE による選択，allowlist と実 membership，実 5 条件評価 child exit，strict 検証と全 member 回収を CPU で照合しました．元の 32 配列と manifest は変更していません．

以前の CBraMod 試用と同じ 3 個の train ID を使い，試用に held-out 窓は使っていません．追加モデルの設計には以前の test 結果を見た経緯があるため，元の UniBa test を使う比較は探索的です．独立した ITU 外部評価は別途の固定契約で扱います．

## 実行環境と計測範囲

CPU float32 で EEG encoder，MPS bfloat16 で凍結 SmolLM2 を実行しました．3 例の計測では P50 544.989 ms，P95 2948.505 ms でした．CPU EEG encoder，転送，projection と 1 回の LLM forward の時間であり，配列 I/O，resampling，tokenization，harness 処理は含みません．3 例の分位点から運用時の遅延分布を推定することはできません．

MPS 観測 allocated は 3,424,401,920 B，driver は 4,310,302,720 B です．これらは sampled 値であり，真の peak としては扱いません．8 GiB の allocator 制限は PyTorch MPS の範囲で，CPU や他プロセスのメモリとは別です．

実行手順は ../../integration/README_TRY_CBRAMOD_SSL_RECORDED_EEG.md，結果と provenance は summary.json，12 件の focused CPU tests と実行検証は validation.json に記録しました．生 EEG，参加者 ID，モデル重み，private path は公開集計に含めていません．
