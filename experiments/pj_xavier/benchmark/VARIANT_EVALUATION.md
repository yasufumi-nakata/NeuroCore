# 凍結checkpointの5条件評価

`evaluate_variants.py` は，validationで選択済みの同じfused checkpointを一度だけ読み込み，同じ凍結test IDに対する5条件を評価します．optimizer，追加学習，checkpointの選び直し，OS操作は行いません．

| 条件 | 実際に変える入力 |
| --- | --- |
| `fused` | 正しいEEGと事前環境を維持 |
| `context_only` | モデルの既存APIでEEG入力をゼロ化し，事前環境を維持 |
| `eeg_only` | EEGを維持し，promptから環境記述を除去 |
| `eeg_shuffled` | 同じdataset・被験者・session・recording内の別のtest EEGに置換 |
| `eeg_time_shifted` | 元の配布BrainVisionから，元EEG窓の正確に10秒前を新しく抽出 |

これらは**同じ学習済みモデルの入力除去・対応破壊実験**です．context専用／EEG専用に別々に学習したbaselineとは区別します．

## 凍結と実sourceの検証

開始時に元JSONL bytes，training membership，test ID数，checkpointに封印された全prepared EEGファイルのhashを確認します．時間ずらしでは元raw binaryに加え，header，marker，元archiveの完了検証，物理channel，sample rate，整数sample intervalを保存します．新しい窓にはcontent由来の別source IDを付け，評価対象のtest IDと正解ラベルは変えません．

strict benchmarkも同じ証拠を独立に検査します．元rawから元窓を再読出しし，checkpointで封印されたprepared arrayと完全一致することを確認したうえで，10秒前のraw窓を再読出しして保存arrayと比較します．別の被験者やsession，異なるheader gain，時刻・sample数の偽装，array改変後の単なる再hash，既存test IDへの偽装は拒否します．この追加検証にはCPUで動くNumPy，PyTorch，MNEが必要です．通常の分類指標と従来のID対応検査には追加依存を要求しません．

元の配布BrainVision自体はofflineで前処理済みです．未配布のXDFを利用したとは扱わず，`online_causality_verified` はfalseのままです．10秒前の区間が範囲外，元波形が未取得，あるいは照合が不成立ならtime-shiftを理由付きblockedとし，部分集合のスコアで1956件の評価を代用しません．同一recording内でshuffleできないsingletonもblockedです．

実際のv2 manifestをworkstationでread-only確認した結果は，SHA `af878638c2fff6d6c7f619cd22005a057e797f1152553517bf05ce795aec2eac`，全7819窓，test1956窓，8channel／256Hz／448sampleです．test開始時刻の最小値は104.21875秒なので，manifestの時刻範囲だけを見る限り，10秒前が負の時刻になる対象は0件です．実rawのhash・値の照合はrunner実行時に行います．

## 本train完了後の実行

本trainの`completed_epochs`とbest checkpointを確認してから，workstation経由のColab CLIで起動します．学習中にこのrunnerのGPU推論を重ねません．

```bash
python /content/xavier/benchmark/evaluate_variants.py \
  --model-module /content/xavier/model/eeg_llm.py \
  --expected-model-sha256 e30811b027414426cdfbed14a822c27695af2ab1be00106d89fe32a5fb740d2f \
  --manifest /content/xavier/prepared/ai_coding_uniba_v2_lead4p5/manifest.jsonl \
  --expected-manifest-sha256 af878638c2fff6d6c7f619cd22005a057e797f1152553517bf05ce795aec2eac \
  --checkpoint /content/xavier/runs/qwen3_8b_aicoding_uniba_fused_v1/best.pt \
  --raw-bids-root /content/xavier/verified_ai_coding_uniba_v2 \
  --artifact-root /content/xavier \
  --output-dir /content/xavier/runs/qwen3_8b_aicoding_uniba_fused_v1/variants_v1 \
  --expected-test-records 1956 \
  --device cuda \
  --memory-budget-gib 21 \
  --candidate-batch-size 4
```

出力先は新しい空のディレクトリにします．既存出力や既存completion receiptがあれば，最終testの無断再実行を避けるため停止します．checkpointのconfigに保存されたimmutable backbone revisionとdtypeを維持します．変更可能なのは実行device，memory budget，candidate scoring batchです．

各条件の`*_predictions.jsonl`と結合した`predictions.jsonl`，`raw_shift_sources.jsonl`，`raw_shift_preparation.json`，`variant_evaluation_report.json`を保存します．100件ごとに`progress.json`を更新します．時間ずらしarrayは各窓のhashを検査できるよう個別に保存しますが，推論後にそのmmapをcacheから外し，1956個のfile descriptorを保持しません．

最終strict評価には結合したprediction JSONLをadapterに渡し，`xavier_benchmark.py evaluate` に **`--artifact-root /content/xavier`** を付けてください．raw由来time-shiftの実ファイル検証に必要です．`method: raw_recording_time_shift` は既存test IDへの単なる時刻ラベル変更とは別の契約です．

## Controllerの完了receipt

既定でcheckpointと同じディレクトリの `variant_evaluation_completion.json` に保存します．`--completion-receipt`でも指定できます．

```json
{
  "status": "completed",
  "exit_code": 0,
  "checkpoint_sha256": "ACTUAL_SELECTED_CHECKPOINT_SHA256",
  "variant_names": ["fused", "context_only", "eeg_only", "eeg_shuffled", "eeg_time_shifted"],
  "report_path": "/content/xavier/runs/RUN/variants_v1/variant_evaluation_report.json",
  "report_sha256": "ACTUAL_REPORT_SHA256"
}
```

reportのトップには同じ`checkpoint_sha256`，`variant_names`，`conditions`があります．各`conditions.NAME.status`はmeasured，または理由付きblockedです．measuredの場合は対象件数・test ID hash・checkpoint hash・prediction実ファイルhashも照合してからreceiptを出します．全条件の終端状態が確定したという意味であり，モデルの性能合格ではありません．エラー終了はstatus failed／exit_code 1となり，正常終了receiptとは区別します．

`autonomous_agent_validated`と`held_out_replay_validated`はfalseを維持します．独立idle記録，実目的達成のハーネスepisode，online因果性，前処理から操作までのend-to-end latencyは別の未実証条件です．`ai_prompt_submit`はtrain2／validation2／test1だけなので，この少数の観測操作から一般的なagent依頼能力の有無を断定しません．

## 検証

```bash
../model/.venv/bin/python -m unittest discover -s . -p 'test_*.py'
```

新規テストは小さな人工BrainVisionファイルを実際にCPUで読んで検証します．GPU，実LLM，実test1956件を実行した結果ではありません．同じcheckpoint／backboneを一度だけloadするCLI配線，5条件のtest ID維持，実raw再抽出，破損と偽装の拒否，receiptの不完全時拒否を確認します．
