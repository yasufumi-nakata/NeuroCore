# PJ-Xavierのベンチマーク

このモジュールは，**実際のEEGから操作を予測できたのか，環境情報だけで答えが分かったのか**を分けて評価します．供給された予測と来歴を検証し，未実施の推論は生成しません．実データによる独立監査は `FINAL_AUDIT.md` に記録しています．テスト内の人工データは計算と漏洩検出の検証専用で，EEG性能の実績として使いません．

## 目標と完了条件

PJ-Xavierの最終目標は，ハーネスから事前に与えられた環境情報と，その後のEEGを使ってユーザーの望む操作を行うことです．このため，次の結果を混同しません．

| 評価 | 分かること | 分からないこと |
| --- | --- | --- |
| EEGの既存タスク分類 | 波形処理・特徴抽出が対象ラベルに役立つか | そのままPC操作できるか |
| 操作前EEGの行動分類 | 観測された次の操作を予測できるか | 記録されていない高次の意図を読めたか |
| 環境付きEEGの負の対照付き評価 | 環境情報に加えてEEGが実際に寄与するか | 未実施の実環境操作の成功率 |
| 独立したハーネス再生 | 評価用の目標・状態遷移を達成するか | 新しい人・実環境での将来の成功率 |
| 前向き運用評価 | 新しく取得したEEGで実操作が成立するか | 未評価の機器・人・タスクへの一般化 |

`autonomous_agent_validated` はこの再生評価だけで `true` にしません．再生の合格は `held_out_replay_validated` に記録し，実際の前向き運用は別の検証として残します．単一被験者EEGlassの時間分割は `within-recording temporal pilot` です．新しい被験者に対する性能とは表示しません．

## ファイル

| ファイル | 用途 |
| --- | --- |
| `xavier_benchmark.py` | split作成，漏洩検査，モデルごとの評価，条件付き集計 |
| `model_adapter.py` | 学習側のflat JSONLを厳密な評価schemaへ変換 |
| `test_xavier_benchmark.py` | 漏洩，棄権，誤作動率，trace重複などの回帰検証 |
| `acceptance.proposed.json` | 最初の再生評価に向けた変更可能な工学的目標値 |
| `evaluate_variants.py` | 同じfused checkpointの5条件評価と，実raw由来10秒前EEG対照 |

5条件runnerの実行，追加のraw証拠契約，controller receiptは [VARIANT_EVALUATION.md](VARIANT_EVALUATION.md) を参照してください．

`xavier_benchmark.py` と `model_adapter.py` の純粋な JSONL／集計処理は，Python 3.9 以降の標準ライブラリだけで動作します．この条件を，表内の全 runner に適用するものではありません．`evaluate_variants.py` や `external_evaluation.py` などの neural runner には [model/requirements.txt](../model/requirements.txt) の PyTorch／Transformers／NumPy／MNE などが必要です．history／external の CPU 集計には NumPy／SciPy／scikit-learn／threadpoolctl も必要です．これらを実測した環境は workstation の Python 3.12 と Colab の Python 3.13 です．各 run の依存版・実機適合性は個別の receipt に記録し，CPU で集計できることを GPU／Mac での推論適合性へ置き換えません．

## レコードschema

以下は**形式説明用の例**で，実在データを表しません．実行時は波形・ラベル・環境のURIとSHA-256を実物から取得してください．`decision_time_s` までに利用可能な情報だけをモデル入力にします．時刻は同じrecordingの時計系にそろえます．

```json
{
  "record_id": "dataset/participant/session/window",
  "dataset_id": "dataset-version",
  "source_dataset_id": "canonical-original-dataset",
  "participant_id": "pseudonymous-participant",
  "subject_group_id": "optional-cross-dataset-person-id",
  "session_id": "session-id",
  "recording_id": "original-recording-id",
  "data_origin": "real_recording",
  "target_kind": "observed_action",
  "decision_time_s": 9.75,
  "minimum_lead_s": 0.25,
  "eeg": {
    "uri": "local-or-provider-original-waveform-uri",
    "sha256": "64-lowercase-hex-digits",
    "start_s": 8.0,
    "end_s": 9.75,
    "sample_rate_hz": 250
  },
  "label": {
    "value": "mouse_click",
    "provenance": "event_derived",
    "evidence_uri": "source-event-log-uri",
    "event_time_s": 10.0
  },
  "environment": {
    "uri": "predecision-context-uri",
    "sha256": "64-lowercase-hex-digits",
    "available_until_s": 7.99,
    "contains_target_label": false
  },
  "clock_sync": {
    "verified": true,
    "method": "documented clock alignment method",
    "max_error_ms": 2.0
  }
}
```

環境情報がない場合は `environment: null` にします．クリック後の画面，未来のログ，事後のAI説明を事前の環境入力に入れません．同期精度が未確認なら `verified: true` を付けません．アクションの発生から作ったラベルは `observed_action` とし，高次の潜在意図に置き換えません．

`label.provenance` は次の4種類です．

| 値 | 用途 | 評価の正解に使用 |
| --- | --- | --- |
| `source_annotation` | 原著のラベル | 出典URIがある場合に可 |
| `event_derived` | 記録イベントからの決定的変換 | 観測行動など，イベントが直接証明する対象で可 |
| `human_verified` | 独立した人の検証 | `reviewer_id` と `independent_review: true` が必要 |
| `ai_pseudo` | AIが補完したラベル | 学習用のみ．validation/testから除外 |

AIラベルには `teacher_id`，`teacher_revision`，`prompt_sha256`，`evidence_uri` を必須にします．同じAIが付けたラベルに同じAIが同意したことを独立検証とは数えません．説明文を増やしても，独立したEEG記録の時間・人数・サンプル数は増えません．

## 分割と漏洩の防止

```bash
python xavier_benchmark.py split \
  --records records.jsonl \
  --unit participant \
  --embargo-s 10 \
  --out split.json
```

既定の60／20／20は**群数**の目安です．レコード数がこの比率になるとは限りません．群のhash順で決めるため，入力行の並びを変えても同じ分割です．被験者，セッション，原典datasetの分割を選べます．元データのversionや転載が違っても，`source_dataset_id` を共通にします．既知の同一人物が別datasetにある場合は `subject_group_id` を共通にします．不明な同一人物や原波形の再圧縮まで自動的に完全同定できるとは主張しません．

被験者が1人しかいない場合は，個人間評価を `blocked` にします．パイロットとして `--unit temporal_block` を指定すると，各元波形を時系列60／20／20に分け，境界の前後を除外します．隣接した重複windowを異なるsplitに入れることはできません．SHA-256が同じ原波形の区間重複は，別のparticipant名が付いていても検出します．split作成後にラベルや時刻を変更するとレコードhashが変わり，その評価は拒否します．

10秒のembargoは初期設定であり，普遍的に十分な長さという根拠ではありません．window長，重複，自己相関，セッション構造に応じて，学習開始前に設定を固定してください．

`temporal_block` は非重複だけでなく，train→validation→testの時系列順を必須にします．`canonical_dataset_id` と `canonical_subject_id` があるflat manifestでは，adapterがそれらを引き継いで元dataset・被験者単位を統一します．元JSONLのbytes hashをtraining membershipと照合するため，subjectやsessionの付け替えだけで既存checkpointの評価分割を作り直すことはできません．

## 前処理の未来情報

判断前でwindowを切るだけでは，事前のzero-phase filterによる未来試料の混入を排除できません．`preprocessing.online_causality_verified: true`，その `evidence_uri`，有限の非負 `max_future_support_s` が記録され，`eeg.end_s + max_future_support_s <= decision_time_s` が成立した場合だけ前処理gateを通します．フィールドがない旧データや，filter supportの仮定しかないデータはunverifiedとしてgoalをblockします．分類指標はretrospectiveな参考値として計算し，未検証を黙ってtrueへ補いません．

EEGlass旧manifestの `decision_time_s == event_time_s` というメタデータ誤りには，明示的な `model_adapter.py --legacy-decision-at-event` のみを使用します．既存EEG終端がeventより規定lead以上前であることを確認して，判断時刻のメタデータだけを既存EEG終端へ正規化します．元manifest bytes・波形・ラベル・splitは変更せず，旧新値と原hashを `migration.json` に残します．評価後に `--annotate-evaluation-report evaluation.json --migration migration.json` を指定すると，その来歴を報告にも付記できます．

## 学習ごとの評価

1. 学習開始前にレコードとsplitを固定します．
2. 各epoch/checkpointでvalidationを評価し，最良checkpointを選びます．
3. 棄権閾値・前処理・モデル選択をvalidationだけで確定します．
4. その学習runの確定checkpointを固定testで評価します．testの結果で選び直しません．
5. 新しい研究判断をtestの結果から行う場合は，確認用の未使用holdoutを別に用意します．

事前学習済み基盤モデルが既存公開データを見たか不明な場合は，`foundation_pretraining_overlap: unknown` として学習manifestに残してください．この実装が検出できるのは，記録された学習レコードとsplitの整合性です．基盤モデルの非公開学習履歴まで確認したとは表示しません．

## 予測と対照条件

各checkpointについて，**同じtestレコード**に対する次の5条件を保存します．

| `variant` | 入力・目的 |
| --- | --- |
| `context_only` | 環境だけで予測できる部分を測る |
| `eeg_only` | EEG単独の寄与を測る |
| `fused` | 正しく同期したEEG＋環境 |
| `eeg_shuffled` | EEG対応を壊しても同じ結果にならないかを測る |
| `eeg_time_shifted` | 時間対応を壊しても同じ結果にならないかを測る |

シャッフルや時間ずらしは `perturbation` に `original_eeg_id`，`eeg_source_id`，手法等を記録します．testをtrainへ移動しません．負の対照だけ別のtest集合を用いません．goal gateでは同じ被験者・セッション内の異なるheldout EEGを要求し，shuffleは固定点のない完全な置換として照合します．time shiftは `method: recording_time_shift`，同じraw recording hash，非重複window，実時刻差と一致する `shift_seconds` を必須にします．対応先を示すだけで実入力を独立再実行して確認したことにはならないため，報告に検証範囲も残します．

```json
{
  "record_id": "same-frozen-test-id",
  "checkpoint_id": "run-001/best-validation",
  "checkpoint_sha256": "64-lowercase-hex-digits",
  "variant": "fused",
  "probabilities": {"mouse_click": 0.7, "keyboard": 0.2, "mouse_scroll": 0.1},
  "abstain": false,
  "latency_ms": 125.0
}
```

全クラスの確率を出し，合計を1にします．`macro_f1`，`balanced_accuracy`，NLL，Brier score，ECE，棄権coverage，採用した予測の正答率を計算します．棄権を通常のaccuracyから単に消して性能を高く見せません．`forced_accuracy` は棄権を無視したargmaxの参考値です．

EEGの追加効果は，fusedと各負の対照のmacro-F1差で出します．参加者単位の対応付きbootstrapを行いますが，独立参加者が5群未満なら区間推定を `unavailable` にします．5群という実装上の下限が統計的十分性を保証するわけではありません．

## 学習・実機manifest

`training.json` は以下を持ちます．

```json
{
  "training_record_ids": ["training-record-id"],
  "teacher_training_record_ids": [],
  "split_manifest_sha256": "canonical-json-hash-returned-by-adapter",
  "checkpoint_sha256": "actual-checkpoint-hash",
  "model_selection_partition": "validation",
  "test_used_for_tuning": false,
  "foundation_pretraining_overlap": "unknown"
}
```

`resource.json` は対象checkpointを**実際に推論した**機器の測定を記録します．

```json
{
  "measurement_mode": "actual",
  "checkpoint_sha256": "actual-checkpoint-hash",
  "machine_id": "workstation-or-smac-node-or-colab-runtime",
  "runtime": "framework-and-version",
  "accelerator": "actual-accelerator",
  "hardware_compatible": true,
  "inference_succeeded": true,
  "measurement_method": "documented-memory-and-latency-measurement",
  "memory_measurement_scope": "process_peak_rss-or-documented-device-scope",
  "peak_memory_bytes": 1000000000,
  "memory_budget_bytes": 10000000000,
  "p95_latency_ms": 1000
}
```

数値は形式例です．Apple Siliconの統合メモリ，GPU専用メモリ，CPU RSSを混同しません．GPU allocationだけを測った値で総プロセスメモリが収まると結論しません．Colabで動いたこととMacで動いたことも別々に記録します．利用可能な環境での実測がなければ実機gateは通りません．

`memory_capacity_bytes` または `device_physical_memory_bytes` を併記すると，設定budgetが実容量を超えていないかも検証します．`host_peak_rss_bytes` と `host_ram_bytes` はGPUとは別の比較として扱います．利用環境の明示的な紐付けは `target_environment_verified: true`，`target_environment_id`，`hardware_evidence_sha256` で記録します．測定機器でのresource passと，この紐付けは報告内で別に表示します．candidate scoringだけの時間をagent全体の応答時間へ置き換えず，goalの遅延gateは `latency_scope: end_to_end` の実測を要求します．根拠がない場合，これらの値を合格用に追加しないでください．

## ハーネス再生と誤作動率

`episodes.jsonl` はepisode ID，評価に使用したtestレコードID一覧，checkpoint hash，variant，goal ID，独立したassertionの出典，traceの相対path/hashを持ちます．

```json
{
  "episode_id": "held-out-episode-id",
  "record_ids": ["test-record-id"],
  "checkpoint_sha256": "actual-checkpoint-hash",
  "variant": "fused",
  "task_scope": "context_conditioned_agent_operation",
  "goal_id": "independently-defined-goal",
  "ground_truth_source": "independent_harness_assertion",
  "environment_goal_seen_in_training": false,
  "action_logging_complete": true,
  "trace_path": "traces/episode.jsonl",
  "trace_sha256": "actual-trace-hash"
}
```

traceには対応goalの `terminal_assertion` を一つ記録します．`passed` はモデルの自己申告ではなく，ハーネスが観測した環境状態から決定します．モデルが生成した計画文を正解にしたり，「操作する予定」と言っただけで成功にしたりしません．外部traceを改変するとchecksum検証で拒否し，同じtraceを重複してepisode数に足しません．

誤作動率は `observed_idle_interval` と，その時間帯に実際に発生した `executed_action` から計算します．独立idle annotationまたは全操作が記録された保証 `action_logging_complete: true` がない場合は `null/unavailable` です．分類のfalse positive数を実運用の誤操作回数/分に置き換えません．重なるidle区間は和集合で集計し，event IDが同じ操作は一度だけ数えます．

## 実行

```bash
python model_adapter.py \
  --records flat_records.jsonl \
  --predictions flat_predictions.jsonl shuffled_predictions.jsonl \
  --training-membership run/training_membership.json \
  --checkpoint-metadata run/run_summary.json \
  --checkpoint-sha256 ACTUAL_CHECKPOINT_HASH \
  --unit temporal_block \
  --embargo-s 10 \
  --out-dir prepared

python xavier_benchmark.py evaluate \
  --records prepared/records.jsonl \
  --split prepared/split.json \
  --predictions prepared/predictions.jsonl \
  --training prepared/training.json \
  --resource resource.json \
  --episodes episodes.jsonl \
  --artifact-root replay_artifacts \
  --acceptance acceptance.proposed.json \
  --out evaluation.json

python -m unittest discover -s . -p 'test_*.py' -v
```

必要な証拠がない場合もJSONを出力し，`blocked`／`unavailable`／`evaluated_goal_unproven` を明記します．CLIの終了コード0は「報告の作成に成功」の意味であり，性能合格の意味ではありません．自動実行はJSON内のstatusと各gateを確認してください．

`acceptance.proposed.json` は初期の工学的目標案です．現時点の達成見込みや先行研究の標準値ではありません．実験開始前にversionを固定し，テスト結果を見て合格線を下げる運用はしません．最終目標の評価データがない場合，既存EEG分類タスクの高い正答率でその代わりを済ませません．
