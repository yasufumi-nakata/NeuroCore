# AI-assisted coding v3：自然な過去操作を含む request cohort

## 準備したもの

**実際の GPT request 送信 150 件をすべて保持しました．** v2 の 5 件に対して 30 倍です．AI が意図ラベルを作った結果ではなく，公開ログの `GPTSentPrompt` だけを正解にしています．`GPTReceiveText`，入力開始，AI 利用許可は request の正解にしません．

| Split | 被験者数 | 全窓 | Request | Keyboard | Mouse click |
|---|---:|---:|---:|---:|---:|
| Train | 18 | 400 | 80 | 160 | 160 |
| Validation | 6 | 155 | 31 | 62 | 62 |
| Test | 8 | 203 | 39 | 82 | 82 |
| 合計 | 32 | 758 | 150 | 304 | 304 |

入力窓は 8 ch × 448 samples，256 Hz，合計 0.368472 時間です．EEG 配列は 10,870,784 bytes，持ち運び用 archive は 10,882,027 bytes です．元記録は v2 と共通の 30.6154 時間であり，v2 と v3 の時間を足して独立データの総量にしてはいけません．

## 選択規則と限界

`natural_cohort_v3_protocol.json` を実準備前に固定しました．既存 v2 の manifest，subject split，clock audit，loader の SHA-256 を照合し，被験者の割当を byte 単位で維持します．32 人の合格記録から，先に request 窓を保持し，その後に同じ recording と過去の AI availability に対応する keyboard／mouse 操作を対照として選びます．各 request に原則 4 対照を割り当て，request がない記録にも最大 8 対照を残します．同一 recording の入力窓は非重複です．

これは **case-control／enriched cohort** です．この標本上の precision，AUPRC，confidence，coverage は，日常利用時の出現率における値を表しません．false actions/minute や実タスク完遂率も推定していません．v2 と v3 の異なる test 標本の精度を直接比較し，モデルが改善したと解釈することはできません．

150 request 窓中 140 窓に，過去のキー／マウス操作が含まれています．その情報で予測できる可能性と EEG が追加する情報を分けて評価するため，操作履歴だけの対照特徴を保存しました．運動・筋電の交絡が消えたとは扱いません．

## 時刻と履歴特徴

EEG 終端を request の少なくとも 4.5 秒前で止め，公開 filter recipe に基づく仮定上の 4.125 秒の特徴量利用遅延を加えます．clock 誤差分も別に確保します．`decision_time_s` とイベントの名目上の差は最低 0.375 秒です．実推論・転送時間はさらに必要です．原 binary と filter recipe の同一性が未確認のため，online causality は未達のままです．

`history_features.jsonl` には，次の 2 系統を保存しています．

- `history_pre_eeg`：EEG 窓開始－clock margin より前の操作履歴．
- `history_pre_decision_guarded`：decision－clock margin より前の操作履歴．より強い履歴対照として別に評価します．

1・5・15・60 秒区間の key-down，key-up，mouse-click，mouse-move の件数と，最後の操作からの時間を保存します．cutoff と同時刻のイベントも除外します．キーの文字，座標，GPT イベント，対象ラベル，未来のイベントを特徴へ入れません．公開 `duration` が次のイベントまでの間隔を含む可能性があるため，duration も使いません．

既存の LLM prompt は従来の過去 oTree context を使います．履歴特徴の sidecar は，そのまま LLM prompt に自動挿入されるものではありません．履歴だけの classifier は cohort 準備後，別の固定 protocol で benchmark 担当が学習・評価しました．実行条件は `../benchmark/protocol.history_uniba_v3.json`，集計は `../benchmark/history_baseline_v3_aggregate.json` を参照してください．EEG 模型の比較では同じ test 標本上の履歴寄与を基準にします．

## 検証と保存

新規 CPU 回帰試験 9 件が通っています．対象イベントを未来に追加しても過去特徴が変わらないこと，case 優先と非重複，選択順の決定性，実 request だけの正解化，旧 phase member が変わると旧 clock を再使用できないことを確認しました．実データについて，32 配列の有限性・hash，入力窓の非重複，全 cutoff，被験者分離を検証しています．phase member の追加 guard 後に再準備した manifest は，初回と同じ SHA-256 になりました．

```text
manifest:
030e8550bd6f4c79005722eb617e9b64f5f80e87b77866b7f17bec4e7e2a1093

registered protocol:
d4e6c5f685dce53e73b3aa88d12ed07c88a8ac08a27fb78392dbd99aed0a6396

prepared archive:
5d0a3968be9313826c1109d0028b6b985dafccf9d7c3a478371ae712cae33c83
```

完全な prepared cohort を workstation の `private/xavier/prepared/ai_coding_uniba_v3_natural_casecontrol_verified` に回収し，archive／manifest／全 32 配列の hash を再照合しました．公開集計は `../reports/ai_coding_uniba_v3_natural.aggregate.json` と `../reports/ai_coding_uniba_v3_validation.aggregate.json` です．

## 固定した実行と最終評価

M1 の SmolLM2-1.7B v3 は3 epochsを完了し，validationのみで選んだbestを固定しました．203 test窓を同じbestで5条件評価する処理を実行中です．元波形から10秒前の区間を取り出す対照は，全203窓でsourceとの一致を検証済みです．評価完了・性能達成を先取りして主張しません．

ColabのQwen8Bは，別に事前保存した `../benchmark/protocol.qwen3_8b_uniba_v3_30e.json` に従い，30 epochs／最大3600秒のrunを開始しました．当初の3-epoch提案はsupersededとしてhandoffへ保全し，重複起動しません．M1とColabでは学習予算が異なるため，差をモデルサイズだけの効果と解釈できません．

各モデルのvalidation選択とtestを分け，履歴classifierの前処理もtrainだけでfittingします．test被験者は8人であり，被験者単位の不確実性を併記します．ITUの追加25人は外部siteの候補として監査した段階で，v3の学習・モデル選択には使っていません．

## Workstation M1 の実学習

固定 SmolLM2-1.7B-Instruct と実 train 窓で，batch 1・2 の各 3 optimizer steps が MPS 上で完了しました．batch 2 の sampled driver memory は 4,420,599,808 bytes，warm throughput は約 1.212 examples/s でした．8 GiB allocator 上限を設定し，CPU fallback を無効にしています．

この結果を受け，別 run `smollm2_1_7b_uniba_natural_v3_fused_v1` を batch 2，candidate batch 2，3 epochs，seed 20261122，1800 秒のループ上限で実行し，600 stepsを完走しました．短い probe の重みは再利用せず，v2 の source と manifest を変更していません．学習中は validation だけを評価します．正確な command，PID，allocator 設定，終了 receipt は workstation の `private/xavier/runs/smollm2_1_7b_uniba_natural_v3_fused_v1/` に保存します．短い probe の成功から長時間の完走を断定せず，終了 receipt と保存 checkpoint を別に確認します．

学習＋各epochのvalidationは1541.72秒（25.70分），全processは1550.64秒でした．観測したMPS driver memoryの最大値は4,437,557,248 bytes，allocatedは3,425,448,960 bytesです．これはsampled値であり，瞬間的な真のpeakとは呼びません．8 GiB allocator上限を維持しています．学習対象は166,656 parameters，1,711,376,384 parametersのLLM本体は凍結したままです．

Validationのbalanced accuracyは1〜3 epochで33.33%，37.63%，33.87%となり，2 epoch目をbestに選びました．この時点でAI送信のrecallは0であり，目的達成を主張できません．best checkpointは2,022,605 bytes，SHA-256は `281716da86293d2b700ec003c967ca2c141515cc565b136a12137c34b73a3475` です．学習集計は `../reports/smollm2_1_7b_m1_natural_v3_training.aggregate.json` にあります．

M1のvalidation scoringは約1秒／例で，仮定上のdecision→event余裕0.375秒より長くなっています．この同じdeadlineでのオンライン操作達成を示すものではありません．5条件の最終評価と履歴対照の比較は別のreceiptで確認します．
