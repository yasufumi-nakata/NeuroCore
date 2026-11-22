# ITU 外部評価用 cohort v3

`prepare_itu_external.py` は，未使用だった ITU の記録を **external_test 専用**に準備します．学習・検証集合は作らず，モデルを読み込んだり，推論・学習・モデル選択を行ったりしません．

## 固定した入力と規則

準備前に `itu_external_v3_protocol.json` を固定しました．SHA-256 は `6dae42f26833eab36028b886907a4cfcf9047f3ad76af182c497129cd885eca0` です．既存 `natural_cohort_v3_protocol.json` の signal，case definition，control definition，selection，history rules を，既存 helper の SHA-256 とともに検証して再利用します．

- Clock audit 全 26 記録のうち，pass status と `clock_sync.verified` の両方を満たす 25 記録をすべて処理します．既存の誤差上限 100 ms，drift 上限 5,000 ppm，phase scale 1 を変更しません．
- 共通 8 channel は CP3，C3，F5，PO3，PO4，F6，C4，CP4 の順です．256 Hz，直近 1.75 秒／448 samples を保持し，追加 filter・resample・rereference は行いません．
- EEG 終端は対象イベントの 4.5 秒＋clock guard 前です．既存の仮定上の filter lookahead 4.125 秒を加えた時刻を特徴量利用時刻とします．実機の online causality は未確認のままです．
- 記録された `GPTSentPrompt` を case とし，既存規則で keyboard と mouse-click の control を選びます．同一記録の EEG 窓は非重複です．窓数や label 数を見て seed・quota・clock 条件を変更しません．
- 操作履歴は EEG 開始前と guarded decision 前の 2 系統です．cutoff と同時刻・未来のイベント，GPT イベント，キー内容，座標，duration を特徴へ入れません．

Site-qualified observation ID は `itu:<participant label>` です．他 site の同名 participant label について，同一人物とも別人保証とも断定しません．凍結 UniBa v3 の全 split が参照する 32 原波形 SHA-256 と，ITU 全 26 原波形 SHA-256 の一致があれば準備を停止します．hash 不一致だけで人の独立性を保証することもありません．

## 保存先と出力

実行済みの Colab 出力は `/content/xavier/prepared/ai_coding_itu_external_v3_natural_casecontrol_v1` です．workstation の `private/xavier/prepared/ai_coding_itu_external_v3_natural_casecontrol_v1` に，全ファイルの SHA-256 を照合して回収済みです．Colab 操作は workstation の `/Users/yasufumi/.local/bin/colab` CLI を経由しました．

| ファイル | 内容 |
| --- | --- |
| `manifest.jsonl` | `split=external_test` の波形・label・context・clock・来歴 |
| `waveforms/*.npy` | 共通 8 channel，448 samples の実 EEG，float32 |
| `labels.jsonl` | 観測された操作 label と元イベント行の証拠 |
| `history_features.jsonl` | 既存 v3 と同じ 2 系統の過去履歴特徴 |
| `external_membership.json` | 外部評価 ID と全 pass 記録．train／validation ID は空 |
| `eeg_artifacts.json` | 全配列のサイズと SHA-256 |
| `source_members_verified.json` | 実体を hash 照合した原ファイルの証拠 |
| `cohort_audit.json` | 全 26 記録の処理・除外・選択理由 |
| `cohort.aggregate.json` | 公開可能な集計，来歴 hash，使用制約 |
| `independent_validation.json` | 全配列・全窓・全 label・全 history の独立検証 |

既存の出力は上書きしません．原 manifest，元波形，clock audit，既存 model と checkpoint も変更しません．正 worktree の private に，実行 argv，launch／exit receipts，回収 archive と全 member hash の証拠を保存しています．

## 検証

```bash
.venv-xavier/bin/python -m unittest discover \
  -s experiments/pj_xavier/model -p test_prepare_itu_external.py -v
```

CPU fixture 8 件が workstation で PASS しました．実 BrainVision の小さな fixture でも，外部評価以外の partition が生成されないこと，全対象記録の監査，clock fail の除外，元ファイルの不変性を確認しています．

`validate_itu_external.py` は固定 manifest の hash を指定して別プロセスで実行します．全配列を再 hash・shape・有限性検査し，元 BrainVision から全窓を再読込みして sample-for-sample で比較します．元イベントから label と case/control 選択を再導出し，2 系統の履歴特徴も全件再計算します．CPU のみを使用し，モデル推論を行いません．

実データでは，25 配列・630 波形窓・630 label・1,260 履歴特徴の組がすべて一致しました．原ファイル 204 個も再 hash して一致しています．詳細集計は `reports/itu_external_cohort_v3_20261122_v1` を参照してください．

## 評価時の制約

この cohort の label は観測された操作です．潜在意図や harness の目標達成を意味しません．case/control 抽出後のクラス構成を，実利用時の自然な発生頻度へ読み替えないでください．元の記録時間と選択窓の時間は同じ EEG の集計なので加算できません．

この固定集合を使った外部推論は，別途固定した評価 protocol で実施します．ITU での fit，checkpoint 選択，閾値調整，label 数に応じた再抽出は許可する設計ではありません．既存の UniBa membership を書き換えず，`external_test` の来歴として別に照合する必要があります．
