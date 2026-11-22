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

既存の LLM prompt は従来の過去 oTree context を使います．履歴特徴の sidecar は，そのまま LLM prompt に自動挿入されるものではありません．履歴だけの classifier は今回の CPU 準備では学習しておらず，解析 plan を handoff に明記しています．

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

## 次の予約枠で実行するもの

具体的な環境変数，学習 argv，5 条件評価 argv，解析 plan は `natural_cohort_v3_handoff.json` に固定しました．**今回 v3 の LLM 学習・GPU 評価は実行していません．**

進行中の v2 学習と test 評価を先に完了・保存し，別の v3 run を固定 seed から開始します．v2 checkpoint の manifest を変更して resume することはしません．batch は 4 を事前指定します．batch 8 は短い capacity probe に合格しましたが，長い v2 実行の 24 steps 時点で reserved memory が予算を超えて停止したためです．この変更は v3 test の成績を見て選んだものではありません．

Validation のみで best を選び，test の 203 窓は固定済みモデル群と対照条件で評価します．履歴 classifier の前処理も train で fitting し，test は使用しません．test 被験者が 8 人であることを明示し，被験者単位の不確実性を報告します．
