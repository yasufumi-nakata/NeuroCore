# PJ-Xavier 最終独立監査

**学習・推論の実行と，EEGから目的に沿う操作を判別できることは分けて評価します．今回の2モデルは実機で動作しましたが，目的達成は未実証です．**

## 凍結データによる実測

| モデル | 学習・validation・testの時間 | test件数 | accuracy | balanced accuracy | macro-F1 | EEG shuffleとの差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolLM2 1.7B | 35.006秒 | 78 | 83.33% | 33.33% | 0.3030 | 0 |
| Qwen3 8B | 227.354秒 | 78 | 83.33% | 33.33% | 0.3030 | 0 |

両方とも78件すべてを mouse_click と予測しています．正解の内訳は keyboard 9，mouse_click 65，mouse_scroll 4です．高いraw accuracyは多数派予測で説明でき，EEGの対応を壊しても分類結果は同じでした．この時間には初回のモデルdownload・backbone loadを含みません．実測資源JSONを渡した最新版benchmarkは，両モデルのL4上のresource gateをpassed，総合を `evaluated_goal_unproven` としました．

## 修正と再検証

前処理の未来支持，negative controlの実出典対応，temporal splitの順序を検証するgateを追加しました．元manifestのsubject/sessionを変えても既存training membershipとhashが一致せず，raw波形hashが同じ区間を名前の付け替えでsplit間に重複させることも拒否します．旧decision timestampの修正は明示flagで行い，元波形・ラベル・split・manifest bytesを保持します．

label workerは `--source-records` を必須にし，trainの申告と実際の凍結所属を照合します．**benchmark 27件，labeling 20件の回帰試験が全件PASS**です．これは実装の検証であり，EEG性能の実証ではありません．

## AIラベルの実体

実際の15件のteacher出力について，元379レコード，training membership，source/prompt/schema/evidence hashを照合しました．225件のtrain窓への再利用を一意に確認し，val/test参照は0，gold変更は0でした．代表jobの引用IDと再利用先のevent出典を別々に保ち，逆mappingを保存しています．

225/225の既存eventラベルとの一致は，event名の正規化整合性です．新しいEEG，独立した225回のteacher推論，潜在意図のgoldが増えたことを意味しません．今回の凍結済み2checkpointが使用したのは原典の `recorded_event` ラベルであり，後から作成した辞書ラベルで既に学習済みとは記載しません．

## 残る検証条件

EEGlassは1人の時系列パイロットで，環境情報，時間ずらし対照，前処理のonline因果性，独立したidle記録，実際の目的達成episodeがありません．これらを欠いたままagentの成功や誤操作回数/分を出しません．公開AIcodingのfilter支持4.125秒という条件付き再構成も，原信号での実証がなければverifiedにはしません．

model側の推論は分類結果のdry-runで，OS操作を実行しません．fleetのgeneric artifact completionはファイルの存在確認までなので，後でlabel/trainを接続するcontrollerでは，同じjobの終了receipt・checkpoint/hash・結果JSONのstatusを照合する必要があります．

詳細な元artifact hash，コードSHA，判定理由は [final_audit.json](final_audit.json)，ラベル来歴は [real_dictionary_propagation_audit.json](../labeling/real_dictionary_propagation_audit.json) と [real_dictionary_membership.jsonl](../labeling/real_dictionary_membership.jsonl) に記録しています．
