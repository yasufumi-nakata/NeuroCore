# UniBa CPU baseline 比較結果

**EEG＋coarse context のtest accuracyは57.36%，macro-F1は0.3795でした．AI送信の正例はtestで1件しかなく，3クラスbalanced accuracyの順位はこの希少クラスに強く左右されます．** 潜在意図や自律agentの実現を示した結果ではありません．

## 固定したデータと評価

| 項目 | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Windows | 4,357 | 1,506 | 1,956 |
| Subjects | 18 | 6 | 8 |
| Keyboard | 1,906 | 549 | 885 |
| Mouse click | 2,449 | 955 | 1,070 |
| AI prompt submit | **2** | **2** | **1** |

8B LLM実験と同じ manifest，ID membership，32個・112,097,280 bytesのprepared EEG artifactをSHA-256で照合しました．subjectと物理EEG sourceをsplit間で分離しています．各windowは8 EEG電極・256 Hz・448 samplesです．録画／行動ログ由来の観測行動ラベルであり，意図の正解ではありません．

3モデルともL2 logistic regression，balanced class weightsを使用しました．6候補のCはtrainにfitし，validation balanced accuracy，macro-F1，小さいCの順で選択しました．testによる再選択はなく，各モデルのtest予測は1回です．

## 3クラスの元評価

| モデル | 選択C | Accuracy | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| EEG spectrumのみ | 0.1 | 52.71% | 34.48% | 0.3443 |
| Coarse contextのみ | 0.001 | 34.46% | 56.68% | 0.2941 |
| EEG＋coarse context | 0.1 | 57.36% | 39.28% | 0.3795 |
| Train最多のmouse固定 | — | 54.70% | 33.33% | 0.2357 |

### クラス別 precision／recall

| モデル | クラス | Test support | Precision | Recall |
| --- | --- | ---: | ---: | ---: |
| EEGのみ | Keyboard | 885 | 47.83% | 41.02% |
| EEGのみ | Mouse click | 1,070 | 56.18% | 62.43% |
| EEGのみ | AI送信 | **1** | 0% | 0% |
| Contextのみ | Keyboard | 885 | 51.40% | 41.36% |
| Contextのみ | Mouse click | 1,070 | 79.53% | 28.69% |
| Contextのみ | AI送信 | **1** | **0.1166%** | **100%** |
| EEG＋context | Keyboard | 885 | 52.03% | 75.14% |
| EEG＋context | Mouse click | 1,070 | 67.60% | 42.71% |
| EEG＋context | AI送信 | **1** | 0% | 0% |

context-onlyは858件をAI送信と予測し，857件が誤りでした．それでも実際のAI送信1件を当てると，このクラスのrecall=100%が3クラスの平均の1/3を占めます．一方，EEG＋contextはその1件を外すため，accuracyとmacro-F1が上回っていてもbalanced accuracyは低くなります．**AI送信の性能は推定不能であり，100% recallを成功率の根拠にはできません．**

## Keyboard／mouse のsubset診断

正解がkeyboardかmouse_clickである1,955件だけを対象とし，予測AIを**誤りとして残します**．予測AIの行を捨てず，keyboardやmouseに再割当しません．元の3クラスモデルのconfusionから求めた記述統計で，二値モデルの再学習，新規test予測，Cの再選択はありません．

| モデル | Accuracy | Balanced accuracy | Macro-F1 | 予測AIによる誤り |
| --- | ---: | ---: | ---: | ---: |
| EEGのみ | 52.74% | 51.72% | 0.5166 | 8 |
| Contextのみ | 34.42% | 35.02% | 0.4400 | 857 |
| EEG＋context | 57.39% | **58.93%** | **0.5693** | 2 |
| Train最多のmouse固定 | 54.73% | 50.00% | 0.3537 | 0 |

この比較ではEEG＋contextが共通2行動のpoint metricsで上回りますが，8人のheld-out subjectsにおける観測結果です．被験者を単位にした信頼区間や独立追試は行っておらず，母集団での有意な改善や因果的なEEG情報の寄与までは確定できません．3-class評価のclass imbalanceと，評価可能なgold label数が重要な制約であることは，今回の出力から確認できます．

## 実行・保存の検証

- WorkstationのColab CLI経由でCPU実行しました．GPUの呼出しや既存8B学習の操作はありません．
- 所要時間25.31秒，単一process peak RSS 1,170,587,648 bytes（約1.09 GiB）でした．観測したBLAS／OpenMP poolはそれぞれ2 threadsです．
- 実測版：NumPy 2.4.6，SciPy 1.16.3，scikit-learn 1.6.1，threadpoolctl 3.7.0です．
- C選択前後・test後にmanifest，split，prepared波形本体を再照合し，変更がないことを確認しました．
- 13成果物をworkstationへ回収し，ZIP全体・各member・report・model係数のSHA-256を確認しました．

元の集計は `comparison_report.json`，全Cのvalidation結果は `selection.json`，今回追加したsubset診断は `binary_keyboard_mouse_subset.json` に保存しています．source codeは実験時から変更していません．subset集計は親の依頼に沿う事後的な記述統計であり，モデル選択には使用していません．

## 解釈の限界

公開EEGはoffline filter済みで，4.125秒の先読み仮定と配布binaryの対応は未検証です．mouse movement／EMG等も交絡し得ます．このcohortはeventに対応して選ばれたwindowで，no-actionを含む連続した操作環境を評価していません．LLMとは表現・最適化・振幅正規化も異なるため，差をモデルサイズだけに帰属できません．`latent_intent_goal_validated=false`，`autonomous_agent_validated=false`を維持します．

データ公開元は [Zenodo 20801101](https://zenodo.org/records/20801101) です．元report SHA-256は `3c93b42fbba07aecc55f551c41bae98e31422a5e743afd245254cef448dbfaaf`，manifest SHA-256は `af878638c2fff6d6c7f619cd22005a057e797f1152553517bf05ce795aec2eac`です．
