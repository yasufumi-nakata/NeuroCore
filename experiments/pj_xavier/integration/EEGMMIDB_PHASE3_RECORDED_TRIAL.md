# PJ-Xavier：追加SSLモデルのMac実機試用

固定済みのEEGモデルを，保存済みの学習例3件で実行するための入口です．EEGから操作クラスを判別し，凍結したSmolLM2-1.7Bが環境の対応表から対象IDを選び，有界のメモリ内状態を更新します．

## 1コマンドで実行

このworktreeのルートで，次を実行します．準備済みの `.venv-xavier` とローカルの固定モデルキャッシュを使います．

```bash
python3 scripts/try_pj_xavier_recorded.py
```

結果は `private/xavier/eegmmidb_phase3_v1/recorded_trials/` に，日時と乱数を含む新しいフォルダとして保存されます．`launcher_completion.json` に全体結果，`result/summary.json` にモデルの出力・資源観測，`result/trace.jsonl` に12回の環境遷移を記録します．既存の試用記録は保全します．

出力には棄権数も表示します．棄権が12件の場合も，モデル読込みと12回のLLM処理が完了した事実を記録します．この実機試用の成否と，課題を正しく遂行できた件数は別の項目です．

## 固定する条件

| 項目 | 条件 |
| --- | --- |
| モデル | `eegnet_ssl_smollm_fused_v1` のvalidation選択best |
| EEG表現 | 全64チャネル，物理単位V，200Hz，3秒 |
| EEG事前学習 | 75名のtrain被験者の連続EEG，元test17名を除外 |
| LLM | SmolLM2-1.7B-Instruct，固定revision `31b70e2e869a7173562077fd711b654946d38674`，凍結 |
| 例の選択 | phase2と同じ，IDの固定ハッシュ順で選んだtrain3例 |
| 環境 | 1例あたり4対応表，合計12エピソード |
| LLMルーティング | 12回すべて実生成，キャッシュ再利用なし，EEG prefixを再注入せず判別クラスを入力 |
| 採用条件 | 2候補の最大確率0.8以上．未校正のモデル出力値 |
| 実行装置 | M1のMPS，8GiBのモデルメモリ予算 |
| 時間 | 試用wrapperは最大300秒 |
| 外部呼出し | オフラインのローカルモデルのみ |
| 状態更新 | メモリ内のテスト環境のみ |

公開データの64チャネル波形による動作確認です．別のEEG装置の入力を使うには，電極配置，物理単位，サンプリング周波数，因果的前処理を揃えた追加検証が必要です．

## モデルと証拠の固定

実行前に，選択モデル，元の3例，配列，科学ソース，学習履歴，最良validation記録，自己教師学習の選択重みとその由来を照合します．独立監査では初期重みも含むvalidation最良選択とtest被験者の除外を確認し，全実行成果の回収完了も必須条件にしています．

| 識別子 | SHA-256 |
| --- | --- |
| 試用契約 | `3ae5f2dfce28994b733adeb93eb4be419a8820455813d4d695c746ab74e34d39` |
| 試用wrapper | `9893da260cc1c01b2e47bd711ce493583ec1baecda278c018df56ee6888fc5aa` |
| 1コマンドlauncher | `fe5db82ec92fe417f6798db6f10a36a318b7f8d2b60fbef977ed6fd40c3939da` |
| EEG＋LLMのbest | `b2373bc054dbdd883236b417d3679744e179f8b067c2289c6191984a8b0de471` |
| 選択SSLエンコーダーのseal | `0ade2cc2ebddb52fe1d53cf2e88957c71fdcfa085263c5b9afed4bac29a7576b` |
| phase3科学ソースのseal | `4c713b5112ebae88885a7c6dae7dc429722df2543d78278a38de7174cf396be7` |

契約は `private/xavier/eegmmidb_phase3_v1/recorded_trial_contract_v1.json` に保存し，launcherはこのSHAを固定しています．モデルや入力を選び直すコマンド引数は設けていません．

## 今回の実測

2026-11-22に，上記の1コマンドから1回だけ実行しました．内側のモデル・routing処理は **18.745秒**，子プロセス全体は **21.194秒**で完了しています．同じtrain3例を判別し，4環境ずつ，**合計12回のfresh routing**を行いました．固定0.8閾値では12エピソードすべて棄権し，成功操作0，誤操作0，不正操作0，実OS操作0，cloud API0，test forward0でした．

EEG3例のbatch呼出しは1,667.972ms，1例あたりの償却値は555.991msです．実際の連続入力1件の端から端までの遅延とは区別します．MPS sampled driverは4,381,786,112 bytes，sampled allocatedは3,423,507,712 bytesでした．true peakではなく，同じworkstationで他の軽いCPU処理も動いていたため，隔離ベンチマークとは扱いません．

今回の出力は `private/xavier/eegmmidb_phase3_v1/recorded_trials/20261122T061619Z_7ed9dfabc696/result/` にあります．summary SHAは `bba97def9083f42b7822cd7444b542df01780d160b5b7dee3c05a7a2ced4551d` です．モデル重みと全476ファイルの必須成果を回収後，所有Colabは停止済みです．

保存した3例はいずれも学習データです．この試用は，未知の被験者への精度，日常の自由な意図の読取り，実環境での自律操作の成否を推定するためには用いません．未学習被験者の分類結果と，EEG除去・入替え・時刻ずらし・休息区間の診断は，別の固定ベンチマーク結果を参照してください．

[同じ未学習被験者での固定5モデル比較](../eegmmidb_phase3/RESULTS.md)と，[回収・試用・停止の完了記録](../reports/eegmmidb_phase3_completion_20261122_v1/README.md)を参照してください．
