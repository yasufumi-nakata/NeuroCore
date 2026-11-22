# Frozen UniBa CPU baselines

既存8B LLM実験と同じ固定 manifest，subject split，prepared EEG本体を使用した，L2正則化 logistic regression の比較です．2026年11月22日に workstation の Colab CLI 経由で CPU 実行を完了しました．GPU処理，元 manifest の変更，新規ラベル生成，test による再調整はありません．

## 依存関係

```bash
python -m pip install -r experiments/pj_xavier/baselines/requirements.txt
python -m unittest discover -s experiments/pj_xavier/baselines -p 'test_*.py' -v
```

通常の依存範囲は `requirements.txt`，今回の Colab 実測版は `requirements.colab_observed.txt` にあります．実測環境は NumPy **2.4.6**，SciPy **1.16.3**，scikit-learn **1.6.1**，threadpoolctl **3.7.0** です．範囲内のすべての組合せを検証した意味ではありません．`threadpoolctl` は必須で，NumPy／SciPy／scikit-learn の native thread pool をそれぞれ2以下に制限します．

## 固定した比較

| モデル | 入力 | 特徴数 | validationで選択したC |
| --- | --- | ---: | ---: |
| EEG spectrum only | 8 EEG電極のスペクトル特徴 | 72 | 0.1 |
| Coarse context only | 決定前の実験page/event，AI利用可否，両者の組合せ | 26 | 0.001 |
| EEG＋coarse context | 上記の連結 | 98 | 0.1 |

Welch PSD は256 Hz，448 samples，Hann window，nperseg=256，noverlap=128，nfft=256を固定しています．各電極の4帯域（4–8，8–13，13–30，30–40 Hz）の log10絶対・相対powerと，3–40 Hzのlog10総powerを使用します．時間・subject ID・record ID・target event・キー／プロンプト内容は特徴に入れません．context は upstream の定型文から2変数を抽出し，辞書の fitting は train のみです．

`class_weight='balanced'` とし，Cの候補は `[0.001,0.01,0.1,1,10,100]` を事前固定しました．scalerとclassifierはtrainだけにfitし，validation balanced accuracy，次にmacro-F1，次に小さいCの順で一意に選択します．train＋valでのrefitはしません．3モデル全ての選択結果を保存してから，各モデルに対してtest予測を一度だけ実行しました．

`protocol.uniba_v2.json` が manifest と LLM側の `training_membership.json`／`eeg_artifacts.json` の SHA-256 を固定します．実行前，モデル選択後，test評価後に，全prepared waveform本体のhashとsplitを再照合します．source code自体も起動時の期待hashと照合します．すでに存在するoutput directoryやtest消費markerを再利用しません．

## 実行結果

| 入力 | Test accuracy | 3-class balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| EEGのみ | 52.71% | 34.48% | 0.3443 |
| Coarse contextのみ | 34.46% | 56.68% | 0.2941 |
| EEG＋coarse context | 57.36% | 39.28% | 0.3795 |
| Train最多クラス固定 | 54.70% | 33.33% | 0.2357 |

**AI送信は train=2，val=2，test=1 のため，性能を推定できません．** context-onlyはtest全体の858件をAI送信と予測し，正解は1件だけでした．AI precisionは0.1166%ですが，その1件のrecall=100%が3-class balanced accuracyの平均の1/3を占めます．この指標単独でモデル順位や実用性を判断しないでください．

正解がkeyboard／mouse_clickの1,955件について，予測AIを誤りのまま残した記述統計では，EEG＋contextのbalanced accuracyは58.93%，macro-F1は0.5693です．これは元のconfusionを集計したもので，二値分類モデルを再学習した結果ではありません．詳細は `../reports/cpu_baselines_uniba_v2_lead4p5_v1/aggregates/` の report と README にあります．

全3モデルのfeature抽出，validation選択，test評価，入力再照合まで **25.31秒**でした．単一processのpeak RSSは1,170,587,648 bytes（約1.09 GiB）で，観測したBLAS/OpenMP poolはすべて2 threadsでした．trained係数とscalerはpickleを使わない `.npz` でprivate回収済みです．

## 限界

このcohortは実際に記録された行動に対応するwindowを選んだものです．潜在的な意図の正解や，no-actionを含む連続運用を検証していません．公開EEGはoffline filter適用済みで，4.125秒の先読み仮定と配布binaryの対応も未検証です．movement／EMG等の交絡が残ります．testは8 subjectsで，1,956 windowsを独立被験者1,956人として扱えません．

LLMと表現・最適化・EEG振幅正規化が異なるため，差をパラメータ数だけの効果に帰属できません．小型baselineの失敗から，EEGに情報が存在しないとも結論できません．`latent_intent_goal_validated` と `autonomous_agent_validated` はともに false です．

## 監査情報

- 本体source SHA-256：`1b74445e9a85c34c86eaa435805b427ba8dbee4d675f6d3e0a4b301b1c10ff08`
- Manifest SHA-256：`af878638c2fff6d6c7f619cd22005a057e797f1152553517bf05ce795aec2eac`
- 元report SHA-256：`3c93b42fbba07aecc55f551c41bae98e31422a5e743afd245254cef448dbfaaf`
- Colab実行前に10 fixture tests成功．追加の二値subset集計は2 tests成功し，実験source／予測／選択結果を変更していません．
- 13成果物をZIP全体と各memberのSHAで照合し，workstationへ回収しました．Gitには集計とコードを保存し，モデル係数や実行logはprivateに保持します．
