# EEGMMIDB：全 run を用いた連続 SSL 入力の準備結果

**学習用 75 群から，固定 QC 後の連続 EEG 29.887500 時間を準備しました．** 検証用は別の 16 群・6.317500 時間です．これは準備済み入力の量です．実際に勾配計算へ投入した ID と，最終採用 checkpoint に反映された量は，学習の履歴から別に集計します．

| 区分 | 学習用 | 検証用 |
|---|---:|---:|
| 読み込んだ原記録 | 1,014 | 224 |
| 原記録の時間 | 32.171944 h | 7.102778 h |
| QC 後の参加者群 | 75 | 16 |
| 採用した 3 秒窓 | 35,865 | 7,581 |
| 採用窓の区間 union | **29.887500 h** | **6.317500 h** |

全 44,664 候補窓のうち，1,218 窓を固定の 500 µV 基準で保留しました．採用は計 43,446 窓，float32・物理 V・64 ch × 600 samples @200 Hz，1,236 配列で **6,673,463,808 bytes** です．2 記録は採用窓がありません．

## 固定した処理と除外

参加者は phase2 と同じ dataset-qualified ID と seed の割当てです．原本 1,526 記録に対し，test 238 記録と注釈監査で要確認の 50 記録を denylist とし，残り 1,238 記録だけを処理しました．test の原波形は開いていません．SSL 用の action label・cue・event annotation の読取りはありません．

各記録の先頭 5 秒を warmup とし，その後を 3 秒・非重複の固定格子で切り出しました．3–40 Hz，4 次 Butterworth の過去だけを参照する SOS と，遅延を保持した因果的 rate 変換を用いています．filtfilt，全記録の正規化，振幅のクリップ，後付けの単位変更はありません．128/160 Hz の未来サンプル変更・prefix 不変テストを，Colab の実 NumPy/SciPy で raw decode 前に確認しました．

QC は finite，全チャネルの母標準偏差 ≥0.01 µV，最大絶対振幅 ≤500 µV です．500 µV は phase2 の train-only 分布から固定した探索的な粗大振幅基準であり，生理学的正常性を保証する閾値でも，既存論文の標準値という主張でもありません．100 µV 超の quality flag は train 27,122 窓，val 4,862 窓に保持しています．val/test の残存率や精度に合わせた変更はありません．

## MI 入力と重複する時間

既存の MI 学習窓 6,359 件は 5.299167 時間ですが，連続 SSL 窓と大部分が重なります．原記録 SHA ごとの native sample 区間を union し，チャネル数や epoch 数を掛けていません．

| 学習用の区間会計 | 秒 | 時間 |
|---|---:|---:|
| SSL 準備窓 | 107,595.0 | 29.887500 |
| MI 準備窓 | 19,077.0 | 5.299167 |
| 両者の重複 | 18,722.8 | 5.200778 |
| **両者を合わせた準備済み union** | **107,949.2** | **29.985889** |

この union は準備済み全 train 窓の上限です．実際の seen-ID による学習入力 union，選択された SSL checkpoint までの exposure，繰り返し提示回数は別の会計です．epoch0 が選ばれた場合，SSL を試行した量と採用 encoder の SSL exposure=0 を分けます．filter の共有履歴全体を新しい target 時間として足していません．

## 検証と保存

原 ZIP と使用した全 1,238 member の SHA を処理前後で照合しました．独立 metadata 監査で raw SHA／EEG payload SHA／参加者の split 排他，denylist，43,446 個の配列 slot の一意かつ完全な対応，因果的な時刻，source ごとの非重複区間，会計一致を確認しました．この metadata 監査自体はリモートの配列を再ハッシュする検査ではありません．配列本体はモデル側 loader の独立検証対象です．

原 ZIP は既取得の同一ファイルを Colab で取得し，全体 SHA を一致確認したコピーです．**新規の独立取得データ量へは加算していません．** 生成配列は Colab に置き，原本は workstation に保持しています．成果の元となる manifest，source allow/denylist，配列 SHA，契約，実行証跡を 4,414,501 bytes の metadata archive で回収し，24 member の全 SHA を照合しました．

- manifest SHA256: `154fd2becdc5bce63a856d69f5a3a48a588cdc7fb688c656d781fec3f0fcd35f`
- preprocessing protocol SHA256: `69716a6ebb740bc43e7dede5aac1309102186037f442d8a215015e26db2500c1`
- prepared corpus summary SHA256: `b54e838b151fdb96c2d1f45aec6cc684e9e23b5bcb577d76f9020a967a0f6de4`

一次資料は [PhysioNet EEG Motor Movement/Imagery Dataset 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/) です．本表の時間と bytes は，公称値ではなく取得済み原本と今回の固定 manifest の実測値です．
