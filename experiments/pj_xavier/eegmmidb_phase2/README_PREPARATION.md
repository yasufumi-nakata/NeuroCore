# EEGMMIDB phase 2：因果的な運動想像入力

この準備処理は，PhysioNet EEG Motor Movement/Imagery Dataset 1.0.0 の取得済み ZIP から，運動想像の二つの課題用に実 EEG の入力を作ります．過去の AI-assisted coding／SSL 実験と，そのデータ・モデル・評価条件は変更しません．新しい出力は `private/xavier/eegmmidb_phase2_v1/prepared/` です．

## 固定した対象と分割

| 項目 | 条件 |
| --- | --- |
| 参加者の割当 | 109 群を train 76／validation 16／test 17 に固定 |
| 割当方法 | `SHA256("20261122|physionet:eegmmidb:1.0.0:Sxxx")` の昇順 |
| 主課題 | runs 4／8／12，T1 左拳の運動想像，T2 右拳の運動想像 |
| 副課題 | runs 6／10／14，T1 両拳の運動想像，T2 両足の運動想像 |
| 注釈の保留 | 既存監査の 50 記録全体を除外，注釈修正は 0 件 |
| 入力 | 全 64 ch，物理単位 V，float32 `[N, 64, 600]`，200 Hz，3 秒 |
| 主窓 | T1／T2 の onset + 1 秒から + 4 秒，一つの注釈につき一窓 |
| 環境情報 | 静的な課題種別だけをモデルに渡す |

割当後に記録や窓が除外されても，参加者の所属を変更しません．各原本 EDF の SHA256 と，注釈・ヘッダを含まない EEG ペイロードの SHA256 の両方で，分割間の同一信号混入を拒否します．T1／T2 は実験で提示された課題条件の記録です．自由な意図や，実際にその運動を想像したこと自体を直接観測したラベルではありません．run 番号，cue，trial 順序，参加者 ID，正解ラベルはモデル入力から除きます．

## 因果的な処理と品質条件

EDF のデジタル値を，元ヘッダの線形校正に従って µV に変換し，V へ一度だけ換算します．65 番目の注釈チャネルは EEG へ混ぜません．記録開始時にゼロ状態から始める 4 次 Butterworth 3–40 Hz の SOS フィルタを，`sosfilt` で順方向に適用します．`filtfilt`，記録全体の平均・標準偏差による正規化，クリッピングは使用しません．

128／160 Hz から 200 Hz への変換は，因果的な `upfirdn` です．最大倍率の 20 倍 + 1 tap，Kaiser β=5 の低域 FIR を固定し，先頭の遅延補償はしません．FIR の群遅延は 160 Hz 入力で 62.5 ms，128 Hz 入力で 78.125 ms です．IIR の遅延は周波数によります．記録先頭 5 秒より前に始まる窓を使用しません．対象の 3 秒区間と，フィルタが参照する記録先頭からの履歴を，別々に保存します．

品質条件は，フィルタとレート変換の後の V 配列で測定し，次のすべてを要求します．

- 全サンプルが有限値であること．
- 全チャネル・全サンプルの絶対振幅の最大値が **500 µV 以下**であること．peak-to-peak の条件ではありません．
- 各チャネルの 3 秒内の母標準偏差（ddof=0）が **0.01 µV 以上**であること．

最初の候補値 100 µV は過去の CBraMod 実験に由来します．今回の scratch encoder／64 ch の独立 run では，train のみの確認により，その条件が対象 72 群のうち 29 群を全窓除外することが分かりました．train の記録だけで得た分布を基に，粗大な振幅の保留条件として 500 µV を探索的に選択しました．この数値は生理学的な正常性の保証でも，既存論文の標準閾値でもありません．100 µV を超えた窓には引き続き品質フラグを付けます．既存の 100 µV 診断と初期 metadata は保存し，最終の決定理由・処理契約を validation／test の波形を読む前に別ファイルへ固定しました．保持数や性能を見て以後の閾値を変えません．

## 対照と休止診断

各主窓の **10 秒前**を，同じ記録の同じ因果的ストリームから取り出します．履歴不足や同じ QC 条件の不合格で対照を用意できなくても，主窓は残します．対照が利用できる主窓 ID を `paired_control_subset.jsonl` に固定し，対応のある fused／shuffle／shift 比較を同じ ID 集合で実行できるようにします．全 test 主窓の成績とは別の集計対象です．

T0 は `idle_manifest.jsonl` に保存する独立した test-only 診断です．左右・両拳／両足の学習やモデル選択には使用しません．全 split の T0 を追加すると保守的な容量見積もりが 4 GiB を超えるため，別の metadata 契約で test-only に限定しました．同じ onset + 1～4 秒，履歴と QC を適用します．この診断で測れるのは実験内の休止区間に対する固定閾値 0.8 での採用割合です．自然な連続操作における誤操作回数／分や自律的なエージェント性能を示すものではありません．

## 成果物と原本の保全

`manifest.jsonl` と `artifacts.json` は主窓だけを指します．`control_artifacts.json` と `idle_artifacts.json` は別々の配列台帳です．配列は recording ごとに分け，ファイル内に partition を混在させません．SHA256，bytes，shape，dtype，原本の SHA，原本・変換後の sample indices，時刻，課題ラベルの由来を保存します．除外した記録・窓と，理由も別台帳に残します．

元 ZIP は最初と最後に全体 SHA256 を確認し，実際に使用した各 EDF member も使用前と最後に再照合します．原 ZIP の一括展開は行いません．モデルの forward／fit は準備プログラムでは実行しません．

処理の上限は CPU 2 threads，登録時から 30 分の絶対期限，このフェーズの増分 4 GiB です．35 GiB の空き領域に加え，既存 bulk 取得の将来増加 7,000,000,000 bytes を予約します．容量・期限を超えて自動的に続行する処理はありません．

## 再現用のコード

- `prep_causal.py`：初期候補 100 µV の train-only 診断を実行した固定コード．
- `prep_causal_v2.py`：500 µV と初期品質フラグを固定した実際の最終準備コード．
- `metadata_protocol.json`：信号を確認する前の対象・分割・窓・予算の契約．
- `final_QC_decision.json`：train-only の品質条件の決定理由．
- `preprocessing_protocol.json`：実処理の code／dependencies／tests／diagnostics の SHA を含む契約．
- `preprocessing_tests_v2.json`：workstation 上の12件のテスト結果．未来サンプルを改変しても過去の出力が bit-exact であること，prefix だけの処理との一致，MNE と独立の単位換算，時刻・境界・容量・上書き拒否を確認しています．

依存は `requirements.preparation.txt` に記載しています．実行した環境の厳密なバージョンは処理契約に保存しています．既存の出力を上書きする再実行は禁止し，新しい出力と事前に固定した契約で行ってください．

## 一次資料

- [PhysioNet EEG Motor Movement/Imagery Dataset 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/)
- [MNE EEGBCI run mapping](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html)
- [SciPy sosfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html)
- [SciPy upfirdn](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html)

これらの資料は原データと信号処理 API の根拠です．今回選択した 500 µV の品質条件を，これらの資料に由来する標準として扱っていません．
