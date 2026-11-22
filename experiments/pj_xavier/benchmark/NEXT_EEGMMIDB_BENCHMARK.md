# 次段階：EEGMMIDB の公開課題ラベルによるデコーダー検証

**状態：設計案のみです．新 benchmark の実装・学習・推論は未実施で，このデータのモデル入力は0件です．停止済み Colab を再起動せず，次回は別の固定 protocol と新しい実行予算から開始します．**

## 1．目的と，確認できた入力

検証対象は，視覚 cue に従った運動想起（MI）の課題クラスを，EEG から未学習の参加者群で識別できるかです．公開 annotation を観測課題の gold として使いますが，実際の想起内容を直接測った正解，自由な agent intent，実課題達成のラベルとは扱いません．視覚誘発反応，眼球・筋活動の寄与も残るため，成績だけで「純粋な意図の解読」と結論しません．[公式説明](https://physionet.org/content/eegmmidb/1.0.0/)

[取得・全記録監査](../reports/eegmmidb_acquisition_20261122_v1/README.md)で，ZIP **1,997,435,063 B**，109公開 subject groups，1,526 EDF，**174,724秒＝48.534444時間**を確認しました．64 EEG channels，全payload内の完全同一波形重複0，実 rate は **160 Hz：1,490記録／128 Hz：36記録**です．全件を160 Hzと仮定しません．この時間は保存波形の合計で，想起窓だけの時間や独立した人の経験時間ではありません．

| 課題の提案 | run と T1／T2 の意味 | 扱い |
| --- | --- | --- |
| 主評価：左右の拳の想起 | 4／8／12，左拳／右拳 | 二値MI分類 |
| 副評価：両拳と両足の想起 | 6／10／14，両拳／両足 | 主評価と別の二値分類 |
| 実運動，開眼／閉眼baseline | 3／7／11，5／9／13，1／2 | MIへ混ぜず，別目的の診断のみ |

対応は [MNE公式表](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html)と run 固有のannotationから固定します．T0は当該課題の休止として別管理します．MIの全記録保存時間は22.425278時間ですが，休止区間も含むため学習可能なMI窓の時間は未確定です．

## 2．ラベル・分割・因果性

- **50／1,526記録を要確認として保持します．** 区間重複547件は5–50 ms，範囲外12件は各25 ms，別にbaseline2記録のT1があります．小さい差を丸めと断定せず，baseline T1を左右クラスへ変換しません．未解決の該当窓は採点対象から外す案とし，元TAL・修正理由・前後時刻・除外数を台帳へ残します．残り1,476記録も信号QC済みではありません．独立MNE代表照合は別の限定監査として確認し，standalone annotation と Raw reader の自動cropを区別します．[ラベル監査](../reports/eegmmidb_acquisition_20261122_v1/header_label_review.json)
- **分割は参加者群単位です．** 案は固定seedとdataset-qualified subject IDのhash順位で train／val／test＝76／16／17群です．全run・全窓・派生配列を同じ所属へ固定し，QC後に別partitionへ移しません．同一trial由来の窓は分割しません．原波形hashと区間で重複を検出し，主評価は1 trial 1窓・窓同士非重複とします．これは新しい案であり，所属一覧はまだ生成していません．
- 窓はcue後に固定長で取り，判定時刻を「最後の入力sampleが利用可能になった時刻」へ置きます．cue前の意図予測と表示しません．sample indexは実160／128 Hzから計算します．filter・resamplingは過去だけを使う実装，状態の初期化とwarm-up，遅延を固定し，未来sampleを変更しても既存出力が不変であることを検査します．zero-phase処理や全記録からの標準化を黙って使いません．
- LLMへ渡せるのは全クラス共通の課題説明と候補集合までです．**当該cueの位置／T1・T2／正解文／annotation channel／正解を推定できるtrial順序・ID・時刻表を入力から除外**します．評価用cue時刻は窓作成にのみ使い，その時刻を知る評価設計であることを明記します．

## 3．次回 protocol で固定する項目

| metadataだけで先に固定 | train信号のQC・容量確認後，test推論前に固定 |
| --- | --- |
| release・ZIP・公式manifest・台帳のhash，課題対応，subject split，50記録のreview規則，cue非開示，主／副指標，候補モデル数，予算 | 実電極名の対応，単位変換，因果的rate統一，filterと遅延，窓長・cue後offset，QC閾値と除外一覧，配列hash，モデル・初期重み・学習設定，計測資源上限 |

QCの統計・変換器・標準化はtrainだけから決め，同じ規則をval/testへ適用します．held-outの性能や保持数を見て規則を変えません．未解決事項が残ればblockedにし，protocolを後付けで上書きしません．

まず新規のCSP＋線形分類器等の小さなdecoderを基準にし，別の新規EEGモデルと比較する案です．既存モデルは読取専用の初期化候補に留め，利用時は事前学習へのEEGMMIDB混入の有無を確認します．不明なら独立holdout性能の主張を限定します．**既存models・UniBa・ITUへ追加fitせず，旧test結果で設定を最適化しません．** 新runの選択は新valのみ，testは最良checkpoint・しきい値を封印後に1回です．testを見た変更は別の探索実験に分けます．

評価には EEG，cue非開示contextのみ，融合，同記録内EEG入れ替え，実波形の固定時間移動を用意します．入力除去と別学習baselineを区別し，同一ID・原波形証拠で比較します．BA／macro F1／二値AUROC，校正，固定採用しきい値のcoverage，参加者群単位の対応付き区間，実メモリと遅延を報告します．連続運用の誤操作／分や実課題完了率はこの評価から算出しません．

## 4．容量・予算と自動再開の順序

展開全体は3,614,967,988 Bです．全EEGをnative rateのfloat32にすると**計算上7,120,224,256 B**，約6.63 GiBで，窓複製・変換・checkpointは別です．全量一括RAM読込みを避け，記録単位またはmemmapで処理します．35 GiBの空きを残し，展開・派生・回収の最大同時占有を事前計算します．学習時間は未測定なので，データ時間から実行時間を推定しません．[容量の根拠](../reports/eegmmidb_acquisition_20261122_v1/summary.json)

次回の候補予算はCPU QC最大30分，GPU枠最大60分（train-only容量試験5分，学習30分，固定評価15分，回収10分）です．これは予約・実行済み予算ではありません．新protocolへ絶対期限，CU上限，機種，GPU上限（L4なら候補21 GiB）を明記し，容量試験の実測で実行可能性を判定します．不成立なら計画とblocked理由を保存して終了します．

**再開順：** 原本／監査hash照合 → 新metadata契約とsubject所属の保存 → label review・train QC・因果性検査 → 配列／学習契約の封印 → 新しい容量・予算確認 → val選択による学習 → checkpoint封印 → test1回 → 回収／hash照合／集計 → 所有session停止．Colabの作成・転送・実行・回収はすべて **workstation上のColab CLI経由**とします．旧sessionの期限や完了receiptを再利用しません．

根拠hash：取得集計 `015cc62a…`，ラベル監査 `bc6105f2…`，全記録台帳 `51b801d0…`．本メモは次回の設計入力であり，学習用manifestや実行許可を代用するprotocolではありません．
