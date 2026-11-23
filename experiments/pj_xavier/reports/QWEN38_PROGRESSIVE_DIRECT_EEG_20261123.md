# Qwen3.8-27B：直接EEG入力の修正と段階的LoRA比較

2026-11-23 JST．本報告は方法・初動の記録です．新実験の有効性や全量学習完了を報告するものではありません．

## 推奨方針

別途学習するEEG encoderを置かず，固定前処理した数値EEGをQwen言語backboneへ入力し，主要linearへ広くLoRAを適用する方式を第一候補とします．4bit基盤を固定して内部の低rank差分を学習します．基盤の全重みを更新するFullFTとは異なります．まず修正版4層rank8の対照を成立させ，その後，同じrankで全64層主要linearの比較へ進めます．FullFTは，入力・データの妥当性とLoRA容量の制約を確認した後の比較候補です．

LoRAに関する根拠は[Thinking Machines Labの一次研究](https://thinkingmachines.ai/blog/lora/)です．小～中規模instruction/reasoning追加学習のlog loss比較であり，EEGやComputer Useの成功を実証した研究として引用しません．[公式Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B)の能力と，本実験で確認した能力も区別します．

## 旧実験の観測値

同じvalidation155件・6被験者です．最終testは使用していません．

| 条件 | 正答率 | Balanced accuracy |
| --- | ---: | ---: |
| 旧Qwen EEGあり80step | 57/155，36.77% | 30.65% |
| 旧Qwen EEGなし80step | 62/155，40.00% | 33.33% |
| 旧QwenのEEG shuffle | 64/155，41.29% | 34.41% |
| train文脈別多数決 | 71/155，45.81% | 38.17% |

旧条件ではEEGの追加効果が確認できていません．文脈別多数決の詳細は[独立baseline報告](QWEN38_CONTEXT_EVENT_BASELINE_20261123.md)を参照します．6被験者の開発用validationでの結果を一般化の確証にしません．

## 確認した不整合

1. 旧trainの既定thinkingと旧evalのno-thinkingが異なりました．各class3件のCPU監査では旧設定9/9でprefixが不一致，新設定9/9で一致しました．ラベル自体は旧lossに残っていたため，完全に教師信号がなかったとはしません．
2. 搭載MLX default lossがpadding境界を含め，id0のpadding1tokenも学習していました．実batchのCPU監査6/6で確認しました．

新runnerは全train/valでno-thinkingを固定し，shift後のtargetが `prefix_length <= position < token_length` の範囲にある場合だけlossへ入れます．最初の正解tokenと終了tokenを含み，paddingを除外します．搭載package・旧run・基盤重みは変更しません．

## 新しい対応比較

[runner](../model/direct_qwen_mlx_progressive.py)と[CPU tests](../model/test_direct_qwen_mlx_progressive.py)が実装です．

| 項目 | 条件 |
| --- | --- |
| 基盤 | mlx-community/Qwen3.8-27B-4bit，revision b935172e4bb5dcd0543774a34748fb92422e3027 |
| 固定split | train400件/18被験者，validation155件/6被験者 |
| 学習 | 最終4decoder層，rank8，scale20，dropout0，Adam，lr1e-4 |
| 更新予算 | batch1，累積1，80step=80提示=0.2epoch |
| 対照 | 同一元データ・例順・seedで独立学習したnoEEG |
| EEG負対照 | 同一被験者内のEEG blockを固定置換，context/labelは保持 |
| 20/40step | 被験者ごと最大6件の固定val，計最大36件 |
| 80step | val155件全件 |
| 追加診断 | train splitから固定12件，基本指示6問 |
| optimizer | 1回のtrain呼出し中，同じAdamを保持して途中評価 |
| 上限 | 各arm14,400秒，memory50GB；完了時刻の予測ではない |

EEG/noEEGは同じ例・更新回数ですが，入力token数や演算量は異なります．train split probe12件は提示済み例とは限らず，「訓練済み例精度」にはしません．今回の80stepも全trainの1周ではありません．旧runとはthinking・mask・順序・採点規則・noEEG変換が変わるため，修正版の対照を主比較とします．

途中評価の不正形式，基本指示sentryの悪化，非finite/過大loss，memory超過を検知して停止します．品質不合格checkpointは比較から拒否します．sequenceはEEG→noEEGを直列実行し，双方完了後に20/40/80の対応比較を作ります．同一GPU lockとsequence lock，既存run拒否で重複起動を防止します．自動的な層数/rank拡大はありません．all64は同一条件のcorrected4比較reportの実SHAと明示した進行根拠を要求します．

## 検証と実行記録

- CPU tests13件が合格し，独立読取りレビューでcorrected4に追加blockerなしと確認しました．
- 17:29 JSTの実tokenizer preflightで，train400/val155のEEG/noEEG全件とval shuffleのprefix/maskを確認しました．元testは開いていません．
- 17:30:21 JSTにsmac35でsequence/controller PID72772，EEG worker PID72774を起動しました．
- 17:31:10 JSTに両PIDのGPU lock保持，4層31linear，学習可能3,647,744parameter，旧4層のtarget選択と一致，基盤重み固定を確認しました．基本指示baselineは6/6です．
- 17:32:34 JST観測では状態training/step0で，初回の数値更新は未確認でした．起動とgradient実行確認を区別します．後続実測は下記の実行記録を参照します．

runtime namespaceはsmac35の `~/pj-xavier/smac35/qwen38_progressive_20261123_v2/` です．`sequence_state.json`，各armの `state.json`／`supervisor.json`／`module_audit.json`，各checkpointの `evaluation.json`，`comparison_0000080.json` が正本です．個別予測・IDはprivateのままです．workstation引継ぎは `private/xavier/direct_qwen_eeg_v1/continuation_20261123_v2/` です．

Source SHA256：`75eecc4f83c61534a8642d57b97978ee2b0b8cefb3930ad5a26bf6c8b0811155`．Test SHA256：`79ea6a1566445631ef43b4af9337b5a545672991bcca95279b693abf0805b72e`．実機manifest SHA256：`078d88ff9c5d453f0ddc80964b34faa920d85194481e6a926276ba3798b2602f`．

## 解釈と次の判断

修正版の正答率・balanced accuracy・被験者別結果・クラス偏りを，独立noEEG，shuffle，固定の文脈多数決と比較します．学習lossだけで成功とはしません．少数被験者のvalidationは探索的であり，複数seed，未学習session/task，独立testと既存能力保持の検証が後続に必要です．最終testは構成を固定するまで保留します．

現データの3クラスは観測操作です．全trainのfeature_availability_verifiedがFalseであり，配布EEGの実時間の因果性や高次意図のgoldは未実証です．現contextも粗い実験状態で，画面画像や詳細履歴ではありません．

cached checkpointにはvision重みが存在しますが，今回のmlx_lm経路はlanguage backboneのみをロードし，vision重みを除外します．画像入力・vision更新・Computer Useの検証ではありません．6問のsentryも広い能力保持の保証ではありません．

base identityは既存receiptのrevision/ファイル名/サイズとconfig/pathで固定し，全weight内容のSHAを今回再計算したものではありません．Adamはrun内では保持しますが，crash後のoptimizer状態復元は未実装です．停止したrunのadapterを使ったfresh optimizer開始を「完全なresume」と呼びません．

37台の取得workerは継続します．既存の毎時継続タスクと[Google Doc](https://docs.google.com/document/d/1kGlNqvZEg_AzSYKGTiGF3Z0HcVIxNGNimNwpR6w1680/edit?tab=t.vnbigpc1omcd)へ新方針・実行namespaceを引き継ぎます．
