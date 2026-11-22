# S Macでの取得・追加学習（2026-11-22）

## 到達点

**37台で公開データの分散取得を継続し，CBraModへ3 source・固有17.3991666667時間のEEGを実際に投入しました．** 初期FIF，CNT追加，InMID追加の3段階を，各10epochsで完了しています．取得・展開・新規学習・checkpointはS Mac側に置いています．smac19は除外，smac31は管理用，smac40はDNS未解決で割当なしです．

全量取得は継続中です．23TBの候補全体を学習済みとする結果ではありません．量は[学習集計](sequential_training_aggregate.json)と[取得snapshot](acquisition_snapshot_final.json)で区別しています．

## 取得量

観測は **2026-11-22 19:51:41.856443 JST** です．

| 指標 | 値 |
| --- | ---: |
| 割当済み取得候補 | 598,556 files／23,071,349,438,764 bytes |
| 取得完了 | 40,187 files／175,035,084,578 bytes |
| 取得途中 | 26,550,729,130 bytes |
| 完了＋途中の保存量 | 201,585,813,708 bytes |
| 稼働supervisor／実データのあるworker | 37／37台 |
| 完了ファイルの欠損／サイズ不一致 | 0／0 |
| 観測エラー | 0 |

取得bytesにはmetadataと非EEG modalityも含みます．snapshotではlive DBのcomplete状態と実ファイルの存在・サイズを照合しました．全payloadをこのsnapshotで再hashした意味ではありません．取得時のchecksum確認と，各学習sourceの再検証は別receiptです．

既知容量・公開条件・pairing・version/path衝突を確認したものを割り当てています．未解決metadata784件，残るpath衝突12件，容量不明42,028 groups，EEG scope未確認2,924件は，理由付きで調査を継続します．各カウントの単位は異なり，合算しません．InMIDの同名MAT衝突はprovider identityによる名前空間化で解消し，URL・サイズ・SHAの組を変えず237 filesを追加しました．

workstationの今回取得分773 files／50,476,724,659 bytesは既に削除し，モデル・評価・台帳を保持しています．過去の49.54GB／255.33hを現在の保持量へ加算しません．rawのworkstationへの一時中継・回収は0です．

## 連続したモデル系統の実学習

全段階で公式CBraModの同じtrainer sourceを使用し，追加段階は直前の選択済みbestを継承しました．旧phase3の小型encoderとは別系統です．

| 段階 | 新規の固有学習時間 | 累積固有時間 | 実fit | 結果 |
| --- | ---: | ---: | --- | --- |
| Zenodo806023，16ch | 0.5491666667h | 0.5491666667h | 10epochs／110steps | 完了 |
| Zenodo2392979，8ch | 0h | 変化なし | 0steps | 固定QCで全新窓不適格 |
| Figshare2061654 CNT，14 EEG | 2.8208333333h | 3.3700000000h | 10epochs／640steps | 完了 |
| InMID，14 EEG | 14.0291666667h | **17.3991666667h** | 10epochs／3,280steps | 完了 |

最終modelが実際に見たtrainは **20,879非重複3秒窓／62,637秒** です．各段階の同じbase窓を固有量へ再加算しません．3段階合計のoptimizer更新は4,030回，反復提示は255,820窓で，これらを固有EEG時間と混同しません．旧phase3の29.985888889hも，このCBraModの入力時間には加えません．

| 実行 | node | batch | fit timer | sampled memory |
| --- | --- | ---: | ---: | ---: |
| 初期 | smac01 | 64 | 18.613秒 | 868,352,000 B |
| CNT追加 | smac10 | 64 | 69.759秒 | 1,137,213,440 B |
| InMID追加 | smac35 | 64 | 332.900秒 | 1,603,256,320 B |

各段階はMPS native／float32，AdamW，LR 1e-4です．登録した全bucketに対しbatch1／8／32／64でbackwardと有限非ゼロgradientを実測し，40GiB枠内のbatch64を採用しました．memory値はsample観測で，true peakではありません．test窓・behavior labelの学習使用は0です．

## 固定validationと解釈

既存806023の327窓・2グループを全段階で保持し，固定maskのグループ平均masked MSEで各段階のcheckpointを選びました．新sourceのvalidation/testを最適化には使っていません．

| checkpoint | 固定validation MSE |
| --- | ---: |
| 公式初期weights | 0.0021402554759976803 |
| 初期806023学習後 | 0.0019222948831695787 |
| CNT追加後 | 0.001595536042884798 |
| InMID追加後 | 0.0016149027567592268 |

**InMID追加後はCNT追加後より誤差が1.213806％増えました．** データ量の増加が，この既存domainのvalidation改善につながったとは言えません．段階別checkpointを全て保持しています．これらはoffline波形再構成の指標で，CNT／InMIDへの汎化，行動decode，自由意図，EEG→LLMや実OS操作の改善は評価していません．

## 適格性，単位，分離

- **806023**：元file SHA・physical-V payload・校正・ordered montageを照合し，train659窓／val327窓を準備．test2記録は信号を開かずdenyへ保持しました．
- **2392979**：36files取得後，重複aliasとmust-linkを維持して21train候補を検査．16件はnative ±0.1V上限，5件は固定window QCを満たさず，新規train窓0でした．test10は未読込です．単位倍率や閾値を変更して通過させず，[blocked receipt](ssvep2392979_preparation_result.json)を保持しています．
- **CNT2061654**：14 EEG＋2 EOG，250Hz．14train記録から3,385窓を受理し，6val＋6testはZIPから展開していません．元int16 payload，header gain/baseline，byte layout，独立MNE校正を照合．EOG・event labelはモデル入力へ加えていません．
- **InMID**：58chからordered14 EEGだけを選び，44auxとMAT/videoをEEG時間へ加算しません．76train記録のdigital／physical-V／校正hashを照合．19,228候補窓から100µV超の2,393窓を除き，16,835窓を受理．16testは未読込です．

窓の最大peak100µV，最小channel標準偏差0.01µV，native ±0.1Vの条件を現物QC前に固定しました．3–40Hz causal SOS，2秒warmup，非重複3秒窓，出力200Hz／V_x_1e4です．128HzのInMIDはupsamplingなのでLP80Hzを用いず，1 native sample遅延の補間を別protocolへ固定しました．元sourceの過去のoffline因果性は未確認のため，online_harness_allowed=falseを維持します．

InMID prepの独立確認は21項目PASS．96 artifacts／717,266,688 bytesを全SHA・shape・dtype照合し，新76 arraysのQCも再計算しました．保存値の最大peak99.9783814µV，最小channel std3.32668884µVです．元rawの信号を再読込した検証ではなく，元校正証跡との照合とprepared配列の検証です．

## 検証と実行環境

取得34，fleet17，trainer9，初期prep6，SSVEP追加prep7，CNT prep8，InMID prep8の **89 focused testsをrootが再実行し，全件PASS** しました．実MPS・実dataのfitとcheckpoint検証は別に実施しています．

初期encoderは33項目，CNTとInMIDは各59項目のread-only独立検証を通過．全210 trainable tensors・12 encoder層の更新，optimizer momentとstep，parent継承，seen-IDと固有区間，固定valの再計算を確認しました．[初期](independent_verification.json)と[CNT](cnt_independent_verification.json)の集計を保存しています．[最終InMID modelの独立検証](inmid_independent_verification.json)も保存しました．最終段階のbestは当該10epochs内での最良値で，既存validation全段階の最良はCNT checkpointです．

初期probeのgit metadata不足，CNTのApple Git shimのXcode license gateは，それぞれ学習前に停止しました．正規shallow checkoutと，公式配布SHAを照合した独立portable Gitで実行環境を整え，同じ科学sourceと入力を使って再開しました．検証処理を省略せず，失敗証拠も保存しています．

## 主要runとSHA256

| 対象 | 値 |
| --- | --- |
| upstream commit | b9e961003214326972c567eff390e75b0287e32a |
| trainer SHA | ad0d8c27352411db52640c077af4b4bb38c6b1652ea20f3e6e965af2c172ca8a |
| official weights SHA | 0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178 |
| 初期best SHA | 3450e90e0c2cdf8b0b7f2fa28d39ae5037f7d9b70fea95cc5d93a885d98e6d10 |
| CNT best SHA | 0d9a9cc83f5aa0a4b25404de286eff7a86fa7af462649a8e3df54f104ff82817 |
| InMID best SHA | 600f1e25e89d5229ea70e5d8a4f873b8653dfc445503850d77ff4739bc184ad5 |
| InMID training protocol SHA | a1ca92383005bafcce2cd877deb56d818712f4d0e368d9ea8eed2fe26240966c |
| 現fleet plan SHA | 155a332f2336913abab7b37052cf0148b6a58a563454147fef4d303e1f5a29f3 |

最終checkpointはsmac35の `/Users/yasufumi/pj-xavier/smac35/runs/smac806023_CNT_plusInMID_ssl_20261122_v1/run/best.pt` です．原EEG，個別ID・予測，weights本体，認証情報はpublic reportへ含めません．

取得は各node1stream／100GiB reserve／600秒cycle／48時間のsupervisor枠で継続します．既存の毎時automationが実jobと台帳を観測し，新たに適格になったsourceを次のimmutable stageへ加えます．新sourceの一般化評価とEEG＋環境→LLM/行動の段階は引き続き未達で，固定した別protocolで進めます．[最新HANDOFF](../../HANDOFF.md)を参照してください．
