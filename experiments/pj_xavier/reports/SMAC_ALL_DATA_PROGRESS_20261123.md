# S Mac 全候補取得・追加学習の進捗

更新：2026-11-23 JST．ユーザーの「約23.07TBの候補を取得して再学習，VPNも自律的に調整」という指示に基づく実行記録です．全量取得・最終unionの再学習は進行中です．

## 完了した追加学習

**BCI8089820の固定train446記録を全件検査し，適格432記録・55元グループから43.6625時間を追加しました．** S Mac smac05で，既存sourceを含む79,493窓／238,479秒＝**66.2441666667時間**を10epoch・12,460step学習しました．独立検証は準備43／43，model69／69項目PASSです．

| 評価対象 | 学習前 | 学習後 | 同一条件の比較 |
| --- | ---: | ---: | --- |
| 既存806023 validation：327窓・2群，seed20261122，batch64 | 0.001429867321 | 0.001436424875 | MSE 0.4586％増加 |
| 新BCIの固定validation：11,709窓・12群，seed20261123，batch32 | 0.002753569855 | 0.000803375829 | MSE 70.8242％低下，12／12群で低下 |

新BCI評価は学習前に固定した96候補記録を検査し，95記録を評価しました．1記録は既定のnative振幅上限で除外，欠測group0です．この評価によるfit・checkpoint選択・test/review信号の読込は0です．旧BDFの過去のscoreはseed20261123であるため，seedをそろえた親・子の再計算を比較に用いました．

best checkpoint SHA256は `f66903e1dc9120da71c6275e70f121662671aadac13f44efaf92d18e4563954f`．全210学習tensor・12encoder層の更新，親の継承，optimizer，実seen-IDを確認済みです．元のcheckpointとv1のseed比較不一致の記録も保持しました．

数値の根拠と限界は[BCI詳細](smac_training_20261122_v1/BCI8089820.md)，[独立検証集計](smac_training_20261122_v1/BCI8089820_encoder_verification_20261123_aggregate_v2.json)，[固定validation集計](smac_training_20261122_v1/BCI8089820_independent_validation_20261123_aggregate_v1.json)，[系統全体の学習集計](smac_training_20261122_v1/sequential_training_aggregate.json)にあります．波形再構成の結果を，意図推定・LLM/agent・実OS操作の成功率とは扱いません．

## 全候補の取得

2026-11-23 **06:07:31～35 JST** の37台観測です．台帳と完了実ファイルの存在・サイズを照合しました．全234,816完了ファイルを今回一括再hashしたという意味ではありません．

| 項目 | 実測 |
| --- | ---: |
| 元の対象ファイル数 | 598,556 |
| 元の既知容量計画 | 23,071,349,438,764 B |
| 公式ZIP輸送容量の訂正後 | 23,058,905,253,931 B |
| 完了ファイル | 234,816 |
| 完了容量 | 2,489,730,344,939 B（約2.490 TB） |
| 部分取得容量 | 762,176,034,748 B（約0.762 TB） |
| 訂正後容量に対する完了率 | 10.7973％ |
| 稼働supervisor | 37／37 |
| 完了実ファイル欠損・サイズ不一致・観測error | すべて0 |

対象は全modalityを含む取得候補であり，全量がEEG，学習適格，学習済みという意味ではありません．容量訂正はKaggle100件の非圧縮member容量を公式ZIP輸送容量へ直した差分で，対象598,556件は減らしていません．元planと訂正の根拠を保存しています．

- 空ファイルの旧retry177件は通常workerで完了し，空payloadのSHAと実サイズを確認しました．
- OpenNeuroの既知404対象29件は，同一source・version・path・元SHA256Eを条件とした公式S3経路で29／29完了．全29の実SHA256を独立再計算しました．
- Kaggle218失敗候補を監査し，100件だけ公式file metadataとZIP transportを照合して台帳を訂正しました．残118件は404／ZIP構造等の根拠不足として保持しました．訂正100件のうち60件が通常workerで取得完了し，公開MD5と台帳SHA256を独立再計算．残40件は順次取得中・待ちです．
- Figshareの誤ったpayload，未解決TLS，guestbook必須342files等は，期待sizeの付け替えや検証の無効化で通しません．取得不能理由とアクセス条件を台帳に保持します．

[取得修復report](smac_acquisition_transport_recovery_20261123_v1.md)と[集計JSON](smac_acquisition_transport_recovery_20261123_v1.json)に，plan/code/config/receiptのhashと確認範囲を示します．

## 接続に依存しない継続

smac31のguardianを配置し，37workerの既存48時間枠を保持しました．期限後は容量・STOP・実PID/lock・manifest/code・backoffを再確認して新しい48時間以内の枠を作ります．100GiB reserveと1node1streamを維持します．guardianの元上限は2026-12-23 05:15 JSTで，今回の更新では延長していません．

最終acquireコードを37台へ原子的に配置・SHA照合し，masterの38コピー，2台のmanifest binding，訂正後planを更新しました．worker停止0回です．guardianだけを所有確認済みmarkerで正常終了・再起動し，journal，seen_windows，元期限を保持．新PID87453のfresh heartbeatで全37window activeを確認しました．

MacBookの既存IPSec VPNは，再接続直後に通信できた後，remote DELETEを受けて短時間で切断しました．インストール済みGlobalProtectはローカル接続サービスが停止し，既存system serviceの起動にはOS管理者認証が必要でした．新たな認証情報は取得せず，既存IPSecを再接続して認証済みSSHを維持する試行を行っています．06:04以降に約20分の連続通信を確認し，その後はrootの有効lease中だけ最大30分の認証済み通信を維持しています．切断の恒久的原因は未確定です．

BCI検証・次のDesign処理・取得修復を一度の認証接続でS Macへ配置しました．各処理はS Mac内でdetached実行し，MacBookのVPN/SSHが切れても継続します．raw，prepared，重みをworkstationへ戻していません．以前削除したworkstation raw約50.48GBは復元していません．

## 次sourceの全件判定と継続

Design Emotion31578517の25EDFを17train／5val／3testへmetadata-onlyで固定し，全17train・7,130窓のQCまで実行しました．**適格窓0のため科学的blockedで終了し，probe／fit／追加gradientは0**です．5val／3testの信号は未読込です．

7,062窓が固定100µV peak上限超過，68窓が固定0.01µV std未満でした．全train integer samplesの98.6049％がdigital endpointに集中し，MNEとの全17件×3区間×32ch読取差は最大3.47e−18Vです．parser・単位換算の不一致は見つからず，配布数値の強い端点集中を確認しました．発生原因は未確定です．README／pre.pyから有効な補正を導けないため，倍率やQCを変更して採用しません．

BCIの親・base corpus・独立検証receiptはcluster内で全SHAを照合して移送しました．private launcherの構文／JSON escapeによるfit前の失敗を保存し，全生成出力のroundtrip13tests後にattempt4／stage-v2でsource全件判定を完了しました．postfit observerはfitが発生しなかったため未配備です．source判定の独立照合は42／42 PASSです．[独立集計](smac_training_20261122_v1/Design31578517_blocked_independent_verification_aggregate_v1.json)，[数値診断集計](smac_design31578517_qc_20261123_v1.json)，[RUNBOOK](../next_multichannel/SMAC_DESIGN31578517_RUNBOOK.md)に根拠を保存しています．

全候補の取得とsource適格性判定を継続し，取得可能な候補が終端状態になった時点で最大の適格train unionを固定して再学習します．既存automationはこの到達目標とB案の実証を継続します．Designの不適格窓やvalidation時間を，66.2441666667hの学習済み量へ加算しません．

## 論文の目的

ユーザーの最新指示では投稿先をCVPRに固定しません．**高性能LLM/VLMへEEGを内部接続し，画面・操作履歴と合わせて高次意図を推定するB案**を主軸に，NeurIPS／ICLRを中心とした適合性を判断します．encoderの追加学習だけでこの目的を達成したことにはしません．[投稿方針と成立条件](../PUBLICATION_PLAN.md)に，必要な対照，実データ，汎化評価と公式募集要項を整理しています．
