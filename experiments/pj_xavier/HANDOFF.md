# PJ-Xavier 実行・引継ぎ

更新：2026-11-24 JST．この文書を現在状態の入口とし，実PID/lock・immutable run・receiptを正とします．古い「最新」が積み重なった記録は[保存版](reports/handoff_snapshots/HANDOFF_20261123_before_full_BCI.md)へ分離しました．

## 2026-11-24 11:05 JST：残差追加20更新は画像gate失敗、早期5更新診断起動、公式版不一致実証

10:27:41 JSTに残差next20 v2実20/20更新完了、worker exit0、peak33.449GB。固定dev36全有効、絶対MSE14.902778→13.090278 µV²（12.16%低下）、線形13.102778とほぼ同等、Hermite10.3125より悪い。numeric gate通過だが10:28:46画像gate失敗でcurriculum failed（base2/2、adapter1/2、画面外座標730,460、実操作0）。adapter 564a53cfd704c1d3f7d198603fc48c75409f389fa751def976bf07e533ecb1b6、numeric receipt db38108a4ab34b851c07a30ac1b9cadf050d8ea89cda0ddd9dba4a7ec411ecea、vision receipt 6e8fbc32d966247c2c8bbd9e1b7b443c6a23782d98b0b47bd0c609c95c2d5ac7を再hash確認。新adapter不採用、親はb1c06fa…の初回残差20更新。両20更新の非重複40例/4群/固有crop6.32秒は実提示済みの実験unionだが、採用モデルの学習は20例/3.16秒。科学的有効性未証明、test未使用。

11:03:47 JSTに `evlm_build_20261124_v1/curriculum_residual_vision_short5_20261124_v1` を新規起動。plan SHA19112fca5376c4bed4301df86fd3c26cf9431a31b588a33316647c298eaeadf6、preflight SHA2cc717a5f0591764fda9cb5d90c8acbfc8aac456430f2c99c3fa4d316bdff47d。旧失敗stageの先頭5例を意図的に再提示し、同一親から更新数だけ20→5へ短縮する単回早期劣化診断（新固有量0、選定5例crop0.76秒）。元の固定dev36/5%数値gate/言語・画像gate/全64層rank8/LR2e-5/54GB/801tokens、fresh Adamを維持。失敗adapterは継承せず、同じrunnerと既存厳格な親receipt拘束を変更していない。11:05:13 controller56845/supervisor56857/worker56859、GPU lock保持、baseline_task_evaluation 11/36、実0/5、peak17.306GB、swap346.56MB。次回は実worker/lock/数値・画像receiptを確認、重複起動禁止。短期診断後に自動拡大せず、劣化時には数値課題と保持学習の再設計を優先し固定dev/smokeへの反復探索を避ける。採用済み親の保存、元test denyを維持。

11:00:40 JST取得は745615件、14.059258/37.095042TB＝37.9006%、残23.035784TB、paused6.218983TB非加算。37台、snapshot旧36+追加37、smac29は直後に旧新PID95971生存確認（手動起動なし）。DB/接続error0、STOP0。前10:00:37から+513.231GB、旧57.004/追加85.433MB/s、現計画限定低信頼ETA旧11/27 06:51・追加11/25 16:38 JST。retry606、integrity2198、access4006、restricted5、最小disk2.879TB、guardian87453 cycle344/feed94063 cycle325。横断重複未排除。
11:02:12 ds008037 complete6152件/47.392GBのうち5043件/2.147GBをread-only低優先実hash照合、DB保存SHA全5043一致、公式版content ID4088一致・955不一致（TSV717/JSON238）、annex SHA256 79/79一致、Git blob4009/4964一致。公式version-pinned GET不一致サンプル4/4は公式hash/size一致。既取得completeに公式版と同サイズ別内容が混在する確証。smac34独立監査receipt SHA4e6d47e33d4dedefef23f50c2634d0b225c1e6955d924babcdd9ff74e8c867e5、pinned probe SHA4c6f5eeeabc33d228688f894b136799e946a9edfa436c85c122260fc8f27101f。未監査1109件、修復候補未適用・source全体学習保留、既存raw/DB/worker変更・削除0。次回は未監査の内容照合と独占移行の安全性を検証。既存DL正常維持。
学校GoogleDoc構築・取得tabとも11:04追記、revision readback済み。workstation hourly_snapshot_20261124/hourly_1100_receipt.jsonが集約証跡。前回efa57bcのCI9/9成功。

## 2026-11-24 10:03 JST：残差20更新のgate通過・次の20更新を起動

09:31:27 JSTに残差matched20は `paused_update_budget`．実20/20，exit0，固定dev36全有効，MSE14.90278 µV²，sentry6/6，画像base/adapter各2/2，peak34.979GB．元baseは全36無効，通常MSEの対応改善は判定不能．線形13.10278，丸めHermite10.3125より悪く，科学的有効性未実証．親adapter SHA b1c06fa677d5948ffa85622e0cc9c2be117c4f7626bf08b1b16eaed4bd5d81ab，numeric receipt cb2fba74a18739ea22d5345cbac5fdcf6b2ade01b1736f8167bdb6b830eb1bd3，画像receipt 20dca3c964b177d9832a545c88964451d0704d50068e42f91b4963126efbccc5．

10:02:40 JSTにsmac35 `evlm_build_20261124_v1/curriculum_residual_next20_20261124_v2` を起動．controller52320/supervisor52322/worker52324，10:02:56 GPU lock保持，baseline_sentry，0/20実更新．curriculum_id末尾v1は起動前停止したv1とはnamespaceが異なる．plan SHA 742a3ba407f3c57ab752256b98c3481186167c5859ad17c6a6d35af6986481a4，runner SHA deca7a69813ae6a1dc0372cecf09c5f7bf2fecb7b03e26d32d6e842015dbda2f．元20例と非重複の次の20例，offset20，固定dev36/54GB/全64層rank8/LR2e-5，2h上限，親adapter warm-start＋fresh Adam．親の数値・画像receipt/offset/source/再現MSEを拘束し，親再評価14.90278の不一致は勾配前停止．v1はreceipt外枠差異で起動前停止，v2で外枠と本体を検証．実preflightと4負例拒否，CPU残差5 tests PASS．次は実worker更新・gate・画像receiptを検証し，強い補間対照との性能差を報告．新bestなしなら漫然と次stageを反復しない．test未使用．

10:00:37 JST取得：731247件，13.546026/37.095042TB＝36.5171%，残23.549015TB，paused6.153831TB非加算，37台/74worker，DB接続error0，STOP0．前時刻から495.875GB増，旧63.100/追加74.689MB/s，現計画限定ETA旧11/27 00:12，追加11/25 21:03 JST（低信頼）．retry601，integrity2198，access4006，restricted5，最小disk2.896TB，guardian87453 cycle332/feed94063 cycle314．ds008037 10:03 6091 complete/40.283GB，1719 integrity_failed，版候補未適用・学習保留．横断重複未排除，既存取得・raw無変更．Google Doc構築・取得tab追記readback済．

## 2026-11-24 09:03 JST：残差表現の独立20更新を実機起動

09:02:55 JSTにsmac35の新規run `evlm_build_20261124_v1/curriculum_residual_matched20_20261124_v1` を開始．09:03:10 controller47654／supervisor47659／worker47661・GPU lock保持を確認．baseline_sentry，実0/20更新（初動），memory free72%，swap362.56MB．前runの失敗adapterを継承せずfresh base/Adam，全64層rank8，LR2e-5，801 tokens，54GB，2h上限．固定manifest・20 train/36 dev indexは旧runと一致，1977 train候補全行一致と981 devの実tokenizer preflight PASS，最大717 tokens．旧と同じ20例を意図的に再提示する独立比較であり，新規固有EEG時間を増加計上しない．残差を絶対µVへ戻したMSE・既存5% numeric gate・format/sentry・画像gateを維持．丸め線形・Hermiteを対照に含む．元test/review未読．20更新到達だけでは採用・改善・意図・Computer Useを主張しない．新plan SHA c320092433ff36059eb225d3c55281f916beb5a6fe1ed53022bfb01b95ceadd0．次回は実state，PID，lock，数値receipt，画像receiptを確認し，active中に重複起動しない．旧失敗runを再開しない．

09:00:38 JSTの取得は714,092件／13,050,151,592,540B／37,095,041,768,046B（35.1803%）．37台・74worker稼働，接続/DBエラー0，STOP0．残24,044,890,175,506B，paused6,062,241,186,263B非加算，横断重複未排除．直近3694.82秒で408.730GB増，旧53.858・追加56.765MB/s，現計画限定の遅い側ETA11/27 11:03 JST（低信頼）．ds008037は09:03:34に6008 complete／1719 integrity_failed，修復候補未適用・学習保留．取得worker/DB/raw無変更．

## 2026-11-24 08:02 JST：低LR比較停止・残差候補準備

07:30:27に低LR比較もstopped_numeric_gate、20/20更新。固定35例MSE25.542857→37.65（+47.4%）、旧2e-5の27.021429より悪化。候補不採用、画像gateなし、test未使用。07:59 GPU lock空き。新規train-only CPU比較と残差候補1977例（全7908値復元、tokenizer全件PASS）はprepared_not_admitted、追加gradient0。次は別runnerで絶対MSE・既存5%gate・画像保持・強い補間対照を維持する。失敗run再起動禁止。詳細：[報告](reports/EVLM_RESIDUAL_DIAGNOSTIC_20261124.md)。07:59取得12.641421/37.095042TB（34.0785%）、74worker、ETA11/27 15:10 JST（低信頼）。ds008037候補未適用、学習禁止。

## 2026-11-24 07:06 JST：通常学習の数値gate停止と独立低LR比較

前の通常学習 `curriculum_initial400_v1` は06:37:30 JSTに20/400更新（5%）で `stopped_numeric_gate`．worker exit0／checkpoint再読込・SHA一致／sentry6/6／出力36/36，peak33.449GB．しかし同じ35有効例のMSE 25.542857→27.021429（+5.79%）が固定5%gateを超過しました．全例の無効罰点込みscore1135.9444→26.4583は形式改善を含み，通常MSE改善ではありません．線形補間MSE13.1028を下回れず，候補不採用・画像gate未実施・最終test未使用．停止receipt SHA256 `c5d611029447498bb3a889353213e31d77c3dbf5490903eda6ff0495e24b7ec7`．旧runを再起動しません．

Train-only監査：20例は20元窓／3群／固有crop3.16秒，80目標値．同source全1977例は659元窓／4群／crop104秒，線形補間MSE20.17165．実測20更新平均52.43秒（中央値47.98秒），評価込み24分49秒．全358260例準備と実学習は別です．

新runはsmac35 `evlm_build_20261124_v1/curriculum_lr1e5_matched20_20261124_v1`．07:05:25 JST起動，controller38185／supervisor38195／worker38197，07:06:07実PID・GPUlock確認，事前評価3/36，optimizer0/20，peak17.128GB，swap386.62MB．新plan SHA `a7ee1d5d3b71ba72b2a4b30a287a39e3588a867c60c47b300ff8314f0b5e486d`．LRだけ2e-5→1e-5，意図的に同一20例を対応比較に再提示します（新しい固有データ量ではなく継続resumeでもない）．不採用adapterは継承せずfresh base/fresh Adam，seed／全64層rank8／801tokens／54GB／固定dev36／数値・画像gate不変．20更新・2h上限，拡張なし．active中は重複起動禁止．終了後は実numeric/vision receiptを確認し，gate通過と有効性を区別します．失敗時は無目的にLR試行を続けず数値課題・表現と強い補間対照を見直します．適格データの順次学習／heldout deny／取得worker維持は継承します．

07:03:26 JST統合取得は690240件／12.277559TB／37.095042TB＝33.0976%，残24.817483TB，旧37＋追加37worker／node・DB error0．前06:00から+505.422GB，速度旧74.622・追加58.220MB/s，現計画限定ETA11/26 13:52 JST頃（低信頼）．横断重複排除未完了，paused5.695592TBは非加算．ds008037 07:06時点5851 complete，1719 integrity_failed，候補未適用・train admission禁止．既存取得・raw無変更．

## 2026-11-24 EVLM 構築：新しい現在状態

ユーザーは推奨方式でのEVLM構築と，取得済み関連データの順次全利用を指定しました．EEGを別の学習型encoderへ戻すことなく，Qwen3.8-27B-4bitの内部LoRAで適応します．[構築報告](reports/EVLM_BUILD_20261124.md)，[モデルカード](model/EVLM_MODEL_CARD.md)，[数値データ変換](reports/EVLM_NUMERIC_DATA_20261124.md)が今回の正本です．下の2026-11-23 v4 running等は履歴です．

smac35 `evlm_build_20261124_v1/candidate_v5_r2` を保存し，画像＋EEGの合成2画面はbase／adapter各2/2でした．新全64層の8ch/1024試験は1更新後58.762GBで54GB guard停止，checkpointなし．6ch/801版は05:56:07 JSTに2更新・peak44.016GB・基本指示6/6・保存再読込を完了し，新adapterの画像gateも05:57:48に両2/2でした．新bundleは `engineering_vlm_probe_6ch_v2/bundles/all64_probe_6ch_v2` です．同じ最長train例の2反復なので学習有効性の証拠ではありません．EEG有効性・実computer useは未実証，実OS操作0，最終test未使用です．

6source・元79,493 train窓は全channelを保持して358,260例へ変換済みです．smac35 `prepared/evlm_numeric_6ch_20261124_v1/source_queue.json` SHAは8218c7d821e4afbd4a6e841129df47236db3d759711ff8eec66991a7a0e94510です．prepared元66.244hとcycle0 crop3.487hを学習済み時間へ加算しません．固定devは806023のみ2group／327窓981例で，通常学習の比較は同じ36例です．元split，単位，QC，deniesを維持し，EEGだけから架空の操作labelを作りません．

現在のcoordinationはprivate `eegmmidb_phase2_v1/coordinator.json` と実PID／GPU lockを正とします．root owner active中は重複起動・コード変更を行いません．通常学習は06:12:39 JSTに `evlm_build_20261124_v1/curriculum_initial400_v1` で開始しました．初期controller34003／supervisor34015／worker34017です．plan SHAは8f66bbb6910f0d54ffbd024e40ac401a67bbf84114cb2d00c9608f2a34bc5478．数値gateを明示し，20→80→300の追加更新，初回400更新または24時間で区切ります．400到達のpaused_update_budgetは全量完了ではありません．詳細・起動receipt・後続判断は構築報告を参照します．stage間はadapter warm-start＋fresh Adamでexact resumeではありません．取得workerは継続します．

## 17:55 JST 追加：現行runはv4，54GB運用予算

現在はsmac35 `~/pj-xavier/smac35/qwen38_progressive_20261123_v4/sequence_state.json` を正本とします．17:55:48開始，初期controller75004／worker75006です．source035f465…／schema v3の学習方法は不変で，v4は新実行namespaceです．active時にEEG/noEEGを重複開始しません．

v3は2更新後peak50.962GBが保守的な50GB guardを超え停止しました．実機GPU推奨55.663GB，物理64GiBを確認し，根拠を記録して両arm54GBへ運用予算を変更しました．推奨値との差は実行時の余裕保証ではありません．4h上限，累積peak guard，active/cacheと実完了更新数の記録を維持し，OS memory pressure/swapも観測します．旧runを保持しfresh base/fresh optimizer開始です．[詳細](reports/QWEN38_PROGRESSIVE_DIRECT_EEG_20261123.md)とprivate `continuation_20261123_v4/` を参照します．

20/40/80stepでEEG/noEEG/shuffleの開発評価を行い，全64層LoRAは結果を見て次に判断します．80stepは0.2epoch，最終testは未使用，VLM画像経路・Computer Use・実時間EEG因果性は未実証です．以下のv3/v2 running記録と50GB条件は履歴です．取得workerは継続します．

## 17:46 JST 追加：直接EEG・省メモリ版Qwenの段階的LoRA比較

最新のユーザー方針は，別学習のEEG encoderを置かずQwen内部で適応し，評価しながら学習を進めることです．推奨は広範囲LoRA/QLoRAで，FullFTや37台分散新設を最初の必須段階にしません．旧80stepのthinking/padding mask不整合を修正した4層rank8のEEG/noEEG比較を進めます．[方法・実行履歴・制約](reports/QWEN38_PROGRESSIVE_DIRECT_EEG_20261123.md)を参照します．

v2は2更新目に50GB peak guardで停止し，noEEG未開始です．履歴を保存し，正解位置だけへLM headを適用する同値の省メモリ修正をv3へ入れました．14件のCPU tests＋実MLX CPUでのloss/勾配同値試験，独立レビュー，全train/val preflightが通過しました．17:46:11に新v3を基盤からfresh optimizerで開始し，旧runのresumeとはしていません．

smac35 `~/pj-xavier/smac35/qwen38_progressive_20261123_v3/sequence_state.json` が現在の正本です．初期controller PID74013，EEG worker PID74015，既存GPU lockを保持します．EEG→noEEGを直列実行し，20/40stepは固定val最大36件，80stepで全155件とshuffleを評価します．80stepはtrain400の0.2epochであり，全量完了ではありません．各arm上限4h/50GBはETAや改善保証ではありません．active sequence中に重複jobを起動しません．

step数，memoryと評価は実stateで確認します．Adamはrun内保持，crash resume未実装です．品質不合格・過大loss・memory超過等で停止し，all64へ自動拡大しません．比較reportの実SHAと進行根拠を読み，次の小規模比較を判断します．最終testは保留，個別予測はprivateです．workstation記録は `private/xavier/direct_qwen_eeg_v1/continuation_20261123_v3/` です．

文脈多数決45.81%も固定対照に含めます．current runnerは言語backbone経路のみで画像/Computer Use未検証，trainのEEG因果性も未検証です．本実験を高次意図や実OS操作の成功と報告しません．旧run・配布重み・取得workerを保持し，外部EEG encoderへ主経路を戻す旧引継ぎより本節と最新ユーザー指定を優先します．

## 07:27 JST 追加：AoyamaLab catalogue bridge，OpenNeuro公式manifest

[ソース照合・実機導入報告](reports/SMAC_CATALOG_BRIDGE_20261123.md)．既存高信頼5,165件は新規ダウンロード候補ではなく，canonical catalogue 5,165件へ全件対応（source-key 5,162件＋title/URL alias 3件）．広域発見22,327件の未対応13,792件は単なるmetadata差分であり，EEG実体・公開取得許可は未確認です．`ds005841`の公式OpenNeuro公開S3 manifest 872 objects／BDF 288／7,889,279,183B／CC0／v1.0.0を独立取得しました．元カタログ003254のfiles0解消の候補ですが，このrunでraw取得もfitもしていません．

smac31の独立領域 `~/pj-xavier/catalog-bridge-v1/` でstdlib-only reader/CLI，CSV2件，公式S3 metadataをSHA照合して導入・実行しました．公開コードは [catalog_bridge](catalog_bridge/README.md) の5件unittestをPASS，公開版コードもsmac31でread-only実行して一致確認済みです．前の37台read-only pilotは全37台完了済みのため重複起動せず，GuardianはPID87453／37 windows activeを確認しました．既存取得DBやraw，重み・GPUは不変，追加gradient0です．次のroot ownerは固定manifestをprovenance・license・EEG scope・splitと照合してからsource単位にadmissionし，B案の同期画面／操作履歴との比較を最優先にしてください．

## 現在の目的と到達点

**約23.07TBの全取得候補をS Macで取得し，適格train全体の学習・評価を継続します．BCI追加までの実累積は79,493窓／66.2441666667h，10epoch・12,460step完了，独立69／69 PASSです．** 全量取得と最終最大unionの再学習は未完了です．

研究の主軸は，EEGを高性能LLM/VLM内部へ接続し，操作前の画面・履歴と統合して高次の目的・意図を推定するB案です．CBraModの自己教師学習は基盤encoderの段階で，研究の完了条件ではありません．投稿先はCVPR固定ではなく適合性で選び，NeurIPS／ICLRを軸に[投稿方針](PUBLICATION_PLAN.md)を更新しました．

raw取得・保存・展開・prepared・重み・学習はS Macのみ．smac19を除外，smac31は管理，37台が取得workerです．smac40はDNS未解決で割当なし．workstationはコード・接続・小さな台帳，Colab学習は再開しません．

## 論文査読と文献取得

ユーザーから論文としてのレビューとworkstation経由のScopus文献取得を依頼されました．既存草稿v1（31745 bytes，SHA256 `159164d4bbce4c2df83fd7d9653559ff22127d53c744efd1b177fa25ab01975a`）を本文・実コード・既存結果・一次文献に照合し，主要改稿と新しい対応付き下流実験が必要との査読を完了しました．[統合査読](literature_review_20261123/REVIEW_REPORT_JA.md)，[独立監査](literature_review_20261123/INDEPENDENT_METHODS_REVIEW.md)，同directoryの文献比較・BibTeX・取得台帳を参照します．原稿・モデルの不実な完成報告はせず，B案の主題を維持します．

文献15件のPDFをS Mac31へ保存し，タイトル・版・ページ・テキスト抽出・SHA256・サイズと転送後の再hashを15／15で確認しました．総108,187,805 bytes／344ページ，既存の正規コピー3件と新規公開取得12件です．重点4件は方法本文まで照合し，全15件の精読完了とはしていません．ScopusはDocuments13件，Preprintsのみ2件で照合しました．一括書誌exportは追加のアカウント認証が必要なため未実施で，EIDを推測生成していません．詳しくは[文献取得・検索記録](literature_review_20261123/README.md)を参照します．

最新encoderの対応付き下流評価，強い画面／履歴対照，予測前の同期入力・意図label，未使用評価群が最優先です．文章で直す事項はMSE式の1/200と入力尺度V×1e4，group macroの定義，checkpoint選択手順，過去13.983h系列のMSE悪化，4条件の履歴baseline，近接研究の追加です．既存testを再分割して未知と呼ばず，観測済み結果に合わせて閾値やQCを変更しません．

既存PR71は既にmainへsquash merge済みでした．28新規ファイルのtreeを保持してmain基点のbranch `feat/pj-xavier-paper-review-20261123` に整理し，新しい[Draft PR73](https://github.com/yasufumi-nakata/NeuroCore/pull/73)を作成しました．旧draft PR72は履歴衝突を記録してcloseし，mainへのmerge／force pushは実行していません．中間commit `d801634182b970ffee748fbf59960d3a1c6c7cc5` は9／9 checks，4 workflowsが成功済みです．その後のreview追加commitのCIは必ず別のexact SHAで確認します．

## 最新の確定モデル

| 項目 | 確定値 |
| --- | --- |
| source／node | BCI8089820追加／smac05 |
| corpus | `/Users/yasufumi/pj-xavier/smac05/prepared/smac_BDF_plusBCI8089820_ssl_20261123_v1` |
| run | `/Users/yasufumi/pj-xavier/smac05/runs/smac_BDF_plusBCI8089820_ssl_20261123_v1/run` |
| 累積固有train | 79,493窓／238,479秒＝66.2441666667h |
| BCI追加分 | 432適格記録／55群／52,395窓／43.6625h |
| 実fit | 10epoch／12,460steps／794,930提示 |
| latest best SHA256 | `f66903e1dc9120da71c6275e70f121662671aadac13f44efaf92d18e4563954f` |
| parent BDF SHA256 | `f2864fc4dba584f6ccda232a43b000f51d15c5124d7cb22a6ca9f95a0b6e8503` |
| corpus completion SHA256 | `a76b870cc24fa543ab8b9310eea2a063872b37cc567f26104fa78334ba85209a` |
| model verifier v2 receipt SHA256 | `adcc130286e6fb97c739a27c1dc0e6f25b4b32b1bc520a3bd29e596e55886a8d` |

固定806023 validationはseed20261122・batch64で親MSE0.0014298673209638358→子0.0014364248753904745（0.4586％増加）．過去のBDF保存値0.001426294969742873はseed20261123です．異なるseedのraw scoreを直接比較しません．最新モデルと旧sourceで良い親モデルは区別し，両方を保持します．

独立BCI validationは事前固定96候補のうち95記録・12群・11,709窓を評価．seed20261123・batch32のgroup macro MSEは0.0027535698545106207→0.0008033758285906217（70.8242％低下），欠測group0，fit/選択/test/review0です．この診断で次のcheckpointを選び直しません．

[BCI詳細](reports/smac_training_20261122_v1/BCI8089820.md)，[全stage集計](reports/smac_training_20261122_v1/sequential_training_aggregate.json)，[全体進捗](reports/SMAC_ALL_DATA_PROGRESS_20261123.md)に根拠を集約しています．旧phase3の別model lineage29.9858889hや反復提示をこの固有時間へ加算しません．

## Design Emotion31578517：全train検査後に科学的保留

smac10の25EDFを17train／5val／3testに固定し，全17train・7,130窓を検査しました．**適格窓0，probe0，fit0，追加gradient0**です．固定100µV peak上限を7,062窓が超過し，68窓が固定0.01µV std未満でした．5val／3testは信号未読込です．

全17件×3区間×32chをMNEと照合し，最大差3.47e−18Vでした．train integer samplesの98.6049％がEDF digital endpointに集中し，native範囲は±0.00983Vです．parser／単位換算の不一致は見つかりませんでした．配布数値の端点集中の原因は未確定で，README／pre.pyから有効な復元倍率を導けません．固定QCを維持して保留します．

最終controllerはsmac31 `/Users/yasufumi/pj-xavier/control_20261122/design31578517_attempt4/controller_state.json`，初期PID89037，科学的blockedで終了しました．最終stageはsmac10 `runs/smac_BCI_plusDesign31578517_ssl_20261123_v2` です．新しい学習modelは作っていません．親base616files・parent8files・receipt1fileはcluster内だけで全SHA照合済みです．

attempt1の親needs_review，attempt2の生成bootstrap構文，attempt3のdeny JSON escapeの各失敗を保存しています．全てfit前です．attempt4は全生成payload・writerの実roundtrip13tests後に実行し，科学用adapter・split・QC・seedは不変です．postfit observerはfitが発生しなかったため未配備です．

source判定の独立照合は42／42 PASS，receipt SHAは `bc3291fc135390dabc597f5b56321744bfec4cd6377912952067c08a5a5e0c2d`．[独立集計](reports/smac_training_20261122_v1/Design31578517_blocked_independent_verification_aggregate_v1.json)，[数値診断集計](reports/smac_design31578517_qc_20261123_v1.json)，[RUNBOOK](next_multichannel/SMAC_DESIGN31578517_RUNBOOK.md)を参照します．private opsは `private/xavier/smac_execution_20261123/design31578517_ops/attempt4/`．根拠の追加なしに同じ不適格sourceを再fitしません．累積学習量はBCIまでの66.2441666667hを維持します．

## 全量取得の現在状態

06:07:31～35 JSTの37台snapshot：complete234,816files／2,489,730,344,939B，partial762,176,034,748B，supervisor37，完了欠損・サイズ不一致・観測error0．全payloadを今回再hashした値ではありません．

元planは598,556files／23,071,349,438,764B，SHA `155a332f2336913abab7b37052cf0148b6a58a563454147fef4d303e1f5a29f3`．Kaggle100件の公式ZIP輸送容量訂正により，**対象ファイル数不変のまま23,058,905,253,931B**へ更新しました．現在plan SHAは `8b7ca80c818d766c8a5356c7cb6f4c006a550a097d906c02bf743693f32186ec`．元planとamendmentは保存済みです．

177空ファイルretryとOpenNeuro既知29対象は通常workerで全件復旧．OpenNeuro全29の元SHA256E一致を独立再計算しました．Kaggle訂正100のうち60completeは公式MD5・DB SHA256一致，残40は順次取得待ち・取得中です．監査218の残118は未修正です．[修復reportとhash](reports/smac_acquisition_transport_recovery_20261123_v1.md)を参照します．

未解決metadata784，path衝突12，容量不明42,028groups，EEG scope未確認2,924は理由付きで調査継続．guestbook必須342files，TLS，404，誤payloadを取得済みと扱いません．追加のフォームや他者への連絡は行っていません．

## 取得継続guardian

管理rootは `/Users/yasufumi/pj-xavier/control_20261122/acquisition/`．

- `guardian_state.json`／`guardian_config_v1.json`／`guardian_windows.jsonl`／`guardian.lock` が正本．新guardian初期PID87453，fresh heartbeatで37window active．
- 現acquire SHAは `d3476952a8d999b9c7113bd79ddb246bb210ea507cb330664fbff769180796e6`，現config SHAは `56e9e505ba707098d8865b218a4c7d4f8548d21bb0fa6245396cf0a08babba15`．
- interval300s，1node1stream，600s worker cycle，100GiB reserve．48h枠が終わればPID/lock/STOP/容量/台帳/code/429条件を再確認して次のbounded枠を登録します．
- guardianの現在horizonは2026-12-23 05:15 JSTまで．今回のreloadは元期限を延長せず，journalとseen_windowsを継承しました．
- `STOP_GUARDIAN` とnode側 `STOP_DOWNLOADS` を尊重します．稼働workerの再起動やDB/partの初期化をしません．既存STOPを削除しません．
- `snapshot_acquisition.py`／`priority_source_snapshot.py`／`resume_acquisition.py` は既存運用を確認して使用します．無条件re-deployは既取得bytesを必要容量へ二重計上し得るので避けます．

今回のdetached更新結果はsmac31 `/Users/yasufumi/pj-xavier/control_20261123/acquisition_transport_amendment_v1/completion.json`．後続観測は `control_20261123/acquisition_after_transport_observer_v1`．workstation receiptは `private/xavier/smac_execution_20261123/provider_reconciliation/` にあります．

## 所有権・接続・科学的条件

正worktreeは `/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training`，branchは `feat/pj-xavier-paper-review-20261123`，[PR73](https://github.com/yasufumi-nakata/NeuroCore/pull/73)です．

最初に `private/xavier/eegmmidb_phase2_v1/coordinator.json` のowner/leaseと実jobを確認します．有効owner中は観測だけにし，lease取得はlock＋atomic writeで行います．古いPIDや既に完了したrunを再起動の根拠にしません．

接続はworkstation通常python3から `private/xavier/smac_execution_20261122/smac_transport.py` の `connect()`／`run_node()`．既存Keychainとsmac31既存cluster credential loader・既知host keyを使い，秘密値を出力・保存・コピーしません．ControlMasterは使いません．smac31のAGENTSとsmac-cluster skillに従います．

MacBookのIPSec VPNは再接続直後に短時間でremote DELETEを受ける事象がありました．既存profileで再接続し，06:04以降の約20分で79／79回の認証済みSSH通信を確認しました．その後はrootの有効lease中だけ最大30分の接続維持を使い，恒久的な再接続ループは設けていません．GlobalProtectは既存サービス停止・起動にOS管理者認証が必要な状態を確認しました．恒久的原因は未確定です．VPN状態表示だけで正常と判断せず数値IPへの正規SSHで確認します．最新診断は `private/xavier/smac_execution_20261123/vpn_keepalive_attempt3_v1/state.json` と `vpn_owner_session_keepalive_v1.json`／`vpn_paper_review_session_v1.json`．再接続の無制限ループは作りません．

prep/fit/transfer/取得controllerはS Mac側detachedで進め，VPN切断で止めません．venv Pythonをresolveしてbase interpreterへ変えません．既存portable GitのPATHとGIT_EXEC_PATHを使用し，source検証を省略しません．sealed trainer SHAは `ad0d8c27352411db52640c077af4b4bb38c6b1652ea20f3e6e965af2c172ca8a`，upstream commitは `b9e961003214326972c567eff390e75b0287e32a` です．

新sourceはunit/scaling，raw/scalar/physical identity，must-link，元group split，祖先val/test/review deny，固定QCを継承します．native異常を恣意的倍率・clipping・閾値緩和で通しません．2392979の0窓，SSVEP9970619のunit等のholdは，根拠が解消するまで保持します．技術例外は修正対象として停止し，科学的blockedや容量不足へ読み替えません．

## 次の担当が行うこと

1. live leaseとguardianの実状態，Designのblocked独立検証を確認します．Designはsource判定が完了した保留であり，fit待ちとして再起動しません．
2. 全候補取得と原因別retryを継続し，取得可能な候補が終端・source適格性判定済みになれば，最大の適格train unionを固定した新runを実学習します．
3. B案の同期EEG・画面・履歴・目的labelとモデル内部接続を点検し，既存の `next_multichannel`／phase3計画を実データの意図推定・強い非EEG対照へ進めます．encoder追加だけを最終目的にしません．
4. cue gold／自由意図，proxy環境／実画面，oracle／decoder，判定前／未来contextを区別し，既存testに合わせて設計を変更しません．
5. 公開code/aggregateだけcommit/pushしてexact SHA CIを確認し，既存Google Docの実行tabへ追記・readbackします．raw，個別ID/予測，weights，認証を公開しません．force push/mergeなし．
6. 実job・期限・次作業・観測時刻をcoordinatorへ反映し，終了時にownerを適切に解放します．

既存automation `Automation_f439325b5d888191bc75e92d1ba97be7` は毎時15分26秒JSTで継続します．raw／colab_raw_archive773files／50,476,724,659Bのworkstation削除は済み，モデル／optimizer／RNG／評価／code／台帳を保持しています．

[Google Doc実行tab](https://docs.google.com/document/d/1kGlNqvZEg_AzSYKGTiGF3Z0HcVIxNGNimNwpR6w1680/edit?tab=t.7cvfsge8pwil)の旧tabsと本文は保持します．phase1～3の履歴は[移行前HANDOFF](reports/handoff_snapshots/HANDOFF_20261122_before_smac_fleet.md)と[phase3結果](eegmmidb_phase3/RESULTS.md)にあります．
