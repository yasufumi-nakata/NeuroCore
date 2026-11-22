# PJ-Xavier 実行・引継ぎ

## 最新：SSVEP9970619 metadata-only splitをS Macで凍結（2026-11-23 03:18 JST）

取得済み44記録を11名義token単位で7train／2val／2testに事前固定し，train28記録／名目18,320秒のFDTについてS Macだけで全SHAと数値診断を完了しました．val/test信号は未読込，metadata SHAは`5a896a2d8031846743dfd8f80379672ecf8879f77f5226d7d8ebf195d40f9f70`．物理Vの校正証拠とcross-ID重複が未解決のため**fit0窓／0steps**で保留します．2026-11-23 03:17:35 JSTの37台実測は**complete192,660files／1,920,312,507,448B**，partial553,929,410,805B，supervisor37/37，欠損・不一致・error0．累積CBraMod学習量は22.581667hのまま．詳細は[SSVEP split/QC実績](reports/smac_training_20261122_v1/SSVEP9970619_METADATA_FROZEN_20261123.md)．

## 最新追加：EEG＋判定前環境→X/Yの漏洩防止スコアラーをS Macで検証（2026-11-23 02:19 JST）

新しい `next_multichannel/score_joint_alignment.py` に固定registry SHAとsource A/B cue goldの独立照合，4配置×同一trialのcounterbalance，6条件のpaired負対照とmust-link単位bootstrapを実装しました．15新規テストがsmac35のtask PythonでもPASS，旧関連23テストもPASS．コードとfixtureのみS Macへ配置し，EEG rawをworkstationへ戻していません．**今回の実gradientは0時間／0stepsであり，EEG→X/Yの実性能やOS成功は未測定です．** [検証内容](reports/smac_training_20261122_v1/JOINT_PROXY_SCORER_CPU_SMOKE_20261123.md)．

S Mac 37台の新snapshot（2026-11-23 02:14 JST頃）は完了**179,003files／1,677,468,786,313B**，partial**483,485,762,153B**，全supervisor生存，完了ファイル欠損・サイズ不一致0．CBraModの累積固有学習量は以前の**22.581667h／27,098窓**を維持．smac19は除外し，取得候補23.07TBを学習済みとして報告しません．


## 最新追加：BDFの未学習validation5群をMPSで検証（2026-11-23）

S Mac smac35でBDF29606906の固定validation5群・327窓を凍結checkpoint2本で比較しました．波形再構成MSEはBDF学習前 **0.001262252863** → 学習後 **0.000975983663**（22.6792％減少）．新たなfit0steps，元test1群の信号読込0です．これは**BDF validationのoffline再構成**に限り，別test・自由意図・LLM/行動・実OS成功ではありません．[詳しい実測](reports/smac_training_20261122_v1/BDF29606906_HELDOUT_VALIDATION_RESULTS.md)．


## 最新：BDF29606906追加学習をS Macで完了（2026-11-23 01:21 JST）

**累積train 27,098窓／22.581667時間**．BDF14記録から1,222窓／1.018333時間を追加し，smac35で10epochs／4,270stepsを完了しました．固定806023 validation最良MSEは **0.001426294970**（直前P300モデル0.001506584226）．BDFの独立val/testとEEG→LLM行動・実OS成功は未評価です．37台の取得は完了168,898files／1,484,540,481,925B，partial407,571,866,481B，supervisor37，欠損/サイズ不一致0（2026-11-23 01:18 JST）．smac19とworkstation/Colabのraw取得・学習は引き続き除外します．

詳細：[BDF追加の実学習結果](reports/smac_training_20261122_v1/BDF29606906_ADDITIONAL_SSL_RESULTS.md)．


## 最新観測：S Mac 37台・SSVEP9970619元構造の独立監査（2026-11-22 22:25 JST）

**37台の新live snapshotは完了99,527files／839,585,508,076B，partial203,790,046,364B（complete欠損・サイズ不一致0，全37supervisor alive）です．** smac02のSSVEP9970619は88files／1,708,501,504B取得済み，smac35で44個の`.set`と代表`.fdt`をcluster内SHA一致転送・別実装で監査しました．先頭64 EEG＋末尾8 EXG/Status，配布200Hz，44連続記録の名目28,817秒，イベント欄0，物理単位明記0です．元論文は512Hz収録→offline 200Hz resamplingを記載しています．unit/split/QC未確定なので新gradient0h，現累積固有学習量**21.563333hのまま**です．BCI8089820は5,241,833,568B／27,475,466,178Bの部分取得中でcompleteにしていません．

詳細：[SSVEP9970619の監査結果](reports/smac_training_20261122_v1/SSVEP9970619_SOURCE_AUDIT.md)．元のP300 validation／封印済みrunは保持し，smac19，workstation raw，Colab学習は使っていません．次は物理電圧校正と因果性の出典検証，別sourceの適格性・固定train-only QC，新学習runです．

## 最新追加：P300未学習1被験者の独立validation確認（2026-11-22 21:28 JST）

**S Macで元DAT9記録／85,040,079Bをsmac05→smac35へ直接SHA一致転送し，未学習のP300 validation1群・609窓（3秒／32ch）で2つの凍結checkpointを実MPS比較しました．** P300追加前の固定mask再構成MSE **0.00177100794** に対し，追加後は **0.00131386130**（同一validationで25.8128%減少）です．元runの再学習はなく，この1被験者群の結果を新たなcheckpoint選択やP300のtestに使いません．元のP300テスト2群・著者除外群の信号読込は0です．独立test，集団汎化，EEG→LLM行動，OS成功は未実証です．

1回目は正しいmetadata-only admissionパスを指定せず信号読込前に失敗しましたが，失敗証拠を保持して一致する元metadata SHAを再指定し，2回目はexit0・9配列全SHA照合で完了しました．関連CPUテスト31件＋13 subtestsがPASSです．21:20:40 JSTの37台live snapshotは **complete78,614files／604,580,599,121B，partial129,871,796,126B**，全supervisor alive，完了ファイル欠損・サイズ不一致0でした．これは取得済みEEG時間ではありません．実学習固有量は前回の21.563333hのまま，今回のgradient追加は0hです．

詳細：[P300追加学習後の別被験者validation検証](reports/smac_training_20261122_v1/P300_HELDOUT_VALIDATION_RESULTS.md)．次段階は取得済み候補の単位・QC・source重複検査と新scientific runの実学習です．BCI8089820はretry partial，SSVEP9970619は単位未検証のまま保持します．workstationへrawを戻さず，smac19を除外します．

## 最新追加：S MacでP300 original DATを4.164167時間分学習（2026-11-22 20:39 JST）

**CBraMod系統の実累積はtrain25,876窓／77,628秒＝21.563333時間です．** smac05のOpenNeuro ds005028のオリジナルDATのうちtrain適格65記録をsmac35へ直接転送し，SHA照合，固定QC，MPS backward4形状，10epochs／4,070stepsを完了しました．P300新窓は4,997／14,991秒で，反復学習258,760提示を固有時間へ加算していません．

固定806023 validation327窓の最良再構成MSEは**0.0015065842255285976**（直前InMIDモデル0.0016149027567592268）です．これはP300新sourceの汎化評価ではなく，行動／自由意図／LLM接続／実OS成功の証拠にもなりません．P300 val9，test19，著者分析除外12記録のDATは開いていません．最良best SHA256は\`9448e34ea1549d406a14b1a3e3eae5603a2ff7882008156d0937e4edb482cc8c\`．別CPU読取検証で配列161件，manifest全26,203窓，optimizer／RNG／checkpointを検証済みです．

取得の新snapshot（20:30:55 JST）は**完了60,732ファイル／395,625,524,078B，partial70,262,345,104B**，37台alive・完了ファイル欠損／サイズ不一致0．約23.07TBは取得候補です．workstationへのraw再取得・Colab学習を行わず，smac19を除外しました．詳細は[今回のP300実績](reports/smac_training_20261122_v1/P300_ORIGINAL_DAT_RESULTS.md)です．

次は既存37台の取得を維持し，未完了BCI8089820 archive／SSVEP9970619 unit／P300新sourceの独立holdout確認，次の適格sourceを新しいscientific runで学習します．既存runの再実行，既存testに合わせた閾値変更はしません．


## 最新状態：S Macで固有17.399時間を実学習（2026-11-22）

最新ユーザー指示は「取得できるデータ全部使って学習進めて、S Macの方で」です．新規の取得・保存・展開・学習はS Macで行い，smac19は除外します．workstationは接続，コード，小さな台帳だけに使用します．raw／preparedのworkstation中継とColab学習は再開しません．

**37台の分散取得を維持し，CBraModの初期・CNT追加・InMID追加を各10epochs完了しました．** 実入力の累積固有量は20,879窓／62,637秒＝17.3991666667hです．旧phase3の別model lineageの29.985888889hを足しません．

| 段階 | 実績 | 状態 |
| --- | --- | --- |
| smac01，806023 | 659 train窓，1,977秒，110steps | 完了，独立33項目PASS |
| smac06，2392979 | 36files取得，21候補すべて固定QC不適格 | 新gradient0，理由付きblocked |
| smac10，CNT2061654 | 新3,385窓／10,155秒，累積4,044窓，640steps | 完了，独立59項目PASS |
| smac35，InMID | 新16,835窓／50,505秒，累積20,879窓，3,280steps | 完了，prep21項目＋model59項目PASS |

最終InMIDモデルの固定806023 validation MSEは0.0016149027567592268です．CNT親の0.001595536042884798より1.213806％増えました．最新学習済みmodelと，既存validationで最良のCNT modelを区別し，全checkpointを保持します．新sourceへの汎化，行動decode，自由意図，EEG→LLMや実OS成功は未評価です．

[実行report](reports/smac_training_20261122_v1/README.md)，[学習集計](reports/smac_training_20261122_v1/sequential_training_aggregate.json)，[取得snapshot](reports/smac_training_20261122_v1/acquisition_snapshot_final.json)を参照してください．

## 取得は稼働中

19:51:41 JSTの1回のsnapshotで，37台すべてのsupervisor／実データを確認しました．complete40,187 files／175,035,084,578B，partial26,550,729,130B，合計201,585,813,708Bです．completeの実ファイル欠損・サイズ不一致・snapshot errorは0．このsnapshotで全payloadを再hashしたわけではありません．

総割当は598,556 files／23,071,349,438,764Bです．これは全modalityの取得候補であり，取得済み・EEGのみ・学習済みの量ではありません．未解決metadata784件，path衝突12件，容量不明42,028 groups，EEG scope未確認2,924件は引き続き根拠を調査します．InMIDの同名MAT衝突は標準namespace migrationで解消しました．

smac31は管理用，smac01～39のうち19・31を除く37台が取得workerです．smac40はDNS未解決で割当なし．1node1stream，空きreserve100GiB，600秒cycle，supervisor起動から48時間です．期限は2026-11-24 18:59～19:04 JST頃で，最後のcycleは期限を少し越えて終了し得ます．429のRetry-After/backoffとTLS検証を維持します．guestbook217必須release342filesは保留し，フォームは送信していません．

controllerは `/Users/yasufumi/pj-xavier/control_20261122/acquisition/` です．

- `snapshot_acquisition.py --name <receipt.json>`：全nodeのlive観測．
- `priority_source_snapshot.py`：学習優先sourceの状態．
- `resume_acquisition.py`：引数なしは確認のみ．`--apply --hours 48` は停止済みで再開可能なnodeだけを再開し，稼働中は二重起動しません．`--hosts smacNN` で限定できます．
- `STOP_DOWNLOADS` がnode rootにある場合は停止を維持します．明示的な再開指示を扱う時だけ `--clear-stop --apply` を使用します．
- 現DB／manifest／partを保持します．無条件のre-deployは既取得量を容量へ二重計上し得るため避け，期限・空き・実PID・lockを再確認して新しい枠を記録します．

詳細はworkstationの `private/xavier/smac_execution_20261122/acquisition/SMAC_ACQUISITION_RUNBOOK.md` と `smac_acquisition_summary_20261122.json` に保存しています．最終全台snapshotは `fleet_snapshot_final_report_20261122T105137Z.json` です．

## 所有権と接続

正worktreeは `/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training`，branchは `feat/pj-xavier-validation-20261122`，repoは `yasufumi-nakata/NeuroCore`，[Draft PR71](https://github.com/yasufumi-nakata/NeuroCore/pull/71)です．

最初に `private/xavier/eegmmidb_phase2_v1/coordinator.json` のowner/leaseと実jobを確認します．有効ownerが作業中なら観測だけ行い，source変更・二重起動をしません．owner解放後に排他的leaseを取り，既存downloadを維持しながら独立した次の適格sourceを処理します．

通常のworkstation `python3` から `private/xavier/smac_execution_20261122/smac_transport.py` を使用します．`connect()` はsmac31，`run_node(node, code, timeout=..., task_python=True)` はtask venvです．task_python省略時は既存cluster Python．ControlMasterはMCP境界でハングしたため使用しません．

既存Keychain→smac31の正規認証と，smac31の既存cluster credential loader／既知host keyを使用します．秘密値は出力・保存・コピーしません．smac31のAGENTSとsmac-cluster skillに従い，既存の他job・サービスを停止しません．node inventoryは `control_20261122/node_preflight.json`，policy/registryはworkstationの `private/xavier/storage_cleanup_20261122/` にあります．

node rootは `/Users/yasufumi/pj-xavier/smacNN`，rawは `raw/<dataset_key>/<relative_path>`，DBは `downloads.sqlite` の `downloads` tableです．smac01/06/10/35にはtask `venv/bin/python` を配置済みです．venv Pythonを `.resolve()` してbase interpreterへ変えません．

## 実行済みchainと再利用する成果物

`control_20261122/training_chain_v2/state.json` はcompletedです．controller PID37852，smac35 pipeline PID70200は過去の実行識別子で，再起動する根拠にしません．v1はCNTのgit前flight失敗を保持したまま終了しています．

| 成果物 | node内path |
| --- | --- |
| 初期corpus | smac01 `prepared/smac806023_ssl_v1` |
| 初期run | smac01 `runs/smac806023_ssl_20261122_v1/run` |
| CNT corpus | smac10 `prepared/smac806023_plusCNT_ssl_v1` |
| CNT run | smac10 `runs/smac806023_plusCNT_ssl_20261122_v1/run` |
| 最新corpus | smac35 `prepared/smac806023_CNT_plusInMID_ssl_v1` |
| 最新run | smac35 `runs/smac806023_CNT_plusInMID_ssl_20261122_v1/run` |

最新best SHAは `600f1e25e89d5229ea70e5d8a4f873b8653dfc445503850d77ff4739bc184ad5`，CNT bestは `0d9a9cc83f5aa0a4b25404de286eff7a86fa7af462649a8e3df54f104ff82817` です．次の学習では目的に沿う親を明示し，この2つを性能上同等とは扱いません．

trainerは `experiments/pj_xavier/model/cbramod_ssl_smac.py`，SHA `ad0d8c27352411db52640c077af4b4bb38c6b1652ea20f3e6e965af2c172ca8a`．旧sealed trainerは無変更です．source追加は別moduleと新corpus/protocolへ分離し，親corpus/run/upstream/weightsはcluster内メモリ中継で全file SHAを照合して転送しました．

upstreamは各訓練nodeの `upstream/CBraMod_pinned`，commit `b9e961003214326972c567eff390e75b0287e32a` です．初期の不完全な `upstream/CBraMod` は使いません．

smac10/35は独立portable Gitを導入済みです．task環境のPATH先頭へ `<node root>/tools/git/bin`，GIT_EXEC_PATHへ `<node root>/tools/git/libexec/git-core` を指定します．公式配布SHAと実rev-parse/showを確認し，source検証を省略せず再開しました．Apple Git／Xcode license／global設定は変更していません．導入receiptは `private/xavier/smac_execution_20261122/portable_git_installation_receipt_v2.json` です．

技術的失敗は停止します．probeはexit2＋kind=probe＋status=failed＋memory_within_budget=falseの明示的な容量超過だけを候補落ちにできます．欠落report・非有限gradient・source mismatch・通常のOOM例外などを容量不足として握りつぶしません．scientific blockedはprepのexit2と正確なstatusが揃う場合だけです．

## 次の具体的な作業

1. 実jobとlease，最新取得DBを確認します．完了した3段階やQC不適格2392979を理由なく繰り返しません．37台の取得は継続させます．
2. 優先取得sourceはBCI8089820／P300 ds005028がsmac05，SSVEP9970619がsmac02などです．取得済み・単位／校正／重複／split監査済みから次のoffline SSL adapterを追加します．新たな取得先へ二重downloadせず，必要なcluster内転送を使います．
3. 2392979は16件のnative過大振幅，5件のwindow QC不適格で0窓でした．公式の単位／変換根拠が得られるまで保留し，恣意的な倍率・clipping・閾値緩和で採用しません．small BCIの単位不明も同様です．
4. raw／physical payload deny，must-link，既存review/val/testと累積partition履歴を継承し，trainだけを追加します．元label，観測event，weak labelは出典と品質を分離し，AI推定を評価goldに昇格させません．
5. 新しいimmutable corpusとprotocolをregisterし，選んだ親の `--parent-run` とSHAを記録します．新形状の実MPS backward/capacity probe，1node1GPUjob，40GiB枠，段階ごとの新しい時間枠を守り，checkpoint/optimizer/RNGと実seen-IDを保存します．古い予算を延長して再利用しません．
6. 新sourceの汎化を評価する際は，既に見たtestへ最適化せず，別の固定protocolで未使用の対象と負対照を定めます．既存806023 validationだけで全sourceの品質を主張しません．
7. EEG＋判定前環境→X/Y，LLM接続，実機試用は既存の次段階設計を引き継ぎます．P300 oracle／decoder出力／未来contextを分離し，cue goldと自由意図を区別します．offline SSLをonline認証や実OS成功と扱いません．
8. 公開code／aggregateだけ既存branchへcommit/pushし，exact commit CIを確認．raw・個別ID・予測・weights・認証はprivate．Google Doc実行tabへ追記しreadback後，実jobを記録してleaseを解放します．

## 継続と過去記録

既存automation `Automation_f439325b5d888191bc75e92d1ba97be7` は毎時15分26秒（Asia/Tokyo）です．S Macだけで取得と学習を進め，未割当metadataの解決も継続します．進捗要約だけで終えず，取得済みの適格sourceを次の学習へ進めます．

workstationのraw／colab_raw_archive773files／50,476,724,659Bは削除済み，モデル／optimizer／RNG／評価／コード／台帳は保持済みです．証拠は `private/xavier/storage_cleanup_20261122/` です．

phase1～3は学習・固定評価・回収・M1試用・Colab停止まで完了済みです．過去の全条件・制限は[移行前HANDOFF](reports/handoff_snapshots/HANDOFF_20261122_before_smac_fleet.md)，[phase3結果](eegmmidb_phase3/RESULTS.md)に保持しています．過去の認証未成立・Colab再開案より，現在のS Mac運用を優先します．

[Google Doc実行tab](https://docs.google.com/document/d/1kGlNqvZEg_AzSYKGTiGF3Z0HcVIxNGNimNwpR6w1680/edit?tab=t.7cvfsge8pwil)の旧tabs・過去記録を保持します．
