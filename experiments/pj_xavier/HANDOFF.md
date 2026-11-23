# PJ-Xavier 実行・引継ぎ

更新：2026-11-23 JST．この文書を現在状態の入口とし，実PID/lock・immutable run・receiptを正とします．古い「最新」が積み重なった記録は[保存版](reports/handoff_snapshots/HANDOFF_20261123_before_full_BCI.md)へ分離しました．

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
