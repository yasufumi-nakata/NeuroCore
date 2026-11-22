# PJ-Xavier 学習・評価の引き継ぎ

更新：2026-11-22，日本時間．**進行中の実ジョブがあります．receipt と実プロセスを照合してから再開してください．**

## 目的・許可・接続経路

PJ-Xavier は NeuroCore の EEG＋LLM AI の研究開発です．ユーザーの目標は，判断前のハーネス環境と EEG に応じて agentic な操作を行い，環境内のタスク完了を検証できるモデルです．操作イベントの分類，dry-run，合成リプレイをこの最終目標の達成として報告しません．

EEG-dataset の自動取得可能な EEG を含む全データ，マルチモーダルの付属物，AI 補助ラベル，学習，checkpoint ごとの評価，実機試用，GitHub push，継続予約は確認不要で許可されています．既存 CU 枠の利用は許可済みです．購入や制限回避は行いません．

**Colab は必ず workstation の `/Users/yasufumi/.local/bin/colab` CLI を経由します．直接の Colab MCP は使いません．** Session は `xavier-eeg-20261122`，remote root は `/content/xavier`．L4 約22.03 GiB，host RAM約52.96 GiB，12 CPUsです．開始時417.49 CU，L4観測率1.94 CU/h．残高は `colab usage` で再確認してください．

SFC VPN の保存構成は接続済みです．smac19 は `133.27.22.29` への名前解決と SSH 応答まで確認しましたが，smac19／smac31 の SSH 認証は未成立です．既存 SSH agent に identity はありませんでした．**SSH パスワード自動入力は以前の自動承認レビューで拒否されています．別スクリプトや秘密の抽出で回避しません．** 認証が成立してから preflight とジョブ投入を行います．38 台の歴史的 RAM 合計を単一モデルの GPU メモリとして使いません．

## 正しい作業場所と GitHub

- Worktree：`/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training`
- Repository：`yasufumi-nakata/NeuroCore`
- 現在の追加 branch：`feat/pj-xavier-validation-20261122`
- 最初の branch：`feat/pj-xavier-training-20261122`（過去の履歴）
- 初回の base：`966907f0087bd38d74b1e1ddae32dda739de986b`．追加 branch は origin/main `63f5ffd` へ付け替え済み．
- 追加 Draft PR：[71](https://github.com/yasufumi-nakata/NeuroCore/pull/71)
- 最初の PR70 は 2026-11-22 00:37:16 UTC に別の操作で squash merge されたことを後から確認．merge commit d3e222bfa120a59214e50981f0f7d64bc7bedb03．root は merge を実行していない．
- 最初に push 確認した commit：`93c952209274721b25acdde4f49cf121b1d4b78e`
- 最初の CI／Dependency Review／Repository Health はsuccess．CodeQL解析job完了後にhostname判定の2件highが検出され，新PRで修正．PR71 head2bd3f08d2e9439f7ed9b2fcba8a19008b3d23c2bではCodeQL alert checkも含め全9checksがsuccess．
- 新branchは最新main63f5ffdへ初回93c9522以後の追加分だけをrebase済み．追加の公開checkpoint commitはea101fb3474c3d501563ef64d2e84375372684b1．2026-11-22 11:16 JST確認で全9CI/CodeQL checks SUCCESS．元branchへforce pushはしていない．以後は現在のgit／PRを読み直すこと．
- 実験コード：`experiments/pj_xavier/`
- 原データ・重み・詳細ログ・台帳：`private/xavier/`（git 除外）
- Python：`.venv-xavier/bin/python`，3.12.13．workstation の既定 Python 3.9 を使わないこと．
- Catalog checkout：`/Users/yasufumi/Documents/GitHub/EEG-dataset`，参照 origin/main は `5594ef98` で始まる commit．ローカル main は更新していない．

original checkout `/Users/yasufumi/Documents/GitHub/NeuroCore` の既存未追跡ファイルを一括 stage／削除／復元しないでください．一時的な出力先誤りの経緯は `private/xavier/model_agent_path_correction_inventory.json` と `model_worktree_copy_receipt.json` に保持しています．正 worktree へ hash 照合してコピー済みで，original は削除していません．

## 最初に確認するジョブ

PID は再利用されます．receipt 内の start time と argv を確認し，Colab では `/proc/PID/stat` の Z／X を終了として扱います．PID 存在や古い summary だけで稼働判定しません．

| 場所・ジョブ | 最終確認 PID | 確認先 |
| --- | ---: | --- |
| workstation bulk download | 8814 | `private/xavier/bulk_launch_v3.json`，`bulk_download.pid/log`，`bulk_downloads.sqlite` |
| workstation metadata resolver | 5571 | `metadata_refresh_launch.json`，`metadata_refresh.pid/log`，`catalog_fresh_resolution.jsonl` |
| workstation recovery controller | 63404 | `controller/gate_update_20261122_ssl_alignment/`，`config.json`，`state.json` |
| Colab CBraMod＋Smol 単一 token 実験 | 初回 supervisor 38446／train child 38907 は学習完了 | `/content/xavier/workflows/cbramod_symbol_v1/`，`runs/cbramod_smollm2_uniba_v3_symbol_v1/` の最新stateを確認 |

workstation の表中の短いパスは `private/xavier/` 配下です．M1 v3 は学習・五条件評価・strict・試用まで完了しました．Qwen v3 も30epochs学習，五条件推論，strict，回収，履歴比較，canonical receipt公開まで完了しています．最新根拠は `private/xavier/variant_ops/v3_workflow_state.json`（status completed）です．Qwen v2 も完了済み，seed23filesの取得・workstation保全も全件完了です．これらの完了ジョブを再投入しません．

CBraMod gateを実daemonへ反映した後，Qwen v3のcanonical receiptを公開済みです．ITU外部評価3gatesを52995へlive反映した段階を経て，実行可能なSSL phase plan・17tests・起動前条件を固定し，新gateをcontroller63404へlive反映しました．live probe1795314619.42987，config SHAbbad4903fd9033c11c0bd08673141e40793c406e8f7139e90570732935d8f36b．元deadline，raw，stateを保持し，元3モデルのITU canonical receiptも公開・回収済みです．public benchmark/itu_external_publication_receipt.jsonを照合してください．新gateだけではGPU起動を許可せず，実corpus・容量・残時間・protocolを別に検証します．無期限holdやdummy gateは使いません．

**Colab sessionの元絶対deadlineは1795321234.65（13:20:34 JST）で，controller終了より早いです．GPU workflowの予算計算はこちらを使います．** bulk の元絶対 deadline は **1795322496.632661**，元 max bytes は **64,062,645,375**，reserve35GiBです．controller の元絶対終了は **1795324658.879168（14:17:38 JST）** です．再起動で5時間を取り直しません．metadata resolver の元 deadline は1795319128.862762です．

## controller の重要な修正

Colab CLI0.7.4の大きな `download` はHTTP JSON／base64全体をRAMへ保持し，ITU4.65GBの回収が進まなくなりました．旧controller12764とその子22431を同定して停止し，state／lease／原本を保存しました．新controller29527は**64MiB chunk**ごとにsource SHA，offset，size，chunk SHAを照合し，localへ1MiB bufferで追記します．中断tailから再開し，全体SHA一致まで保全完了にしません．installed CLI本体は変更していません．

元 `launch_receipt.json` は保持し，chunk実装の起動は `launch_receipt_v3_gate.json` です．その後，CBraMod gate追加のため29527を本人性照合後に停止し，43294へ切り替えました．新起動・live gate根拠は `controller/gate_update_20261122_cbramod/completion.json`，config SHAは755fbeb2184b8114d1ba5447e7f0fc9510cfbe20519255c829950321f5e82357．元deadlineとstateを維持しています．公開根拠は `reports/controller_v3_gate_20261122/` とcontroller/validation.json．36testsと実転送を確認済みです．

新completion gateは `uniba_natural_v3_five_variant_evaluation` です．receiptは
`runs/qwen3_8b_aicoding_uniba_natural_v3_fused_30e_v1/variant_evaluation_completion.json`，checkpointは同runのbest.pt．live daemonで未完了gateが反映された確認後，v2 canonical receiptを公開しました．v2 receipt SHA：
`75d988106771fcb861cc3239262f7973c6edf91ffdfcfb9285353696ae8533e8`．

新しい GPU ジョブを追加する前にはconfigに必要なcompletion gateを加え，**実行中daemonへ読み込まれたこと**を検証してください．config fileの編集だけでは反映されません．全raw／run成果物の保全，実process終了，GPU idle grace，completion receiptを満たすまで停止しません．dummy keepaliveやgate無効化で処理を隠しません．

## 全目録と取得キュー

catalogは5,165行で，ID集合の不足・余剰0です．CSV SHA：
`142a7e8da1ca3c8439082cf3af768cab9ffcdb0f967c003f86dd5ee71f1ee5f8`．

最初のcacheではEEG候補405releases／29,229files／4,673,956,432,245Bでした．最新の固定snapshotから206releases／438,834jobsを追加済みです．bulkは計609releases／468,040jobs／既知15,821,047,310,466B，別管理のseedは2releases／23files／11,221,153,522Bです．これはrelease全体の候補で，純EEG・重複排除済み記録時間・全件取得済みという値ではありません．

固定fresh snapshot：
`private/xavier/fresh_queue_snapshots/catalog_ef5aa95b592bc88c45588c9fe6ea9d39bf776bc724ceaa91150fc9e55d547a84.jsonl`．
private planは`fresh_queue_plan_20261122_01/`，apply/verification receiptは`fresh_queue_append_receipt_01.json`／`fresh_queue_append_verification_01.json`．公開集計は`reports/fresh_queue_append_20261122_01/`．

追加時は既存29,206＋新438,834件のURL／provider identity／size／checksumを再照合し，不一致0．importerはINSERT-onlyで既存row更新・削除0，稼働workerや元budgetを変えていません．38filesはunknown sizeでmetadata_blocked，1releaseはstorage key長がfilesystem上限を超えるため保留しています．2,925行のscope review候補は，official description EEG231，title-only1902，metadata不足778，矛盾14に分けています．未確認scopeを自動で学習承認しません．

Dataverse DOI10.7910/DVN/K5EDB4の342filesは，公式APIが必須Guestbook応答を要求するためaccess_action_requiredです．HTTP400を一律にrestricted扱いせず，実応答を記録しました．元running／completeは変更していません．Guestbook修正時のacquire.py SHAは0e265a3c...．その後CodeQLの指摘を修正し，正規hostnameまたは区切り付きsubdomainでFigshare／PhysioNetを判定する現source SHAは1c3c2099d0ded2009c098c2f3b3eabbe5148f6b0e50fdb22754f0dbaad33f597です．稼働bulkは以前の043031...をロード済みですが，保留rowは選択しません．hostname修正＋append＋試用の51tests／27subtestsとCodeQLの再解析が通過しています．

最新の完全取得量は `reports/acquired_progress_003.json` の10:58〜10:59観測で，39 unique file contents／14,569,523,189Bです．workstation16files＋Colab23files．全completeをSHA再検証し，replicaとpartial9,081,331,952Bを加算していません．次回は新しいsnapshot番号で再観測し，既存時点を上書きしません．`acquisition/report_progress.py` は468k行をstream処理する版です．7tests通過．

全468,040file jobsのread-only構造監査も完了．EEG明示path＋信号形式は62,303files／7,209,327,960,212B／135releases．これは未検証のpath候補で純EEG実測ではありません．未展開archive，MEG同時EEG，iEEG，その他modalitiesを別分類に残します．`reports/queue_structure_20261122_v1/` に全609releasesと分類の根拠があります．キュー変更0，8tests通過です．

## 検証済みの EEG と前処理制約

`reports/verified_eeg_quantities_20261122.json`：118記録，**87.90746536458333時間**，物理EEG scalar **10,924,347,024B**．EEGlass1記録1.08882222h，UniBa34記録30.61540256h，ITU26記録21.30617947h，portable57記録34.89706111h．原記録／filtered copy／channel時間／学習窓を二重加算しません．ZIP容量から時間を推定しません．

UniBa32／34人，28.58953451hが固定100ms／5000ppmの時計監査に通過しました．P003／P015を保留．ITU25／26人，20.51673720hが同基準に通過し，P012は最大LOO4.30216秒で保留．ITUは全員外部test候補として予約，学習やモデル選択に使っていません．共通8EEGとphase補助は全26記録にあり，時計監査と数量監査は独立照合済みです．

ITU archive4,650,970,671B，SHA：
`1e192b5e65931b9a44858acd987d540d3436de3fddf99454170b93434c2c99b5`．
Colab raw root：`/content/xavier/verified_ai_coding_itu_v1`．workstationの全監査資料は `ai_coding_itu_quantity_audit_v1/` と `ai_coding_itu_external_clock_v1/`．archive本体を含む23seed files／11,221,153,522Bの64MiB保全コピーとSHA照合は全件完了しました．controllerのraw_preservation_receiptはall_complete=true，held=[]です．コピーを新規corpus容量へ加算しません．

公開filter recipeからの未来側supportは条件付き4.125秒．窓終端→eventを4.5秒＋clock guard，feature availabilityは終端＋4.125秒，decision→eventは最低0.375秒です．元XDFと配布binary／recipe同一性が未確認なのでonline_causality_verified=falseを維持します．実推論の遅延も別に測ります．

## 凍結した cohort／model source

model/eeg_llm.py SHA：
`e30811b027414426cdfbed14a822c27695af2ab1be00106d89fe32a5fb740d2f`．
実行中のrunがこのsourceを使うため，その場でmodel本体を改変しません．新architectureは別ファイル／別protocol／別runで行います．

v2 manifest：`/content/xavier/prepared/ai_coding_uniba_v2_lead4p5/manifest.jsonl`，SHA：
`af878638c2fff6d6c7f619cd22005a057e797f1152553517bf05ce795aec2eac`．
train4357／val1506／test1956，18／6／8人．AI送信2／2／1例でこのclassの性能推定は困難です．総7,819窓／3.80090278h．

v3 manifest：`prepared/ai_coding_uniba_v3_natural_casecontrol_verified/manifest.jsonl`，SHA：
`030e8550bd6f4c79005722eb617e9b64f5f80e87b77866b7f17bec4e7e2a1093`．
workstationとColabに同一bytes・32配列hash検証済み．train400（AI80）／val155（AI31）／test203（AI39），同じ18／6／8人．全758窓8ch×448／256Hz，0.36847222h．AI全150イベントと自然な直前操作を保持したcase-control cohortです．通常利用時precision／誤作動毎分を推定できません．v2と原記録を共有するので時間を合算しません．

自然cohort protocol SHA：d4e6c5f685dce53e73b3aa88d12ed07c88a8ac08a27fb78392dbd99aed0a6396．subject split SHA：bdf214b16c92b33c889b5bcb8b4ec7b888877c40d7992be8a796cdbc8ae23dae．

## モデルと評価の到達点

EEGlassのSmol1.7／Qwen8Bは3epochs完走．test78は全mouse_click予測，balanced accuracy1/3，shuffle差0．Smol best_pinnedは元のimmutable revision根拠付き．3train窓をM1で推論し，NeuroCore実router＋DryRunActionSinkの接続も3件成功，OS操作0／cloud call0．READMEに実行コマンドがあります．

Qwen v2は3epochs／1635steps，attempt2学習・validation1,332.96秒，CUDA peak reserved20,631,781,376Bで完走しました．初回memory guard停止も記録保存済みです．best SHA：
`29f349c199bd99232a2431f83179085e136a1509beb9c84d5b25dd25edba21f7`．
5条件GPU／strict証拠検証／private回収完了．fused BA39.54%，macroF1.37370，shuffleとの差macroF1+.00171，参加者95%CI[-.00700,+.00946]．固定confidence.8のfused coverage0%．goal=evaluated_goal_unproven．公開詳細はbenchmark/QWEN8B_UNIBA_V2_RESULTS.md／qwen3_8b_uniba_v2_aggregate.json．private archive49,002,964B，SHA：d9646d138f00a81c6893c1dc5e636d5df0caaac697d7f23cd3b9082c0f53af9b．

M1 Smol v3は3epochs／600stepsを1,541.7197秒で完走，8GiB allocator quota，MPS driver sampled最大4,437,557,248B．validation bestはepoch2，SHA：
`281716da86293d2b700ec003c967ca2c141515cc565b136a12137c34b73a3475`．
run：`private/xavier/runs/smollm2_1_7b_uniba_natural_v3_fused_v1`．固定203testの5条件評価は1回完了済みです．MPS true peakは測れていないためresource reportにsampledをtrue peakとして埋めません．実行成功とstrict peak gateの証拠不足を区別します．validation candidate scoring約1秒は名目lead0.375秒より遅いです．

その後M1の5条件評価は1177.71秒で正常終了しました．fused BA35.37%，context-only35.77%，EEG-only33.33%，shuffle35.37%，−10秒34.55%．fusedのfixed.8採用率0%，AIvsotherのfixed.5はrecall0/39，BA50%，AUROC.6165．strict integrity errors0，shuffle/raw shift203件すべて証拠passed，goal未実証．M1実機は他のdownload/metadata/controller/軽CPU処理と並行しておりisolated latencyではありません．M1 strict SHA8f837c0c0baedc513ed25730e4181b92b88a45aaf3f434c58e5bd4042478ddbc．

Qwen v3は`Qwen/Qwen3-8B` revision b968826d9c46dd6066d109eabc6255188de91218．run：`/content/xavier/runs/qwen3_8b_aicoding_uniba_natural_v3_fused_30e_v1`．protocol SHA：
`3c0c702d66024df9dee302536f6a643ad6397a1584936f63b532d29a5e87b95b`．
30epochsまたは3600秒，batch4／candidate4，lr.001，seed20261122，prefix8，width64，21GiB，expandable_segments．testを学習中に使わずvalidation bestだけで選択．正常なtime-budget停止も事前に許可された完了条件で，実epochsを報告します．OOM／integrity failureは保留．完走後samebestの5条件とhistory比較を実行します．Smol3epochsとの予算差を明記します．

Qwen v3は30epochs／3000steps／1340.03秒で完走，CUDA reserved peak18,492,686,336B．best SHAb7c8e8f37a1f9276cc58b35609a6e21191faacc138a6f37e37296976be2ed924．五条件のforced BAはfused35.77%，context-only33.74%，EEG-only32.52%，shuffle35.37%，−10秒36.59%．fused fixed.8採用率2/203．strict／回収／履歴比較は全て完了．archive107,137,145B，SHA dd608399c8098503f8ce314b924a707b17c07069690ab09bc38ffceb571d1f3d，canonical receipt SHA22aac8e5d686fb5c1e8cd3d9e52a684f42c7afc4d02d7ef0448055a719614c5c．

両v3モデルのAIvsother fixed.5 recallは0/39．AUROCはSmol.616479，Qwen.621482，差+.005003の参加者95%CI[-.051595,.061366]．全14条件の固定予測比較は `benchmark/V3_FIXED_MODEL_COMPARISON.md` とv3_fixed_models_history_aggregate.json．再学習・閾値選択・再推論はしていません．

v2 CPU spectrum／context／combined baselineは学習・評価・回収完了．公開はreports/cpu_baselines_uniba_v2_lead4p5_v1/aggregates/．v3 history baselineも完了，10.35秒，2threads．AIvsother BA：preEEG70.41%，guardedpredecision83.11%，guardedhistory＋context86.59%（TP39,FN0,TN120,FP44，precision46.99%は濃縮標本のみ）．benchmark/history_baselines.pyのscore-predictionsで固定LLM予測を再推論せず比較できます．historyモデルにEEGは使っていません．

## AI ラベルと次の encoder

Ollama qwen3.5:2b（実体2.3B/Q8_0）の15種event表記正規化を225train窓へ伝播済み．`weak_labels_propagated_v1/`．heldout参照0，元gold変更0，新しい潜在意図gold0．モデル自身の予測をtest正解へ使いません．

公式CBraModをprivate/upstream/CBraModへcommit b9e961003214326972c567eff390e75b0287e32aで取得済み．weights revision500543c7e30bda1b22bfd51a49301b238dee21fd，SHA0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178．weights_only／strict load，4,883,800parameters，実train3窓をCPU2threadsで通し0.033秒．GPU calls0，学習0，test0．公式source／19.8MB weightsのhashはreports/cbramod_cpu_capacity_probe.json．入力は最後1秒のみ200Hzへresample，V×1e4．これから結合学習をする場合は独立protocolを固定してください．

## 次の独立実験と外部 cohort

新module `model/cbramod_token_llm.py`（SHA b0c397176f8cb9e6dd759ed7169c0029c870f756c501ce317c2e6c7c41f32505），protocol（SHA fddc652d9cf963154d1dc07ff5903b3c7e1b4cfaf7f1d3b22ba00b0085e79bcb）を固定して実行済みです．frozen CBraModの8×200 featureをLayerNorm＋LinearでSmol1.7Bの8prefix tokensへ投影し，412,048parametersのみ学習．EEG encoderはCPU FP32，LLMはL4／M1です．元eeg_llm.py不変．A/B/C token IDs330/389/340，train400窓の2,400境界検証を通過し，同じ3logitsのCEとsoftmaxで統一しました．13CPUtestsをworkstation／Colab双方で通過（cloud scratchはfixture不在で12pass/1skip）．旧label長・EOS，encoder，入力長，正規化，予算を同時に変えた探索実験です．

run：`runs/cbramod_smollm2_uniba_v3_symbol_v1`．gate：`cbramod_smollm2_uniba_v3_symbol_evaluation`，receiptは同runの`variant_evaluation_completion.json`，checkpointはbest.pt．容量probe8/16/32が各50stepsを通過しbatch32を選択，probe weightsは本学習に再利用していません．本学習は30epochs／390steps／429.33秒で正常完了．最大CUDA reserved9,219,080,192B．best epoch23，SHA3019cddd288a9052fa91f9af0a7bf81b5662106178f16367525a98e6667f3a4c．後続wrapperの評価script path誤りはtest推論前に停止し，失敗receiptを保持して評価段階だけ再開しました．学習繰返し0．203×5条件評価は61.35秒で完了．strictのmemory_measurement_scope欄は初稿を保全して実測値を補完し再集計，追加推論0．

CBR forcedBA fused35.09%，context33.33%，EEG-only31.48%，shuffle30.05%，−10秒34.80%．fused固定.8 coverage47.29%，selective accuracy43.75%で未校正．AI AUROC.6053，fused−shuffle差.1609［.1004,.2548］，−10秒差.1293［.0521,.2227］は探索的信号ですが，context-onlyとの差CI［−.0399,.1158］は0を含みます．固定.5はAI10/39，FP36/164．L4 p95 46.92msは部分処理のみ．strict integrity0，shuffle/shift203件passed，resourcepassed，goalunproven．publicはreports/cbramod_symbol_final.aggregate.json とbenchmark/CBRAMOD_UNIBA_V3_RECORDED_COMPARISON.md．

CBR archive177,230,987B，SHAcb0fd60b4a2488abacdbc7418d71061d5bd8d51bd45a9a018ec2a804bd0d1878，371members全SHA一致でworkstation回収完了．同best4,963,861B．LLM／CBraMod元weightsは別の固定依存です．新M1 wrapperはintegration/try_cbramod_recorded_eeg.py，focused9tests通過．train各class1例のM1試行もexit0，checkpoint／閾値／source不変，3例の分類は一致しましたがtrainでのruntime確認です．全例proxy_label_has_no_explicit_intent_mappingでrouter block，OS0／cloud0．private/xavier/demos/cbramod_uniba_v3_train_examples_v1/summary.json．contract SHAa55521c8360bfe7b19dfc8b1b7a58b93f07e22ca790df9284e576ac013de9490．実施済みtrialを再実行せず，ユーザー用の新しい出力先はREADMEにあります．

ITU外部test-only cohortは `model/prepare_itu_external.py` で準備・独立検証・workstation回収済み．`/content/xavier/prepared/ai_coding_itu_external_v3_natural_casecontrol_v1`，manifest SHAcfd93863b8f8e263174e6ab59395496efbd397e821542fa3a1f0c1e5faa503b0，protocol SHA6dae42f26833eab36028b886907a4cfcf9047f3ad76af182c497129cd885eca0．25clock-pass記録全て，630窓（AI126,key252,click252），25arrays9,034,880B，0.30625h，train/val0．630原波形窓全サンプル，630実label，1,260history，204source membersを独立再照合．49files回収hash一致．公開reports/itu_external_cohort_v3_20261122_v1/．原波形hashのUniBaとの重複0ですが，unqualified participant labels21件共有のためglobal human identity分離の証明ではありません．1外部siteのためsite-cluster CIは不能です．

ITU外部評価はbenchmark/external_evaluation.pyとscore_external_predictions.pyの別契約で全推論・集計・回収・独立監査まで完了しました．旧protocol f1633...はhistory adapterの入力schema誤想定（availability=trueを要求）を予測0件の時点で検出したため保全し，offline false＋4.125s＋過去境界を検証する修正版を予測前に固定しました．有効protocolはprotocol.itu_external_evaluation.v2.json，SHA1acd857958d034125c4a69d222b7564c15531f17cdf89a509d6319beb625eb02．外部cohort／checkpoint／thresholdは不変．3 neural×5条件＋4固定history＝19条件，各630例，計11,970予測を最終QA済みです．

外部fused3classBAはSmol34.52%，Qwen36.11%，CBR34.13%．AI AUROCは.559366［.514256,.607123］／.534218［.470577,.599478］／.515920［.459164,.572050］．fixed.5 recall0/126，0/126，7/126（CBR FP35/504）．3モデル全9対照差CIが0を含み，UniBaのCBR信号の外部再現は未確認．public benchmark/ITU_EXTERNAL_THREE_MODEL_RESULTS.md，itu_external_three_models_aggregate.json（SHA058d8a6255b57057520262206ffae1663cf2aba9baa978195249defee4258e86），itu_external_completion_audit.json（SHA524a00074112fd696b1ea9e8a2a405d16f4f562b26135d6126495c81831aa94d）．private1955members回収照合，archive SHA92aab5641e54d3f95193b1f83fc37bd41c950747d1c25c9f4a97c2c8b5f143c5．stateはprivate/xavier/external_itu_ops/postprocessing_state.json．新SSL gateのlive反映後にcanonical公開・回収も完了しました．元3モデルを再推論しません．

ITU history4モデルはCPU推論済み，fit0／閾値変更0，元UniBa1,624確率を差0.0で再現．AI BAはpre-EEG history68.55%，＋context67.36%，guarded decision history77.48%，＋context76.98%．public reports/itu_external_history_20261122_v2/．EEG無しで得られる成績であり，EEG寄与比較の強い対照です．濃縮標本の20%AI比率を自然利用の頻度と混同しません．

追加の未ラベル学習候補はUniBa train18人15.991899h＋portable19人34.897061h，75記録50.888960h，物理EEG samples5,454,564,672Bです．portableの3装置を同じsubject groupにして15人train／4人SSL reconstruction validationを先に固定し，ITU全量とUniBa val/test・clock不合格を除外しました．この50.889hは品質検査前の候補量です．UniBa18記録のBIDS V／header0.1µVの矛盾は台帳保存．fresh MNEと独立header換算で元train400窓の全1,433,600値がsealed配列に一致し，既存尺度は保持しました．inventoryはprivate/xavier/ssl_source_inventory_v1/．

前処理は196.48秒で完了し，全6,040候補窓の排他的会計も通過しました．固定QC通過は2,200窓／18.333333h，内訳はtrain1,678窓／13.983333h，SSL validation522窓／4.35hです．166配列1,685,013,248B．UniBaは1,902候補から8窓／240秒／3原記録のみ採用され，portable2,192窓が大半です．固定条件はresampling後の全sample最大絶対値100µV以下，各channel std0.01µV以上，finiteです．QC後に条件や尺度を変更していません．大きなdomain偏りを結果に付記してください．全166配列・421,248,000値の独立検査，実8/32chのtrain勾配finite/nonzeroも通過．最大abs99.955µV／最小channel std0.391µV．

corpus：private/xavier/prepared/ssl_portable_uniba_v1/．manifest SHA216729840806ca20040331baf59ddeadea4e62bc48a19a57c183810e13dd9043，artifacts SHA3dc4599eef7093584e806c15c39c99be42b0fef33851a60e2f5c7ade6f883be2，allowlist SHAe2c44f7135afa8a3d5a896d6ad7f8f335d418d77a34d2815497e1335ef736215，preprocessing protocol SHA642246adb9bf39a9652a2ccb9fdbead834c611fbd4f5325b8dfc08d54fba45a5．採用時間を実際にモデルが見た時間と同一視せず，runのseen IDs／区間和集合で報告してください．

SSLは30秒非重複窓，1秒200Hz patches，実8/32EEG channels，mask.5，公式4.924M encoder/headのmasked raw reconstruction MSEです．validation maskはID+seed固定．SSL1800秒＋同じ400窓へのalignment900秒，旧CBRの末尾1秒・A/B/C・seed・batch32・30epochs等を維持し，追加SSL重みだけを変更する別moduleです．phase plan SHA b547fb49aaac9b60c10aec6c48c668f632ff8b366b849a68c307ed10796959e3．trainer47f24f9429bdc9af9495b121c4bf2cbc94ce62dbeb6b1033eeff6a4a88fe60fd，newalignmentf246128e909d3581168fbda7a5d062dc31491f27e96b6cae999579795a04fe10．実公式8/32ch backprop，array index完全対応，再開後val非改善で旧bestを保持するcaseを含めCPU17tests全PASS．public reports/cbramod_ssl_cpu_verification.json．corpus実SHA入り最終protocol model/cbramod_ssl_training_protocol.jsonはSHA9f264097ae6699fa4320e89afc26a5eac999bee0113788325c6001f6d249a8b0で封印済みです．入力検証終了時点ではworkstationCLI転送準備で，GPU起動はまだ未確認．4200秒のadmission reserve，実容量，GPU idle，元session期限を確認してから起動します．新しい実験・予算・gateを固定し，既存deadlineを無記録で延長しません．

rootの `reports/selected_eeg_interval_audit_20261122.json` で全manifestの原記録SHA別interval unionを独立確認．UniBa trainは元700秒／CBraMod末尾400秒，val271.25／155秒，test355.25／203秒．ITUは1102.5／630秒．どちらの窓長でもoverlap0．epoch反復提示時間や取得原記録87.9hとは合算しません．

M1の一括試用は実行済みです．`private/xavier/demos/m1_natural_v3_train_examples_v1/summary.json`．train各class1例，heldout使用0，全3例abstainでrouter block，OS0／cloud0．元32配列，checkpoint，model，backendのhash不変．そのprepared dirは再実行せず，新しい試用はREADMEの未使用output名を使います．公開集計は `reports/recorded_eeg_demo_20261122_v1/`．

## 次の作業と継続予約

追加コードを含むlocal validation04は，38 test files／89 Python source filesのsnapshotで365件＋127 subtestsが通過しました．初回はSSL用fixture環境変数名の違いで2件skipとなりましたが，既存の公式weightsを指定してその2件だけを追加実行し通過，未実行skipは0です．前後source SHA不変，AST／Ruff／diff checkも通過．public reports/local_validation_20261122_04.json，SHA77349647cdf393ce460f6f9da2f475ddf6dee766ee5b48d72d0c275e65e43494．後から追加されるharness等のコードをこのsnapshotで検証済みとは扱いません．

1. 完了済みの M1／Qwen v3／CBraMod symbol学習・五条件評価・strict・回収・M1試用を再実行しない．現在のSSL workflow stateと起動receiptを先に確認する．
2. 全23seed原本・checkpoint・予測・reportのSHA回収receiptを照合する．chunkや保全コピーを新しいcorpus bytesへ足さない．大容量CLI downloadへ戻さない．
3. ITU外部630例での元3モデル×5条件＋保存済み履歴4モデルは計19条件を完了済み．新SSLモデルは別の探索的external protocolを予測前に固定し，元評価を変更・再実行しない．外部cohortをtrain/val扱いに書き換えず，語彙・scaler・係数の再fitや閾値選択も行わない．
4. 固定した13.983333hのtrain EEGでSSLを実行し，同条件のalignment，5条件評価，別protocolのITU評価，strict，回収，M1試用を完了する．ITU全量と既存val/test原記録を除外．元checkpointを保存し，容量/CU・元Colab期限13:20:34 JST・回収余裕を守る．
5. 実ハーネスの目的達成・idle誤作動・因果的live EEGの評価は未成立として残す．公開coding資料にUML・初期APIはあるが自動Java test本体と判断前の編集状態記録がないため，遡及的task completion採点は不能．別に独立Java plumbing benchmarkを整備する場合，人工fixtureの成功をEEGモデルによる自律操作成功と扱わない．
6. metadata新規解決分をimmutable snapshotからINSERT-onlyで追加し，容量・アクセス保留を明示する．未解決scope，Guestbook，SSH認証，storageを成功扱いしない．smac19認証後は本人性と実資源を確認して計算投入する．
7. 新結果を明記してcommit／pushし，現在のbranch・PR71の状態と最新workflowを確認する．PR70は既に外部でmerge済み．force push／mergeは行わない．raw／重み／秘密をgitへ入れない．

同一threadの一回の継続予約は **2026-11-22 14:15:26 Asia/Tokyo**，enabled確認済み．ID：`Automation_f439325b5d888191bc75e92d1ba97be7`．利用モデルをProへ固定するパラメータは公開されていないため，Pro固定済みとは報告しません．予約はこの文書，git，receiptを読んで既存ジョブを重複せず継続する設定です．全て完了した場合は不要な予約を解除し，残件があれば具体的な次の再開条件を残してください．
