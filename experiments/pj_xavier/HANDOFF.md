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
- Branch：`feat/pj-xavier-training-20261122`
- Base：`966907f0087bd38d74b1e1ddae32dda739de986b`
- Draft PR：[70](https://github.com/yasufumi-nakata/NeuroCore/pull/70)
- 最初に push 確認した commit：`93c952209274721b25acdde4f49cf121b1d4b78e`
- 最初の CI／Dependency Review／Repository Health／CodeQL は全て success．以降の commit は最新の git／PR を読み直すこと．
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
| workstation recovery controller | 29527 | `controller/launch_receipt_v3_gate.json`，`config.json`，`state.json`，`controller.log` |
| workstation M1 v3 五条件評価 | supervisor 29035／child 29036 | `m1_natural_v3_final_evaluation/`，`evaluations/smollm2_1_7b_uniba_natural_v3_fiveconditions_v1/` |
| Colab Qwen v3 30 epochs 学習 | supervisor 28620／child 28621 | `runs/qwen3_8b_aicoding_uniba_natural_v3_fused_30e_v1/` と `private/xavier/variant_ops/` の実行 receipt |
| workstation Qwen v3 一連処理 supervisor | 32691 | `private/xavier/variant_ops/v3_workflow_state.json`，`v3_workflow_launch.json` |

workstation の表中の短いパスは `private/xavier/` 配下です．M1 v3 training PID20716 は3epochs／600stepsで正常終了済みです．Qwen v2 の学習と五条件評価・strict検証・private回収も終了済みです．seed取得は23files全てcompleteです．完了ジョブを再投入しません．

Qwen v3 workflow supervisorは学習終了→best凍結→5条件test1回→strict／archive→workstation回収→既存history予測とのCPU比較→公開集計→canonical receiptを進めます．エラーはblockedで停止し，元controller期限内で動作します．このsupervisorが動いている間に同じ評価を手動で重複起動しません．

bulk の元絶対 deadline は **1795322496.632661**，元 max bytes は **64,062,645,375**，reserve35GiBです．controller の元絶対終了は **1795324658.879168** です．再起動で5時間を取り直しません．metadata resolver の元 deadline は1795319128.862762です．

## controller の重要な修正

Colab CLI0.7.4の大きな `download` はHTTP JSON／base64全体をRAMへ保持し，ITU4.65GBの回収が進まなくなりました．旧controller12764とその子22431を同定して停止し，state／lease／原本を保存しました．新controller29527は**64MiB chunk**ごとにsource SHA，offset，size，chunk SHAを照合し，localへ1MiB bufferで追記します．中断tailから再開し，全体SHA一致まで保全完了にしません．installed CLI本体は変更していません．

元 `launch_receipt.json` は保持し，新起動は `launch_receipt_v3_gate.json` です．当初のconfig hashはbb7370d22be14d85b9b62cf2734fd18e1460ce91edb0a276af0ec6d33aeef13e．remote_coreのACK cleanup修正は次cycleで読み込まれます．公開根拠は `reports/controller_v3_gate_20261122/` とcontroller/validation.jsonにあります．36 testsと実転送を確認済みです．

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

Dataverse DOI10.7910/DVN/K5EDB4の342filesは，公式APIが必須Guestbook応答を要求するためaccess_action_requiredです．HTTP400を一律にrestricted扱いせず，実応答を記録しました．元running／completeは変更していません．新acquire.py SHAは0e265a3c8c35f70ffe3e7b3eef76045468ac2883f8a68e12d7252825beaa70c5．稼働bulkは以前の043031...をロード済みですが，保留rowは選択しません．

最新の完全取得量は `reports/acquired_progress_002.json` の10:08〜10:09観測で，39 unique file contents／14,569,523,189Bです．workstation16files＋Colab23files．全completeをSHA再検証し，replicaとpartialを加算していません．次回は新しいsnapshot番号で再観測し，既存時点を上書きしません．`acquisition/report_progress.py` は468k行をstream処理する版です．7tests通過．

## 検証済みの EEG と前処理制約

`reports/verified_eeg_quantities_20261122.json`：118記録，**87.90746536458333時間**，物理EEG scalar **10,924,347,024B**．EEGlass1記録1.08882222h，UniBa34記録30.61540256h，ITU26記録21.30617947h，portable57記録34.89706111h．原記録／filtered copy／channel時間／学習窓を二重加算しません．ZIP容量から時間を推定しません．

UniBa32／34人，28.58953451hが固定100ms／5000ppmの時計監査に通過しました．P003／P015を保留．ITU25／26人，20.51673720hが同基準に通過し，P012は最大LOO4.30216秒で保留．ITUは全員外部test候補として予約，学習やモデル選択に使っていません．共通8EEGとphase補助は全26記録にあり，時計監査と数量監査は独立照合済みです．

ITU archive4,650,970,671B，SHA：
`1e192b5e65931b9a44858acd987d540d3436de3fddf99454170b93434c2c99b5`．
Colab raw root：`/content/xavier/verified_ai_coding_itu_v1`．workstationの全監査資料は `ai_coding_itu_quantity_audit_v1/` と `ai_coding_itu_external_clock_v1/`．archive本体の64MiB保全コピーはcontroller receiptを読み直してください．

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
run：`private/xavier/runs/smollm2_1_7b_uniba_natural_v3_fused_v1`．固定203testの5条件評価を1回実行中．MPS true peakは測れていないためresource reportにsampledをtrue peakとして埋めません．実行成功とstrict peak gateの証拠不足を区別します．validation candidate scoring約1秒は名目lead0.375秒より遅いです．

Qwen v3は`Qwen/Qwen3-8B` revision b968826d9c46dd6066d109eabc6255188de91218．run：`/content/xavier/runs/qwen3_8b_aicoding_uniba_natural_v3_fused_30e_v1`．protocol SHA：
`3c0c702d66024df9dee302536f6a643ad6397a1584936f63b532d29a5e87b95b`．
30epochsまたは3600秒，batch4／candidate4，lr.001，seed20261122，prefix8，width64，21GiB，expandable_segments．testを学習中に使わずvalidation bestだけで選択．正常なtime-budget停止も事前に許可された完了条件で，実epochsを報告します．OOM／integrity failureは保留．完走後samebestの5条件とhistory比較を実行します．Smol3epochsとの予算差を明記します．

v2 CPU spectrum／context／combined baselineは学習・評価・回収完了．公開はreports/cpu_baselines_uniba_v2_lead4p5_v1/aggregates/．v3 history baselineも完了，10.35秒，2threads．AIvsother BA：preEEG70.41%，guardedpredecision83.11%，guardedhistory＋context86.59%（TP39,FN0,TN120,FP44，precision46.99%は濃縮標本のみ）．benchmark/history_baselines.pyのscore-predictionsで固定LLM予測を再推論せず比較できます．historyモデルにEEGは使っていません．

## AI ラベルと次の encoder

Ollama qwen3.5:2b（実体2.3B/Q8_0）の15種event表記正規化を225train窓へ伝播済み．`weak_labels_propagated_v1/`．heldout参照0，元gold変更0，新しい潜在意図gold0．モデル自身の予測をtest正解へ使いません．

公式CBraModをprivate/upstream/CBraModへcommit b9e961003214326972c567eff390e75b0287e32aで取得済み．weights revision500543c7e30bda1b22bfd51a49301b238dee21fd，SHA0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178．weights_only／strict load，4,883,800parameters，実train3窓をCPU2threadsで通し0.033秒．GPU calls0，学習0，test0．公式source／19.8MB weightsのhashはreports/cbramod_cpu_capacity_probe.json．入力は最後1秒のみ200Hzへresample，V×1e4．これから結合学習をする場合は独立protocolを固定してください．

## 次の作業と継続予約

1. M1／Qwen v3 の進行中ジョブの終了を確認し，凍結bestの5条件評価，strict証拠検証，historyとの対比，private回収を完了する．testを見て同じcohortへ再調整しない．
2. controllerの全23seed原本・checkpoint・予測・reportの最終SHA回収を確認する．chunk数を新しいcorpus bytesへ足さない．大容量CLI downloadへ戻さない．
3. 一括試用CLIが追加されていれば，M1評価終了後にtrain各class1例だけで1回実試行し，確率・abstain・dry-run理由を保存する．held-out評価と区別する．
4. 未使用ITU外部testは，公開metadata・common montage・時計・cohort抽出規則を先に固定してから使用する．元データのonline causalityが不明なままlive性能を主張しない．
5. CBraMod等の事前学習EEG encoderとLLMの結合を別protocolで比較する．現行source／既存checkpointは保存する．
6. metadata新規解決分をimmutable snapshotからINSERT-onlyで追加し，容量・アクセス保留を明示する．未解決scope，Guestbook，SSH認証，storageを成功扱いしない．
7. 新結果を明記してcommit／pushし，PR70と最新workflowを確認する．force push／mergeは行わない．raw／重み／秘密をgitへ入れない．

同一threadの一回の継続予約は **2026-11-22 14:15:26 Asia/Tokyo**，enabled確認済み．ID：`Automation_f439325b5d888191bc75e92d1ba97be7`．利用モデルをProへ固定するパラメータは公開されていないため，Pro固定済みとは報告しません．予約はこの文書，git，receiptを読んで既存ジョブを重複せず継続する設定です．全て完了した場合は不要な予約を解除し，残件があれば具体的な次の再開条件を残してください．
