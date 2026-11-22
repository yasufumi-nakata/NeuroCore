# PJ-Xavier 学習・評価の引き継ぎ

更新：2026-11-22，日本時間．**実行中です．既存ジョブと台帳を読んでから継続してください．**
PJ-Xavier は NeuroCore の AI の研究開発です．目的は，ハーネスから判断時点以前の環境を受け取り，EEG に応じた意図の推定と agentic な操作を実現・検証することです．単なるクリック分類や合成リプレイを目的達成と表示しません．

## 許可と接続経路

ユーザーは EEG-dataset から自動取得できる EEG を含む全データ（マルチモーダルを含む）の取得，AI 補助ラベル，学習，各 checkpoint の評価，実機試用，GitHub への push，継続予約を確認不要で許可しています．**Colab は必ず workstation の `/Users/yasufumi/.local/bin/colab` CLI を経由します．直接の Colab MCP は使いません．** 現在のセッションは `xavier-eeg-20261122`，remote root は `/content/xavier` です．

既存 CU 枠のみ使用します．開始時 417.49 CU，L4 の観測消費率 1.94 CU/h でした．現在残高は `colab usage` で読み直してください．自動購入はしません．

SFC VPN の保存構成は接続できましたが，smac31 と，ユーザーが追加指定した smac19 の SSH 認証は未成立です．smac19 は DNS と SSH 到達まで確認しました．過去に SSH パスワード自動入力が自動承認レビューに拒否されたため，別経路で秘密を注入して回避しません．認証済み SSH が使えるようになれば preflight し，実測した容量だけを割り当てます．38台の歴史的合計 RAM を一つの GPU メモリと見なさず，今回は cluster job を投入済みと報告しません．

## 正しい作業場所

- Repository：`yasufumi-nakata/NeuroCore`
- Worktree：`/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training`
- Branch：`feat/pj-xavier-training-20261122`
- Base：`966907f0087bd38d74b1e1ddae32dda739de986b`
- 実験コード：`experiments/pj_xavier/`
- 原データ・重み・詳細台帳・ログ：`private/xavier/`（git 除外）
- 目録 checkout：`/Users/yasufumi/Documents/GitHub/EEG-dataset`，読取対象 origin/main は `5594ef98` で始まる commit．ローカル main は更新しません．

**original checkout `/Users/yasufumi/Documents/GitHub/NeuroCore` の既存未追跡ファイルを一括削除・上書き・stage しないでください．** model 担当が一時的に original 側へ新規ファイルを保存したため，正しい worktree へコピー・hash 照合しました．`private/xavier/model_agent_path_correction_inventory.json` と `model_worktree_copy_receipt.json` に経緯・birthtime・hash が残っています．original 側は削除や推測復元をしていません．

## 実行中プロセスの確認

PID は再利用されます．次の receipt と実際の command，開始時刻を照合してください．Colab の `/proc/PID` 存在だけを生存判定にせず，Z/X 状態を除外します．

1. workstation bulk：`private/xavier/bulk_launch_v3.json` と `bulk_download.pid/log`．最新起動時 PID 8814．全403 EEG候補 release，29,206 file entries のキュー．元5時間 deadline を保持し，35 GiB reserve を守ります．
2. workstation metadata resolver：`metadata_refresh_launch.json`，`metadata_refresh.pid/log`．起動時 PID 5571．全目録の未解決取得先を再照合中で，新規 resolved を `catalog_fresh_resolution.jsonl` に残します．
3. workstation controller：`private/xavier/controller/launch_receipt.json`，`config.json`，`state.json`，`controller.log`．初回 PID 12764．`--apply --interval-seconds 120 --max-seconds 18000` で起動済みです．26 tests と実機 readonly tick は PASS です．priority 台帳の照合，checkpoint／report／complete raw の回収を継続します．
4. Colab seed download：`/content/xavier/download.pid/log`，`downloads.sqlite`．起動時 PID 2312．EEGlass と AI-assisted coding の計23ファイルを取得．
5. Colab priority EEG：`priority_eeg.sqlite` の UniBa EEG は complete．元 DB の `delegated_priority` は controller が実 SHA・MD5・identity を照合して reconcile します．同じファイルを2回数えません．
6. Colab 本学習：`train_uniba.pid`，`train_uniba_launch.json`，`train_uniba_child.json`，`train_uniba_exit.json`，`train_uniba.log`．下記を参照してください．

controller は train を自動でやり直しません．failed／partial／completed を区別します．追加 GPU ジョブは今の学習プロセスが終了したことを確認してから起動します．CPU の前処理・軽量 baseline は2 thread以下で並行可能です．

## データ台帳と取得の修正

目録は5,165行，最大ID011520です．`catalog_latest.csv` と queue の ID 集合は完全一致，不足・余剰0でした．CSV SHA：
`142a7e8da1ca3c8439082cf3af768cab9ffcdb0f967c003f86dd5ee71f1ee5f8`．

- 解決済み全 file entries：50,696，延べ10,431,903,294,358 B．非EEG・不明scope・付随データ・重複を含みます．
- 明示的な EEG 候補：405 releases，29,229 files，4,673,956,432,245 B．取得完了量・重複排除済み純EEG量ではありません．
- 2 seed releases を Colab，残り403を workstation に分けました．容量不足やアクセス制限を理由付きで残し，無条件に成功扱いしません．
- `catalog_resolved_queue_v2.jsonl` SHA：
  `1cd07762fdcd14e589f4852b05577d31967a9466314d3cdd6060d88df562ce2c`．
- 同一releaseの同名異fileを provider file ID の namespace で分離．2,201 entries／32 releasesを変更し，50,696 jobs の全件 ingest，再投入増分0，重複path0を検証しました．sidecar対応の曖昧な release は学習保留です．
- 巨大ZIPのストリームが途中でEOFになる実障害を検出しました．`partial < expected` は checksum failure と区別して再開し，受信 bytes を計上します．最終サイズ超過・完成checksum不一致は integrity_failed のままです．稼働 bulk は修正版へ更新済みです．
- 現取得コード SHA：
  `0430315971c1e77cff2872b46b67dc1921ac656eea1b6ae04a81063302e144e7`，27 tests PASS．
- Colab の稼働 seed worker は旧取得コードをロード済みです．更新する場合は自分の PID を同定して終了・partial保持・lease修正を行い，元 deadline を保って再開します．無条件の二重起動をしません．

## 検証済みデータ

EEGlass：元記録3,919.76秒＝1.088822時間．raw／filtered の二重計上をしません．379非重複窓（train225，val76，test78），3ch×438 samples，総入力664.008秒．原manifest SHA：
`87e98a0d50f6288bc9b293de32a40e118431eaf3431514b87271da6b3a9daf5a`．

UniBa EEG（Zenodo20801101）：34人／34記録，30.61540256時間．archive 881,786,830 B，公開MD5確認済み，SHA：
`801e3e29ce6c58151c7a3a7aa1be17a6a0ee64d3cd355d43dd09418368a8e195`．
workstationの `private/xavier/colab_raw_archive/zenodo-20801101-20801101/bids_eeg_uniba.tar.gz` にもcopy・SHA検証済みです．32人（28.58953451時間）が時計監査に通過，2人は drift／残差で保留しています．EEGと行動ログの原点差を補正せず使わないでください．

固定v2学習manifest：
`/content/xavier/prepared/ai_coding_uniba_v2_lead4p5/manifest.jsonl`
SHA：`af878638c2fff6d6c7f619cd22005a057e797f1152553517bf05ce795aec2eac`．
8物理EEGch×448 samples，7,819窓，3.80090278時間，112,097,280 Bの配列．train18人4357窓／val6人1506窓／test8人1956窓．補助の `otree_phase`／`STI_artificial` は入力禁止です．

公開recipe由来のfilter supportは条件付き4.125秒です．窓終端からeventまで4.5秒＋clock guardを取り，feature availabilityは窓終端＋4.125秒，decisionからeventのminimum leadは0.375秒です．binaryとrecipeの同一性未確認のため online_causality_verified=false を維持します．推論時間を含む実際の deadline 達成も別評価です．

元ログには150のGPTSentPromptがありますが，v2の静かな入力窓・非重複条件では train2／val2／test1 の計5例です．agent委任の精度を推定できません．この不足を偽ラベルで水増ししないでください．v2と進行中モデルは固定し，別の自然作業 case-control v3 cohort を事前プロトコルで準備中です．進捗は model/natural_cohort_v3_protocol.json と担当のreportを確認し，別 cohort のデータを v2 学習へ混ぜません．

## 完了したモデルと実機試用

1. SmolLM2-1.7B-Instruct＋EEG接続部：3 epochs，171 steps，35.006秒．L4 peak reserved 3,900,702,720 B．
2. Qwen3-8B＋EEG接続部：3 epochs，675 steps，227.354秒．L4 peak reserved 16,821,256,192 B．

両方とも EEG 接続部の学習で，LLM本体は凍結です．時間に download／backbone load は含みません．EEGlass test は全78件mouse_click，balanced accuracy=1/3，macro F1=.3030，shuffle差0です．accuracy83.3%をEEG解読成功と記載しません．

checkpointと107ファイルのrun資料を workstation `private/xavier/colab_snapshot/` へ回収し，ZIPと全file SHAを照合済みです．ZIP `xavier_artifacts_v2.zip` SHA：
`59959cbce189c4a8b6fae0b835a6053d47b32a2c4d1a33f7cad73d37dcbc0566`．
controllerの新snapshotにはさらに新しいデータが含まれます．

Smol1.7B `best_pinned.pt` を M1 / 16GiB で実EEG3窓に対して推論しました．BF16，budget8GiB，candidate batch1．driverの観測値4,310,319,104 B，p50 776.4ms．n=3・cold呼出を含む実機確認であり，steady-state latencyや精度評価ではありません．
`private/xavier/local_m1_predictions.jsonl` と metadata に結果があります．NeuroCoreのrouterへのdry-runも3件成功し，OS操作0・cloud agent call0です．

## 進行中の UniBa 8B fused 学習

モデル：`Qwen/Qwen3-8B`
revision：`b968826d9c46dd6066d109eabc6255188de91218`
source SHA：`e30811b027414426cdfbed14a822c27695af2ab1be00106d89fe32a5fb740d2f`
出力：`/content/xavier/runs/qwen3_8b_aicoding_uniba_fused_v1`

train-onlyの3step容量probeではbatch8が21GiB以内でしたが，本実行24step時にreservedが23,290,970,112 Bとなりguardで停止しました．allocatedは20,423,472,128 Bでした．失敗と `epoch_001_step_0000024.pt` を保存しています．同じbatch・seed・optimizer・cursorで `PYTORCH_ALLOC_CONF=expandable_segments:True` を設定し再開しました．再開supervisorは初回観測PID16643です．プローブ結果を完走証明と扱わないでください．

3epochs，batch8，candidate batch4，21GiB，max_seconds3600，fused，subject split，`--finalize-test` なしです．最終testは学習中に採点せず，validation bestを固定してから実施します．再びmemory guardに達したら同じ失敗を反復せず，前試行を保持した別runでbatch4等へ変更してください．

5条件評価runnerは完成し，CPU40 testsを通過しています．benchmark担当が現行trainの終了を確認してから実行します．same best checkpointで fused／context_only／eeg_only／within-recording shuffle／実原波形の10秒前を評価します．これは同一学習モデルの入力除去ablationで，別学習baselineと区別します．raw派生time-shiftには実sample interval，hash，同じsubject/session/recordingの証跡が必要です．CPU logistic baselineもEEG/context/結合の3条件で準備中です．

完了receipt contract：
`runs/qwen3_8b_aicoding_uniba_fused_v1/variant_evaluation_completion.json`．
status=completed，exit_code=0，同checkpoint SHA，5 variant名，同report本体SHAを必須とします．reportのschemaは `pj-xavier-variant-evaluation-v1`，各conditionsは measured または理由付きblocked．goal=falseでも評価実行の完了は認めます．

## 弱教師ラベル

workstation Ollama qwen3.5:2b（実体2.3B，Q8_0）で，225train窓の15種類のevent表記を正規化済みです．teacher digest，prompt/source/schema hashを保存しています．凍結manifestと実際のtrain所属を照合して225レコードへ伝播済み：
`private/xavier/weak_labels_propagated_v1/`．
heldout参照0，元gold変更0，新EEG0，新しい潜在意図gold0です．225/225の一致はevent名の正規化整合性で，EEG decoding精度ではありません．label worker の新CLIには `--source-records` が必須です．

## 継続と報告

同一threadの再開予約は **2026-11-22 14:15:26 Asia/Tokyo** に設定・enabled確認済みです．予約IDは `Automation_f439325b5d888191bc75e92d1ba97be7`．利用モデルをProに固定できたとは報告しません．一回の継続で，まず現状態・git・controllerを読み，完了済みのジョブを重複させず残件を進めます．

本turnの最終 commit/push と新しい報告は後段の追記を参照してください．無条件merge，force push，rawや秘密のgit公開は行いません．取得候補のTB，実取得bytes，header検証済み時間，学習に採用した時間を区別して報告してください．
