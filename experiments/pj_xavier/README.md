# PJ-Xavier：EEG と LLM の学習・評価

NeuroCore の EEG デコーダと LLM 接続部を開発する実験コードです．目標は，判断以前の環境情報と EEG から操作を推定し，実ハーネスでタスク完了まで評価することです．現在の実装では，学習可能な EEG encoder／prefix を既存 LLM の `inputs_embeds` に渡し，出力トークンの損失から逆伝播します．

**実 EEG による 1.7B・8B モデルの学習，checkpoint 保存，held-out 評価，workstation での推論，NeuroCore router への dry-run を実行しました．目的の自律エージェント能力と EEG の追加寄与は，まだ実証できていません．** 最初の基盤を追加した [PR #70](https://github.com/yasufumi-nakata/NeuroCore/pull/70) はマージ済みで，追加監査・評価・修正は [PR #71](https://github.com/yasufumi-nakata/NeuroCore/pull/71) で継続しています．最新の実行場所と再開手順は [HANDOFF.md](HANDOFF.md) を参照してください．

## データ量：候補・取得済み・検証済みを区別

2026-11-22 日本時間の観測です．GB／TB は十進，GiB は二進です．

| 範囲 | 確認した量 | 意味 |
| --- | ---: | --- |
| EEG-dataset の目録 | 5,165 行 | 全 ID と取得先解決結果を照合し，不足・余剰 0 |
| workstation の取得キュー | 609 releases，468,040 file jobs | EEG を含む候補の release 全体．認証・規約操作・metadata の保留を含む |
| 上記キューの既知の容量 | 15,821,047,310,466 B，約 15.82 TB | 動画・補助データ・版の重複を含む候補容量．38 files はサイズ未確定 |
| 取得完了し SHA-256 を再検証した内容 | 39 files，14,569,523,189 B，約 14.57 GB | 10:58〜10:59 の workstation＋Colab 観測．同一内容の複製を二重加算しない |
| 元 EEG の header／binary を数量検証した範囲 | 118 記録，87.90746536 時間 | チャンネル数を掛けた時間ではなく，記録の長さの合計 |
| 上記の物理 EEG 値の非圧縮容量 | 10,924,347,024 B，約 10.92 GB | 補助チャンネル・動画・header・窓配列・モデル重みを除く．元の保存形式のまま |

取得状況の根拠は [acquired_progress_003.json](reports/acquired_progress_003.json)，全キュー更新は [fresh_queue_append_20261122_01](reports/fresh_queue_append_20261122_01/README.md)，数量の定義と hash は [verified_eeg_quantities_20261122.json](reports/verified_eeg_quantities_20261122.json) にあります．未完了の partial 9,081,331,952 B は取得済み容量に加算しません．取得候補全体の EEG 時間は未確定です．ZIP 容量から EEG 時間への一律換算はしません．

さらに全 468,040 file jobs を固定規則で構造監査しました．そのうち EEG と明示された path の信号形式は 62,303 files／7,209,327,960,212 B，135 releases でした．これは header を検証した純 EEG 量ではなく，取得順序を検討するための候補量です．未展開 archive，EEG を含み得る MEG container，動画・視線等の companion は別々に残し，EEG の有無を確かめる前に一律除外していません．詳細と全 609 releases の分類は [全キュー構造監査](reports/queue_structure_20261122_v1/README.md) にあります．

| 元データ | 記録数 | EEG 時間 | 物理 EEG チャンネル |
| --- | ---: | ---: | ---: |
| EEGlass | 1 | 1.08882222 h | 3 |
| AI-assisted coding：UniBa | 34 | 30.61540256 h | 8 |
| AI-assisted coding：ITU | 26 | 21.30617947 h | 64 |
| Portable EEG systems comparison | 57 | 34.89706111 h | 32 |
| 合計 | 118 | 87.90746536 h | 記録ごとに異なる |

raw と filtered export，同じ archive の保全コピー，原波形から切り出した学習窓を新しい独立記録として数えません．記録時間の検証は，信号品質・意図の正解・オンライン処理の因果性の証明とは別です．異なる corpus 間の同一参加者照合も未実施です．

Colab の優先取得は EEGlass と AI-assisted coding の全 23 files／11,221,153,522 B が完了しています．AI-assisted coding は EEG に加えて gaze，screen，behavioral，補助情報も取得しています．モデルへ実際に入力しているのは EEG と判定前の page／AI availability です．画面動画を vision encoder で学習した実験ではありません．

## 実際に学習したモデル

全て LLM 本体を凍結し，EEG 接続部を学習した実測です．基盤重みは immutable revision で固定しています．時間は原則として学習・検証 loop の範囲で，ダウンロードや環境構築を含みません．

| モデル／cohort | 実機 | 完了した学習 | 実測時間 | メモリの観測 |
| --- | --- | ---: | ---: | --- |
| SmolLM2-1.7B／EEGlass | Colab L4 | 3 epochs，171 steps | 35.01 秒 | CUDA 最大予約 3.90 GB |
| Qwen3-8B／EEGlass | Colab L4 | 3 epochs，675 steps | 227.35 秒 | CUDA 最大予約 16.82 GB |
| Qwen3-8B／UniBa v2 | Colab L4 | 3 epochs，1,635 steps | 再開後 1,332.96 秒 | CUDA 最大予約 20.63 GB |
| SmolLM2-1.7B／UniBa v3 | M1／16 GiB | 3 epochs，600 steps | 1,541.72 秒 | MPS driver の観測最大 4.44 GB，allocator quota 8 GiB |
| Qwen3-8B／UniBa v3 | Colab L4 | 30 epochs，3,000 steps | 1,340.03 秒 | CUDA 最大予約 18.49 GB |
| CBraMod＋SmolLM2-1.7B／UniBa v3 | Colab L4 | 30 epochs，390 steps | 429.33 秒 | CUDA 最大予約 9.22 GB |

MPS のメモリ値はサンプリング観測で，瞬間最大値の測定ではありません．実際の M1 上で学習が完走した事実と，true peak の未測定を分けて記録します．UniBa v3 の Smol は 166,656 parameters，v2 の Qwen は 299,776 parameters を学習しました．frozen backbone はそれぞれ約 1.71B／8.19B parameters です．

Qwen v2 は初回 step 24 で予約メモリが 21 GiB の設定を超えたため停止し，optimizer／RNG／cursor を含む checkpoint を保存しました．同じ実験を `PYTORCH_ALLOC_CONF=expandable_segments:True` で再開し，完走しました．短い容量 probe の成功だけを完走証拠にはしていません．

### 完了した held-out 評価

EEGlass は 1 人の時間分割です．両モデルとも全 78 test 例をクリックと予測しました．accuracy 83.3% はクラス比率によるもので，balanced accuracy は 33.3%，EEG shuffle による差は 0 です．

Qwen v2 は別の 8 人／1,956 test 窓で，同じ凍結 checkpoint の入力を変えて比較しました．

| 入力条件 | Forced-choice balanced accuracy | Forced-choice macro F1 |
| --- | ---: | ---: |
| EEG＋判定前の環境 | 39.54% | 0.3737 |
| EEG を除く入力 ablation | 39.31% | 0.3502 |
| 環境を除く入力 ablation | 33.33% | 0.2077 |
| 同じ記録内の EEG shuffle | 39.42% | 0.3720 |
| 原記録から実抽出した 10 秒前の EEG | 38.68% | 0.3622 |

fused minus shuffle の macro F1 差は +0.00171，参加者単位の補助 bootstrap 95% 区間は −0.00700〜+0.00946 でした．EEG の時点に固有の追加寄与は確立していません．固定 confidence 0.8 に達した fused 予測は 0 件で，採用率 0% です．forced-choice の値は採用された操作の成功率ではありません．この補助 bootstrap は partial score 閲覧後の記述解析で，test による再学習・選択には使っていません．[完全な結果と制約](benchmark/QWEN8B_UNIBA_V2_RESULTS.md) を保存しています．

## v3：AI への送信イベントを保持する評価

v2 の静かな窓条件では AI 送信が train 2／val 2／test 1 例しか残らないため，別の事前プロトコルを作りました．v3 は実ログの 150 AI 送信を保持し，過去の AI availability を合わせた keyboard／click の対照を取ります．参加者 split は v2 と同一です．

| 分割 | 参加者 | 全窓 | AI 送信 | keyboard | click |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 18 | 400 | 80 | 160 | 160 |
| validation | 6 | 155 | 31 | 62 | 62 |
| test | 8 | 203 | 39 | 82 | 82 |

758 窓の入力長は 0.36847222 時間です．元記録 30.6154 時間と，選択した窓を区別します．v2 と v3 は同じ原記録に由来し，合算して新しいデータ時間を作りません．AI 送信を増やした case-control 標本のため，通常利用時の precision や誤作動／分は推定できません．

**EEG を使わない強い対照も実測しました．** 判定前の key／mouse の回数・経過時間だけの CPU モデルでは，AI 送信対その他の balanced accuracy が 83.11%，さらに粗い環境を加えると 86.59% でした．後者は AI 39／39 を拾う一方，その他 164 例中 44 例を誤検出し，この濃縮標本内の precision は 46.99% です．意図を解読した成績ではありません．[history baseline](benchmark/HISTORY_BASELINES.md) を EEG モデルの比較対象にします．

Smol v3 は M1 上で学習と全 203 例の 5 条件評価を完了しました．Qwen v3 も，別途固定した予算で 30 epochs の学習と同じ 203 例の 5 条件推論，strict 検証，workstation への hash 照合付き回収を完了しました．Qwen と Smol の学習予算・実機は異なるため，成績差をモデルサイズだけの効果とみなしません．

| v3 の入力条件 | Smol1.7B：forced-choice BA | Qwen8B：forced-choice BA |
| --- | ---: | ---: |
| EEG＋粗い環境 | 35.37% | 35.77% |
| EEG を除く入力 ablation | 35.77% | 33.74% |
| 環境を除く入力 ablation | 33.33% | 32.52% |
| 同じ記録内の EEG shuffle | 35.37% | 35.37% |
| 原記録の 10 秒前の EEG | 34.55% | 36.59% |

fused の固定 confidence 0.8 における採用率は Smol が 0／203，Qwen が 2／203 です．AI 対その他の主評価は両モデルとも閾値 0.5 で全て「その他」となり，AI recall 0／39，BA 50.0% でした．AUROC は Smol 0.6165，Qwen 0.6215 で，Qwen minus Smol の参加者単位 bootstrap 95% 区間は −0.0516〜+0.0614 です．8B にした効果も，EEG を使わない強い履歴対照に対する優位性も確立していません．[14 条件の固定予測による比較](benchmark/V3_FIXED_MODEL_COMPARISON.md)，[Qwen v3 の全結果](benchmark/QWEN8B_UNIBA_V3_30E_RESULTS.md)，[自然作業 cohort の仕様](model/NATURAL_COHORT_V3.md) を参照してください．

別の探索実験として，事前学習済み CBraMod を EEG encoder にし，凍結 Smol1.7B へ投影する構成の学習・5 条件評価・strict 検証・workstation 回収も完了しました．ラベルを単一 token の A／B／C へ固定対応させ，同じ 3 logits で学習・評価する構成です．学習したのは 412,048 parameters の投影部で，4,883,800 parameters の EEG encoder と LLM 本体は凍結しています．検証集合で選択した epoch 23 の checkpoint を全条件で固定しました．[固定プロトコル](model/CBRAMOD_SYMBOL_EXPERIMENT.md) と [学習・評価・回収の実測](reports/cbramod_symbol_final.aggregate.json) に詳細があります．

| CBraMod＋Smol の入力条件 | Forced-choice BA |
| --- | ---: |
| EEG＋粗い環境 | 35.09% |
| EEG を除く入力 ablation | 33.33% |
| 環境を除く入力 ablation | 31.48% |
| 同じ記録内の EEG shuffle | 30.05% |
| 原記録の 10 秒前の EEG | 34.80% |

AI 送信対その他の fused AUROC は 0.6053 です．fused minus shuffle の差は +0.1609［参加者 95% CI：+0.1004，+0.2548］，fused minus −10 秒は +0.1293［+0.0521，+0.2227］で，この探索的標本内には EEG 整列に関する信号があります．一方，fused minus context-only は +0.0413［−0.0399，+0.1158］で，EEG 追加の優位性は確立していません．閾値 0.5 では AI 10／39 を検出し，その他 164 件中36件を誤検出しました．固定 confidence 0.8 の採用率は47.29%ですが，採用された96件の正解率は43.75%にとどまり，confidence を意図の正解確率として使えません．[19条件の記録済み予測比較](benchmark/CBRAMOD_UNIBA_V3_RECORDED_COMPARISON.md) を参照してください．

L4 の fused 推論 p95 は46.92 ms，最大CUDA予約は3.47 GBでした．これはCPU EEG encoder・転送・投影・LLM scoringの範囲で，読み込み・resampling・tokenization・harnessを含む端から端までの遅延ではありません．旧結果を閲覧した後の探索実験であり，encoder，出力方式，入力長，正規化，学習予算，実機を同時に変えているため，差を個別要素の効果とは同定しません．

**87.91 時間を全てこの LLM 接続部へ学習させたわけではありません．** UniBa v3 の train は 400 窓で，元構成では 700 秒＝11 分40秒，新 CBraMod 構成では末尾 1 秒ずつの計 400 秒＝6 分40秒を使用しています．全窓を原記録 SHA ごとに照合した時間区間の和集合も同じで，窓の重なりは 0 でした．新構成で 30 epochs 反復した 12,000 秒は，新しい EEG の追加時間ではありません．[時間区間の独立監査](reports/selected_eeg_interval_audit_20261122.json) に記録しました．

### 連続EEGによる追加学習の入力

encoderの自己教師学習に使う連続EEGを，ITU全量とUniBa validation／test・時計不合格の原記録を除外して準備しました．75記録50.888960時間の候補に，事前固定した30秒窓と品質検査を適用した実測です．

| 範囲 | 窓数 | 原波形の採用区間の合計 |
| --- | ---: | ---: |
| 品質検査前の30秒窓 | 6,040 | 50.333333 h |
| 品質検査通過・学習用 | 1,678 | 13.983333 h |
| 品質検査通過・SSLモデル選択用 | 522 | 4.350000 h |
| 品質検査通過・合計 | 2,200 | 18.333333 h |

166配列，1,685,013,248Bを生成し，全421,248,000値の独立検査が通過しました．学習用とモデル選択用は参加者groupを分けています．品質基準はresampling後の最大絶対振幅100µV以下，各channelの標準偏差0.01µV以上，finiteです．元の50.889時間との差には端の除外・不完全窓も含みます．

**UniBaは1,902候補から8窓・240秒しか残らず，採用データの大半はportable EEGです．** このdomainの偏りを保持した実験として評価し，歩留まりを上げるために品質基準や単位を後から変えていません．[固定前処理](model/SSL_PREPROCESSING.md)，[全75原記録の品質検査と集計](reports/ssl_corpus_preparation_20261122_v1/README.md)に根拠を保存しています．2026-11-22の入力検証終了時点では，最終protocolの封印まで完了しました．この表は準備済みの量であり，実学習の完了時間・成績ではありません．学習後にはモデルが実際に見たIDと原記録区間から実量を報告します．

## ラベリングと拡張可能性

元データの gold，観測された操作イベント，AI の弱教師ラベルを別々に保持します．workstation の Ollama qwen3.5:2b で train の 15 種類のイベント表記を正規化し，225 窓へ出典付きで伝播しました．teacher digest／prompt hash／元ラベルを保存し，held-out のラベル作成には流用していません．これは新しい EEG や潜在意図の正解を増やしたという意味ではありません．

UniBa は 34 記録中 32 記録，28.58953451 時間が時刻同期検査に通過しました．ITU は 26 記録中 25 記録，20.51673720 時間が同じ基準に通過し，外部 test 専用の 630 窓を抽出しました．AI 送信 126／keyboard 252／click 252，train・validation は 0 です．原波形 630 窓の全サンプル，実ログラベル 630 件，過去の履歴表現 1,260 件，source members 204 件を独立に再照合し，49 files を workstation へ hash 一致で回収しました．[ITU 外部 cohort の検証結果](reports/itu_external_cohort_v3_20261122_v1/README.md) を保存しています．サイト名を含む ID は分かれますが，UniBa と ITU 間の同一人物不存在を証明したものではありません．外部サイトが 1 つのため，サイト間一般化の信頼区間も求めません．

### 別施設 ITU での完了した評価

固定した3モデルの5条件と，学習済み履歴4モデルの計19条件を，各630例で評価・回収・独立再検証しました．ITUでの追加学習，閾値の調整，保存済み履歴モデルの再fitはありません．

| 固定したモデルの fused 条件 | 3-class BA | AI送信 AUROC［参加者95%CI］ | 固定0.5で検出したAI送信 |
| --- | ---: | ---: | ---: |
| Smol1.7B | 34.52% | 0.5594［0.5143，0.6071］ | 0／126 |
| Qwen8B | 36.11% | 0.5342［0.4706，0.5995］ | 0／126 |
| CBraMod＋Smol1.7B | 34.13% | 0.5159［0.4592，0.5721］ | 7／126 |

各モデルの fused minus context-only／shuffle／−10秒という計9比較のAI AUROC差は，全て95%区間に0を含みました．UniBaで見えたCBraModのEEG整列の信号は，この外部評価で再現を確認できませんでした．EEGを使わない履歴対照のAUROCは，EEG窓開始前の履歴で0.7239，判断直前までの履歴で0.8352でした．これらを超えるEEGの実用的な追加効果は未実証です．[外部評価の全結果と限界](benchmark/ITU_EXTERNAL_THREE_MODEL_RESULTS.md)，[全19条件の独立完了監査](benchmark/itu_external_completion_audit.json) に記録しています．

先行する互換性確認では，公式 CBraMod の source／weights を固定し，M1 の CPU 上で実 train EEG 3 窓を通しました．約 19.8 MB の weights，2 CPU threads の 3 例 batch が 0.033 秒で完了しています．この数値は encoder 単体の資源確認で，LLM と結合した推論時間や解読性能は表していません．[単体実測 JSON](reports/cbramod_cpu_capacity_probe.json) と，その後に実行した上記結合学習を区別してください．

## オンライン利用と実機の条件

AI-assisted coding の公開 EEG は offline filter 済みです．公開 export recipe を再構成すると未来側の support は条件付きで 4.125 秒です．窓終端からイベントまで 4.5 秒＋時計誤差を確保しても，feature の可用時点からイベントまでは最低 0.375 秒です．配布 binary と recipe の同一性，元 XDF，online causality は未確認です．M1 の validation candidate scoring は約 1 秒であり，この名目 lead を満たすオンライン動作は実証できていません．

Colab は workstation の既存 CLI を経由しています．SFC VPN は接続でき，smac19 は DNS／SSH 到達まで確認しましたが，smac19／smac31 の SSH 認証は未成立です．今回 Mac cluster への計算投入はできていません．学習・推論が実証できた実機は workstation M1 と Colab L4 です．

workstation の取得は空き容量 35 GiB を残して進めます．Colab からの大きな保全コピーは，CLI の一括メモリ使用を避けるため，64 MiB ごとの SHA 検証・再開可能な転送にしました．保全コピーは新しい取得データ量へ加算しません．Dataverse の必須 Guestbook 応答がある 342 files は `access_action_required` として保持しています．

## 再現と試用

リポジトリのルートで Python 3.12 を用います．実験用依存は core runtime の標準依存と分離しています．

```bash
uv venv --python 3.12 .venv-xavier
uv pip install --python .venv-xavier/bin/python -e '.[server,dev]'
uv pip install --python .venv-xavier/bin/python -r experiments/pj_xavier/model/requirements.txt
```

### CBraMod＋Smol の実機試用

新構成も，Colabで選択した同じcheckpointをM1へ読み込み，実EEG→モデル→NeuroCore routerまで3例で実行確認しました．trainから事前に固定した各class1例であり，この3例の一致を汎化精度とは扱いません．routerは全件を `proxy_label_has_no_explicit_intent_mapping` として保留しました．現在の出力は「AIへ送信／keyboard／click」という観測操作の分類で，何を入力し，どの画面で何を達成するかという実行可能な意図までは含みません．

正 worktree のルートで，未使用の出力先を指定すると準備と試用を一度に実行できます．この例は実施済みの試用先を上書きしません．モデル本体・source・全32配列・評価・回収証拠を実行前後に照合します．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/try_cbramod_recorded_eeg.py \
  --contract private/xavier/demos/cbramod_train_demo_contract_v1.json \
  --expected-contract-sha256 a55521c8360bfe7b19dfc8b1b7a58b93f07e22ca790df9284e576ac013de9490 \
  --output-dir private/xavier/demos/cbramod_train_trial_user_01
```

既に実行したtrialは `private/xavier/demos/cbramod_uniba_v3_train_examples_v1/summary.json` にあります．上のユーザー向け新しい出力先では，こちらから重複実行していません．[固定した入力と実行方法](integration/README_TRY_CBRAMOD_RECORDED_EEG.md) を参照してください．

### 元の CNN＋Smol 構成の試用

**新しい UniBa v3 checkpoint の一括試用を M1 上で実行確認しました．** 学習集合から固定規則で各 class 1 例を選び，実 EEG→学習済みモデル→NeuroCore router→dry-run sink を通しました．3 例とも confidence 0.8 未満で棄権し，router が保留しました．OS 操作・cloud agent 呼出は 0 件です．全 32 元配列と checkpoint は実行前後で hash が一致しました．[試用結果](reports/recorded_eeg_demo_20261122_v1/summary.json) と [詳細な実行手順](integration/README_TRY_RECORDED_EEG.md) を保存しています．

workstation の正 worktree で，未使用の出力名を指定して実行できます．このコマンドの 3 例は train からの試用で，汎化精度は上の held-out 評価を参照してください．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/try_recorded_eeg.py \
  --output-dir private/xavier/demos/m1_natural_v3_train_examples_local_01
```

以下は実行確認済みの，保存した EEGlass train 3 窓を使う推論例です．精度評価とは別の runtime probe です．

```bash
.venv-xavier/bin/python experiments/pj_xavier/model/eeg_llm.py predict \
  --checkpoint private/xavier/colab_snapshot/runs/smollm2_1_7b_eeglass/best_pinned.pt \
  --manifest private/xavier/local_runtime_probe_manifest.jsonl \
  --output private/xavier/local_m1_predictions.jsonl \
  --device mps --memory-budget-gib 8 --candidate-batch-size 1 \
  --dtype bfloat16 --max-examples 3
```

NeuroCore の実 router と dry-run sink への接続も実行確認済みです．不明な keyboard 文字や scroll 方向を補完せず，明示した mouse click proxy だけを `select` に対応させます．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/replay_to_neurocore.py \
  --predictions private/xavier/local_m1_predictions.jsonl \
  --output private/xavier/neurocore_dry_run.jsonl \
  --allow-mouse-click-proxy --human-armed
```

## 構成

| ディレクトリ | 用途 |
| --- | --- |
| `acquisition/` | 取得先解決，公開データの再開可能な取得，checksum，容量・時間監査 |
| `model/` | EEG prefix＋frozen LLM，前処理，学習・推論，公式 EEG encoder の容量 probe |
| `benchmark/` | split・時刻・波形の漏洩検査，負の対照，history baseline，実機・ハーネスの判定 |
| `labeling/` | train のみを使う AI 補助ラベル，teacher／出典／元ラベルの保持 |
| `fleet/` | 認証済み node の preflight，容量を考慮した割当 |
| `baselines/` | EEG spectrum／環境／結合の CPU baseline |
| `controller/` | ジョブ監視，raw と成果物の検証付き回収，終了条件 |
| `integration/` | モデル出力と NeuroCore router の dry-run |
| `reports/` | 実測値，失敗・未達事項，成果物 hash |

生 EEG，checkpoint，詳細ログ，イベント，ジョブ台帳は git 除外の `private/xavier/` に保持します．継続予約と実行中ジョブは [HANDOFF.md](HANDOFF.md) に記録しています．

## 一次資料

- [EEG-dataset 目録](https://github.com/AoyamaLab/EEG-dataset)
- [EEGlass 固定 release](https://zenodo.org/records/4453396)
- [AI-assisted coding 固定 release](https://zenodo.org/records/20801101)
- [Portable EEG systems comparison 固定 release](https://zenodo.org/records/15698441)
- [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [CBraMod 公式実装](https://github.com/wjq-learning/CBraMod)
- [UniMind 公式実装](https://github.com/kaleidoyao/neuroverse)
