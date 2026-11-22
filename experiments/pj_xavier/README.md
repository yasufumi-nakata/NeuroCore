# PJ-Xavier：EEG と LLM の学習・評価

NeuroCore の EEG デコーダと LLM 接続部を開発する実験コードです．目標は，判断以前の環境情報と EEG から操作を推定し，実ハーネスでタスク完了まで評価することです．現在の実装では，学習可能な EEG encoder／prefix を既存 LLM の `inputs_embeds` に渡し，出力トークンの損失から逆伝播します．

**実 EEG による 1.7B・8B モデルの学習，checkpoint 保存，held-out 評価，workstation での推論，NeuroCore router への dry-run を実行しました．目的の自律エージェント能力と，負対照まで含めた安定したEEGの追加寄与は，まだ実証できていません．** 最初の基盤を追加した [PR #70](https://github.com/yasufumi-nakata/NeuroCore/pull/70) はマージ済みで，追加監査・評価・修正は [PR #71](https://github.com/yasufumi-nakata/NeuroCore/pull/71) で継続しています．最新の実行場所と再開手順は [HANDOFF.md](HANDOFF.md) を参照してください．

## EEGMMIDB：学習・評価完了と追加学習

2026-11-22 14:30 JST．CSP＋LDA，scratch EEG encoder，EEG＋凍結SmolLM2の学習と固定test評価，負対照，T0診断，合成環境ハーネス，全成果物回収，M1試用を完了しました．主左右想起の被験者平均balanced accuracyは **59.92%／59.09%／57.44%** で，この条件ではLLM融合による上積みは確認できませんでした．[全結果と区間推定](eegmmidb_phase2/RESULTS.md)を保存しています．

train6,359／val1,360／test1,489の分割を固定し，同じ被験者が分割をまたがないようにしています．固有train入力は3秒×6,359＝5.299167時間です．モデルを固定してからtestを評価し，rootが保存済み予測から3モデル全5条件の主要点指標を独立に再計算して一致を確認しました．bootstrap区間は参加者単位の探索的な不確実性で，学習seedや多重比較まで含みません．

fusedの主課題採用率は固定0.8で14.11%，採用内正解率63.21%，T0二択誤採用23.80%です．合成環境の主episode成功7.99%と，正解classを渡したrouting上限62.5%を分けて保存しました．M1のtrain3例は20.53秒で12fresh routingを実行できましたが，全件棄権のため操作成功は0でした．[実機試用手順](integration/README_TRY_EEGMMIDB_RECORDED.md)を参照してください．

さらに，phase2 test閲覧前に固定したphase3で，**29.8875時間の連続train EEG**を使うmasked SSL→同じMI教師あり学習→同じSmol接続を実行中です．SSLとMIの準備済み区間unionは29.985889時間で，最終実学習量は選択checkpointまでの実seen-IDから別途監査します．既存test17群は学習へ入れません．[準備済み全入力](reports/eegmmidb_phase3_preparation_20261122_v1/README.md)と[固定科学設定](eegmmidb_phase3/protocol.scientific.v1.json)を公開しています．

## データ量：候補・取得済み・検証済みを区別

2026-11-22 日本時間の観測です．GB／TB は十進，GiB は二進です．

| 範囲 | 確認した量 | 意味 |
| --- | ---: | --- |
| EEG-dataset の目録 | 5,165 行 | 全 ID と取得先解決結果を照合し，不足・余剰 0 |
| workstation の取得キュー | 783 releases，650,483 file jobs | EEG を含む候補の release 全体．認証・規約操作・metadata の保留を含む |
| 上記キューの既知の容量 | 24,172,262,716,513 B，約 24.17 TB | 動画・補助データ・版の重複を含む候補容量．42,398 files はサイズ未確定 |
| 取得完了し SHA-256 の検証記録がある内容 | 175 files，44,432,880,979 B，約44.43GB | 14:17 JSTのsnapshot005．同一内容は重複加算せず，P300は取得途中の66filesを含む |
| 元EEGのheader／binaryを数量検証した範囲 | 2,405記録，210.50324878時間 | 記録長の合計．channel-hoursではない |
| binary EEG scalarの保存容量 | 43,382,368,656 B，約43.38GB | 補助channel・動画・header・窓配列・重みを除く．校正の正しさを保証する名称ではない |
| UTF-8のEEG数値tokenの保存容量 | 31,065,967 B | 元TXTの数値表現．binary値のbytesとは区別 |

取得の根拠は[acquired_progress_005.json](reports/acquired_progress_005.json)，数量は[verified_eeg_quantities_20261122_005.json](reports/verified_eeg_quantities_20261122_005.json)です．既存部分は過去の全量SHAと保全記録を照合し，新規compactデータとBCI archiveは全ファイル／memberを新たに検証しました．約44GB全てをこのsnapshotで再hashしたという意味ではありません．partialは完了容量やEEG時間に加算しません．P300の未完release，新規MI releaseは完了・header監査後のsnapshotへ反映します．ZIPのbytesからEEG時間を一律換算しません．

前段の468,040 file jobsは固定規則で構造監査済みです．これは新たに加えた182,443件を含む全650,483件の構造監査ではありません．そのうち EEG と明示された path の信号形式は 62,303 files／7,209,327,960,212 B，135 releases でした．これは header を検証した純 EEG 量ではなく，取得順序を検討するための候補量です．未展開 archive，EEG を含み得る MEG container，動画・視線等の companion は別々に残し，EEG の有無を確かめる前に一律除外していません．詳細と全 609 releases の分類は [全キュー構造監査](reports/queue_structure_20261122_v1/README.md) にあります．

| 元データ | 記録数 | EEG 時間 | 物理 EEG チャンネル |
| --- | ---: | ---: | ---: |
| EEGlass | 1 | 1.08882222 h | 3 |
| AI-assisted coding：UniBa | 34 | 30.61540256 h | 8 |
| AI-assisted coding：ITU | 26 | 21.30617947 h | 64 |
| Portable EEG systems comparison | 57 | 34.89706111 h | 32 |
| EEGMMIDB：運動実行・運動想起 | 1,526 | 48.53444444 h | 64 |
| Kinesthetic force MI | 60 | 0.70076172 h | 14 |
| Right hand／feet MI | 8 | 1.10371962 h | 16 |
| BCI users profile MI | 693 | 72.25685764 h | 27 |
| 合計 | 2,405 | 210.50324878 h | 記録ごとに異なる |

新規EEGMMIDBは1,997,435,063BのZIPを477.51秒で取得し，公式全3,058 member＋チェックサム一覧，1,526 EDFを23.46秒で監査しました．実headerは160Hzが1,490記録，128Hzが36記録です．注釈区間の5〜50msの重なり・25msのはみ出しやbaselineのT1を含む50記録を，ラベルreview対象としました．[全取得・数量・注釈監査](reports/eegmmidb_acquisition_20261122_v1/README.md)と，[9代表記録・209注釈のMNE独立照合](reports/eegmmidb_mne_representative_audit_20261122_v1/README.md)を参照してください．この追加データは前段のUniBa／ITUモデルを変更せず，独立したphase2のデータ準備・CSP学習に使いました．初期案の[EEGMMIDB benchmark設計](benchmark/NEXT_EEGMMIDB_BENCHMARK.md)から具体化した新protocolをeegmmidb_phase2に保存しています．

raw と filtered export，同じ archive の保全コピー，原波形から切り出した学習窓を新しい独立記録として数えません．記録時間の検証は，信号品質・意図の正解・オンライン処理の因果性の証明とは別です．異なる corpus 間の同一参加者照合も未実施です．

Colab の優先取得は EEGlass と AI-assisted coding の全 23 files／11,221,153,522 B が完了しています．AI-assisted coding は EEG に加えて gaze，screen，behavioral，補助情報も取得しています．モデルへ実際に入力しているのは EEG と判定前の page／AI availability です．画面動画を vision encoder で学習した実験ではありません．

## 実際に学習したモデル

LLMとの接続時は本体を凍結し，EEG接続部を学習しました．別の自己教師学習段階ではEEG encoderを更新しています．基盤重みは immutable revision で固定しています．時間は原則として学習・検証 loop の範囲で，ダウンロードや環境構築を含みません．

| モデル／cohort | 実機 | 完了した学習 | 実測時間 | メモリの観測 |
| --- | --- | ---: | ---: | --- |
| SmolLM2-1.7B／EEGlass | Colab L4 | 3 epochs，171 steps | 35.01 秒 | CUDA 最大予約 3.90 GB |
| Qwen3-8B／EEGlass | Colab L4 | 3 epochs，675 steps | 227.35 秒 | CUDA 最大予約 16.82 GB |
| Qwen3-8B／UniBa v2 | Colab L4 | 3 epochs，1,635 steps | 再開後 1,332.96 秒 | CUDA 最大予約 20.63 GB |
| SmolLM2-1.7B／UniBa v3 | M1／16 GiB | 3 epochs，600 steps | 1,541.72 秒 | MPS driver の観測最大 4.44 GB，allocator quota 8 GiB |
| Qwen3-8B／UniBa v3 | Colab L4 | 30 epochs，3,000 steps | 1,340.03 秒 | CUDA 最大予約 18.49 GB |
| CBraMod＋SmolLM2-1.7B／UniBa v3 | Colab L4 | 30 epochs，390 steps | 429.33 秒 | CUDA 最大予約 9.22 GB |
| CBraModの連続EEG追加事前学習 | Colab L4 | 10 epochs，280 steps | 169.79 秒 | CUDA 最大予約 7.84 GB |
| 追加学習CBraMod＋SmolLM2-1.7B／同じUniBa v3 | Colab L4 | 30 epochs，390 steps | 432.71 秒 | CUDA 最大予約 9.22 GB |

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

**前段で取得していた136.44時間を全てこのLLM接続部へ学習させたわけではありません．** UniBa v3 の train は 400 窓で，元構成では 700 秒＝11 分40秒，新 CBraMod 構成では末尾 1 秒ずつの計 400 秒＝6 分40秒を使用しています．全窓を原記録 SHA ごとに照合した時間区間の和集合も同じで，窓の重なりは 0 でした．新構成で 30 epochs 反復した 12,000 秒は，新しい EEG の追加時間ではありません．[時間区間の独立監査](reports/selected_eeg_interval_audit_20261122.json) に記録しました．

### 連続EEGによる追加学習の入力

encoderの自己教師学習に使う連続EEGを，ITU全量とUniBa validation／test・時計不合格の原記録を除外して準備しました．75記録50.888960時間の候補に，事前固定した30秒窓と品質検査を適用した実測です．

| 範囲 | 窓数 | 原波形の採用区間の合計 |
| --- | ---: | ---: |
| 品質検査前の30秒窓 | 6,040 | 50.333333 h |
| 品質検査通過・学習用 | 1,678 | 13.983333 h |
| 品質検査通過・SSLモデル選択用 | 522 | 4.350000 h |
| 品質検査通過・合計 | 2,200 | 18.333333 h |

166配列，1,685,013,248Bを生成し，全421,248,000値の独立検査が通過しました．学習用とモデル選択用は参加者groupを分けています．品質基準はresampling後の最大絶対振幅100µV以下，各channelの標準偏差0.01µV以上，finiteです．元の50.889時間との差には端の除外・不完全窓も含みます．

**UniBaは1,902候補から8窓・240秒しか残らず，採用データの大半はportable EEGです．** このdomainの偏りを保持した実験として評価し，歩留まりを上げるために品質基準や単位を後から変えていません．[固定前処理](model/SSL_PREPROCESSING.md)，[全75原記録の品質検査と集計](reports/ssl_corpus_preparation_20261122_v1/README.md)に根拠を保存しています．2026-11-22 12:05 JST時点で，このうち学習用1,678窓の全てを使う追加事前学習が完了しました．10 epochs・280 steps，学習ループと検証169.79秒，CUDA予約の実測最大7,841,251,328Bです．学習用float32値は1周あたり1,284,096,000Bで，反復提示を新しいEEG量には加えません．採用重みは固定validationのMSEが学習後候補で最小だったepoch 8で，固有の入力13.983333時間，採用時までの反復提示111.866667時間です．ただし，採用時MSE0.001495は更新前の0.001183より高く，復元誤差が改善したとは言えません．[実測集計](reports/cbramod_ssl_pretraining.aggregate.json)に全epochと選択根拠があります．LLMとの同条件alignmentも30epochs・390steps・432.709秒で完走し，UniBaの5条件評価を終えました．fusedの3-class balanced accuracyは34.20%で，更新前35.09%を上回りませんでした．別施設ITUの評価，全成果物509件の回収，M1での試用まで完了しました．SSL50,340秒とalignment末尾400秒の原記録区間を照合し，重複3秒を除いた実学習入力は50,737秒＝14.093611時間です．[完了した学習・実機検証](model/CBRAMOD_SSL_COMPLETED_RESULTS.md)と[完了時の入力区間集計](reports/ssl_alignment_completed_train_interval_union.json)に根拠を保存しています．この時間は取得原記録136.44時間やepoch反復提示とは異なります．

UniBaのBIDS V表記とBrainVision headerの0.1µV表記は，公開されたMNE-BIDS／pybvの通常の保存処理でも生じる組み合わせです．表記差だけで尺度の誤りとは判断できません．元のXDFと機器の校正情報がないため，入力段階の単位と極端な公称振幅の由来は未確認です．[出力コードまで追跡した追加監査](reports/uniba_amplitude_provenance_20261122_v1/README.md)に，確認できた変換と残る不足を記載しています．

### 公開課題に基づく独立ハーネス

公開されたConnection課題の説明・UML・APIの形から，新しい8ケースの実行テストを構成しました．既存JDKで正解fixture8/8，優先順位の誤り7/8，常時成功2/8，偶発的なNPE0/8，構文エラーはcompile failureを確認し，失敗case集合も照合しています．[ハーネスと実行方法](benchmark/connection_harness/README.md)，[元資料の不足と将来の実課題評価の条件](benchmark/CODING_HARNESS_SOURCE_AUDIT.md)を保存しています．これは手製fixtureによる工学検証であり，EEGモデルの課題達成率ではありません．元データには判断前の正確な編集状態と実行可能な元oracleが揃わないため，遡及的な自律操作成功は採点できません．

## ラベリングと拡張可能性

元データの gold，観測された操作イベント，AI の弱教師ラベルを別々に保持します．workstation の Ollama qwen3.5:2b で train の 15 種類のイベント表記を正規化し，225 窓へ出典付きで伝播しました．teacher digest／prompt hash／元ラベルを保存し，held-out のラベル作成には流用していません．これは新しい EEG や潜在意図の正解を増やしたという意味ではありません．

UniBa は 34 記録中 32 記録，28.58953451 時間が時刻同期検査に通過しました．ITU は 26 記録中 25 記録，20.51673720 時間が同じ基準に通過し，外部 test 専用の 630 窓を抽出しました．AI 送信 126／keyboard 252／click 252，train・validation は 0 です．原波形 630 窓の全サンプル，実ログラベル 630 件，過去の履歴表現 1,260 件，source members 204 件を独立に再照合し，49 files を workstation へ hash 一致で回収しました．[ITU 外部 cohort の検証結果](reports/itu_external_cohort_v3_20261122_v1/README.md) を保存しています．サイト名を含む ID は分かれますが，UniBa と ITU 間の同一人物不存在を証明したものではありません．外部サイトが 1 つのため，サイト間一般化の信頼区間も求めません．

### 別施設 ITU での完了した評価

元3モデルの保存済み5条件へ新しいSSLモデルの5条件を追加し，履歴4モデルも含む計24条件・各630例を回収・独立再検証しました．元19条件の再推論は行っていません．ITUでの追加学習，閾値の調整，保存済み履歴モデルの再fitはありません．

| 固定したモデルの fused 条件 | 3-class BA | AI送信 AUROC［参加者95%CI］ | 固定0.5で検出したAI送信 |
| --- | ---: | ---: | ---: |
| Smol1.7B | 34.52% | 0.5594［0.5143，0.6071］ | 0／126 |
| Qwen8B | 36.11% | 0.5342［0.4706，0.5995］ | 0／126 |
| CBraMod＋Smol1.7B | 34.13% | 0.5159［0.4592，0.5721］ | 7／126 |
| 追加EEG学習CBraMod＋Smol1.7B | 38.10% | 0.5997［0.5449，0.6547］ | 38／126 |

新しいSSLモデルのAI AUROCは，旧CBraModとの差が+0.0838［対応付き95%CI +0.0056〜+0.1514］でした．一方，shuffleとの差+0.0868［−0.0010〜+0.1804］，−10秒との差+0.0381［−0.0091〜+0.0836］は0を含みます．UniBaでの追加SSL前後のAUROC差も+0.0098［−0.0503〜+0.0781］で，安定した利益はまだ確認できません．先行結果を見た後の探索比較であり，単一学習・多重比較未補正という範囲です．

新モデルの固定0.5判定は，AI送信38/126件の検出と非送信102/504件の誤検出で，この濃縮標本内の適合率は27.14%でした．3class最大確率0.8以上で採用した111/630件の正解は46/111＝41.44%です．EEGなしの操作履歴はAUROC0.7239〜0.8352で，新モデルを上回りました．AUROCは順位づけの指標で，作業の完了率や校正済みの意図確率ではありません．[両施設の24条件比較と限界](benchmark/SSL_FOURTH_MODEL_RESULTS.md)，[第4外部モデルを含む独立完了監査](benchmark/itu_external_ssl_completion_audit.json)を参照してください．旧3モデルの数値・報告も保全しています．

先行する互換性確認では，公式 CBraMod の source／weights を固定し，M1 の CPU 上で実 train EEG 3 窓を通しました．約 19.8 MB の weights，2 CPU threads の 3 例 batch が 0.033 秒で完了しています．この数値は encoder 単体の資源確認で，LLM と結合した推論時間や解読性能は表していません．[単体実測 JSON](reports/cbramod_cpu_capacity_probe.json) と，その後に実行した上記結合学習を区別してください．

## オンライン利用と実機の条件

AI-assisted coding の公開 EEG は offline filter 済みです．公開 export recipe を再構成すると未来側の support は条件付きで 4.125 秒です．窓終端からイベントまで 4.5 秒＋時計誤差を確保しても，feature の可用時点からイベントまでは最低 0.375 秒です．配布 binary と recipe の同一性，元 XDF，online causality は未確認です．M1 の validation candidate scoring は約 1 秒であり，この名目 lead を満たすオンライン動作は実証できていません．

Colabはworkstationの既存CLIを経由して使用し，全成果物回収後の12:26:40 JSTに所有sessionを正常停止しました．SFC VPN は接続でき，smac19 は DNS／SSH 到達まで確認しましたが，smac19／smac31 の SSH 認証は未成立です．11:45〜11:48 JSTのユーザー再承認後には実際のパスワード入力まで進めましたが，既存設定・過去の接続履歴のログイン指定とも認証は成立しませんでした．現在の障害は承認ではなく，SSHの認証情報またはサーバー側のアクセス設定です．今回 Mac cluster への計算投入はできていません．学習・推論が実証できた実機は workstation M1 と Colab L4 です．

workstation の取得は空き容量 35 GiB を残して進めます．Colab からの大きな保全コピーは，CLI の一括メモリ使用を避けるため，64 MiB ごとの SHA 検証・再開可能な転送にしました．保全コピーは新しい取得データ量へ加算しません．Dataverse の必須 Guestbook 応答がある 342 files は `access_action_required` として保持しています．

## 再現と試用

リポジトリのルートで Python 3.12 を用います．実験用依存は core runtime の標準依存と分離しています．

```bash
uv venv --python 3.12 .venv-xavier
uv pip install --python .venv-xavier/bin/python -e '.[server,dev]'
uv pip install --python .venv-xavier/bin/python -r experiments/pj_xavier/model/requirements.txt
```

### 追加EEG学習済みモデルの実機試用

追加学習したencoderとLLM接続部も，workstationのM1上で試用を完了しました．事前に固定した同じtrain3例を読み，予測は3例とも観測ラベルと一致しましたが，汎化性能の証拠にはしません．routerは具体的意図が未対応のため全件を保留し，OS操作・cloud agent呼出しは0です．MPS driverの観測最大は4.31GB，quota8GiBで動作しました．瞬間最大メモリやオンライン全工程の遅延は未確定です．

正worktreeのルートで，未使用の出力名を指定して実行できます．このユーザー向け出力先は未実行です．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/try_cbramod_ssl_recorded_eeg.py \
  --contract private/xavier/demos/cbramod_ssl_train_demo_contract_v1.json \
  --expected-contract-sha256 bde21873e6c0c2c173658eae47f4ce7f38d5be8575944dcf2ce7cf08365797cb \
  --output-dir private/xavier/demos/cbramod_ssl_train_trial_user_01
```

[実行方法・検証条件](integration/README_TRY_CBRAMOD_SSL_RECORDED_EEG.md)と[実試用の記録](reports/cbramod_ssl_recorded_eeg_demo_20261122_v1/README.md)を保存しています．モデル本体とEEGはworkstationのprivateディレクトリに保持しており，公開コードだけのcloneでは実験成果物が不足します．

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
