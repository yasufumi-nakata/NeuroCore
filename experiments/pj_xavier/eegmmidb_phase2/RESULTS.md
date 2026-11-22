# EEGMMIDB Phase 2：脳波デコーダと有界な操作環境

## 実測した結論

**3モデルの学習・固定test評価，休止診断，合成環境での状態更新が完了しました．** 公開cueの左右運動イメージ条件を分類する信号は，元波形を入れ替えた負対照との差として観測されました．今回の点推定では，LLM融合はCSP＋LDAやEEG専用モデルを上回っていません．固定0.8での採用率，休止時の誤採用，言語モデルのroutingにも残る問題があり，自発的な意図や実OSでの自律操作を検証したとは扱いません．

完全な集計，checkpoint SHA，区間推定，校正指標，条件別の分母，実メモリと出典は [protocol.measured_results.v1.json](protocol.measured_results.v1.json) に保存しています．集計SHAは `3ac844ba0c4c689b52464449b303007fe2fe089b2071ce15319c68ce122e035a` です．個別予測，脳波配列，学習済み重みはprivateに保持しています．

### 固定testでの分類

全testは17 participant群，左右想起751窓，両拳・両足想起738窓です．以下のBAはparticipantごとに算出したbalanced accuracyの平均です．括弧内はparticipant単位の1,000回bootstrapによる95%区間で，学習のやり直しによる不確実性や多重比較補正は含みません．

| モデル | 左右想起BA・主評価 | 両拳・両足想起BA・副評価 | 主評価AUROC |
| --- | ---: | ---: | ---: |
| CSP＋LDA | 59.92%（53.00–66.84） | 59.97%（53.27–67.40） | 0.6349 |
| EEG専用・scratch | 59.09%（53.52–65.73） | 65.10%（57.80–73.21） | 0.6313 |
| EEG＋凍結SmolLM2 1.7B | 57.44%（53.18–61.73） | 59.85%（52.97–67.38） | 0.5911 |

モデル間の優劣を確定する反復学習は実施していません．fusedモデルはscratchの選択済みencoderをさらに更新しており，追加学習予算も異なります．差をLLM本体の有無だけに帰属させません．

共通の負対照subsetは全体1,254窓，主左右課題635窓です．同じ635窓を用いたfused通常入力のsubject平均BAとの差は，EEGなしに対して **+7.45ポイント（2.75–12.30）**，同一記録内shuffleに対して **+9.90ポイント（3.23–16.63）**，実波形の10秒前への差替えに対して **+11.45ポイント（5.48–17.48）** でした．context-onlyの主BAは50%です．これは公開cue条件の分類にEEG入力が寄与するという範囲の結果です．

### 採用率と休止時の挙動

| モデル | 主課題の採用率（固定0.8） | 採用した主課題だけの正解率 | T0休止の二択行動への誤採用・左右 | T0誤採用・両拳／両足 |
| --- | ---: | ---: | ---: | ---: |
| CSP＋LDA | 5.86% | 72.73% | 5.37% | 5.39% |
| EEG専用・scratch | 5.99% | 77.78% | 6.10% | 15.57% |
| EEG＋Smol | 14.11% | 63.21% | 23.80% | 25.00% |

採用内の正解率は，全testに対する成功率ではありません．T0はtest-onlyの公開指示休止1,357窓（左右689，両拳／足668）です．自然な連続無操作のground truthや誤操作毎分は算出していません．閾値は結果に合わせて変更していません．

### 操作環境を最後まで実行した結果

凍結Smolで16種類のrouting promptを実推論し，同一prompt cacheを使って **104,388件の合成episode** の状態更新とoracle照合を実行しました．各EEG記録を4mappingに展開した数であり，EEGサンプル数や新しいLLM呼出し回数ではありません．

fusedモデルの全episode成功率は，左右 **7.99%**，両拳／足 **14.09%** でした．棄権も分母に含みます．正解controlをrouting入力に与える上限比較でも，左右のclass-balanced成功率は **62.5%**，両拳／足は **100%** でした．この上限比較はデコーダの性能ではなく，固定promptとSmolのroutingが達成した値です．デコーダの誤りとroutingの誤りを分けて保存し，routingを規則で完全に行うreference oracleも別集計しています．実OS操作は0件です．

uncached routingの実測wall latencyは中央値64.48ms，p95 105.32msです．保存予測からの全モデルの合成環境CPU集計は4.15秒でした．

## 学習とデータの内訳

実データ準備，評価実装の25試験，実入力の独立照合を完了し，CSP＋LDA／scratch／fusedのtrain・validation選択をすべて固定してからtestを一度ずつ評価しました．以前のUniBa／ITU／SSL実験の値は転用していません．

準備済みの主入力はtrain 6,359窓，validation 1,360窓，test 1,489窓です．各窓3秒で，trainへ渡す固有窓は19,077秒，約5.2992時間です．これは因果filterが保持する記録先頭からの過去状態や全原記録時間とは異なる量です．共通の負対照比較はtest 1,254窓，別契約のtest-only T0診断は1,357窓です．全主testで同一記録内shuffleが成立しています．

CSP＋LDAはCPU2threads，3.70秒でtrain／validationのみを完了しました．以下は**validationでの選択値であり，test精度ではありません**．

| 課題 | train窓 | validation窓 | 選択components | validation participant平均BA |
| --- | ---: | ---: | ---: | ---: |
| 左手・右手の想起 | 3,182 | 679 | 16 | 63.73% |
| 両拳・両足の想起 | 3,177 | 681 | 4 | 57.53% |

元のsubject hash split，main/control/idle全配列のSHA，因果的な入力時刻，元波形のSHA，別ledgerを独立に照合済みです．`protocol.csp_selection.v1.json` に選択済み重みのSHA，`protocol.implementation_verification.v1.json` にfixture試験と実入力照合の出典を保存しています．

scratchは60epoch，1,500optimizer steps，学習とvalidationのwall time304.28秒で完走しました．fusedは元のfit cutoffに従って正常なpartial exit 2で停止し，完了済み17epochと続く61steps，計1,761steps，1,394.16秒です．30epoch完走とは表示しません．両モデルともtrain 6,359窓すべてを観測しました．繰り返し提示した時間と固有脳波時間は分け，両段階を足して固有窓の量を二重計上しません．

Colabの2モデルの独立評価workflowは203.62秒で完了し，元の共有評価上限900秒内です．推論中のL4 CUDA reserved peakはscratch **1.35GiB**，fused **3.58GiB** で，上限21GiBを満たしました．主課題のmodel-call amortized p95はscratch1.81ms／例，fused4.42ms／例です．配列I/O，計測前の転送，prompt encoding，因果stream準備，harnessは含まないため，オンライン端末間の応答遅延とは異なります．

モデル・optimizer・RNG・全checkpoint・学習／validation履歴・予測・成功／失敗capacity probeを294member，展開135,633,776Bとして回収し，全SHAを再照合しました．private recovery receiptのSHAは `81d7814c76c4d22da76ec5e67460bc97376109720e69a5f9c9e02b79bf6a788e` です．全phase2所有processの終了を確認してから，同じ物理sessionを別に登録したphase3予算へ明示引継ぎしました．phase2終了時点でsessionを停止したという記録にはしていません．

## Macでの記録済みEEG試用

回収したfused bestをM1／16GiB RAMで読み，metadataのID hash順だけで事前選択したtrain3例と4mappingずつを実行しました．**20.53秒，exit 0，12回のfresh LLM routing，cache再利用0** で完了しました．3例すべてが固定0.8未満だったため，12episodeとも棄権し，選択操作と成功episodeは0件です．これは実機での推論経路と棄権処理が動いた証拠であり，操作成功やheld-out性能の証拠ではありません．実OS操作・cloud API呼出し・追加test forwardはすべて0です．

PyTorch MPS上限は8GiB，観測時点のallocatedは3.19GiB，driverは4.07GiBでした．MPS値はsampleであり，真のpeakを測った値ではありません．実測の出典は [protocol.M1_trial_measured.v1.json](protocol.M1_trial_measured.v1.json)，確定済み契約SHA・実行パス・再試用方法は [Mac試用README](../integration/README_TRY_EEGMMIDB_RECORDED.md) です．

## 実機互換性の修正

最初のCUDA容量probeは，Torch 2.14のallocator APIがindex無しのdeviceを受け付けず，model構築前，optimizer更新0，test推論0で停止しました．device indexを明示する最小修正とregression試験の後，benchmark execution sealをv2へ新規発行しました．全データ，解析計画，閾値，CSP科学sourceと選択済み重みは同一です．CSPは再学習していません．元のsource／契約を保持し，差分と互換性は `protocol.execution_amendment.v2.json` に記録しています．

## 固定した問いと評価

主課題は公開された左手・右手の運動イメージ条件の分類，副課題は両拳・両足の運動イメージ条件の分類です．主指標は左右課題の participant ごとの balanced accuracy の平均です．CSP＋LDA，scratch EEG encoder，同 encoder と prefix projection を学習して SmolLM2 1.7B 本体を凍結するモデルを，同じ participant 分割で比較します．checkpoint 選択は validation のみ，test は各凍結モデルで一度だけです． CSPのfilterと線形判別器，scratchのEEG encoderと二つの分類head，fusedのEEG encoderとprefix projectorが学習対象です．Smol本体は更新しません．fusedはscratchの最良encoderから追加学習するため，scratchとの性能差をLLMの追加だけの効果と断定せず，学習予算とencoderの追加更新も併記します．

同一 checkpoint の5条件は，通常入力，EEGを除いた入力，静的課題説明を除いた入力，同一記録内のEEG入替え，元記録の実際の10秒前のEEGです．CSP と scratch では通常入力と EEG-only が同じ task-specific head を使うため，同値になり得ます．Smol の EEG-only は課題説明を汎用 A/B prompt に置き換えます．context-only は同じモデルへの zero-EEG ablation で，独立に学習した context-only baseline ではありません．CSP の context-only は等しい事前確率です．

主評価は全 test 窓を保持します．10秒前のデータが存在し品質基準も満たす，事前固定した共通ID集合で対応のある負対照比較を行います．比較条件ごとに分母を変えません．participant 単位の bootstrap，macro F1，AUROC，Brier，NLL，ECE，固定0.8の採用率，処理時間，実メモリを記録します．二つの logit に限定した softmax の0.8は校正済みの「意図の確信度」ではありません．

## 因果性と前処理の範囲

全64 channel，物理単位V，200Hzの3秒窓を用います．元160／128Hzの各記録の先頭から，因果的な3–40Hz SOS filter と遅延を戻さない FIR resampling を適用します．cue の1秒後から始まる600試料を取り，実際に利用した最後の試料が判定時刻を越えないことを検証します．annotation review 対象の記録は保留し，自動修復しません．

100µVの候補基準では train の72群中43群だけが残ることが train-only 診断で分かり，validation／test の信号処理前に500µVの粗大振幅保留，channel標準偏差0.01µV以上，finite を最終基準としました．100µV超過の flag は保持し，単位の変更，振幅の再尺度化，clip は行いません．500µVは今回の探索的前処理選択であり，正常脳波の保証や標準的な artifact 除去基準ではありません．残留 artifact の影響を結果の限界として扱います． 公開cue条件の分類には，運動イメージに関連する成分のほか，cueへの反応や眼・筋由来成分が寄与し得ます．本実験だけでこれらの寄与を分離したり，自由な意図を読み取ったと判断したりしません．

## 操作ハーネスの意味

独立に生成する環境には候補ID X/Y，左右の位置，各候補に割り当てた運動イメージ control を提示します．元の正解 cue，participant，trial ID，記録順をLLM入力へ渡しません．各EEG窓に対して4種類の完全に均衡した mapping を用います．予測された control を凍結Smolへ渡し，返答を厳密な X/Y として解釈して，メモリ内の環境状態を実際に更新します．不正な返答と低確信度は状態を変えません．正解 control は独立の oracle と上限比較にだけ使います．

2課題×4mapping×2予測classの16種類の prompt を実際に greedy 生成し，完全一致する prompt の結果を明示的に cache します．全episodeではこの cache を用いた状態遷移を実行します．LLM推論16件と全episodeの replay を別々に数え，uncached推論時間とcached環境時間を分けます．4mappingは4件の独立EEG観測ではありません．decoderの誤りとroutingの誤りを別集計します．

この実験が測るのは，公開cueに応答した運動イメージから二択controlを選び，合成環境の候補へ対応付けることです．自由な意図の読取り，自発的な操作要求，実OSでの課題達成とは区別します．実OS操作はありません．

## 休止診断

同じMI記録のT0から一つの区間につき固定3秒窓を取り，test-onlyの別manifest／配列として扱います．学習，checkpoint選択，threshold調整には使いません．固定0.8で二択行動として採用される割合を測ります．公開cueによる休止であり，自然な連続無操作のground truthや誤操作毎分は推定しません．

## 一次資料と再開インターフェース

課題・T0/T1/T2の意味は [PhysioNet公式説明](https://physionet.org/content/eegmmidb/1.0.0/) と [MNEのrun対応表](https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html) を確認しました．CSP＋LDAの構成は [MNE公式例](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html) を参照していますが，その例の分割，前処理，精度は本実験と同一ではありません．[CBraMod原著](https://arxiv.org/html/2412.07236v6) ではTUEG事前学習とPhysioNet-MIのdownstream評価が記述されています．これはcheckpoint単位の全記録非重複を認証する証拠ではありません．今回の主比較はscratchから始めます．URLと限定した確認範囲は `protocol.primary_sources.json` に保存しています．

- `benchmark_seal.py`：新予算，元subject split，準備済み配列と因果性試験，コードをhashで固定します．
- `benchmark_csp.py`：train fitとvalidation選択のみを行います．
- `benchmark_evaluate.py`：同じ凍結test IDで5条件と独立T0診断を一度評価します．fused runだけ `--collect-routing` を指定して16 promptを生成します．
- `benchmark_audit.py`：保存済み予測のID，gold，subject，checkpoint，負対照の元波形を独立に照合します．
- `harness_evaluate.py`：3モデルの保存予測と16件のcacheから実状態遷移を行い，個別のprivate traceと集計を保存します．

Colabへはworkstation CLI経由でのみ接続し，新しい期限・最大5CU・GPU21GiB・workstation空き35GiBを守ります．以前のモデルや評価データは更新しません．
