# EEGMMIDB Phase 2：脳波デコーダと有界な操作環境

## 実施状況

実データ準備，評価実装の25試験，実入力の独立照合，CSP＋LDAのtrain／validation学習が完了しました．neuralモデルは別の固定計画で準備中です．全モデルの学習・validation選択が固定するまで，test，休止診断，LLM routingの実測は開始しません．以前のUniBa／ITU／SSL実験の値はこの実験の結果に転用しません．

準備済みの主入力はtrain 6,359窓，validation 1,360窓，test 1,489窓です．各窓3秒で，trainへ渡す固有窓は19,077秒，約5.2992時間です．これは因果filterが保持する記録先頭からの過去状態や全原記録時間とは異なる量です．共通の負対照比較はtest 1,254窓，別契約のtest-only T0診断は1,357窓です．全主testで同一記録内shuffleが成立しています．

CSP＋LDAはCPU2threads，3.70秒でtrain／validationのみを完了しました．以下は**validationでの選択値であり，test精度ではありません**．

| 課題 | train窓 | validation窓 | 選択components | validation participant平均BA |
| --- | ---: | ---: | ---: | ---: |
| 左手・右手の想起 | 3,182 | 679 | 16 | 63.73% |
| 両拳・両足の想起 | 3,177 | 681 | 4 | 57.53% |

元のsubject hash split，main/control/idle全配列のSHA，因果的な入力時刻，元波形のSHA，別ledgerを独立に照合済みです．`protocol.csp_selection.v1.json` に選択済み重みのSHA，`protocol.implementation_verification.v1.json` にfixture試験と実入力照合の出典を保存しています．

## 固定した問いと評価

主課題は公開された左手・右手の運動イメージ条件の分類，副課題は両拳・両足の運動イメージ条件の分類です．主指標は左右課題の participant ごとの balanced accuracy の平均です．CSP＋LDA，scratch EEG encoder，同 encoder と prefix projection を学習して SmolLM2 1.7B 本体を凍結するモデルを，同じ participant 分割で比較します．checkpoint 選択は validation のみ，test は各凍結モデルで一度だけです． CSPのfilterと線形判別器，scratchのEEG encoderと二つの分類head，fusedのEEG encoderとprefix projectorが学習対象です．Smol本体は更新しません．fusedはscratchの最良encoderから追加学習するため，scratchとの性能差をLLMの追加だけの効果と断定せず，学習予算とencoderの追加更新も併記します．

同一 checkpoint の5条件は，通常入力，EEGを除いた入力，静的課題説明を除いた入力，同一記録内のEEG入替え，元記録の実際の10秒前のEEGです．CSP と scratch では通常入力と EEG-only が同じ task-specific head を使うため，同値になり得ます．Smol の EEG-only は課題説明を汎用 A/B prompt に置き換えます．context-only は同じモデルへの zero-EEG ablation で，独立に学習した context-only baseline ではありません．CSP の context-only は等しい事前確率です．

主評価は全 test 窓を保持します．10秒前のデータが存在し品質基準も満たす，事前固定した共通ID集合で対応のある負対照比較を行います．比較条件ごとに分母を変えません．participant 単位の bootstrap，macro F1，AUROC，Brier，NLL，ECE，固定0.8の採用率，処理時間，実メモリを記録します．二つの logit に限定した softmax の0.8は校正済みの「意図の確信度」ではありません．

## 因果性と前処理の範囲

全64 channel，物理単位V，200Hzの3秒窓を用います．元160／128Hzの各記録の先頭から，因果的な3–40Hz SOS filter と遅延を戻さない FIR resampling を適用します．cue の1秒後から始まる600試料を取り，実際に利用した最後の試料が判定時刻を越えないことを検証します．annotation review 対象の記録は保留し，自動修復しません．

100µVの候補基準では train の72群中43群だけが残ることが train-only 診断で分かり，validation／test の信号処理前に500µVの粗大振幅保留，channel標準偏差0.01µV以上，finite を最終基準としました．100µV超過の flag は保持し，単位の変更，振幅の再尺度化，clip は行いません．500µVは今回の探索的前処理選択であり，正常脳波の保証や標準的な artifact 除去基準ではありません．残留 artifact の影響を結果の限界として扱います．

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
