# CBraMod＋SmolLM2：単一 token の探索実験

## 固定した変更

公式の CBraMod を凍結した EEG encoder にし，8 チャネルの特徴を小さな projection で SmolLM2-1.7B の入力 embedding へ接続します．LLM 本体も凍結します．学習するのは `LayerNorm(200)` と `Linear(200, hidden_size)` のみです．

同時に，回答の符号化を固定した A／B／C に変えます．旧モデルの回答長は EOS を含めて AI 送信 6 token，keyboard 2 token，mouse click 4 token でした．新しい出力はそれぞれ ` A`＝330，` B`＝389，` C`＝340 の 1 token です．400 train 窓の fused／EEG-only prompt で，2,400 通りの連結境界を実 tokenizer で確認しました．対象ラベルを prompt に入れず，最後の prompt 位置の 3 logits だけで cross entropy を計算し，評価にも同じ logits の softmax を使います．1 forward で全候補を比較し，回答 token や EOS を入力に足しません．

これは v3 の既存 test 結果を見た後に設計した**探索実験**です．encoder とラベル方式を同時に変えるので，もし改善しても原因をどちらかに特定しません．既存 v3 の 400／155／203 窓と被験者割当を保持します．test は新しい確認用標本ではなく，探索的な再使用です．この run 内の選択は validation だけで行い，ITU は使いません．

## EEG の単位と実行経路

入力元の manifest と配列は 8 ch × 448 samples，256 Hz，1.75 秒のままです．その**末尾 1 秒の 256 samples**を切り出し，volts を 10,000 倍，`scipy.signal.resample_poly(up=25, down=32)` で 200 Hz にします．各チャネルを 1 patch とし，公式 encoder から得た 8 × 200 の特徴を 8 prefix tokens に投影します．元窓の外の EEG を読まず，per-window z-score も加えません．原データに対するオンライン因果性は未確認のままです．

CBraMod は全環境で明示的に CPU／float32 に固定します．`.to(cuda)`／`.to(mps)` を親 model に適用しても EEG branch は CPU に残ります．LLM と projection が指定 accelerator 上で動きます．MPS の FFT 対応に依存せず，同じ経路を L4 と M1 で利用するための構成です．これは暗黙の CPU fallback ではありません．

## 事前固定した資源・学習条件

`cbramod_symbol_protocol.json` が唯一の今回の protocol です．新 run は `runs/cbramod_smollm2_uniba_v3_symbol_v1` です．

- 容量候補は batch 8／16／32 で，各候補を新 process・固定 seed で測ります．train だけを使って 50 optimizer steps かつ最低 1 epoch 相当の 400 例を処理します．validation／test のスコアは使いません．
- 21 GiB 以内で完了し，最後の 10 step の reserved memory の変動幅が 256 MiB 以下の，実測済み最大 batch を選びます．失敗した候補も記録します．短い資源試験の成功から本学習の完走を断定しません．
- 本学習は容量試験の projection を使わず，seed 20261122 で初期化し直します．最大 30 epochs／900 秒，AdamW，lr 0.001，weight decay 0.01，gradient clipping 1.0，BF16，LoRA なしです．validation balanced accuracy が最良の checkpoint を選び，同点は早いものを残します．
- 容量選択 receipt，protocol，source，prepared 配列，公式 weights の hash を照合します．途中停止では checkpoint と epoch 内 cursor を保存し，再開時間は累積で数えます．

新しい controller gate `cbramod_smollm2_uniba_v3_symbol_evaluation` の live tick と，先行 Qwen 評価の完了・GPU 空きを確認するまで GPU ジョブを起動しません．

## 評価と解釈

既存 `evaluate_variants.py` の API を保持し，同じ best で fused，context-only，EEG-only，記録内 shuffle，元 raw の 10 秒前 EEG を各 1 回評価します．context-only はゼロ EEG を固定 encoder に通す入力除去であり，分布変化を伴う ablation です．別に最適化した単一 modality モデルとは呼びません．time shift は checkpoint が封印した元配列と raw を再照合します．

報告する 3-class 指標，固定 0.8 閾値の abstention 指標，AI 送信対その他の補助指標，操作履歴 baseline との比較を残します．A／B／C の softmax は，日常の出現率で校正された意図確率ではありません．この case-control 標本から日常の precision，false actions/minute，エージェントの目標完遂率を推定しません．

L4 の評価後に，同じ best を M1 で train 各 class の固定 1 例，計 3 例だけ実行し，既存の dry-run router へ渡します．model／閾値を変えず，OS 操作と cloud agent 呼出は 0 のままです．

## CPU 検証

新規 `test_cbramod_token_llm.py` は，projection への勾配，LLM／EEG encoder の凍結，CPU branch の device／dtype 保持，checkpoint 復元，単位・shape，末尾 1 秒以外の値からの独立性，target を prompt に入れないこと，single-forward の loss／評価整合，protocol 固定と容量条件を検証します．公式 weights と実 train EEG 3 窓を使う試験も含みます．小型ランダム LLM による plumbing 試験は，SmolLM 本体の実性能評価ではありません．

実装は `cbramod_token_llm.py` です．既存 `eeg_llm.py`，既存 checkpoint，v3 manifest，過去の protocol は変更していません．

## 実行済みの結果

L4上でbatch8／16／32を各50steps測定し，事前候補の最大32を選びました．batch32のpeak reservedは9,219,080,192 bytes，最後の10stepの変動幅は0でした．probeの重みを使わず，本学習を30epochs／390steps，429.33秒で完了しました．validationだけで選んだbestは23epoch目，balanced accuracy45.16%です．

固定bestの8人・203窓の5条件評価は61.35秒で完了しました．forced balanced accuracyはfused35.09%，context-only33.33%，EEG-only31.48%，shuffle30.05%，raw10秒前34.80%です．fusedは時刻ずらし対照とほぼ同じで，EEGの有益な追加寄与は確立していません．固定0.8閾値のfused coverageは47.29%ですが，これは校正済みの意図確率や行動成功率ではありません．

L4のfused scoringはp50 45.02ms，p95 46.92ms，評価時peak reserved3,466,592,256 bytesでした．array読出し・resampling・token化・harnessを除くcomponent値です．strict検証はintegrity errors0，shuffle／rawshift各203件passed，CUDA resource passed，agentic goal unprovenです．全30checkpointを含む177,230,987-byte archiveをworkstationへ回収し，371membersを再hashしました．M1の3train例試用と保存予測によるAI二値比較は別の完了receiptで確認します．

### 比較時に同時変更として扱う条件

sealed protocolの記載を保ち，結果解釈ではencoderとラベル方式に加えて，入力1.75秒から末尾1秒への変更，window z-scoreからV×1e4への変更，trainable parameters166,656から412,048への変更，旧Smol3epochから30epochへの予算差，M1からL4への実行環境差を明示します．個別要因の効果は同定していません．実装の可用性と回答長の対称性が改善したことを，EEGの予測寄与の実証と混同しません．

実投入の窓長合計はtrain400秒です．独立のraw-source単位interval union監査でも400秒，overlap0と一致しています．元1.75秒窓では700秒，30epochsの反復提示は12,000秒です．取得済みの全原記録時間，採用時間区間のunion，反復提示量は別々に報告します．

集計は ../reports/cbramod_symbol_final.aggregate.json，容量試験は ../reports/cbramod_symbol_capacity.aggregate.json，時刻union監査は ../reports/selected_eeg_interval_audit_20261122.json です．

## Completed M1 train-example trial and AI ranking follow-up

The recovered selected checkpoint completed the same CPU EEG encoder to MPS Smol pathway on three deterministic training examples, one per class. All three matched their recorded action labels; this is a training-example implementation check and cannot estimate generalization. All three downstream routes remained blocked because these proxy labels do not specify a usable agent intent. OS actions and cloud-agent calls were zero.

Observed model-call p50 was 426.924 ms and p95 1,663.342 ms, including a cold first example and only n=3. MPS sampled driver memory was 4,310,302,720 bytes and sampled allocated memory 3,424,401,920 bytes; CPU process peak RSS was 593,936,384 bytes. The explicit MPS allocator quota was 8 GiB. These sampled MPS values are not true GPU peaks, and the shared workstation was not isolated from acquisition/metadata work.

The separate saved-prediction comparison gives fused AI-vs-other AUROC 0.605300 (participant-bootstrap 95% CI 0.535555–0.667713), AP 0.255807, and at probability 0.5, 10/39 AI events detected with 36 false positives among 164 other selected actions. The AUROC differences against shuffled EEG and raw −10 s EEG were positive in this reused exploratory cohort; the difference against context-only has a CI spanning zero. The 19-condition report retains exact paired intervals. They are unadjusted for multiple comparisons and conditional on these fitted checkpoints. This does not establish natural-use precision, command semantics, or autonomous-agent readiness.
