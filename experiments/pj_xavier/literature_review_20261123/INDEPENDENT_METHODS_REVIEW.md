# PJ-Xavier 原稿に対する独立方法論レビュー

**評価日：2026年11月23日 JST**
**対象：別保存の英文研究原稿 v1 と，対応する実験コード・公開集計・投稿方針**
**結論：読んで査読できる原稿は存在します．現在の原稿は，複数 EEG データ源への適応と過去の行動分類を区別した，慎重な予備研究報告として成立しています．一方，B案の中心である「予測前の画面・操作履歴を条件に，EEG を高性能 LLM/VLM の内部へ統合することで，意図・次操作の予測が改善する」という主張を支える直接実験は未実施です．既存の数式・方法記述・結果掲載を修正することと，B案を実証することを分けて進める必要があります．**

本レビューは採択確率や架空の査読点を付けるものではありません．原稿に実際にはない強い主張を仮定して批判することも避けます．実験を追加実行せず，原稿，コード，保存済み結果を独立に照合しました．本レビューでいう「独立」は，原稿の主張をそのまま採用せず根拠を確認したという意味です．同じプロジェクトの研究記録に依拠しており，外部研究室による再現実験ではありません．

## 1．対象と証拠の範囲

### 1.1 実際にレビューした原稿

リポジトリの初期検索では manuscript 本文を発見できませんでしたが，別保存の以下の原稿を取得し，**全文を読んで対象に追加しました**．「原稿がない」という評価ではありません．原本は変更していません．

| 資料 | 識別・範囲 |
| --- | --- |
| 英文原稿 | *Toward EEG-Conditioned Language Agents: An Audited Study of Multi-Source Brain-Foundation-Model Adaptation*，PJ_Xavier_research_manuscript_v1.md，31,745 bytes，210行 |
| 原稿 SHA-256 | 159164d4bbce4c2df83fd7d9653559ff22127d53c744efd1b177fa25ab01975a |
| 別保存の投稿方針 | PJ_Xavier_submission_strategy_ja.md，6,109 bytes，SHA-256 82619c87b95c35cfca0f7202cabf99d5c0bf469d55c30f9270e2e0429fce7418 |
| コード・記録の基準 | HEAD d801634182b970ffee748fbf59960d3a1c6c7cc5，branch feat/pj-xavier-paper-review-20261123．レビュー期間中の PUBLICATION_PLAN 改訂も差分を確認しました |
| 現行の研究目的 | [PUBLICATION_PLAN.md](../PUBLICATION_PLAN.md)．B案を主軸とし，現時点の SSL をその基盤に位置付けています |
| 今回作成したもの | このレビューのみ．原稿・投稿方針・学習コード・既存結果を変更せず，学習・再評価・外部投稿を行っていません |

原稿は自ら preliminary / not submission-ready と明記し，RQ3 を未回答としています．この自己評価は証拠と整合します．レビューの対象は現在の文章と実験の完成度であり，投稿先の将来の判断を予測するものではありません．

### 1.2 区別して扱う三つの系列

| 系列 | 実際に測ったもの | 混同してはいけないもの |
| --- | --- | --- |
| 現行 S Mac CBraMod 系列 | 6採用源，79,493個の非重複3秒窓，66.2441667時間．複数の source 内再構成評価 | 新しい 66.244時間 encoder の行動・意図・LLM/VLM 評価はありません |
| 過去の CBraMod＋Smol/Qwen 行動系列 | UniBa 203 test 窓・8 dataset IDs，ITU 外部サイト，記録済み操作の分類と EEG 入力の対照 | 現行 66.244時間 encoder の下流評価ではありません．入力は実画面の pixel 系列ではありません |
| より以前の EEGMMIDB Phase 2/3 | 小型 EEG encoder，二値 motor imagery，合成画面対応と保存済み予測の replay | UniBa 三分類や現行 encoder と同じモデルではありません．replay 回数は独立 EEG 標本数ではありません |

原稿はこの系列の区別を既に行っています．ただし，表だけを読む読者にも分かるよう，モデル・親 checkpoint・SSL corpus・下流 cohort を対応させる一枚の表を本文に置くと誤読が減ります．根拠は [BCI8089820.md](../reports/smac_training_20261122_v1/BCI8089820.md)，[過去の追加 SSL 完了報告](../model/CBRAMOD_SSL_COMPLETED_RESULTS.md)，[Phase 3 の比較](../eegmmidb_phase3/ADDED_DATA_COMPARISON.md) です．

## 2．現時点で支持できる主張

| 主張 | 確認した証拠 | 主張の上限 |
| --- | --- | --- |
| 1本の現行系列で 66.244時間の適格 EEG 窓を実際に学習へ投入しました | 238,479秒，79,493窓．最終 BCI stage は10 epochs，12,460 updates．[系列集計](../reports/smac_training_20261122_v1/sequential_training_aggregate.json) | 取得予定の約23 TB，全データ取得，潜在参加者数，別系列の学習時間とは同一ではありません |
| BCI 追加後の新 source 内 validation 再構成誤差は低下しました | 12 groups，95採用記録，11,709窓．0.0027535698545106207 → 0.0008033758285906217，−70.8242%．[BCI 報告](../reports/smac_training_20261122_v1/BCI8089820.md) | 未使用 test の成績，未知データ源への汎化，意図推定ではありません |
| 同じ BCI 追加後に旧 source の再構成誤差は小幅に悪化しました | 同じ327窓・2 groups，同じ seed 20261122で親 0.0014298673209638358 → 子 0.0014364248753904745，＋0.4586% | 機構としての catastrophic forgetting や母集団での有意な悪化までは示しません |
| P300 と BDF でも新 source 内の対比較が改善しました | P300 1 group・609窓で −25.8128%，BDF 5 groups・327窓で −22.6792%．各 [P300](../reports/smac_training_20261122_v1/P300_HELDOUT_VALIDATION_RESULTS.md)／[BDF](../reports/smac_training_20261122_v1/BDF29606906_HELDOUT_VALIDATION_RESULTS.md) 報告 | 共通の test，source 外評価，十分な参加者数の統計的証明ではありません |
| 過去の行動分類では，EEG なしの強い履歴対照を上回る一貫した効用を示せていません | 同じ UniBa cohort で行動モデル AUROC 0.6053～0.6215，履歴対照 0.7844～0.8885．[行動比較](../benchmark/CBRAMOD_UNIBA_V3_RECORDED_COMPARISON.md)，[履歴対照](../benchmark/HISTORY_BASELINES.md) | EEG が一般に無効という結論ではありません．入力情報，encoder，学習予算が異なります |
| 実行と来歴の検証は相当に整備されています | BCI 69/69 checks，親 tensor 継承，入力 membership，旧 validation の同 seed 再計算．[BCI 報告](../reports/smac_training_20261122_v1/BCI8089820.md) | 69個の独立した科学的仮説の検定でも，外部再現でも，神経学的因果性の証明でもありません |

この部分は論文の長所です．原稿は，負の結果や制約を隠して成功した agent と称していません．前処理，親子 checkpoint，入力台帳，検証時の seed，評価と学習の役割を分けた点も適切です．方法論上の弱点は，これらの監査が不十分というより，**監査された計算結果が，B案の科学的問いをまだ直接検定していないこと**にあります．

## 3．投稿前に必須となる修正・追記

以下の M1 は数式と実装の不一致です．M2 以降には，不足した記述，結果の掲載範囲，新たな証拠が必要な箇所が含まれます．これらを一律に「誤り」とは扱いません．

### M1．損失関数の式は，patch 内の標本平均を含む形へ修正してください

**対象：Data and methods / Audited multi-source EEG adaptation の数式．**

原稿は masked patch ごとのベクトル二乗ノルムを合計し，masked patch 数だけで割っています．一方，[cbramod_ssl.py](../model/cbramod_ssl.py) の masked_loss は masked positions を選んだ後，残る全要素の二乗誤差に mean() を取ります．[cbramod_ssl_smac.py](../model/cbramod_ssl_smac.py) の load_corpus が要求する配列は [N,C,P,200] です．通常の L2 ノルムの定義では，原稿の式に patch 長 T＝200 による除算が足りません．

1窓について実装に対応する式は，例えば次です．

$$
\mathcal L_{\mathrm{SSL}}
=
\frac{1}{|\mathcal M|T}
\sum_{(c,p)\in\mathcal M}
\sum_{\tau=1}^{T}
(\hat E_{c,p,\tau}-E_{c,p,\tau})^2,
\qquad T=200.
$$

さらに，入力は物理 V を一度だけ 10^4 倍した model units です．表の MSE はこの入力単位の二乗であり，物理 V² に戻すときは 10^−8 倍です．物理単位との対応と，1 patch が200 Hzで1秒であることを明記してください．

validation の集約も同じ式の後に記述が必要です．実装の validate は「各窓の MSE → 各 group 内の平均 → group 間の等重み平均」です．全窓を pooled mean にした値とは一般には異なります．学習時の batch の mean と，報告する group macro validation を分けて定義してください．**これは原稿の式・単位の修正であり，保存済み MSE 数値や相対変化が誤計算されているという指摘ではありません．**

### M2．何を使って best checkpoint を選んだかを明示してください

**対象：Data and methods / Reconstruction validation，Results / Reconstruction，Limitations．**

原稿は legacy validation を「安定した旧領域のチェック」と説明していますが，この327窓・2 groups は，各 stage の学習後 checkpoint 選択にも使われています．[cbramod_ssl_smac.py](../model/cbramod_ssl_smac.py) の execute は epoch ごとに validate を呼び，mean_subject_masked_mse が小さい checkpoint を best とします．これは許容される開発手順ですが，選択に使った validation を，独立した最終評価と同じ意味で扱うことはできません．

次の点を Methods に追加してください．

- 各追加 stage は親の選択済み weights から始め，AdamW は新しく初期化します．既存 stage の中断再開では optimizer/RNG を復元しますが，新 source 追加とは別です．
- initial_validation は記録されますが，新 stage の best_score は infinity で始まります．したがって best は「更新後 checkpoint の中で最良」であり，親の初期状態を候補に含めた最良ではありません．
- 旧 validation を複数 stage・複数 epochs で再利用した回数と役割を開示してください．この選択規則により，親より旧 validation が悪い子が選ばれること自体は実装矛盾ではありません．
- BDF 親の saved score の seed は20261123，BCI 子は20261122です．原稿が同 seed で再計算した比較を採用した点は正しいです．異なる mask seed の元の値を単純に並べて改善率を出さないでください．

新 source validation の凍結時点も分けて表記してください．BCI は protocol を05:08:44 JSTに固定し，fit は05:18:20 JSTに開始した記録があります．P300 報告は事後に固定した diagnostic としています．BDF は source partition を入力前に固定していますが，それだけで評価分析全体が fit 前登録だったことにはなりません．**全 source の評価を一括して「事前登録済み」と広げないこと**が必要です．根拠は [BCI protocol](../next_multichannel/BCI8089820_VALIDATION_PROTOCOL_V1.json)，[BCI 報告](../reports/smac_training_20261122_v1/BCI8089820.md)，[P300 報告](../reports/smac_training_20261122_v1/P300_HELDOUT_VALIDATION_RESULTS.md)，[BDF 報告](../reports/smac_training_20261122_v1/BDF29606906_HELDOUT_VALIDATION_RESULTS.md) です．

### M3．過去の追加 SSL は，再構成成績も悪化していたことを本文へ載せてください

**対象：Results / Recorded-action classification and language-model alignment，Discussion / What has not been established．**

過去の追加 SSL 系列では，適格 train が1,678個の30秒窓，13.983333時間で，UniBa 由来は8窓・240秒，残る1,670窓は portable experimental dataset 由来でした．この系列の同一 fixed-mask portable validation MSE は次のとおりです．

| 同じ過去系列の比較 | 公式初期 encoder | 選択された追加 SSL encoder |
| --- | ---: | ---: |
| portable validation MSE | 0.001183398643337276 | 0.001494813073512124 |
| 相対変化 | 基準 | ＋26.315%，悪化 |
| 同じ行動 alignment 設定による UniBa 三分類 BA | 35.09% | 34.20% |

根拠は [CBRAMOD_SSL_COMPLETED_RESULTS.md](../model/CBRAMOD_SSL_COMPLETED_RESULTS.md) です．現在の原稿は下段だけを載せています．上段を載せないまま現行系列の再構成改善と並べると，「同じ encoder の再構成が改善したのに行動能力は悪化した」という，実際には行っていない媒介関係を読者が推測しやすくなります．

原稿は系列が異なることを明記しており，虚偽の実験を主張してはいません．それでも，**過去系列では SSL 自体の評価も改善していない**という情報は，負の下流結果の解釈に不可欠です．「再構成改善は agent 改善を保証しない」は一般的な論理として妥当ですが，本プロジェクト内でその因果的断絶を同一系列・同一条件で証明したとは書けません．

ITU の結果も併記の均衡が必要です．過去の追加 SSL は，外部 ITU で旧 CBraMod に対して AUROC が＋0.083782，未補正の participant bootstrap 区間が［＋0.005627，＋0.151406］でした．同時に，shuffle と実 −10秒対照との差の区間はゼロを含み，強い履歴対照には届いていません．この限定的な正の結果と，UniBa での非改善を両方報告し，単純な「一切効果なし」にもしないでください．同じ完了報告に数値と制約が保存されています．

### M4．履歴 baseline は4条件を揃え，入力可能時刻を明記してください

**対象：Table 3，Separate EEG-to-language-model action experiments．**

Table 3 は AUROC 0.7844 の一条件だけを載せています．同じ203 test 窓・8 dataset participant IDs，AI送信39件・他操作164件を使った事前固定の4条件は以下です．train は400窓・18 IDs，validation は155窓・6 IDsです．

| EEG なしの入力 | 時刻の上限 | AI-vs-other AUROC | binary BA |
| --- | --- | ---: | ---: |
| 過去の操作履歴 | EEG 窓開始より前 | 0.7844 | 70.41% |
| 同履歴＋coarse context | 同上 | 0.7933 | 72.98% |
| guarded decision 前の操作履歴 | 予測時点の guarded cutoff より前 | 0.8885 | 83.11% |
| 同履歴＋coarse context | 同上 | 0.8870 | 86.59% |

根拠：[HISTORY_BASELINES.md](../benchmark/HISTORY_BASELINES.md)，[NATURAL_COHORT_V3.md](../model/NATURAL_COHORT_V3.md)．

この表の binary BA と，EEG＋LLM の三分類 BA は別の指標です．AUROC の対比較では同じ AI-vs-other target と標本集合を使えますが，入力情報量が同じという意味にはなりません．後半2条件は EEG 窓開始後から予測 cutoff までの履歴も利用できるため，前半2条件と同じ時間窓の比較としてまとめないでください．

履歴には過去1/5/15/60秒の event 数，直近 event からの時間，source/missing indicators を使い，GPT events，target，キー文字，座標，duration，絶対時刻，IDを入力から除いています．L2 logistic の C，scaling，vocabulary は train/validation のみで決定し，test は固定閾値0.5です．コードは cutoff，禁止特徴，future onset，欠損と件数の矛盾などを検証します．ただし，この baseline 自体は原 TSV から履歴を再構築せず，固定された prepared manifest に依拠します．これらの guard は確認できますが，元記録の時刻付与まで独立に証明し直したわけではありません．

もう一つ重要なのは，旧 EEG＋LLM にはこの豊富な history sidecar がそのまま入力されていない点です．[eeg_llm.py](../model/eeg_llm.py) の tokenized_prompt は coarse environment_text を渡します．したがって，既存比較が支持するのは「現行の記録済み行動モデルより，強い履歴モデルの成績が高い」という記述です．**同じ V,H を与えた際に EEG を足す利益がない，という条件付き効果の検定にはなっていません．**

context_only は固定済み EEG モデルへゼロ信号を渡す対照であり，単独で最適化した画面・履歴モデルではありません．[旧 SSL 完了報告](../model/CBRAMOD_SSL_COMPLETED_RESULTS.md) も distribution-shift ablation と明記しています．これを唯一の non-EEG baseline にしない方針は，原稿の Appendix B と整合します．

### M5．採用・除外と実入力の品質を，結果の一部として掲載してください

**対象：Data and methods，Table 1，Limitations / Retrospective source heterogeneity．**

固定 QC を守ったことと，採用された分布が研究目的を代表することは別です．過去の13.983333時間 SSL のほぼ全てが portable source に偏ったことは，モデルが受けた分布を理解するための主要情報です．

さらに，過去の alignment が使った400 train 入力の最後1秒を，同じ resampling 経路で診断した記録では，各入力の nominal 最大絶対振幅の中央値303.74 µV，95 percentile 2,139.97 µV，100 µV以内36/400，peak と descriptive minimum-std の両方を満たすもの24/400でした．これは [実入力振幅集計](../reports/cbramod_alignment_train_input_amplitude.json) と [完了報告](../model/CBRAMOD_SSL_COMPLETED_RESULTS.md) にあります．

この数値は，物理単位の誤実装や低成績の原因を確定しません．元 XDF・device calibration の未確認部分が残り，波形の神経成分と artifact の割合も未確定です．ただし，入力品質の問題が大きい可能性を残したまま，一般的な EEG の有用性を否定するのは強すぎます．**既存 test を成績を見て除外し直さず，この診断を報告し，次の独立評価用 cohort では前処理・QC を事前に固定する**のが適切です．

今回の evidence freeze では，Design Emotion 31578517 の terminal 判定も追加できます．[公開 QC 集計](../reports/smac_design31578517_qc_20261123_v1.json) によれば，固定 train/val/test＝17/5/3のうち train 17記録を判定し，7,130候補窓から採用0，peak閾値超過7,062，flat68，probe0，fit0です．元整数標本の98.6049%が宣言端点に集中し，17記録×3区間×32 channels の物理 V 読取りは MNE と最大絶対差3.47×10^−18 Vで一致しました．原因は未確定ですが，今回の検証では parser・V変換の不一致は見つかっていません．

この判定を，旧2392979などの別の blocked source とまとめて曖昧にせず，除外フローに追加してください．新たな学習時間は0です．元配布ファイルの問題を自動的に作者の研究の誤りと断定したり，根拠のない倍率補正・clippingで採用可能としたりする根拠はありません．固定 QC を変えなかった判断は妥当です．

### M6．split，時間境界，評価閲覧履歴を一つの表で追えるようにしてください

**対象：Reconstruction validation，Limitations / Sampling and identity，Appendix A．**

本レビューで，新 source の test が現行適応の fit に入ったという証拠は見つかっていません．[trainer の load_corpus/check_history](../model/cbramod_ssl_smac.py) は raw/payload/partition と入力 array の seal を検証し，[BCI evaluator](../next_multichannel/evaluate_smac_bci8089820_validation.py) は validation だけを選んで test/review を読まない経路を持ちます．これらは評価すべき実装です．

それでも，以下は異なる意味を持ちます．Methods では列を分けてください．

| 境界 | 確認できること | 残る不確実性・必要な記述 |
| --- | --- | --- |
| 現行 source 内の train/val/test | 固定 group と既知 raw/payload deny，interval 非重複，train-only admission | 別 release の変換済み alias や全提供元をまたぐ真の個人同一性までは証明されません |
| 公式 pretrained checkpoint | 公開された CBraMod checkpoint を strict に読み込みます | 公式事前学習 corpus と今後の downstream test の照合範囲を別に明記すべきです．本適応で未使用であることと，過去の全事前学習で未使用であることは同じではありません |
| 旧327窓 validation | 現行 stage の checkpoint 選択と監視に使用 | 完全に未使用の最終 test と呼べません |
| P300/BDF/BCI の新 source validation | 親子を同じ入力・maskで比較．BCI は fit 前に評価 protocol を固定 | dataset partition 凍結と，解析計画の登録時点を区別します |
| UniBa と旧 EEGMMIDB test | ID 単位の分割はあります | 既に複数の試行・分析で閲覧した test です．追加の発見的比較を，初回の独立確認と呼べません |
| UniBa/ITU の EEG 時刻 | guarded extraction，実 −10秒信号対照 | 4.125秒の将来 filter 支持は公開 recipe に基づく仮定であり，元配布物の全処理と一致したことまでは未検証です |
| 4種類の合成 layout/replay | 入力 EEG と対応関係を固定し，比較を揃えられます | 同じ EEG の変形・再生は独立参加者・独立試行を増やしません |

「教師なしなので test EEG を事前適応に混ぜても無条件に問題ない」という扱いは避けてください．明示した transductive protocol を研究する余地はありますが，未見参加者への inductive generalization とは別の問いです．現在の既知 deny を守る方針は維持し，公式 pretrained weights に関する未確認範囲を明示するのが適切です．

時間的な有効性も同様です．後段の forward-only filter は，publisher が以前に行った非因果的処理を取り消しません．原稿のこの留保は正しいです．将来 B案を検定する際は，予測時点以前に「計測された」だけでなく，処理遅延込みで「利用可能だった」EEG・画面・履歴であることを同じ時計で確認する必要があります．

### M7．データ量の増加と因果的な改善を区別してください

**対象：Abstract，Results / Reconstruction，Discussion．**

現行系列で BCI の52,395窓は最終79,493窓の約65.91%です．[epoch_batches](../model/cbramod_ssl_smac.py) は montage/reference/native sampling rate/patch shape ごとに batch を作り，各行を1 epochに一度ずつ提示します．source や参加者を等重みに抽出する方式ではありません．窓数の比は把握できますが，異なる batch サイズ端数や geometry を含むため，65.91%をそのまま勾配寄与率とは呼べません．

[系列集計](../reports/smac_training_20261122_v1/sequential_training_aggregate.json) では，全採用 stage 合計24,830 updates，繰返しを含む1,580,490窓提示です．各 stage は累積 corpus を再提示し，optimizer を新規作成します．したがって親子差には，新 source の種類と量だけでなく，追加 compute，旧 source の再学習，source の投入順，optimizer の再初期化が同時に変化しています．

現在の原稿が「追加 stage の前後で source 内 MSE が改善した」と記述することは妥当です．一方，「EEG を多く集めたことが改善の原因」「規模を増やせば意図性能が向上」という主張には，少なくとも同じ親からの **旧 corpus のみの追加学習対照と，旧＋新 corpus の追加学習を compute を合わせて比較**する設計が必要です．source バランスや投入順に関する主張をする場合は，それぞれ対応する対照を追加してください．

MSE 自体も，信号振幅，帯域，reference，domain 統計への適応で低下し得ます．表の各 cohort を混ぜて平均しない原稿の方針は妥当です．再構成の価値を強く主張する際は，zero/mean/interpolation など適切な再構成対照，信号エネルギーで正規化した誤差，凍結 encoder の下流 probe を合わせると，「何を学んだか」を区別しやすくなります．これは現在の実測値の否定ではなく，解釈を広げるための追加実験です．

### M8．近接先行研究を追加し，新規性を比較可能な形に絞ってください

**対象：Introduction，Related work，Discussion / A testable next stage．**

「EEG と画面 context を LLM へ渡して操作意図を推定する」という組合せそのものには近接先行例があります．今回取得した一次資料を本文まで確認しました．

| 一次資料 | 確認できた内容 | 本プロジェクトへの含意 |
| --- | --- | --- |
| UniMind，*UniMind: Unleashing the Power of LLMs for Unified Multi-Task Brain Decoding*，arXiv v2，[一次資料](https://arxiv.org/abs/2506.18962v2) | 本文§2/3の LaBraM encoder，spatial/temporal cross-attention queries，LLM embedding への射影，task-aware query selection を確認しました．InternLM2.5-7Bを凍結し，他の構成要素を学習します | EEG embedding の学習可能な内部接続は既存手法です．本稿は予測前の画面・履歴を条件にした追加情報，未知条件，融合方式の差分を直接検証する必要があります |
| Shiinoki, Iwama, Ushiba，*Human intention inference with a large language model can enhance brain-computer interface control: A proof-of-concept study*，bioRxiv v2，2025-06-10，[DOI](https://doi.org/10.1101/2025.06.01.657282) | Results の decoder framework，Methods の Decoder Architecture を確認しました．EEG 特徴，眼球運動，UI情報をテキスト化し，知識・few-shot例とともに LLM/ReAct に渡して操作を出力します | EEG＋環境＋LLMで操作するという課題だけを新規性とできません．本稿で学習可能な EEG embedding の内部融合を主張するなら，この特徴量テキスト化方式との違いと利点を検証してください |
| Chiossi et al.，*Anticipation Before Action: EEG-Based Implicit Intent Detection for Adaptive Gaze Interaction in Mixed Reality*，CHI 2026，[DOI](https://doi.org/10.1145/3772318.3790523) | 28名の EEG・eye tracking，Select/Observe と feedback を操作した SPN 研究です．abstract の75～97%は person-dependent．本文§4.10/5.2.2には person-independent 設定もあります | 意図と観察の識別，時間窓，個人内適応と未知個人の区別に直接関係します．数値を自然な自由意図や汎用 desktop agent の成績へ読み替えないでください |
| Zhang et al.，*A multimodal human-computer interaction dataset for neurocognitive user state evaluation*，Scientific Data 13:1439，2026，[DOI](https://doi.org/10.1038/s41597-026-07726-x) | SENSE-42は42名のdesktop操作，EEG等，keyboard/mouse，5分ごとの状態報告を含み，主目的は tonic alertness です | 同期した生理・操作記録の設計には参考になります．自由な高次意図の正解ラベルを提供する資料として扱う根拠はありません |

Shiinoki 論文の79%は responder subgroup の結果であり，全登録者にそのまま適用する数字ではありません．資料は bioRxiv preprint として引用してください．取得担当が確認した Scopus の Preprints collection と Documents の区別は，査読済み出版版の確認と混同しないようにしてください．本レビューでは正式出版版が存在すると推定していません．

一方，本プロジェクトにも **内部接続の実装自体は既にあります**．[eeg_llm.py](../model/eeg_llm.py) の EEGCausalModel と [Phase 2 model](../eegmmidb_phase2/model.py) の FusedDecoder は，学習可能な EEG prefix/projector を frozen LM の inputs_embeds へ渡します．そのため，「これから初めて EEG を LLM 内部へ接続する」という説明も正確ではありません．未実証なのは，現行大きい EEG encoder と実際の予測前画面・履歴を同じ高性能 LLM/VLM に統合したときの，条件付き追加効用です．

原稿が UniMind などを挙げて「connector を作るだけでは新規でない」としている点は妥当です．さらに踏み込むなら，次の候補を明確に仮説として選び，対照で検証してください．

- 同じ画面・履歴では曖昧な意思決定に対して，EEG が未知参加者・未知 task でも追加情報を与えるか．
- 同じ EEG・画面・履歴・学習予算で，内部の学習可能な融合が，特徴量のテキスト化や late fusion より有効か．
- 異種 EEG source への適応が，単なる再構成の向上を超えて，上記の条件付き意思決定に寄与するか．

ここでいう追加情報の検証と，脳活動が行動を因果的に生んだことの証明は別です．shuffle/時刻ずらしで出力が変わっても，モデルが利用した信号に眼球・筋活動や task の手掛かりが混在する可能性は残ります．「EEGセンサーから得られる追加情報」と「神経由来の固有機構」を区別してください．

## 4．B案を論文の中心にするための最小の実験設計

以下は新しい実験の提案であり，既に実施した結果ではありません．すべてを一度に巨大な agent benchmark に広げるより，まず明確なオフライン条件付き予測の問いを検定し，その後に実操作の効用へ進む方が反証可能性を保てます．

### 4.1 まず一つの target と，一つの主比較を固定します

高次の目標，次の観測操作，実験者の cue，AI送信イベントは異なる target です．B案の中心をどれにするかを定義し，正解の取得方法，参加者に与える指示，正解と入力が混ざらない時刻規則を固定してください．AI が作った説明文を正解として循環的に評価する設計は避けます．既存の観測操作ラベルで出来る問いと，新規収録が必要な問いを分けます．

主要比較は，**同じ V,H を入力する，独立に学習した強い非 EEG モデルと，V,H＋E のモデルの対差**です．同一の test cohort，target，token/compute budget，訓練回数，選択規則，利用できる tools を揃えます．EEGをゼロにしただけの固定済みモデルは補助対照です．

### 4.2 対照群

| 比較目的 | 最低限必要な条件 | 固定すべきもの |
| --- | --- | --- |
| EEG の条件付き寄与 | V,H のみ／V,H＋正しい E | 同じ入力可能時刻，同じ backbone，target，学習・選択予算 |
| encoder 適応の寄与 | 公式 CBraMod／中間 checkpoint／現行66.244時間 checkpoint | 同じ connector と下流訓練，最終評価群，parent lineage |
| 内部融合の寄与 | 学習可能 prefix/resampler／特徴量テキスト化／late fusion | EEG情報，画面・履歴，容量と学習可能 parameter の差を記録 |
| EEG表現自体の基本能力 | EEG-only の単純 classifier／同じ EEG encoder の probe | 同じ split・target．LMを足した効果と分離 |
| 時間対応と input dependence | 正しい E／正当な block 内の derangement／実測した過去区間の E | participant/session/recording，非 EEG 入力，共通試行集合 |
| 信号特異性 | 必要に応じた eye/EMG/artifact 対照，EEG channel 条件 | 「神経固有」と主張する場合に対応する設計 |

前述の Shiinoki 方式や原稿が挙げる既存 EEG-to-LM connector は，関連研究として列挙するだけでなく，中心的な新規性に近いところを実験で比較する対象です．すべての既存大規模モデルを網羅する必要はありませんが，自案の主張を直接反証できる baseline は欠かせません．

原稿の prospective 部分の subject-wise shuffle は曖昧です．既存実装の [within_recording_derangement](../model/eeg_llm.py) のように，同じ dataset/subject/session/recording 内で対応を壊す条件を明示してください．他人や他施設の波形へ置き換えるだけでは，時間対応の効果に participant/domain shift が混ざります．複数の対照を置く場合は，それぞれ何を壊すかを事前に定めます．

### 4.3 split と評価の終点

新たな最終評価では，学習・適応・前処理の調整・checkpoint 選択に使っていない評価群を定めます．未知参加者，未知 session，未知 task，未知画面の一般化は，それぞれ違う主張です．一つの分割で全てが検証できたとせず，主張に合わせた評価軸と十分な group 数を設計してください．

既存 test をさらに探索に使う場合は，探索用であることを明示して利用履歴を残します．新しい確認用 cohort と混ぜません．crop，layout augmentation，replay，連続する窓は元の participant/session と結び付け，同じ group が別 split に流れないようにします．

主指標は target に応じた paired NLL，BA，goal/action accuracy などから事前に一つ定め，改善量と不確実性を報告します．後から有利な metric，threshold，checkpoint を選ばないために，副指標と選択に使わない diagnostic を分けます．screen/history だけで高精度になる task と，EEG によって候補が区別され得る曖昧な task を，結果を見ずに定義することが重要です．

### 4.4 PC操作の応用を主張する段階

オフラインの観測操作分類だけで論文の限定的な問いを完結させることは可能です．しかし，「PC操作に役立つ agent」を主張するなら，sandbox の実 task 完了，誤操作，abstention，自然な idle，end-to-end latency を測る必要があります．命令の意味がない3クラス出力を直接 executable goal とみなすことはできません．

現行の enriched cohort では AI送信の比率が日常利用の比率と異なります．その precision，average precision，Brier score を日常の false actions/minute に変換してはいけません．連続利用での母数を伴う評価が別に必要です．また，EEG の nominal decision margin 約375 msと，モデル呼出しだけの速度は別です．前処理，データ待ち，tokenization，model，tool 実行を含む遅延を測り，正解の定義と実際に利用可能な予測時刻を揃えてください．根拠は [NATURAL_COHORT_V3.md](../model/NATURAL_COHORT_V3.md) と [旧 SSL の実機試行を含む報告](../model/CBRAMOD_SSL_COMPLETED_RESULTS.md) です．

## 5．統計についての必須事項と，現状の限界

1. **独立単位を参加者または保守的な must-link group に置いてください．** BCI の11,709窓は12 groupsに属します．窓を独立として狭い区間を作らないでください．P300 の1 group は参加者母集団への不確実性を推定するには不足しています．BDF 5 groups，legacy 2 groupsも制約が大きいです．
2. **paired comparison を維持してください．** 同じ原試行，共通の有効 mask，同じ評価 seed，同じ preprocessing で親子を比較します．除外によって群が減った場合は候補と実採用の group/record/window 数を全て報告します．BCI の96候補→95採用の区別は良い例です．
3. **学習の乱数と評価標本の不確実性を区別してください．** 現行の participant bootstrap は固定済みモデルに条件付けたものです．学習をやり直すばらつきは含みません．中心的な B案の主張には複数 training seeds が必要で，利用可能な group 数と最小有用差に基づいて設計してください．
4. **多くの比較をした後の区間を，事前の確証的検定と呼ばないでください．** 旧 UniBa 19/24条件の保存済み比較や17 groupを再使用した Phase 3は探索的です．原稿は既にこの点をかなり適切に留保しています．
5. **bootstrap の退化に注意してください．** 履歴対照で39/39のAI送信を検出すると，観測された group を再標本化しても recall の区間が100–100%となり得ます．未知群に対する完全感度の証明ではありません．[履歴報告](../benchmark/HISTORY_BASELINES.md) のこの注意は原稿にも引き継いでください．
6. **効果の方向，規模，ばらつき，利用価値を分けてください．** 全12 BCI groupsで再構成が改善したという記述は有用です．それだけで EEG 固有の情報が agent に有用になった，という結論には進めません．confidence interval を追加するだけでも因果問題は解消しません．

旧 Phase 3 の追加 SSL 条件は，EEG-only で participant-macro BAが＋3.37 points，区間［＋0.31，＋6.41］でしたが，融合条件の差は−0.72 points，区間［−4.98，＋3.69］です．さらに融合条件では batch/update budget も異なります．この履歴を簡潔に示すことは有用ですが，二値・三分類・新しい source の結果をひとまとめにして「全て悪い」「全て良い」と総括することはできません．根拠は [Phase 3 comparison](../eegmmidb_phase3/ADDED_DATA_COMPARISON.md) です．

## 6．原稿の各節に対する具体的な編集方針

| 原稿の箇所 | 必須の変更・追記 | 維持すべき良い記述 |
| --- | --- | --- |
| Abstract | 66.244時間系列と行動系列の区別を維持．within-source validation の範囲と，旧 validation が選択用である点を本文へ明確に誘導 | RQ3未実施，OS成功なし，無理なSOTA主張なし |
| Introduction | Shiinoki，CHIの先行研究と，新規性を検定する具体的比較を追加．large-scale は比較基準を示すか multi-source とする | encoder能力と条件付き意思決定を分けた問題設定 |
| Related work | EEG-to-LM，生理信号＋環境＋意図，画面操作 benchmark を対比．dataset 論文を task gold と誤認しない | CBraMod architecture の独創を主張しない点 |
| Data and methods | M1数式，scaled units，group macro，AdamW設定，mask seed，best候補範囲，source別 native fs/reference/QC/採用率を追加 | source identity，labels非使用，publisher preprocessing の留保 |
| Table 1 | source原著/版・適格群・実採用数を付録で対応．Design等の除外フローを更新 | unique時間と repeated epochs の区別 |
| Table 2 | selection用legacy と diagnostic用newsource を明示．凍結・閲覧・fitの時系列を付す | 各cohortを合算しない，同 seed の親子比較 |
| Table 3 | 履歴4条件，時刻cutoff，binary/3-class区別．旧 SSL MSE悪化と実入力QC診断を別表で追加 | 現行66.244時間の下流結果ではないという注意 |
| Discussion | 書ける結論をsource内適応に限定し，量/計算/順序の交絡を追記．限定的なITU改善も公平に扱う | 成績の非改善をEEG一般の無効としない点 |
| Prospective protocol / Appendix B | 同じ V,H のpaired test，強い学習済みnon-EEG，fused-vs-text/late fusion，shuffle block，fresh final cohort を具体化 | prospective と明記し，未実施を結果として扱わない点 |
| Reproducibility / Appendix A | source引用・版・条件，親子checkpoint，protocol/code/corpus seal，評価の閲覧履歴をまとめる | 個人データと公開aggregateを区別する点 |

現行 Methods に対応する最低限の再現設定として，learning rate 1×10^−4，weight decay 0.05，gradient norm clipping 1，train random masking と val ID/seed依存fixed mask，各 geometry の batch，選択済み epoch，group定義とcountsを記載できます．実行ログを全て本文に載せる必要はありません．[trainer](../model/cbramod_ssl_smac.py) と各固定 protocol に結び付いた付録表が適しています．

別保存の投稿方針にある「LLM内部への学習可能な接続の実装」を次の優先作業とする説明は，既存の EEG prefix 実装を無視しないよう補足してください．次の仕事は，既存接続の存在確認ではなく，現行 encoder・実画面/履歴・選択した VLM を結ぶ比較可能な実験です．

## 7．任意の改善

以下は，現在の実測数値を正しく報告するための最低条件とは分けます．

- source×stage の再構成行列と，旧領域・新領域の曲線を掲載すると，正の適応と trade-off が読みやすくなります．
- 取得量→EEG候補→単位確認→QC→独立train窓→実seen窓の流れを一枚にすると，取得の規模と学習の規模を混同しにくくなります．
- 最終 corpus の source/participant 構成を図示し，source-balanced sampling 等の効果を将来の対照として切り分けられます．
- baseline と code/protocol/aggregate の対応を機械的に検証する小さな再集計 entry point は再現性に役立ちます．今回のレビューのために生データを再取得する必要はありません．
- 本文では研究上の問いと対比較を中心に据え，fleet の台数，PID，復旧の詳細は必要な再現性付録に置くと，論文の中心が明確になります．
- 将来の agent 評価では calibration，risk–coverage，失敗分類，遅延分布を併記できます．現時点で未測定の値を推定して埋める必要はありません．

## 8．最終的な評価と次の完了条件

**現在の草稿を「監査された multi-source EEG adaptation と，別系列の探索的 action-proxy 実験の報告」としてレビューすることは可能です．その範囲では，多くの限定表現が適切で，中心的な数値も実記録と整合しています．**

投稿前の原稿修正としては，MSE式，checkpoint選択，過去 SSL の再構成悪化，履歴baseline全条件，実入力QC，最新の除外判定，近接研究の追加を先に完了できます．これらは新たな大規模学習を待たずに直せます．

B案を中心貢献にするための完了条件は，**同じ高性能 LLM/VLM・予測前の画面/履歴・target・学習予算・未使用評価群で，EEG を足すことの追加効用と，学習可能な内部融合を選ぶ理由が示されること**です．現在の SSL 再構成結果はその実験の出発点です．追加取得量，監査項目数，GPU学習の完走だけでは，この条件の代替になりません．

### 照合に用いた主要な固定識別子

| 資料 | SHA-256 |
| --- | --- |
| レビュー時の PUBLICATION_PLAN.md（未commit改訂を含む） | d8062fdf6a5d6d6e7c4577fece5f7d634880b1c58a209a775c132511446bdff8 |
| BCI validation protocol | 65e33394e56f05ce03fb5f79c41486d0e781af81d555ff18780178ec6458f9ab |
| 旧 SSL 完了報告 | 493a0214c85f0570afe03ab94006ccd5161bd0e68bd644a48fac0f9c5b8e6b87 |
| 履歴baseline報告 | 7ea1a79f45e193ce5ca4374e376f3e730b0e6f117dbe3572f41607a1e34dcab5 |
| Design QC aggregate | 93d6fb4ff5d9657779508730c166a168ca29efc11499ce36ab4c561289b4e4cb |
| UniMind arXiv v2 取得PDF | 776eb747bbb42470d5fc0cff8d8f817f3dc23f6fd9164d2cc220ad9893b2fa03 |
| Shiinoki 2025 v2 取得PDF | 80a079e0d1884b14d77f61a2098ef8e6ca066f44d828c53c6e50db94e40bf8ef |
| Chiossi 2026 CHI 取得PDF | d22349acaaeeafae3eea7c6ae9f641bed70cacf0682ee65a7c9973f42af9014b |
| Zhang 2026 SENSE-42 取得PDF | 80abe7e67ddbf30ef7e5c32dfc93bf468fb395d37bf7163530eb58fe77253170 |

個人ID，個別予測，EEG raw，認証情報はこのレビューへ収録していません．上の実験根拠は既存の公開可能な集計・コードへの参照です．原稿外に保持されている元資料の真正性・未確認の publisher preprocessing・cross-release の実個人同一性を，このレビューだけで独立に認証したものではありません．
