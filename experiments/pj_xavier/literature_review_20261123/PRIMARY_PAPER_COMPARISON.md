# B案に対する一次論文比較

更新日：2026-11-23．対象は[文献台帳](reference_manifest.json)の15件です．以下の「B案への含意」「必要な比較」は今回のレビュー判断であり，先行論文がB案を検証済みという意味ではありません．

## 研究位置の結論

**「EEGをLLMへ接続する」だけでは新規性を成立させにくいです**．NeuroLMは離散神経トークンをLLMへ入力し，UniMindは学習可能なNeuro-Language ConnectorでEEGとLLMを接続しています．さらにShiinokiらは神経特徴・眼球情報・画面文脈をLLMへ渡して意図推定を行っています．B案はこれらを主要比較対象として明示する必要があります．[NeuroLM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html)，[UniMind v2](https://arxiv.org/abs/2506.18962v2)，[Shiinoki v2](https://www.biorxiv.org/content/10.1101/2025.06.01.657282v2)．

研究として有望なのは，**同一の画面・操作履歴・推論予算を与えた強い非EEGモデルに対して，判定前のEEGを内部表現として加えることが，未学習の人・セッション・課題で意図推定と実行成功を改善するか**という問いです．これは今後検証すべき仮説です．encoderの再構成誤差が下がったことだけでは，この仮説の成立を示せません．

## 一次論文15件の役割

| 文献・採用版 | 確認した主な内容 | B案で必要になる比較・制限 |
|---|---|---|
| [Shiinokiら，bioRxiv v2，2025](https://www.biorxiv.org/content/10.1101/2025.06.01.657282v2) | EEG・眼球・UI文脈を特徴量のテキストとしてLLMへ渡す意図推定．本文ではLightGBMによる8特徴の選択とReAct方式を扱う | 最も近い文脈統合の先行例．同じ非EEG条件で特徴テキスト方式と内部embedding方式を比較する．79%は20人中11人のneural-responsive群の値で，全員の成功率として引用しない．査読済み出版版は未確認 |
| [UniMind v2，2026](https://arxiv.org/abs/2506.18962v2) | LaBraM系EEG表現から空間・時間の二枝connectorとtask-aware queryを通してLLMへ接続 | internal fusion自体の先行例．同一encoder・LLM・データ量のconnector比較が必要．画面＋履歴を使う自由な意図推定が検証済みとは扱わない．平均gainの版内表記差は引用しない |
| [NeuroLM，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html) | テキストに整合させるVQ神経tokenizer，LLM上のEEG自己回帰，複数EEG課題のinstruction tuning | 「LLMがEEGを理解する初の方式」という主張を避ける．離散トークン方式と連続connector方式の比較対象 |
| [CBraMod，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbbd6d915cb90be21c1254a82d45cedd-Abstract-Conference.html) | 時空間criss-cross attentionを使うEEG基盤モデル | 現在使うencoderの出典．公式重み・継続SSL後・必要ならscratchを同一の下流splitで比較し，既存architectureを新提案として扱わない |
| [LaBraM，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html) | チャネルと時間patch，神経スペクトルtokenizerを用いた大規模EEG事前学習 | encoder選択とconnectorの効果を分離するための代替encoder．別のpretraining量による差をarchitecture差だけに帰さない |
| [EEGPT，NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html) | 表現整合と再構成による自己教師学習，汎用EEG表現の評価 | 低コストな線形probeやencoder比較に有用．同名の別論文や応用論文と混同しない |
| [BENDR，2021](https://doi.org/10.3389/fnhum.2021.653659) | Transformerとcontrastive SSLによるEEG表現学習・転移 | 早期の強いSSL比較対象．「大量の未ラベルEEGを使った」こと自体の新規性を主張しない |
| [DeWave，NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/1f2fd23309a5b2d2537d063b29ec1b52-Abstract-Conference.html) | 離散EEG表現を用いるEEG-to-text | 自由生成時の評価とnoise controlを伴う比較が必要．テキスト復元と画面上の意図・操作予測を同一課題とみなさない |
| [Joら，Scientific Reports 16:350，2026](https://doi.org/10.1038/s41598-025-29587-x) | EEG入力をnoiseと比較し，言語モデルの事前分布や評価手順で得点できる問題を検討 | EEGが本当に条件付き情報を与えるかをnoise・shuffle・非EEG条件で検証する根拠．関連arXiv版を別の独立再現研究として二重計上しない |
| [COFETT，ACL 2026](https://aclanthology.org/2026.acl-long.61/) | 神経心理学に基づくEEG-to-text benchmarkとteacher-forcingなしの評価 | EEG-to-text全体を一括して不可能と断定しないための肯定的比較．生成設定・課題・データ品質が異なるのでJoらの結果と単純な勝敗にしない |
| [Chiossiら，CHI 2026](https://doi.org/10.1145/3772318.3790523) | gaze interactionに先行するEEGのimplicit intent／anticipationを評価．個人内と個人非依存の設定を区別 | 判定前信号の比較に直結する．要旨の75–97%は個人内の値で，本文§4.10／§5.2.2には個人非依存評価もある．制約されたselect/observeと自由意図を区別する |
| [SENSE-42，Scientific Data 13:1439，2026](https://doi.org/10.1038/s41597-026-07726-x) | 42人のdesktop課題にEEG・生理・キーボード／マウス等を対応づけた多モダリティdataset | 計測・同期設計の参考．alertness等のuser-state資料であり，自由意図の正解ラベルが既に揃うdatasetとは扱わない |
| [Liら，TPAMI 43(1)，2021](https://doi.org/10.1109/TPAMI.2020.2973153) | blockの時間相関に依存したEEG分類と，実験分割の問題を検討 | 連続窓のランダム分割を避け，人・session・block・近接時間を管理する．特定block-design批判を全EEG decodingの不可能性へ一般化しない |
| [OSWorld，NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/5d413e48f84dc61244b6be550f1cd8f5-Abstract-Datasets_and_Benchmarks_Track.html) | 実computer環境でのmultimodal agentを実行結果に基づき評価 | 非EEG agentの強い比較と行動成功指標の参考．明示指示によるtask実行と，隠れた利用者意図の推定は分ける |
| [SeeClick，ACL 2024](https://aclanthology.org/2024.acl-long.505/) | screenshotからのGUI groundingとvisual GUI agent | 画面だけでも高い予測性能を得うるため，弱いcontext-only baselineではEEGの追加価値を示せない |

## 次の実験で事前に固定すべき比較

以下は文献から導いた実験設計上の提案です．今回これらのB案実験を実施したという記録ではありません．

### 1．非EEG条件を揃えて，EEGの追加効果を測る

最低限，画面のみ，操作履歴のみ，画面＋操作履歴，EEGのみ，画面＋履歴＋EEGを同じsplitで評価します．gaze・EOG・EMGが得られる場合は，それらを加えた非EEG条件とEEGを含む条件も比較します．モデル容量，学習例，履歴長，操作候補，推論予算を記録し，追加されたEEG以外の条件が変わる比較を主効果と混同しません．Shiinoki方式の特徴量テキスト化と，UniMind／NeuroLMに対応する内部接続も主要比較に含めます．

### 2．入力が判定時点より未来を見ないようにする

入力EEGの終端，画面の取得時刻，利用可能な操作履歴の終端，正解行動の時刻を固定します．クリック後の画面や正解操作を含む履歴，中央化された窓，未来側まで使うfilterを含めない設計が必要です．最終操作前の予測であることを，単に「EEGを使う」と独立に確認します．収録元のoffline前処理が不明なデータは，そのままonline因果評価の証拠にしません．

### 3．脳以外の説明と時間漏洩を検討する

同じtensor形状のnoise，trialをずらしたEEG，許容される範囲のshuffle，眼球・筋電の補助測定を用意します．ノイズ置換で性能が変わらなければ，EEG由来情報の寄与は支持されません．一方，低下しただけで脳活動への因果帰属が確定するわけでもなく，眼球・筋活動や課題構造の説明が残ります．連続EEG窓がtrainと評価へ跨がらないsplitを保持し，試行の近さだけで正解を復元できないかを調べます．[Joら](https://doi.org/10.1038/s41598-025-29587-x)，[Liら](https://doi.org/10.1109/TPAMI.2020.2973153)．

### 4．未学習の人と実行結果で評価する

参加者・session・task/templateの分割を先に固定し，参加者単位のmacro集計と不確実性を示します．窓数を参加者数の代わりに扱いません．必要なcalibration量も条件として明記します．自由生成なら評価時の正解prefixを使わず，意図分類や次操作予測と，実際のtask成功・誤操作・保留率・遅延を分けて報告します．[Chiossiら](https://doi.org/10.1145/3772318.3790523)，[COFETT](https://aclanthology.org/2026.acl-long.61/)，[OSWorld](https://papers.nips.cc/paper_files/paper/2024/hash/5d413e48f84dc61244b6be550f1cd8f5-Abstract-Datasets_and_Benchmarks_Track.html)．

## 現段階の主張の境界

CBraModの継続SSLや新sourceでの再構成評価は，encoder準備の証拠として整理できます．B案について主張を強められるのは，以上の条件で**同一文脈にEEGを加えた差分**が示され，未学習参加者と実行結果に届いた後です．「LLM内部への接続」「多量のEEG」「意図に関係しそうな信号」の三点だけを組み合わせても，先行研究との差や有効性の検証にはなりません．
