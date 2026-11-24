# PJ-Xavier 論文査読報告

**対象：Toward EEG-Conditioned Language Agents: An Audited Study of Multi-Source Brain-Foundation-Model Adaptation，草稿 v1**

**査読日：2026年11月23日，日本時間**

**判定：主要改稿と，新たな対応付き下流実験が必要です．**

## 1．総合評価

現稿は，EEG基盤モデルの追加学習を監査可能な形で整理した研究草稿です．学習量と反復量，波形再構成と行動分類，過去モデルと最新モデル，記録された行動と潜在的意図を区別している点は適切です．一方，主題である「EEGをLLM/VLM内部へ統合すると，画面・操作履歴だけでは識別できない意図や次操作を推定できるか」は，現在の実験では回答されていません．

**現状のまま，B案の有効性を実証したNeurIPS／ICLR本会議論文として投稿することは勧めません．** 必要なのは，未実証事項の但し書きを増やすことではなく，中心仮説を直接検証する一組の実験です．慎重に書かれていることと，論文の主要貢献が成立していることを分けて判断します．これは今回の資料に基づく事前査読判断であり，会議・誌の公式査読や採否予測ではありません．

研究の主軸は，ユーザーが指定したB案を維持します．CBraModの再構成改善だけを主題に置き換えたり，「忘却を防いだ」という別の主張を現結果に追加したりする根拠はありません．全候補の取得・適格データの学習は基盤整備として継続し，B案に必要な同期データと対照実験を並行して整えるのが妥当です．

## 2．査読対象と証拠の範囲

対象原稿は31745 bytes，SHA256は159164d4bbce4c2df83fd7d9653559ff22127d53c744efd1b177fa25ab01975aです．本文・表・付録を読み，NeuroCoreのコードと実験報告，追加の一次文献に照合しました．実験証拠の基準コードはcommit d801634182b970ffee748fbf59960d3a1c6c7cc5です．これは旧379470f0の公開成果物と同一treeで，既に統合済みのmainを基点に整理したcommitです．

今回，新しいモデルの学習，ヒト実験，原稿の投稿は行っていません．過去の実行検証を再実験や第三者による科学的再現と呼び替えていません．取得文献の出版版・プレプリント，Scopus上の照合，全文取得は文献台帳で区別します．

| 草稿にある結果 | 根拠から許される解釈 | 現段階では許されない解釈 |
| --- | --- | --- |
| 79493窓，66.244時間の追加学習 | 当該checkpoint lineageで実際に学習した固有区間のunion | 約23TBを全て学習済み，あるいは同量の意図教師データがある |
| BCI validationのMSE 70.82%低下 | 追加sourceの固定validationにおける再構成改善 | 意図推定が70.82%改善した，未知データ全般へ汎化した |
| 旧validationのMSE 0.46%上昇 | seedを揃えた点推定での小さな悪化 | 有意な破滅的忘却，または忘却防止の実証 |
| 過去の行動分類35.09%／34.20% | 別系統モデルによる探索的な3クラス行動proxy評価 | 最新66.244時間モデルの下流評価，実PC操作成功率 |
| 69／69のモデル検証 | 実行・重み・入力・記録などの整合確認 | 69個の独立した科学的仮説検定，意図推定の再現実験 |
| Design sourceの適格窓0 | 固定QCにより追加学習を見送った結果 | 条件を緩めれば有効データになる，または新しいfitが完了した |

根拠：原稿のAbstract，Methods，Results，Appendix A，およびrepo内BCI8089820.md，sequential_training_aggregate.json，DesignのQC・独立検証集計です．

## 3．維持すべき長所

第一に，原稿は最新encoderの学習結果と，過去のSmolLM／Qwenを用いた行動分類を同じ実験として扱っていません．この区別をAbstract，表のcaption，結論まで維持してください．

第二に，元データ・単位・重複・split・実際の学習入力を追えることは，再現性の基礎として有用です．特に，容量を学習時間に読み替えないこと，元のvalidation／testをtrainに移さないこと，科学的理由による除外を残すことは，Methodsと付録の強みになります．

第三に，旧validationの悪化と，EEGなしの履歴モデルが強い結果も記載されています．不都合な比較を消さず，「何がまだ示されていないか」を明示している点は維持すべきです．ただし，これらの記録自体から新しい学習原理や汎用的な否定結果が導かれるわけではありません．

## 4．主要指摘

### M1．中心仮説と，実際に比較したモデルがつながっていません

**該当箇所：Abstract，Research questions，Results，Discussion．**

最新の66.244時間encoderで実証されたのはmasked reconstructionです．一方，行動proxyの35.09%／34.20%は別のモデル系統です．両者を並べても，「追加学習がLLM/VLMの意図推定に寄与したか」は検証できません．現在の慎重な記載は正しいものの，論文の中心を担う対応付き実験が欠けています．

必須修正は，同一の下流モデル，画面・履歴，学習予算，label，splitで，少なくとも①非EEG，②公式pretrained encoder，③最新adapted encoderを比較することです．旧encoderと最新encoderの下流差を，同じ未使用の評価集団で測定してください．実行するまではRQ3を「未回答」としたままにします．

### M2．近接研究との比較が不足し，独自性の境界が曖昧です

**該当箇所：Introduction，Related work，貢献の3点．**

UniMindだけでなく，NeuroLMもEEGをLLMへ内部接続する既存研究です．さらにShiinokiらは，神経・眼球信号と画面文脈をLLMで統合し，コンピュータ上の選択意図を扱っています．したがって，「EEG＋LLM」だけでなく，「画面文脈を含む意図推定」という問題設定にも直接的な先行例があります．[R1–R3]

差分は，入力信号の有無だけでなく，融合位置，学習対象，予測時点，labelの根拠，未知ユーザー・環境への評価，強い非EEG比較，除外された利用者の扱いで記述してください．「内部接続」と「言語化した特徴をpromptへ与える方式」の違いを主張する場合も，接続形式の違いだけで十分とせず，それが必要になる条件を実験で示します．

有力な研究上の問いは，「同じ画面と履歴の下で残る意図の曖昧さを，EEGが追加的に解消し，その改善が未見ユーザー・セッション・環境でも残るか」です．これは提案する仮説であり，今回の結果では未実証です．

### M3．EEGの独自寄与を，現在の対照だけでは確定できません

**該当箇所：Recorded-action classification，Discussionのnext stage．**

原稿では，EEGありモデルのAI-vs-other AUROCは0.6053，履歴baselineは0.7844と記録されています．実験報告には，同じ203窓・8 IDsで事前固定した履歴対照が4条件あります．EEG窓開始前の履歴が0.7844，同＋coarse contextが0.7933，guarded decision前の履歴が0.8885，同＋contextが0.8870です．後二者はEEG窓開始後から予測時点までの履歴も使い得るため，入力cutoffを区別して全条件を掲載してください．

これらは特徴集合とmodel familyが異なるため，数値差からEEGの有害性を結論することはできません．一方，同一EEGモデルのcontext-only ablationに対する差の95%区間は−0.0399～＋0.1158と記載され，正の追加価値を確証するものでもありません．根拠はbenchmark/HISTORY_BASELINES.mdです．

EEGありモデルの入力をゼロ化しただけの比較は，非EEG専用に学習・選択した強いbaselineの代用になりません．独立に調整した画面＋履歴モデルを主比較に置き，入力置換のnegative controlを補助比較として使ってください．接続token数，学習可能parameter数，検索・tool利用権限，観測時間の差も記録します．

shuffleや時間ずらしによる低下は，信号とlabelの対応が重要であることを支持し得ますが，皮質活動に固有の情報の証明ではありません．視線・EOG・EMGを測定できる場合は対応する比較を加え，測定できない場合は「EEG計測系から得た予測情報」という限定を残します．[R4，R5]

### M4．観測された行動，高次目的，実行成功を別々のlabelとendpointにする必要があります

**該当箇所：Problem formulation，Separate action experiments，prospective protocol．**

クリック・キー入力・AI送信という3クラスは行動形式です．例えば同じクリックでも，検索，資料作成，承認，誤操作では目的が異なります．次の操作を当てたことを，その人の高次目的を理解した証拠として扱えません．この区別は現稿で明示されており，実験設計にも落とす必要があります．

最低限，①記録された操作，②実験で指定した目標，③本人が選択・報告した目標，④decoderの推定，⑤実環境での達成結果を別フィールドにします．AIが既存EEGへ推定で付与した「目的」は独立したgoldにしません．目標を示したcueや説明文がそのまま画面・履歴入力へ残る場合，EEGの追加情報を試す課題として成立するか確認が必要です．

SENSE-42は同期した生理・PC行動を含む有力な候補ですが，主目的は覚醒度などの状態評価であり，その論文だけを根拠に自由な高次意図のgoldがあるとは言えません．ChiossiらのSelect／Observeも，高次目標の自由推定と同一ではありません．[R5，R6]

### M5．未知条件と独立評価の範囲を，より厳密に定義してください

**該当箇所：Abstractのpreviously unseen，Reconstruction validation，Limitations．**

「今回の追加学習から分離したvalidation」と，「公式foundation modelの事前学習でも一度も使われていないデータ」は別です．前者は台帳で確認できますが，後者は公開pretraining corpusとの重複確認が必要です．確認できない場合は，held out from our adaptation corpusの範囲に限定します．

既に複数phaseで観測したtestを，次の設計変更後も未使用のconfirmatory testと呼ぶことはできません．既存結果は探索用として保持し，次の中心仮説には未使用の参加者・session・taskまたは新収録の評価群を固定します．新しい分割を作るだけで，既に見たデータが未知になるわけではありません．

時間的に近い窓や，同じtrialから作ったaugmentationをtrain/testへまたがせないこと，刺激・layout・記録blockによる予測を点検することも必要です．EEG視覚分類でblock構造と時間相関が成績を支配した先行例があるため，participant splitだけで全ての交絡が解消すると考えないでください．[R7]

過去UniBa評価における公開EEGの将来filter支持4.125秒は，実験報告上も仮定のままです．guarded cutoffを使ったことと，元信号が本当にオンラインで利用可能だったことは別です．原稿では一般的な注意だけでなく，この具体的な未検証点をMethodsと限界に示してください．

### M6．再構成MSEの数式・単位を修正し，集計方法を明確にしてください

**該当箇所：Data and methodsの損失式，Table 2の説明．**

原稿の式は，masked patchごとの二乗ノルムを|M|で割っています．実装のmasked_lossは，masked patch内の全scalarについてsquare().mean()を取ります．patch lengthが200なら，通常の二乗ノルム表記では追加の1/200が必要です．

実装に対応する表記は，例えば次です．

L_SSL = [1 / (|M| × T)] × Σ_(m∈M) Σ_(j=1…T) (prediction_mj − target_mj)^2，T = 200．

入力はV×10^4という数値尺度です．報告MSEを物理単位V^2の値と混同させないよう，入力尺度，masked patch数，channel数が異なるときの集約，group／record／windowの重みを明記してください．これは本文の定義と再現性の問題であり，報告値が誤計算されたという指摘ではありません．

根拠コードは，model/cbramod_ssl.pyのmasked_loss，model/cbramod_ssl_smac.pyのcorpus unit・shape確認と学習呼出しです．詳細な独立監査は同梱のINDEPENDENT_METHODS_REVIEW.mdを参照してください．

### M7．checkpoint選択とvalidationの役割を明文化してください

**該当箇所：Methods，Reconstruction validation，Table 2．**

どのvalidationがepoch選択に使われ，どの評価が選択に使われない事後比較だったかを，runごとに示してください．新BCI validationをfit開始前に固定したことは有用ですが，旧validationでcheckpointを選んだ場合のそのscoreは，独立した最終testではありません．

現実装では，初期の親checkpointをepoch 0の候補に含めず，更新後epochの中からbestを選びます．そのためbestという名称だけから，親より悪化していない保証は得られません．現結果の選択を後から変更せず，本文へ仕様を追記してください．将来の維持性能を目的にする実験では，親モデルを含めるか，複数domainで何を選択基準とするかを結果を見る前に決めます．

Table 2の4行は，実装の独立照合により，いずれも「各group内の平均window MSEを求め，group間を等重みで平均」する方式と確認しました．legacyの327窓・2群だけがwindow-weightedという不一致はありません．本文のgroup-weighted aggregationを，an unweighted mean of per-group mean window MSEsと明記すれば再現しやすくなります．新sourceのP300／BDF／BCIは本適応系統のfit・checkpoint選択に用いないcohortですが，legacyはcheckpoint選択に使っているため，同じ未使用評価という扱いにはしません．

### M8．独立単位と不確実性が，論文の主張に対して不足しています

**該当箇所：Table 2，Table 3，Statistical uncertainty．**

BCIは12 metadata groups，行動proxyは8 participant IDs，P300は1 validation groupです．11709窓を11709人と同等の独立観測として扱えません．12群すべてでMSEが下がったことは記述できますが，1回のfitから学習の不確実性まで推定したことにはなりません．

主比較は元参加者または保守的なmust-link group単位で対応を保ち，複数seedの変動と被験者間の変動を分けます．cluster bootstrap等は集団一般化の仮定を明記し，群数が少ない場合は個別群の差も併記します．primary endpointを絞り，多数のAUROC／accuracy／shuffle比較を後から最も良い一つへ置き換えません．

新しい被験者数は，目標とする最小の改善量とcluster内相関を使って設計してください．現段階で「30人なら十分」などの固定人数や採択確率を根拠なく置くことは勧めません．

### M9．文献と版の管理を更新してください

**該当箇所：Related work，References．**

現稿の6文献では，直接近接するHCI/BCI研究と，EEG-to-textの評価上の批判を十分にカバーしていません．NeuroLM，Shiinokiら，Chiossiら，noise baselineの評価研究，block-design批判を加える必要があります．否定的文献だけで結論を固定せず，teacher-forcingを用いない評価を提案したCOFETTも含めます．[R1–R8]

UniMindは版を固定してください．元稿の著者列はv1系ですが，2026年5月のv2では著者順・人数が更新されています．年，著者，URL，本文で参照した版を対応させます．同じ研究のarXiv版と掲載版を，独立した二つの証拠として数えないことも必要です．

### M10．過去の追加SSLの否定的結果と入力診断も記載してください

**該当箇所：Separate action experiments，Recorded-action classification，Limitations．**

過去の13.983333時間SSL系列では，固定maskのportable validation MSEが公式初期0.001183398643337276から学習後0.001494813073512124へ26.315%悪化しています．同系列の下流3クラスbalanced accuracyも35.09%から34.20%でした．現稿は後者を記載していますが，前者も併記し，最新66.244時間系列との違いを明示してください．これらの別系統データを混ぜて「最新の再構成改善が下流へ転移しなかった」とは結論できません．

同じ過去系統の400 train入力に関する振幅診断では，最大絶対nominal振幅のmedianは303.74µV，p95は2139.97µV，100µV以内は36／400でした．元XDF/device calibrationが未確認なので，数値だけで単位誤りや性能低下の原因とは断定できません．ただし，SSL側の固定QCと下流入力分布の関係を説明するうえで省けない診断です．根拠はmodel/CBRAMOD_SSL_COMPLETED_RESULTS.mdとreports/cbramod_alignment_train_input_amplitude.jsonです．

一方，同じ過去系統のITU外部cohortでは，旧CBraModに対するAUROC差が＋0.083782，未補正のparticipant bootstrap区間が＋0.005627～＋0.151406でした．この限定的な正の結果も公平に掲載してください．同cohortではshuffle・実−10秒対照との差の区間がゼロを含み，強い履歴対照にも届いていません．探索的なsite依存の改善を，安定した条件付きEEG効果や「一切効果なし」のどちらにも広げないことが必要です．

## 5．章ごとの改稿指示

| 箇所 | 修正内容 | 新規実験の要否 |
| --- | --- | --- |
| Title／Abstract | 「language agents」は目標であることを維持し，実証済み貢献をadaptationと別系統proxy評価に限定．large-scaleは比較対象を示すか削除 | 不要 |
| Introduction | 主貢献を「EEGの条件付き追加価値を検証する問い」へ整理．現稿で未回答であることを保つ | 最終主張には必要 |
| Related work | 内部接続，文脈を使うBCI，意図・注意の分離，厳格な評価の4群を追加 | 不要 |
| Methods | 損失の1/T，V×10^4，optimizer・scheduler・head初期化・mask・seed・選択基準・集計単位を追記 | 不要 |
| Table 1 | source追加量と累積unionを維持し，反復回数を固有時間に加算しない | 不要 |
| Table 2 | cohort別集計方式，選択使用／不使用，群数，matched seedをcaptionへ | 不要 |
| Table 3 | 同じ比較条件がある行を明確化し，異なるfeatureやbudgetの行から因果比較しない | 対応付き比較には必要 |
| Discussion | 独立した下流実験なしに「MSE改善が行動へ転移しない」と一般化しない | 必要 |
| Appendix | exact commit／config／source admission／除外／checkpoint／評価codeの対応表 | 不要 |
| References | 最新版と正式書誌を照合し，preprintと掲載版を識別 | 不要 |

原稿の英語表現を磨くだけでM1～M5は解決しません．文章だけで直せる項目を先に修正し，結果が必要な箇所には予定と未実施を明示します．

## 6．B案を検証する最小の実験構成

### 6.1 主要な比較

主endpointは，あらかじめ定義した目標／次操作の性能について，参加者単位で平均した「EEGあり − 独立に学習した画面＋履歴のみ」の差とします．高次目的を評価する場合は，そのlabelの取得方法を先に固定します．

| 条件 | 目的 |
| --- | --- |
| 画面＋履歴のみの強いモデル | EEGの追加価値を測る主baseline |
| 同じモデル＋公式pretrained EEG encoder | 追加SSL前の比較 |
| 同じモデル＋最新adapted EEG encoder | 今回の追加SSLの下流効果 |
| 同じEEGを使うlate fusion | 内部接続という設計の寄与 |
| 同じ内部接続＋対応を崩したEEG | 対応関係への依存を確認 |
| EEG-only | 文脈とEEGの相補性を確認 |
| 真の目標を与えるoracle | decoder以降のagentが達成できる上限 |

parameter-efficient tuningや複数LLMは，主比較が成立した後の一般性確認に位置付けます．最初から多数のbackboneを並べるより，同じ観測条件でEEGの寄与が残ることを先に確かめます．

### 6.2 時間・分割・対照の固定

EEGの取得終了時点，screen frame時刻，履歴cutoff，実操作時刻，filter／resampling遅延を同じclockへ対応させます．反応後の情報や，予測対象の操作そのものを含む履歴を入力へ混ぜません．元データのoffline filteringの因果性が不明な場合は，オンライン予測という主張を限定します．

主splitは未使用ユーザー／sessionを保持し，taskやlayoutの一般性は別軸で評価します．shuffleは被験者・記録条件の違いそのものを検出するだけにならないよう，交換可能性とdonor範囲を事前指定します．実際の過去EEGによる時間ずらしも，同じ意図が持続していないかを点検します．

文章生成を評価する場合は，正解の前tokenを逐次渡すteacher forcingを，利用時の自由生成成績として扱いません．no-EEG／noise条件でも同程度の文章が出る可能性を検証します．一方，この批判から全てのEEG-to-textを不可能と結論することも避けます．[R4，R8]

### 6.3 判定の分岐

主baselineに対して改善があり，対応を崩したEEGでは改善が消え，未知条件でも再現する場合，EEGの追加予測価値を主張できます．それでもEOG／EMGを切り分けていなければ，神経活動に固有という解釈は限定します．

内部接続がlate fusionを上回らない場合，EEGの効用と接続方法の新規性を分けて結論します．意図推定は改善してもtask successが上がらない場合は，decoder以降の制約をoracleとの差から分析します．全ての比較が不確かな場合は，powerと信頼区間を示し，「EEGは無効」と過大に一般化しません．

## 7．投稿先の判断

NeurIPS／ICLRを研究計画の候補として維持しますが，現在の草稿がそこへ投稿可能な完成度に達しているという評価ではありません．投稿先は，完成した貢献に合わせます．

| 主に実証できた貢献 | 優先して検討する投稿先 | 必要な根拠 |
| --- | --- | --- |
| 条件付き情報，融合，汎化に関する一般的MLの知見 | NeurIPS／ICLR／ICML | 強い比較と，複数条件で再現する知見 |
| 人の意図・注意とPC操作を結ぶ新しいinteraction | CHI | 近接HCI研究との差，人にとっての効果，適切な利用評価 |
| 視覚的対象・状態へのgroundingが中心 | CVPR | 視覚問題としての独自性と未知画面への検証 |
| 専門分野に重要な科学的発見 | Nature Communications等 | 十分な独立実証と科学的な意義 |

CHIは，脳波を含む生理信号によるinteractionを扱うため，今回の近接研究を踏まえると明示的に比較すべき候補です．会議の公式説明は，実装と将来構想の区別や，貢献に適切な検証も重視しています．[R10] h5順位は，この適合性や証拠の不足を埋めるものではありません．

2027年の各会議へ今から新規投稿できるとは一括して扱わず，既存のPUBLICATION_PLAN.mdにある公式日程を再確認します．本査読では投稿・同時投稿・対外連絡を行っていません．

## 8．優先順位と完了条件

**最優先は，B案の入力とlabelが成立する同期データの確認です．** 全量取得の完了を待たず，既存の適格データで何を検証できるかを確定します．自由意図のgoldがない場合は，その事実を保持し，行動proxyまたは指定目標の範囲で先に成立する比較を定義します．

次に，正式な文献比較，MSE式・単位・選択手順の修正，強い非EEG主baselineと対応付きencoder比較を実行します．最後に，未使用集団での独立評価，群単位の不確実性，必要な実環境評価を揃えます．論文の完成判定は取得容量ではなく，主要主張に対応する証拠が揃ったかで行います．

追加の全量取得・学習は継続対象ですが，本レビューをもって未完了の取得・再学習が完了したとは報告しません．Designは17train・7130窓を全て検査して適格0だったためfit0であり，既存66.244時間モデルが最新の検証済みlineageです．

## 9．主要一次文献

以下は査読で特に重要な文献です．著者全員・正式書誌・Scopus照合状態・取得版・ファイル確認は別添references.bibと文献台帳を参照してください．

- **[R1] UniMind**，Luら，arXiv:2506.18962v2，2026年5月改訂．EEGとLLMの内部接続に関する近接研究．https://arxiv.org/abs/2506.18962v2
- **[R2] NeuroLM**，Jiangら，ICLR 2025．EEG tokenとLLMを用いるmulti-task学習．https://proceedings.iclr.cc/paper_files/paper/2025/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html
- **[R3] Human intention inference with a large language model can enhance brain-computer interface control: A proof-of-concept study**，Shiinoki，Iwama，Ushiba，bioRxiv v2，2025．直接近接する文脈付き意図推定．https://doi.org/10.1101/2025.06.01.657282
- **[R4] Evaluating EEG-to-text models through noise-based performance analysis**，Joら，Scientific Reports 16，350，online 2025／collection 2026．noise対照と評価方法の検証．arXiv:2405.06459は同系統研究として管理．https://doi.org/10.1038/s41598-025-29587-x
- **[R5] Anticipation Before Action: EEG-Based Implicit Intent Detection for Adaptive Gaze Interaction in Mixed Reality**，Chiossiら，CHI 2026．Select／Observeとgaze・EEGの関係．https://doi.org/10.1145/3772318.3790523
- **[R6] A multimodal human-computer interaction dataset for neurocognitive user state evaluation**，Zhangら，Scientific Data 13，1439，2026．SENSE-42のデータ説明．https://doi.org/10.1038/s41597-026-07726-x
- **[R7] The Perils and Pitfalls of Block Design for EEG Classification Experiments**，Liら，IEEE TPAMI．block構造と時間相関に関する検証．https://doi.org/10.1109/TPAMI.2020.2973153
- **[R8] Is EEG-to-Text Feasible in Real-World Scenarios? An In-Depth Analysis Using a Neuropsychology-Inspired Benchmark**，Zhangら，ACL 2026．COFETTとteacher-forcingなしの評価．https://aclanthology.org/2026.acl-long.61/
- **[R9] CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding**，Wangら，ICLR 2025．今回の基盤モデルの原論文．https://proceedings.iclr.cc/paper_files/paper/2025/hash/bbbd6d915cb90be21c1254a82d45cedd-Abstract-Conference.html
- **[R10] Contributions to CHI**，ACM CHI 2027公式説明．候補会議との適合性を判断する一次資料．https://chi2027.acm.org/contributions-to-chi/

## 10．文献取得の実施結果

workstation上の既存ChromeからScopusの検索を実行し，今回選定した15件の研究を照合しました．全文PDFは15件ともS Mac31の専用references_20261123/pdfsへ保存済みです．合計108187805 bytes（約108.19 MB），344ページです．各ファイルについてPDF signature，全ページのtext抽出，タイトル・版，SHA256とサイズを確認しました．

| 確認項目 | 完了した範囲 |
| --- | --- |
| 選定・全文PDF保存 | 重複版をまとめた15 unique papers |
| Scopus Documents | タイトル・著者・掲載先が一致した13件 |
| Scopus Preprintsのみの一致 | Shiinoki論文とUniMindの2件 |
| 同一研究の複数collection | COFETTはDocuments／Preprints両方で一致し，1研究として集計 |
| 実record URL／数字ID | 5件で取得 |
| 明示的なEID文字列 | 0件．未表示の識別子を推測して補っていません |
| ScopusのCSV／BibTeXエクスポート | アカウント認証要求のため未実施 |
| 参考文献ファイル | 一次資料の正式書誌を基にBibTeXと照合台帳を作成 |

Shiinokiのexact-title検索はDocumentsで0件，Preprintsで1件でした．この検索結果を「論文が存在しない」とは解釈せず，引用に用いたbioRxiv v2として管理しています．Scopusの検索・照合に成功したことと，Scopus公式exportを取得したことは区別しています．BibTeXを公式exportとは称していません．

重点的に本文の方法まで確認したのは，Shiinoki，Chiossi，SENSE-42，UniMindの4件です．その他は一次abstractと関連する方法箇所を照合しました．全15件の全ページtextを抽出したというファイル確認を，全15件の精読と呼び替えていません．本調査はB案に近接する研究を優先した文献レビューであり，検索期間・全index・全除外を事前登録したsystematic reviewではありません．

再利用した既存PDFと今回の追加取得を台帳で区別し，全文PDFそのものは公開gitへ含めません．公開成果物はPRIMARY_PAPER_COMPARISON.md，references.bib，取得版・hash・照合状態のreference_manifest.jsonです．検索式・日時・collection・取得元・除外／未取得の扱いは文献README.mdとprivate取得receiptに記録しています．
