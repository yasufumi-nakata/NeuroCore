# EVLM 構築・実機検証記録

確認日：2026-11-24 JST．本記録は実行 receipt に基づき，準備済みデータ，技術試験，通常学習，科学的な有効性を分けて記録します．

## 結果

画像・数値 EEG・指示文を同じモデルへ渡せる研究用候補 `PJ-Xavier-EVLM-Qwen3.8-27B-v0.1` を構築しました．既存 v5 の EEG adapter を使う候補と，新しい全64層 LoRA の短い技術試験候補を保存し，それぞれを実機で確認しました．数値 EEG が意図推定や computer use を改善したことは，まだ示していません．実 OS 操作は0です．

基盤は `mlx-community/Qwen3.8-27B-4bit`，revision `b935172e4bb5dcd0543774a34748fb92422e3027` です．固定の数値変換を使い，独立した学習型 EEG encoder は追加しません．既存の画像 tower と基盤重みを保持し，言語 decoder の内部 LoRA を学習します．[モデルカード](../model/EVLM_MODEL_CARD.md)に入力契約・実行コマンド・保存物を記載しました．

## 実機試験

| 試行 | 実測結果 | 保存・解釈 |
|---|---|---|
| v5 adapter＋画像＋EEG | 05:29:37–05:30:28，base 2/2・adapter 2/2，推論 peak 17.292 GB | 末尾4層・31対象 module．既存候補を保存しました |
| 全64層，8ch block，1,024 token枠 | 05:34–05:38，1更新後 peak 58.762 GB，54 GB guard停止 | checkpoint なし．失敗履歴を保持しました |
| 全64層，6ch block，801 token枠 | 05:51:29–05:56:07，2更新完了，peak 44.016 GB，基本指示6/6 | adapterの数値一致・再読込後の生成一致を確認しました |
| 新全64層 adapter の画像＋EEG | 05:56:49–05:57:48，base 2/2・adapter 2/2，推論 peak 17.579 GB | 496対象 module／992 LoRA tensor，333 vision tensor の不変を確認しました |

画像試験では，同じ EEG と指示に対して START ボタンの位置が左右で変わる合成2画面を使いました．単に固定のクリック座標を返す実装では両方を通過できません．ただしこの2画面は画像経路と adapter 接続の動作確認であり，EEG の寄与や実環境の操作成功率を評価するものではありません．新全64層候補では6ch×32点の数値 EEG を入力しています．

全64層の技術試験は，最長 train 例1件を同じ801 token枠で2回提示しました．独立した訓練例は1件です．通常の学習 epoch や開発用 validation 評価の完了とは扱いません．検証生成は0，最終 test 読込みは0です．実 update 時間は126.613856秒と129.364201秒，損失は1.121489→0.523196でした．同じ例への適合であり，汎化の改善を示す値ではありません．

## 全データを順次利用する準備

品質確認済み6 source の元 train 79,493窓を，元の全 channel を保持した最大6chの prompt 群へ変換しました．準備済み train は358,260例です．全359,241例（train＋development）の最大長は710 tokensで，801を超えた例，追加の除外，clipping，無言の切捨てはありません．[変換・データ監査報告](EVLM_NUMERIC_DATA_20261124.md)に全source内訳とSHAを記録しました．

元 train 窓の長さは66.244167時間ですが，現在の cycle 0 で各窓から取り出した時間位置の合計は3.487211時間です．全3秒の各時点を覆うには19 cycles必要です．channel block を増やした分を独立時間へ加算しません．この準備量を新EVLMの学習済み時間として報告しません．

元の train／validation／test／review の分割，物理単位，固定QC，元記録・被験者の除外制約を維持します．今後ダウンロードが完了する関連人EEGも，同じ確認を通ったものから次の固定キューへ追加します．EEGだけのデータには数値再構成を使い，架空の画面・操作・意図の正解は作りません．

開発用データは806023の2 group・327元窓／981例で，全6 sourceの未知データへの汎化を測るものではありません．通常学習中は同じ固定36例を使い，学習前・学習後・単純な局所補間・ゼロ基準を比較します．

## 通常学習の運用方針

最初の通常学習は技術試験の adapter を引き継がず，fresh base・fresh Adamから開始する方針です．20更新，累積100更新，累積400更新を確認点にし，初回の計算予算を400更新に区切ります．各stageで保存・再読込，数値評価，基本指示，画像保持を確認します．次stageは合格したadapterを引き継ぎ，Adamは新規開始します．同じstage内ではoptimizerを保持します．停止後は完了したsource内の順序位置を記録し，新しい明示的な継続planで再開します．

数値の継続判定は，学習を開始する前に設定します．有効出力だけを残して平均誤差を良く見せないため，無効出力に40,000の失敗罰点を加えた開発用scoreを通常MSEと区別します．同じ36例で過去のbestより5％を超えて悪化した場合，共通して有効な30例以上のMSEがstage開始時より5％を超えて悪化した場合，または2stage続けて改善がない場合は次stageへ進みません．量子化単位に対応する最小許容差も設けます．これは2 groupの開発用データで計算の無駄を避ける運用上の基準で，科学的な効果判定の閾値ではありません．

54 GB はMLXの観測値に対する運用予算です．OSの絶対的なメモリ上限ではありません．累積peak，実更新数，損失，更新秒数を保存し，超過時に止めます．GPU lockと実PIDを使い，同じnodeの学習と画像推論を直列化します．既存のダウンロードworkerは維持します．

### 通常学習の実行状況

2026-11-24 **06:12:39 JST** にsmac35で通常学習を開始しました．controller PID 34003，最初のsupervisor 34015／worker 34017です．06:13の初回観測は学習前の36例評価中で，optimizer更新は0でした．技術試験2更新とは別のfresh base開始で，初期adapterはnullです．この初回snapshotは通常学習の完了・改善を示すものではありません．

実行rootは `evlm_build_20261124_v1/curriculum_initial400_v1`，現況はその `curriculum_state.json` と，current stageの `state.json`／`supervisor.json` を照合します．controllerの完了更新数は合格stageの集計だけでなく，数値停止前に完了したstageも含みますが，実行途中のstage内進捗はworker stateで確認します．最初の3stageは20／80／300更新で，まずsource806023内の固定順序400例を使います．残る5sourceをこの初回400更新で学習したとは扱いません．

固定planは `configs/curriculum_initial400_v1.json`，SHA256は `8f66bbb6910f0d54ffbd024e40ac401a67bbf84114cb2d00c9608f2a34bc5478` です．全6source・358,260例を登録し，このrunの上限を400更新／24時間に設定しました．数値gate，schedule [20,80,300]，合格scope64層／rank8／801tokens，元54GB予算を明示し，起動前のplan検証を通過しています．400更新では `paused_update_budget` となり，全量完了とは呼びません．

最終実装は，数値変換19件，学習runner30件，VLM runtime20件，curriculum36件のCPUテストを各対応環境で通過しました．curriculumの最後の36件は配備したsmac35で再実行し，2stageの実process継続，数値悪化後の停止，bestの継承，予算停止を含めてPASSでした．コードと手順は別担当の読取りレビューも受けています．

## 時間の見方

技術試験の固定800入力tokenにおける2更新平均は127.989秒でした．同じ速度だけを機械的に掛けると20更新42.7分，100更新3.56時間，400更新14.22時間，358,260例の1回提示は約530.7日です．これには評価・保存・source走査の時間を含めません．通常学習は例ごとに短いpaddingを使うため，この数字を全量完走ETAにはしません．

少なくとも，全量を数日で学習できるとは言えません．最初の20更新から通常学習の実測速度と改善量を確認し，400更新の終了時または早期停止時に，次の計算予算・課題・入力量を見直します．全358,260例の固定キューは保持しますが，改善がないまま全量へ無条件に進む設定にはしません．

## 保存物と検証識別子

S Mac35上の作業rootは `/Users/yasufumi/pj-xavier/smac35/evlm_build_20261124_v1` です．

| 保存物 | 相対path／SHA256 |
|---|---|
| v5画像動作確認候補 | `candidate_v5_r2` |
| v5画像receipt | `d780899ce0b5be79c1b2bbe01c536bc4b3667d71dc23c442f3ddffca62761e5f` |
| 全64層技術試験 | `engineering_all64_probe_6ch_v2` |
| 全64層adapter | `34e3e22b31d6fd2a3c8807537c56ef5badf3ebe54b2dca10c500e99020cb8af0` |
| 全64層checkpoint metadata | `c5f19d6942afe73113b0dc819690869dddeb57172ae408776fd2f3fd6598df3d` |
| 全64層engineering evaluation | `db38a3e99de518b353e2878f4eeb72bcda5d7e02a9cb537431907e8cdc05ce77` |
| 全64層画像動作確認bundle | `engineering_vlm_probe_6ch_v2/bundles/all64_probe_6ch_v2` |
| 全64層画像receipt | `75e54092be08ecf335681b7603ac3d46bb8f081b9a4e35fe5688a397f5a8fe5d` |
| 全64層画像bundle manifest | `8b1ec2408c83d19905ef2d21b798d7031ac3948e375f44b9386d8c34bfecae6f` |
| 全6sourceの固定queue | `8218c7d821e4afbd4a6e841129df47236db3d759711ff8eec66991a7a0e94510` |
| 全量変換QC | `09c2b418a346eb0c7729a5bc192f913a4b106430925c7ff39efdd2f9c9a6115d` |

Bundleは同じS Mac上の固定baseを参照します．bundle directoryだけの配布では動作せず，対応baseと専用環境が必要です．元信号・重み・個別予測はS Mac側に保存し，公開記録にはコードと集計のみを含めます．
