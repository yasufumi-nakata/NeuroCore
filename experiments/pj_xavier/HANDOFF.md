# PJ-Xavier 実行・引継ぎ

更新：2026-11-22 15:35 JST．最新の公開状態はこのbranchのHEAD，運用所有権はprivate coordinatorと実jobを優先してください．

## 現在の到達点

**phase3の学習・固定評価・全成果物回収・M1実機試用・Colab停止と，今回開始した追加取得sourceの全件監査を完了しました．** 取得済みは重複を除き49.54GB，連続EEG255.33時間です．phase3選択モデルの実入力unionは29.99時間で，新規追加sourceは未学習です．M1の12例は全棄権，成功操作0でした．環境とEEGから自由な意図を推定して自律操作する目標は未達です．

ユーザーは取得量の拡大，学習・評価・試用，GitHub／Google Doc保存，定期的な継続を許可済みです．workstationとworkstation経由Colab CLIを使用し，smac19は除外します．SFC認証待ちを作業の依存条件にしません．今回clusterへ計算を投入した事実はありません．

前段／phase2／phase3の完了run，評価，失敗probe，元データを保持します．[12:40](reports/handoff_snapshots/HANDOFF_20261122_1240.md)，[13:36](reports/handoff_snapshots/HANDOFF_20261122_1336.md)，[14:30](reports/handoff_snapshots/HANDOFF_20261122_1430.md)の引継ぎを保存しています．旧状態の「進行中」は各観測時点の記録です．

## 作業場所と所有権

| 項目 | 場所 |
| --- | --- |
| 正worktree | `/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training` |
| repository／branch | `yasufumi-nakata/NeuroCore`／`feat/pj-xavier-validation-20261122` |
| Draft PR | [71](https://github.com/yasufumi-nakata/NeuroCore/pull/71) |
| Python | `.venv-xavier/bin/python`，3.12.13 |
| Colab CLI | `/Users/yasufumi/.local/bin/colab`，0.7.4 |
| 原本・重み・詳細証拠 | `private/xavier/`，git除外 |
| coordinator | `private/xavier/eegmmidb_phase2_v1/coordinator.json` |
| 運用証拠 | `private/xavier/eegmmidb_phase3_v1/` |
| 取得・監査証拠 | `private/xavier/expansion_candidates_v1/` |

rootが公開allowlistだけをcommit／pushします．原EEG，個別予測，参加者ID，認証情報は公開しません．個別の科学source・protocol・checkpointはhashを固定し，次の変更は新しいrunへ分けます．有効leaseと実jobを確認し，active ownerがいる間は観測だけにしてください．rootの最終公開後にcoordinatorを解放し，後続ownerが排他的leaseを取って継続します．

venv Pythonを起動する際にsymlinkを `.resolve()` しないでください．base interpreterへ解決するとvenvのnumpy等が見えません．直接のpathまたは `.absolute()` を使用します．rootの最初のmetric確認がこの理由で計算前に失敗し，同一source／入力のまま正しいinterpreterで成功しました．失敗receiptを保持しています．

## phase3：完了した学習

fresh同seed小型EEG encoder→連続EEG masked SSL→MI教師学習→凍結SmolLM2の順です．CBraModやphase2学習済みencoderを初期化に使った実験ではありません．科学設定はphase2 test閲覧前に固定しました．

| 段階 | 実績 | fitタイマー |
| --- | --- | ---: |
| SSL | 20epochs／1,420steps，正常完了 | 694.855秒 |
| MI | 60epochs／1,500steps，正常完了 | 306.493秒 |
| fused | 20完了epochs＋43steps，計1,043steps，登録1,800秒による正常partial | 1,800.418秒 |
| 合計 | 準備・probe・最終評価・回収は別 | 2,801.766秒／46.70分 |

SSL train35,865非重複3秒窓／75群＝29.8875時間，validation7,581窓／16群＝6.3175時間．MI train6,359窓／72群＝5.299167時間．test17群と50注釈review記録をSSLから除外しました．選択checkpoint経路のseen-IDを監査し，SSL＋MIのnative区間とphysical payloadのunionが**107,949.2秒＝29.985888889時間**で一致しました．epoch反復597.75時間を独立量には加えません．

SSLは初期epoch0を含む21候補の全validation ID／loss，最小値とtie，encoder tensor，heldout排除を独立に照合し，epoch20を選択しました．subject平均MSEは0.0901938974→0.0671713159です．融合bestはepoch11／550stepsで，最終1,043steps状態とは別です．

LLMはSmolLM2-1.7B-Instruct revision `31b70e2e869a7173562077fd711b654946d38674`，更新対象72,416 parameters，凍結LLM1,711,376,384 parameters．EEG分類器は4,900 parametersです．L4最大予約18,593,349,632B．2行head最適化は実機BF16の固定gradient許容差を満たさず不採用とし，元full-vocabulary headを使用しました．許容差は変更していません．

[モデルカード](reports/eegmmidb_phase3_training_20261122_v1/MODEL_CARD.md)，[完了証拠](reports/eegmmidb_phase3_completion_20261122_v1/README.md)，[parameter実数](reports/eegmmidb_phase3_completion_20261122_v1/parameter_counts.json)を参照してください．

主要private receipt／SHA：

- 科学protocol：`984b516475b06f4e7641074596790c43f76e34d33b6f047239a4ddbd2d0d4016`
- `source_seal_v1.json`：`4c713b5112ebae88885a7c6dae7dc429722df2543d78278a38de7174cf396be7`
- `benchmark_execution_seal_v1.json`：`564017c4cba3d01fe6a9d47fe152b52aee8fc0df84b51c4276b25dc006a660cc`
- SSL best：`d2f40f6ba1ac5b8ee3fe9b9063ead48bb74af2e375dc346ad535400c69480ed2`
- EEG best：`3981052a799072e384be0ac003d1e272821ae080c2f55369e5f1764e76aa4138`
- fused best：`b2373bc054dbdd883236b417d3679744e179f8b067c2289c6191984a8b0de471`

## 固定評価と追加データ比較

同じ17 test群，主課題751窓／副課題738窓．参加者平均BAです．

| モデル | 主左右BA［95%参加者bootstrap区間］ | 副両拳／足BA |
| --- | ---: | ---: |
| CSP＋LDA | 59.92%［53.00，66.84］ | 59.97% |
| phase2 EEG | 59.09%［53.52，65.73］ | 65.10% |
| phase2 EEG＋Smol | 57.44%［53.18，61.73］ | 59.85% |
| phase3 SSL→EEG | 62.45%［56.85，69.00］ | 64.80% |
| phase3 SSL→EEG＋Smol | 56.72%［52.21，61.70］ | 63.12% |

新2モデルの全5条件，T0，16実LLM routingを独立SSL監査と共通900秒枠内の230.29秒で完了しました．旧3モデルの再推論は0です．全36,050保存予測と16 routingから173,980回のcached状態遷移を比較し，新fusedの主episode成功は2.16%，正解class routing上限は62.5%でした．4環境配置を独立EEGの4倍と数えません．

新EEGの主採用率は固定0.8で8.79%，採用内正解率83.33%．新fusedは3.60%／70.37%です．0.8は二候補softmaxの閾値であり，校正済みの意図確信度ではありません．T0主誤採用の参加者平均は新EEG15.53%［7.12，26.57］，新fused3.68%［1.30，6.81］です．自然idle誤操作／分とは区別します．通常−shuffleの主BA差は＋11.89［5.91，18.10］／＋8.19［1.66，15.11］ポイントです．

rootが新2モデル×5条件×2課題の点指標を保存済み予測から独立に再計算し，最大差2.22e−16で一致しました．bootstrapの再計算ではありません．phase2の旧3モデルも別途独立確認済みです．[全結果](eegmmidb_phase3/RESULTS.md)を参照してください．

15:03:03 JSTにphase3 test readout前の比較仕様を固定し，17群のpaired bootstrap1,000回／seed20261122を適用しました．主課題の新−旧はEEG**＋3.3673ポイント［＋0.3064，＋6.4127］**，融合**−0.7189［−4.9788，＋3.6936］**．副課題はEEG−0.3035［−4.0793，＋3.5105］，融合＋3.2704［−0.5985，＋7.5843］です．4比較は多重比較未調整，同じtest再利用・1seedの探索的比較です．融合ではbatch／更新回数／時間枠が変わり，SSLの単独効果と同定できません．[比較証拠](eegmmidb_phase3/ADDED_DATA_COMPARISON.md)を参照してください．

## 回収，M1試用，停止

全476member／489,554,060Bを回収し，担当者とrootが全件SHAを再照合しました．ZIPは77,992,715B．既にworkstationへ保存した2GB原ZIPと再生成可能な6.67GB配列だけを明示hash台帳付きで回収対象から除外し，checkpoint／optimizer／RNG／validation／test／失敗probe／sourceは保持しています．

モデルは `private/xavier/eegmmidb_phase3_v1/runs/` の `masked_ssl_v1/`，`eegnet_ssl_supervised_v1/`，`eegnet_ssl_smollm_fused_v1/` にあります．試用はrepo rootで次の1コマンドです．

```bash
python3 scripts/try_pj_xavier_recorded.py
```

固定train3例×4mappingの12 fresh routingを18.745秒，exit0で実行しました．全棄権，成功0，誤操作0，OS／cloud／test forward0です．同じ入力を使う再試用は新日時付きoutputを生成します．[契約と実測](integration/EEGMMIDB_PHASE3_RECORDED_TRIAL.md)を参照してください．MPSメモリはsample観測でtrue peakではありません．

物理session `xavier-eegmmidb-20261122-p2`／boot `d2656e10-f70e-4eb0-8915-009c4aa629bd` はphase2から明示移管してphase3で使い，**15:18:59 JSTに停止・active一覧から消失**を確認しました．15:19:42の残高405.38CU，rate0，active assignments0．登録408.93CUとの差3.55CUは，fitだけに帰属する費用ではありません．前段sessionも12:26:40に停止済みです．

private phase3主要receipt：

- `phase3_recovery_receipt_v1.json`：`1bf122bba4744bd86c9a57d434bcc0c6d193f162afca03bdc465229a657e0c90`
- `root_recovery_integrity_check_v1.json`：`60216274f57829c3c4bb5c764063742b2504930435127bb5d1c774cf072b936d`
- `root_saved_predictions_check_v1.json`：`4ca9cf884ad3c31d5b852838156b9b81ad4760613442e1a42dda64145ee09fab`
- `phase3_session_stop_receipt_v1.json`：`ebe049607eadfa8608c3a7a20aea62c8b1703b8ee385702a03ea41c8d06025c5`

M1後の空きが35GiB未満になり，最初のSTOPはCLIへ送る前のguardで終了しました．旧helper・失敗ログを保持し，全476SHAとowned terminal／GPU空を確認するSTOPだけの追補で停止しました．新しい取得・学習の容量guardは緩めていません．M1前後の2,320,908,288Bの空き減少は原因未特定です．

## 追加取得006とラベルの制約

[取得006](reports/acquired_progress_006.json)，[数量006](reports/verified_eeg_quantities_20261122_006.json)，[確度補足](reports/quantity_scope_addendum_006_20261122_v1.json)は15:21 JSTの観測です．

- 完全コピー766／49,563,745,287B，SHA重複除去後760内容／**49,539,668,378B**．companionを含みます．
- 原形式として確認した連続EEGは2,638記録／**244.802595659722h**．Brainstorm62候補chの.469455556hを保留にすると244.333140104167hです．
- 処理済み連続は73 unique記録／**10.526006944444h**．nativeと合わせ255.328602604166hです．
- 処理済みepochは1,688×3.5秒＝**1.641111111111h**．原区間の重なり不明のため別掲．
- Starlab5列tableは公称1.295405556h相当ですが列対応未確定で確認済みEEGへ非加算．
- 新規開始したP300／CNT／SSVEP／InMID／Brainstorm／small3 releasesは全件checksum・header監査・reconciliation完了．担当workerは全て終了．旧partial2件912,924,912Bは未完として保全．
- [最終source監査](acquisition/FINAL_SOURCE_AUDITS.md)に18focused tests，MNE独立校正probe，入力／成果物のhashを保存．旧全49GBをこのsnapshotで新たに再hashした意味ではありません．

BCI8089820は694 GDFから完全EEG重複1つを除き693記録／72.256857639h．eventrate0をMNEの512Hz fallbackで正解時刻へ確定しません．著者flag群と跨ID重複をsplit前に扱います．

P300は原DAT105記録／7.488324653h，派生EDFを加算しません．SelectedTargetはdecoder出力，TextToSpell／StimulusTypeは教師側oracle，終了時TextResultは過去contextにできません．CNT24記録のMOABB旧版code1／2／3対応を固定sourceで確認しましたが，2追加記録のcode4は未確定です．SSVEPは原BDFなし，空eventsとfractional Statusを意味labelへ変換しません．

InMIDは237files全取得，92 EDF／19.478333333h，14 EEGと44auxを区別．MAT／動画を独立EEG時間に加えません．Zenodo2392979のcross-ID同一EEG7組を除いた29 FIF／2.521284722hとし，7名義IDの3 must-link成分をsplit制約にします．8制約群を8実参加者と呼びません．MI-BMPIのSET／MAT全値一致を確認し，epochsは一度だけ加算．単位，code1／2の意味，原区間，商用許可noticeは未解決です．

キューの783releases／650,483jobs，既知24.17TBは取得候補で，取得済みではありません．42,398 jobsは容量不明．snapshot006の空きは37,020,954,624B＝34.48GiBで，35GiB reserve＋必要な回収枠を満たしません．**新しい大量取得・展開は停止し，実ストレージ計画を先に更新します．** 古いbulk workerや期限切れbudgetを再起動しません．原データ・重み・ユーザーの既存ファイルを消しません．

## 次の実行可能な作業

[多channelの引継ぎ設計](eegmmidb_phase3/protocol.next_multichannel_handoff.v1.json)と[P300漏洩防止追補](eegmmidb_phase3/protocol.next_P300_label_context_addendum.v1.json)は設計済みで未実行です．

1. coordinatorを読み，有効ownerと実jobがなければ排他的leaseを取ります．終了済みのphase1／2／3を再実行せず，最新git状態・容量・到達可能な計算資源を確認します．現在GPU sessionはありません．
2. 新規large allocation前に，実在する永続保存先と容量，回収枠，帯域を測り，新たな上限を登録します．task所有の再生成可能containerのみ，保全原本とbyte-exact復元検証を伴う整理を検討できます．課金追加・他者領域削除は行いません．
3. CPU上で新sourceの可用性をmanifest化します．原label／観測event／AI weak labelを区別し，train-onlyの教師推定を評価goldへ昇格させません．元のval／test／レビュー保留と原payloadのdeny，cross-ID must-linkを新splitへ継承します．
4. 別moduleで14／16／27／32／64chを扱うcausal shard builder，source別単位・QC，同montage batching，variable-channel resamplerを実装します．既存CBraMod encoderは動的C/Pを受けられますが，旧trainerの8／32ch制約と接続部8ch制約は別修正が必要です．元の非因果処理は後段normalizationでは修復できません．
5. 新しい固定protocolで，EEG＋判定前の環境配置からX/Yを直接出すalignmentを設計し，train側で配置をcounterbalanceします．cueの正解と自由意図の正解を区別します．環境のみ／EEGなし／shuffle／原時刻shift／配置同変性／T0／正解class routing上限を固定します．必要に応じて新しい未使用holdoutを作り，同じ17群への最適化を避けます．
6. 実機のbackward・最大形状・回収容量probe，新たな時間／CU／メモリ予算と単一CLI ownerを登録してから学習します．ColabへはworkstationのCLI経由でアクセスします．checkpointごとのvalidation，固定bestのtest，回収hash，M1試用，停止まで一つの工程として扱います．
7. 未達，実測量と時間，次の阻害要因をREADME・HANDOFF・PR71・[学校Google Doc](https://docs.google.com/document/d/1kGlNqvZEg_AzSYKGTiGF3Z0HcVIxNGNimNwpR6w1680/edit?tab=t.7cvfsge8pwil)へ反映します．過去Doc2tabsは保持します．

automation `Automation_f439325b5d888191bc75e92d1ba97be7` は毎時15分26秒，Asia/Tokyoです．15:15 runは15:16:15に完了し，root activeを確認して競合操作を行いませんでした．次は16:15:26 JST．最終更新promptとreadbackはphase3 privateのautomation receiptを参照してください．Proモデル固定指定はAPIで保証できません．継続は進捗の要約だけで終えず，上記の独立して実行できる実装・検証・記録を進めます．
