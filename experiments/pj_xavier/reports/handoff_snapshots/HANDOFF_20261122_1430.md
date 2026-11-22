# PJ-Xavier 実行・引継ぎ

更新：2026-11-22 14:30 JST．進行中の実状態はprivate receiptとcoordinatorを優先してください．

## 現在の到達点

**EEGMMIDB phase2の3モデルは学習・固定test・負対照・T0診断・操作ハーネス・全成果物回収・M1試用を完了しました．phase3の追加学習と，公開EEGの取得を進めています．** 最新指示は，smac19を除外して学習・評価・試用まで完了し，現環境に収まる取得量と実学習量を増やすことです．workstationとworkstation経由のColab CLIを使います．SFC認証待ちを再開条件にしません．

前段のSmol1.7B・Qwen8B・CBraMod・追加SSL CBraModの実験は完了済みで，再実行しません．旧sessionは12:26:40 JSTに停止済みです．[前段](reports/handoff_snapshots/HANDOFF_20261122_1240.md)，[phase2開始時点](reports/handoff_snapshots/HANDOFF_20261122_1336.md)，[全体README](README.md)に履歴を保全しています．自由な意図の推定や実OS作業成功は未実証です．

## 作業場所と所有権

| 項目 | 場所 |
| --- | --- |
| 正worktree | `/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training` |
| repository／branch | `yasufumi-nakata/NeuroCore`／`feat/pj-xavier-validation-20261122` |
| Draft PR | [71](https://github.com/yasufumi-nakata/NeuroCore/pull/71) |
| 最新結果を公開したcommit | `47a4a9942623ecb08faeb3eb6a242914c4ec5ec8` |
| Python | `.venv-xavier/bin/python`，3.12.13 |
| Colab CLI | `/Users/yasufumi/.local/bin/colab`，0.7.4 |
| 原本・重み・詳細証拠 | `private/xavier/`，git除外 |
| 進行管理 | `private/xavier/eegmmidb_phase2_v1/coordinator.json` |

rootが公開対象を限定してcommit／pushします．他の作業変更，原EEG，個別予測，認証情報を公開しません．freezeしたsource／protocolはhashで保持し，後続実験は別runにします．有効なcoordinator leaseと実jobがある間は，automationも観測だけを行い，競合するkernel操作・編集・二重起動を避けてください．

## phase2：完了したモデルと評価

同一被験者が分割をまたがない固定splitです．train6,359窓／72群，validation1,360／16群，test1,489／17群．各3秒，train固有窓は19,077秒＝5.299167hです．全64ch，native160／128Hz，因果3–40Hzと200Hz変換，cue＋1〜4秒．50注釈review記録を保留しています．品質条件500µV・0.01µV・finiteはtrain-only診断で固定した粗い条件で，正常脳波の保証ではありません．

| モデル | 学習完了 | 主左右想起test subject平均BA［95%CI］ |
| --- | --- | ---: |
| CSP＋LDA | CPU fit3.70秒 | 59.92%［53.00，66.84］ |
| scratch EEG | 60epochs，1,500steps，304.28秒 | 59.09%［53.52，65.73］ |
| EEG＋凍結Smol1.7B | 完了17epochs＋61steps，計1,761steps，1,394.16秒 | 57.44%［53.18，61.73］ |

fusedは元fit期限14:04:54 JSTに従う正常partial exit2で，30epoch完走ではありません．LLM融合による上積みは確認できていません．fusedの同一635窓でのBA差はEEGなし＋7.45ポイント，同記録shuffle＋9.90，実−10秒＋11.45です．公開cue分類の信号として扱い，自発意図や因果的な生理機序を同定した結果とはしません．

固定0.8でfused採用率14.11%，採用内正解率63.21%，T0二択誤採用23.80%です．16 promptの実LLM routingと104,388合成episodeを分けて計数しています．fused主episode成功7.99%，正解classをroutingへ渡す主課題上限62.5%．自然idleの誤操作／分や実OS成功ではありません．全結果は[RESULTS.md](eegmmidb_phase2/RESULTS.md)と[実測JSON](eegmmidb_phase2/protocol.measured_results.v1.json)を参照してください．

M1／16GiBでtrain3例・12fresh routingを20.53秒，exit0で実行済みです．全12episodeは固定0.8未満のため棄権し，選択・成功0でした．test forward，OS操作，cloud APIは0です．[試用README](integration/README_TRY_EEGMMIDB_RECORDED.md)の既存契約と新output pathを使って再試用できます．

private phase2 base：`private/xavier/eegmmidb_phase2_v1/`

- `benchmark_execution_seal_v2.json` SHA `e1847ae39ef284a49cdc34dbe1695593b6d9a3c46b8d4db046842c403b26eb7f`
- `phase2_recovery_receipt_v1.json` SHA `81d7814c76c4d22da76ec5e67460bc97376109720e69a5f9c9e02b79bf6a788e`：294member／135,633,776B展開，全SHAをagentとrootが再照合済み．
- `root_saved_predictions_check_v1.json` SHA `1594cda0abc38b77fbe6238a13a9133843bb5f25ff42653ba215dffda751c2d9`：全3モデル5条件の主要点指標を保存済み予測から独立再計算．追加model forward0．bootstrap CIの再計算ではありません．
- fused best SHA `13df40e53dab6087bfef2fac1a4e413a2d00178413f05f4759a0aa1ea849d4ee`，scratch best `3d041e1a8c07e97ae1bebd873873964241d3856f8b6d7461f9134924f8c3f54d`．

## phase3：実行中の追加学習

phase2 testを見る前に科学設定を固定した，fresh同seed EEG encoder→連続EEG masked SSL→同じMI教師あり学習→同じ凍結Smol接続です．SSL初期epoch0も選択候補に含め，subject平均masked MSEの最小，tieは最初を選びます．下流は従来どおりvalidation主左右BAで選択します．

準備・独立照合済みのSSL入力はtrain35,865窓／75群／29.8875h，validation7,581窓／16群／6.3175h．計43,446窓，1,236配列，6,673,463,808Bです．test17群／原238記録とreview50記録を除外しています．MIとSSLの準備済みtrain区間unionは29.985889hですが，最終実学習量は実seen-IDと選択checkpoint経路から再集計してください．[準備証拠](reports/eegmmidb_phase3_preparation_20261122_v1/README.md)を参照してください．

phase2の学習・評価・全回収・全process終了後，14:12:55 JSTに同じ物理sessionを明示的にphase3へ移管しました．旧fit期限は変更していません．

| 項目 | 値 |
| --- | --- |
| 物理session | `xavier-eegmmidb-20261122-p2` |
| phase3 remote root | `/content/xavier_phase3` |
| workflow | 14:14:41 JST開始，PID16598，実receipt優先 |
| boot ID | `d2656e10-f70e-4eb0-8915-009c4aa629bd` |
| 新GPU予算 | 最大7200秒／5CU／21GiB，登録時408.93CU |
| 絶対期限 | `1795331083.750155`＝16:04:43.75 JST |
| 最終fit cutoff | 15:29:43.75 JST，評価900秒・回収900秒・停止300秒を確保 |
| stage caps | capacity300／SSL1200／MI900／fused1800秒 |
| CLI owner | model担当．training terminal後にbenchmark担当へ明示引継ぎ |

L4 BF16の2行symbol-head最適化は事前固定のgradient許容誤差を満たさず不採用となりました．許容誤差を緩めず，従来のfull-vocabulary headを使います．CPU小fixtureでの合格だけを実機同値性と扱いません．

private phase3 base：`private/xavier/eegmmidb_phase3_v1/`

- `handover_from_phase2_v1.json` SHA `1486c70467e0300d4047015dc15a6517e6029f86d22031dfb1d4ea2e196002c3`
- `gpu_budget_v1.json` SHA `018cdaa6c81eba94178ce8e465b89360e425691c4d083c5e25855b6d5c6d25a5`
- `source_seal_v1.json` SHA `4c713b5112ebae88885a7c6dae7dc429722df2543d78278a38de7174cf396be7`
- 科学protocol SHA `984b516475b06f4e7641074596790c43f76e34d33b6f047239a4ddbd2d0d4016`
- `benchmark_execution_seal_v1.json` SHA `564017c4cba3d01fe6a9d47fe152b52aee8fc0df84b51c4276b25dc006a660cc`．15CPU fixtures合格．原phase2 evaluatorのbytes・分析条件を維持した新runtime admissionです．

学習後は独立SSL監査→新2モデルの全test／5条件／T0／16routingを**共通900秒**内で実行します．旧3モデルを再推論せず保存予測を使う5モデルharness比較を行います．同じtestを再利用する追加比較は探索的です．重み・閾値・prompt・QCをtestに合わせて変更しません．

全checkpoint／optimizer／RNG／val／test／失敗probe／sourceを回収し，raw ZIPと再生成可能な配列だけをexact hash付きで除外します．原ZIPはworkstationに保存済みです．全回収hash照合・M1試用を完了し，この所有sessionの停止とactive一覧からの消失を確認します．未回収状態で停止を完了扱いにしません．

## 追加取得と量

[取得snapshot005](reports/acquired_progress_005.json)，[元EEG数量005](reports/verified_eeg_quantities_20261122_005.json)は14:17 JSTの観測です．

- 完了175内容，44,432,880,979B．このうちP300は66filesの途中時点で，release全完了とは扱いません．
- 元EEGを数量監査した範囲は2,405記録，210.503248785h．binary EEG scalar43,382,368,656BとUTF-8数値token31,065,967Bを区別します．
- BCI8089820の27,475,466,178B ZIPは全1,071member検証済み．694GDFから完全EEG重複1つを除いた693記録72.256857639hです．元event rate0の694記録は時刻確定と扱わず，27ch入力の新protocolも必要です．[BCI監査](reports/bci_8089820_acquisition_20261122_v1/README.md)を参照してください．
- P300 OpenNeuro ds005028は版固定S3 object216files，1,563,795,662Bを取得中．原DATと派生EDFを両方保存し，時間は重複加算しません．deadline15:40:23 JST，新budget SHA5390924e4b4db084275b776a3f5077ee2113819a6591f3f31baeebe3ab81b1f6．
- Figshare2061654 v1のMI完全release156,181,084Bも新規予算で取得中．現空きから未完P300・35GiB reserve・1GiB成果物回収枠を引き，全量が収まるものだけを追加します．

全catalog5,165行は不足・余剰0，固定queueは783releases／650,483jobs，既知24,172,262,716,513Bです．EEG・companion・複数版を含む候補量で，取得量ではありません．42,398jobsは容量未確定です．古いbulk PID8814と期限は終了済みで再起動しません．取得担当の新budget・delegation・checksum完了・reconciliationを確認してください．ライセンス／認証保留は維持し，他人の領域削除や追加購入を行いません．

## 継続と公開

automation `Automation_f439325b5d888191bc75e92d1ba97be7` は毎時15分26秒，次回15:15:26 Asia/Tokyo，enabled=trueです．14:15の初回は観測のみで終了しました．Proモデル固定指定はAPIで保証できません．最新保存はphase3 privateの `automation_scope_hourly_1418_v1.json` です．

1．coordinator・実PID／argv／bootを照合し，active ownerと競合しないようにします．
2．phase3 fitを同じprotocol・上限で完了します．同じ失敗や旧modelを理由なく再実行しません．
3．独立SSL選択／実入力union監査，新2モデル評価，全5モデルharness，完全回収，M1試用，session停止を実行します．
4．取得中releaseの全checksum・全記録header・duration／units／原event・重複を監査し，最新snapshotへ進めます．
5．実測結果と未達をREADME・HANDOFF・PR71・[学校Google Doc](https://docs.google.com/document/d/1kGlNqvZEg_AzSYKGTiGF3Z0HcVIxNGNimNwpR6w1680/edit?tab=t.7cvfsge8pwil)へ反映し，push後のCIを確認します．旧Doc2tabsは保持します．

ユーザーの最終目標は環境情報とEEGから自律的に操作できるモデルです．現実験のcue分類，合成環境，train例での実機smoke checkをその達成と同一視しません．不足は具体的な次の実験・入力条件として引き継いでください．
