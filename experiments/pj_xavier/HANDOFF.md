# PJ-Xavier 実行・引継ぎ

更新：2026-11-22 13:36 JST．進行中の実状態は以下の receipt と coordinator を優先してください．

## 現在の指示と到達点

最新指示は，smac19を使わず学習・評価・試用まで進め，さらに取得可能なEEGと実際に学習する量を増やすことです．workstationと，workstation上のColab CLIを使用します．SFC接続や追加承認を待ち条件にしません．購入や利用条件の回避は行いません．

前段のSmol1.7B・Qwen8B・CBraMod＋Smol・追加SSL CBraMod＋Smol，両施設の比較，M1試用は完了済みです．旧Colabは12:26:40 JSTに停止しました．その詳細は[前段の引継ぎ記録](reports/handoff_snapshots/HANDOFF_20261122_1240.md)と[README](README.md)を参照し，再実行しません．目的の自由意図の推定・実OS作業成功は未実証です．

現在は別のEEGMMIDB phase2です．準備した9,208試行からCSP＋LDA，scratch EEG encoder，EEG＋凍結SmolLM2を比較し，二択の運動想起を有界な環境操作へ接続します．前処理は完了，CSPのtrain／val fitも完了しました．ニューラルモデルは全入力2,881,756,917Bを新Colabへ転送・照合しました．Torch2.14のallocator引数の互換エラーをmodel forward前に検出して明示index指定へ修復し，科学設定・入力不変をv2 sealで確認しました．実L4容量試験はbatch256×50steps，19.12秒，CUDA予約最大2,080,374,784Bで通過しました．13:35:38 JST，本学習supervisor PID6955を起動しました．実学習開始・終了はrun receiptで確認してください．test成績はまだありません．

さらにphase3として，同じ元train群の全14runsを使うマスク自己教師学習を準備しています．科学設定はphase2のtest成績を見る前に固定しました．追加のGPU実行予算も別登録済みで，phase2完了・回収後に同じ物理sessionを後続leaseへ明示的に移管します．現phase2を黙って変更しません．

## 作業場所

| 項目 | 場所 |
| --- | --- |
| 正worktree | `/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training` |
| repository／branch | `yasufumi-nakata/NeuroCore`／`feat/pj-xavier-validation-20261122` |
| Draft PR | [71](https://github.com/yasufumi-nakata/NeuroCore/pull/71) |
| 開始時のcommit | `c966eedb4341c4901228a1259064b8d9cc1b9af7`，当時9 CI／CodeQL成功 |
| Python | `.venv-xavier/bin/python`，3.12.13 |
| Colab CLI | `/Users/yasufumi/.local/bin/colab`，0.7.4 |
| 原本・重み・詳細証拠 | `private/xavier/`，git除外 |
| 進行管理 | `private/xavier/eegmmidb_phase2_v1/coordinator.json` |

元checkoutの無関係な変更は触りません．rootが今回の公開対象だけをstage／commit／pushします．PRの現在状態を確認し，merge／force pushは実行しません．実行済みsource／protocolはhashで保持し，後続実験は別ファイル／別runにします．

## phase2：固定済みデータ

[前処理の説明](eegmmidb_phase2/README_PREPARATION.md)と[公開集計](reports/eegmmidb_phase2_preparation_20261122_v1/README.md)を参照してください．

| 分割 | 採用窓 | 採用subject groups | 3秒の固有入力区間 |
| --- | ---: | ---: | ---: |
| train | 6,359 | 72 | 19,077秒，5.299167 h |
| validation | 1,360 | 16 | 4,080秒，1.133333 h |
| test | 1,489 | 17 | 4,467秒，1.240833 h |
| 合計 | 9,208 | 105 | 27,624秒，7.673333 h |

元109群のhash splitは76／16／17で固定しています．QC後に群を移動していません．左右MIはruns4/8/12，両拳／両足MIは6/10/14です．50注釈review記録を保留し，自動修正は0です．EEG全64ch，native160/128Hzを尊重して因果的3–40Hz filterと200Hz rate変換を行い，cue＋1〜4秒を使います．未来の入力変更に対するprefix不変性，独立MNE単位fixtureを検証しました．

最大絶対500µV，channel標準偏差0.01µV以上，finiteというQCはtrainだけの診断後・val/testの波形を読む前に固定しました．100µV初期候補を超えたflagを保持しています．500µVは探索的な粗い振幅条件で，生理的正常の保証や一般標準ではありません．後から成績に合わせて緩めません．

−10秒の実波形対照は7,880窓あり，対照不足を理由に主9,208窓を減らしていません．testの固定共通subsetは1,254窓です．別のtest専用T0休止1,357窓は，fit／選択に使わず，固定confidence0.8での二択誤採用を診断します．自然idleや連続誤操作／分の評価ではありません．

private base：`private/xavier/eegmmidb_phase2_v1/`

- `prepared/manifest.jsonl` SHA `d673ffa0bc54506fc5866a7092fa5ff81deffdcfdf0754b10902cb6fda8657a9`
- `prepared/preprocessing_protocol.json` SHA `b8fc0c0def124e28e312b43818aecad68f7da2e5728c1833c7b18d25961cc1f4`
- `prepared/corpus_summary.json` SHA `0ff7ae2a74eed31341602b9506594fe75e917cb364d69a223fdce7fe55e01706`
- `prepared/completion.json` SHA `5041351f4fd80a80e4956034a7a479983cd921c6701062906a5dd68c5f30936c`
- `benchmark_execution_seal_v2.json` SHA `e1847ae39ef284a49cdc34dbe1695593b6d9a3c46b8d4db046842c403b26eb7f`（旧v1とCSPの選択結果は保存済み．互換receipt SHA `7c19d204274d0b219d284ff9363364a8e3b5c8d6d5a79ea569d18da68bab78aa`）

## phase2：計算・評価・回収

新sessionは `xavier-eegmmidb-20261122-p2`，remote rootは `/content/xavier_phase2` です．13:13 JSTに作成し，L4の実VRAM23,659,151,360B，空きdisk約253GBを確認しました．開始CU409.45，観測rate1.94 CU/hです．

登録予算 `gpu_budget_v1.json` SHA `49c5c108685fab74befc17085e99a98940674444cd792ceed323a1940c2bbc84`．絶対期限は `1795325994.596278`＝14:39:54 JST，90分／5CU，GPU memory21GiBです．fitは評価900秒・回収900秒・停止300秒を残す時刻で止めます．旧予算を起動し直して延長しません．追加ユーザー指示に対応する後続実行は，別の明示した予算と引継ぎreceiptを要します．

`operations/transfer/progress.jsonl`，`operations/transfer/*receipt.json` とworker PIDの開始時刻／argvを照合してください．約56MiBのpackごとにSHAと全memberを検査します．初回はremote incoming directory不足，再開時には一部CLI応答timeoutがあり，保存済みreceipt／現物を確認して未完了位置から再開しています．sessionのJupyter kernelを共有するCLI execは直列に操作します．

CSPは `runs/csp_lda_v1/` でtrain／val fit完了，CPU2threads，3.70秒です．左右のval subject BA63.73%／16 components，副課題57.53%／4 componentsを選択しました．これはtest性能ではありません．

neural local run予定は `runs/eegnet_scratch_v1/` と `runs/eegnet_smollm_fused_v1/` です．scratchは最大60epochs，fusedは最大30epochs．同じ主左右MI validationのsubject平均BAで選び，初期EEG事前学習重みを使いません．SmolLM2-1.7B revision `31b70e2e869a7173562077fd711b654946d38674` は凍結し，encoderと8prefix投影を学習します．

全best／閾値の固定後に，各モデルの全test・5条件・共通subset・T0診断を実測します．fusedのLLM routingは16種類の異なるpromptを実生成し，同一prompt cacheを使って各EEG試行の新しいin-memory状態を更新します．EEG判別誤りとrouting誤りを分け，実OS操作や自由意図の証拠にしません．[評価設計と進捗](eegmmidb_phase2/RESULTS.md)を参照してください．

全checkpoint，optimizer／RNG，validation履歴，予測，source／data hash，workflow終了，回収後の照合を保存します．試用はM1／最大8GiBでmetadata固定train3例を使い，held-out評価と区別します．同じrawやcheckpointを完了済みという理由なく再実行しません．

## phase3：追加学習の準備

科学protocol `eegmmidb_phase3/protocol.scientific.v1.json` SHA `984b516475b06f4e7641074596790c43f76e34d33b6f047239a4ddbd2d0d4016`．現phase2のtestを見る前に，fresh同seed encoder，50% channel-time patch mask，masked-only再構成MSE，AdamW，最大20epochsを固定しました．SSL validationで初期epoch0も選択候補に含めます．その後，同じMI6,359 train窓でsupervised学習し，同じSmolへ接続します．

trainの全14runsはreview除外後1,014記録／75群／32.171944hです．5秒warmupと3秒非重複窓による上限は36,600窓＝30.5hで，品質検査後の採用数は未確定です．SSL validationは既存16群／224記録から最大8,064窓＝6.72h．test17群／238記録はfitへ投入しません．最大派生float32 payload6,860,390,400BはColab側で生成する候補で，workstation35GiB reserveを維持します．

private `eegmmidb_phase3_v1/contracts/` のallowlist／denylist／元splitを使います．取得済み原ZIPをremoteへコピーするか，公式配布元から同一SHAで取得します．同じ内容を新たな取得EEG時間に加えません．CPU前処理は13:30:07に別登録し，14:00:07 JSTまで／8GiB派生出力／CPU2threadsです．receiptは private/xavier/eegmmidb_phase3_v1/cpu_resource_v1.json，SHA383b9eb38ce5452ff7312be955d4ab6008816377dac2feadd52826b07d80f994．追加GPU予算は `private/xavier/eegmmidb_phase3_v1/gpu_budget_v1.json`，SHA `018cdaa6c81eba94178ce8e465b89360e425691c4d083c5e25855b6d5c6d25a5`．絶対期限 `1795331083.750155`＝16:04:43 JST，phase3 GPU最大2時間，登録時408.93CUから最大5CUです．phase2 fit／評価／回収が全て終わってから所有権を移し，物理sessionが継続している事実を記録します．phase2の科学設定・fit期限は変えません．実行・時間／CU・保存／回収予算は最新receiptで確認してください．

## 取得状況と量

全catalog5,165行のcoverageは不足・余剰0です．metadata resolver5571は終了しました．固定metadata snapshot SHA `01efee3f07e76d22d1553211b7ee9d347d15b407be10330ccf79bcb086221b60` からappend02を実行し，174 releases／182,443 jobsを追加しました．旧468,040件を含む全650,483件の固定metadataを独立照合し，不一致0です．再appendは不要です．

現在の候補は783 releases／650,483 files，既知24,172,262,716,513B（約24.17TB）．サイズ未確定42,398 filesを含みます．動画等のcompanionや版の重複も含む候補総量で，取得完了量・純EEG量ではありません．[append02証拠](reports/fresh_queue_append_20261122_02/README.md)を参照してください．

取得完了snapshot004は40 unique contents／16,566,958,252B，header／payloadを数量検証したEEGは1,644記録／136.441909809hです．EEG sample部分は14,484,459,152B．既存39内容は前回full SHAと保全記録，新規EEGMMIDBは全archive／memberのfresh SHAを照合した観測です．

bulk PID8814は元deadline `1795322496.632661`（13:41頃），上限64,062,645,375B／reserve35GiBで継続しています．13:15頃のBCI Database.zipは23.49／27.48GBで未完です．partialを取得完了容量やEEG時間へ加えません．元worker終了・lease解放後に，最新空きと保存枠から新しい予算を明示して再開します．max-bytesはrun内受領量で，既存partial量を含む累計ではありません．他workerとの同じpartialへの二重書込みを防ぎます．

EEGMMIDB原ZIP SHAは `0ff5f7f38d1d14b8e485581ad4cf5b327e6494bc2ac7c7d7d4f7e9518a950cbe`（64文字）です．前段引継ぎ文の62文字転記誤りは修正済みで，元監査と実checksumは最初からこの64文字値です．取得原時間，品質通過窓，実際に勾配へ入力した区間union，epoch反復は分けます．

## 継続と公開

予約 `Automation_f439325b5d888191bc75e92d1ba97be7` は14:15:26 Asia/Tokyo，enabled=trueで，最新の追加量指示を反映済みです．APIにはProモデル固定指定がないため，Pro固定は保証していません．有効なcoordinator leaseと実jobがあれば，競合編集／二重起動をせず現進捗を観測してください．

1．coordinatorと所有jobのPID／argv／開始時刻／boot IDを照合します．
2．未完の転送→train-only容量測定→同じprotocolでの学習を進めます．
3．bestを固定し評価・回収を完了し，M1で試用します．
4．phase3は科学protocol・allowlist／denylistを維持し，別予算で実行します．
5．完成した新データを監査し，新snapshotの記録数・時間・量を保存します．
6．今回の成果をGitHubへpushし，最新CIと実験固有の検証を別々に記録します．

学校Google Doc [実行・検証記録](https://docs.google.com/document/d/1kGlNqvZEg_AzSYKGTiGF3Z0HcVIxNGNimNwpR6w1680/edit?tab=t.7cvfsge8pwil)の新tabへ追記し，旧2tabsを保持します．この進行中記録を完了報告として扱わず，結果と未達を実receiptから更新してください．
