# PJ-Xavier 次段階：多channel EEG＋事前環境の直接alignment（CPU着手）

更新：2026-11-22 16:24 JST．**phase1/2/3は完了済みで再実行していません．** このディレクトリは別の次実験向けです．新checkpointのGPU学習・評価・LLM実機試用は未実行です．

## 今回実行した作業

- 過去のfixed test／review holdと新sourceの公開監査，private原台帳を照合し，**1,121件**の記録metadataをprivateでindexしました．発行した184のmust-link componentは実人数ではありません．sourceはBCI GDF694，P300 DAT105，InMID EDF92，CNT26，compact TXT/FIF68，processed SSVEP44，Brainstorm10，小型BCI関連82です．同一配布物の異なる表現や同一波形の重複は独立量に加算しません．
- `admission.py`はmetadata-only split提案とcross-ID同一EEGのfold結合を行います．既存の76/16/17 splitと旧holdは維持します．**全1,121件をfit・validation gold・test gold未承認**として出力します．信号QC・単位・source間payload deny・著者flag・時刻・元実人確認が未完了のためです．AI weak label欄は独立で空，goldへ昇格しません．P300のTextToSpell／StimulusTypeはoracle側，SelectedTargetはdecoder出力側です．
- `causal_shards.py`は14/16/27/32/64ch向けに，raw chronological physical Vから3–40Hz causal SOS，必要な80Hz causal anti-alias，200Hz遅延補間，2秒warmup，連続chunk状態と3秒shardを実装しました．各出力が依存する最新のnative source sampleを保存できます．異なるmontage・reference・native fsは混合できません．真の欠損で状態を隠れて持ち越せません．このアルゴリズムが因果的でも**sourceが以前に非因果offline処理されたかは別問題**です．
- `ChannelQueryResampler`は可変C/Pの新encoder特徴から8 learned prefixへ集約する独立Torchモジュールです．14/16/27/32/64ch全形状でCPU backward済みですが，**CBraMod L4/BF16・LLM接続・実fitは未検証**です．既存8/32ch専用trainerと8ch専用adapterは変更していません．
- `joint_alignment.py`は公開cue A/Bと独立4配置からproxyのX/Yを構成します．train側4配置は元EEG1試行の4viewsであり，独立EEG量4倍ではありません．正解cue，target，participant，run，TextToSpell，未来TextResult等を環境入力から除外します．環境のみ／EEG除去／shuffle／実−10秒／mapping同変性／T0／正解class routing上限をbenchmarkに列挙しました．**自由意図・任意画面理解・実OS成功は評価対象ではありません．**
- 15 focused CPU tests PASS（streaming chunk等価性・no-future・形状別逆伝播・must-link・context denyを含む）．さらに元SHAを検査した**新sourceの16ch FIFの先頭4,096 native sampleだけ**を読み，3秒shard2件のCPU処理を試しました．この試用は提案train群の非学習safety smokeであり，QC合格・online causality・gradient exposure・holdout evaluationを意味しません．preflightの個別証拠はprivateに保存しています．

## 容量と実行gate

16:17時点のworkstation APFS Dataのavailableは**39,075,487,744B**．35GiB reserve＋1GiB回収枠を維持すると，未割当headroomは**420,782,080B**です．新規allocation上限をさらに64MiBにしています（都度再計測）．追加の書込可能な永続volumeは確認できていません．**大量download，全展開，新しいColab sessionとGPU学習は起動していません．** 旧physical sessionは停止済みのままです．容量減少原因をswapと断定していません．原EEG・optimizer・RNG・重み・旧成果物は削除していません．

## 封印と残る実行順序

固定CPU段階protocol `protocol.cpu_stage_v1.json`，benchmark `benchmark.next_joint_v1.json`，集計 `../reports/next_multichannel_admission_20261122_1615.aggregate.json`．個別記録manifest・source SHA・storage計測・失敗とsmokeは `private/xavier/phase4_automation_1615/` に保存しています．GPU budgetではありません．

次回は既存leaseと実jobを再確認し，train群の原source identity・cross-source payload denyを照合してから，rawでphysical-V証明できるrecordのsource別QCをtrain-onlyで固定します．新しい未使用group testを先に確保し，新scientific protocolとbudget，最大形状のreal backward/CU/disk recovery probeを封印してからColab CLIを**workstationから直列使用**します．全checkpoint validation→best固定→独立test/負対照→全回収SHA→M1試用→physical停止を新receiptに残してください．今回のmetadata-only提案splitをそのまま独立評価の証明と扱わないでください．

再実行コマンド（git repo root）：` .venv-xavier/bin/python -m unittest discover -s experiments/pj_xavier/next_multichannel -p 'test_*.py' -v `．公開資料には個人別EEG・参加者ID・予測・重み・認証情報を含めていません．
