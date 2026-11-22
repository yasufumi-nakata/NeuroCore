# PJ-Xavier 実行・引継ぎ

## 最新の到達点：S Macで取得と学習を実行済み（2026-11-22）

ユーザー最新指示「取得できるデータ全部使って学習進めて、S Macの方で」に基づき，新規の取得・保存・展開・学習はS Mac側で行います．smac19は除外します．workstationは接続，コード，小さな運用台帳に使用します．公開データのraw／preparedをworkstationへ一時中継せず，Colab学習も再開しません．

**正規認証を確認した38台のうち37台で分散取得を開始し，smac01では追加sourceのCBraMod自己教師学習を10epochs／110steps完了しました．** smac31は管理用，smac40はDNS未解決で割当なしです．

| 状態 | 実績 |
| --- | --- |
| 割当済みの取得候補 | 598,319 files／23,069,939,036,185 bytes，全modalityの候補 |
| live取得完了 | 5,908 files／21,313,810,836 bytes（19:07:50 JST snapshot） |
| 取得途中 | 3,228,565,504 bytes，同時点，完了へ非加算 |
| worker | 37台でsupervisor・実取得を確認，完了ファイル欠損／サイズ不一致0 |
| 初期学習用source | Zenodo806023，8 files／138,420,368 bytes |
| 準備・実学習量 | train659窓／1,977秒＝0.5491666667h，val327窓／981秒 |
| 実学習 | MPS native，batch64，10epochs／110steps，反復提示6,590窓 |
| 固定validation | 初期masked MSE0.002140255476 → best0.001922294883 |
| 独立検証 | 全210 trainable tensors・12 encoder層の更新，optimizer step110，val再計算を33項目で確認 |
| 次段階 | 2392979の8ch追加準備をsmac06へ配備，rawはsmac08からcluster内直接転送 |

取得候補・実取得・prepared・gradient exposureを混同しないでください．MSEはoffline波形再構成のvalidationで，行動性能や未知sourceへの汎化を実証する指標ではありません．初期の実入力0.549hと，別lineageのphase3選択モデル29.985888889hは足しません．

[現実行reportとhash](reports/smac_training_20261122_v1/README.md)を参照してください．以前の全文は[移行前HANDOFF snapshot](reports/handoff_snapshots/HANDOFF_20261122_before_smac_fleet.md)に保持しました．その中の認証未成立・新source未学習・Colab再開案は当時の記録です．最新の実job，private coordinatorと本引継ぎを優先します．

## 作業場所，所有権と接続

| 項目 | 場所 |
| --- | --- |
| 正worktree | `/Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training` |
| repository／branch | `yasufumi-nakata/NeuroCore`／`feat/pj-xavier-validation-20261122` |
| Draft PR | [71](https://github.com/yasufumi-nakata/NeuroCore/pull/71) |
| coordinator | `private/xavier/eegmmidb_phase2_v1/coordinator.json` |
| workstation transport | `private/xavier/smac_execution_20261122/smac_transport.py` |
| smac31 controller | `/Users/yasufumi/pj-xavier/control_20261122` |
| verified inventory | controller配下の `node_preflight.json` |
| 各worker root | `/Users/yasufumi/pj-xavier/smacNN` |
| raw／DB | worker rootの `raw`／`downloads.sqlite` |
| task Python | smac01／smac06の `venv/bin/python` |
| 運用証拠 | `private/xavier/smac_execution_20261122/`，取得担当のprivate運用dir |
| policy／registry | `private/xavier/storage_cleanup_20261122/` |

有効owner/leaseと実PID/argv/bootを確認します．rootまたは別ownerが実作業中なら観測・記録だけ行い，コード変更や二重起動をしません．owner解放後は排他的leaseを取得し，既存download jobを維持しながら独立した次の学習準備を進めます．全download完了まで待つ必要はありません．

workstationの通常 `python3` からtransportをimportし，`connect()` はsmac31，`run_node(node, code, timeout=..., task_python=True)` はworkerのtask venvを使用します．task_python省略時は既存cluster Pythonです．通常venvへparamikoを追加する必要はありません．

OpenSSH ControlMasterはMCP実行境界でハングしたため使用しません．smac31の既存credential loaderと観測済みhost keyを保持し，秘密値は出力・保存・コピーしません．smac31のAGENTSとsmac-cluster skillに従います．system python3がXcode license要求を返すworkerでは既存standalone cluster Pythonを明示します．license変更やsudoは不要です．venvのPython pathは `.resolve()` せず，venv pathのまま起動します．

## 全量取得の実運用

controllerの `acquisition/full_fleet_plan.json` は容量・公開条件・pairing・衝突を確認した候補を37台へ割当済みです．`planning_receipt.json` と `source_blocker_summary.json` に根拠を保存しています．未解決metadata784件，path衝突13件，容量不明42,028 groups，EEG scope未確認2,924件は未割当であり，引き続き解決します．権利や単位の不明点を推測で承認しません．

観測はcontrollerの `acquisition/snapshot_acquisition.py --name <new_receipt.json>` で行います．最新の運用memo／safe resume controllerを優先し，node DBと実ファイルを照合してください．

各nodeは `download_supervisor.py`，`download.json`，`download_state.json`，`bulk_supervisor_state.json`，`download.log`，`bulk_supervisor.log` を保持します．1node1stream，100GiB空きreserve，600秒cycle，supervisor起動から48時間の枠です．429のRetry-Afterとbackoffを尊重し，TLS検証を無効化しません．Dataverseのguestbook217必須releaseは同一release全体を既存access_action_requiredに沿って保留し，フォームを送信せず次の公開sourceへ進めました．

再開時は生存PIDとlockを照合し，既存DB／manifest／partを保持してsupervisorだけを安全に継続します．期限終了後の無条件re-deployは既取得bytesを容量へ二重計上し得るため避けてください．node rootの `STOP_DOWNLOADS` は現在cycle後に停止する仕組みです．新しい期限・空き容量・実jobをreceiptへ記録して再開します．

## 完了した初期CBraMod実学習

smac01内のcorpusは `prepared/smac806023_ssl_v1`，runは `runs/smac806023_ssl_20261122_v1/run` です．実行codeは `training_code_v1` に独立保存し，取得code更新の影響を受けません．

- 公式CBraMod commit `b9e961003214326972c567eff390e75b0287e32a`，公式weightsから開始しました．
- 有効checkoutは `upstream/CBraMod_pinned` です．元の不完全な `upstream/CBraMod` は使用しません．
- test2記録は波形を開かずdenyへ保持．元file SHA，physical-V payload，ordered montage，単位，split，固定QCを照合しました．
- 16ch／3秒／200Hz／float32 V_x_1e4，causal追加処理．配布sourceの過去のoffline因果性は未確認で，online_harness_allowed=falseです．
- 実probe batch1／8／32／64が全PASS．本学習は公式weightsに戻してbatch64，10epochs，110stepsを実行しました．probeの学習済み重みを本学習に流用していません．
- best checkpoint SHAは `3450e90e0c2cdf8b0b7f2fa28d39ae5037f7d9b70fea95cc5d93a885d98e6d10`．
- training protocol SHAは `51a5c1b15f407405da0e7d9eb58f60f23978c9008e21ff4b2bf649de42048f0b`．
- 全210 trainable tensor，12 encoder層，AdamWの有限非ゼロmomentとstep110を独立確認．val327窓のみno_grad再計算し，初期・bestの保存MSEを再現しました．新testアクセス0，新training step0．
- private receiptは `stage1_encoder_verification_20261122_v1.json`．

最初のprobe attemptはupstreamのgit metadata欠落でforward前に停止しました．git検証を回避せず，正規shallow repositoryを転送したattempt2で完了しています．失敗の証拠を削除しません．

## 次に実行すること

1. coordinatorと実jobを確認し，最新stage2 stateを読みます．完了runを再実行せず，進行中jobを維持します．
2. `next_multichannel/SMAC_SOURCE_ADDITION_RUNBOOK.md` に従い，2392979の取得完了をsmac08で確認し，smac06へcluster内直接コピーします．36file／110,071,616Bを二重downloadしません．
3. smac06の `training_code_v2` で `prepare_smac_ssvep.py` を実行します．base806023 corpus・parent run・公式upstream・weightsはクラスタ内コピーして全SHA照合します．旧test10記録と重複alias5を開かず，21train候補から固定QCで選びます．native source guardを通り得る5記録もwindow QCは別です．全落ちならblocked receiptを保存して次sourceへ進み，単位倍率・閾値を調整しません．
4. 適格な新corpusは新protocolでregisterし，`--parent-run` で選択された直前bestを引き継ぎます．新C/Pを含む全bucketの実MPS capacity probe後にfitします．旧val/test/review deny，must-link，source/payload/native intervalのunionを継承し，baseの反復を新規固有量に加算しません．
5. InMID／BCI8089820／CNTなど，取得とsource監査が完了したものから次adapterへ進めます．EDF/BDF/GDF/CNT/SET等の単位・時刻・native範囲・校正・duplicate制約を確認します．元label，観測event，weak labelを区別し，不明なlabelを評価goldへしません．
6. EEG→LLM／判定前環境→X/Yは既存の新protocol設計を引き継ぎ，未使用holdoutと負対照を固定して別段階で実行します．P300のTextToSpell／StimulusTypeは教師oracle，SelectedTargetはdecoder出力，未来TextResultは事前contextにできません．既に見た17test群へ最適化しません．
7. 公開コード・aggregateだけcommit/pushし，exact commitのCIを確認します．raw／個別ID／個別予測／weights／認証はprivate．既存Google Docの実行tabへ追記しreadbackを確認します．最後に実jobを記録してleaseを解放します．

## 過去run，記録と継続

phase1～3は学習・固定評価・回収・M1試用・Colab停止まで完了しています．phase3の選択モデルはfresh小型encoder lineageで，CBraModと混同しません．主左右BAはphase3 EEG62.45%，fused56.72%，同17test群を使う探索比較です．自由意図や実OSの成功は未実証です．過去の完整な条件・制限・CI・回収hashは[移行前snapshot](reports/handoff_snapshots/HANDOFF_20261122_before_smac_fleet.md)，[phase3結果](eegmmidb_phase3/RESULTS.md)に保持しています．

workstationのraw／colab_raw_archive773files／50,476,724,659 bytesは依頼どおり削除済みで，学習済みモデル／optimizer／RNG／評価／コード／台帳は保持しています．削除証拠は `private/xavier/storage_cleanup_20261122/` です．

既存automation `Automation_f439325b5d888191bc75e92d1ba97be7` は毎時15分26秒（Asia/Tokyo）です．promptはS Macだけで取得と学習を進める実経路へ更新しました．稼働workerがある時も，重複を避けつつ取得済みsourceの学習を前へ進めます．未解決候補のmetadata調査も継続し，進捗の要約だけで終了しません．

[Google Doc実行tab](https://docs.google.com/document/d/1kGlNqvZEg_AzSYKGTiGF3Z0HcVIxNGNimNwpR6w1680/edit?tab=t.7cvfsge8pwil)と旧tabsの記録を保持します．
