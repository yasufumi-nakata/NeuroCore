# S Macでの追加取得とCBraMod自己教師学習

## 到達点

2026-11-22に保存先と新規学習先をS Macへ統一し，正規認証を確認した38台のうち37台で分散取得を開始しました．smac31は管理用，smac19はユーザー指定で除外，smac40はDNS未解決で割当なしです．workstationへのraw一時中継は行っていません．

Zenodo 806023の取得・固定QC・準備をsmac01で実行し，公式CBraModの全encoderを使うMPS自己教師学習を10epochs／110steps完了しました．これは追加データの最初の段階です．全取得候補の学習完了を意味しません．

## 量の区別

| 指標 | 確認値 | 範囲 |
| --- | ---: | --- |
| 割当済み取得候補 | 598,319 files／23,069,939,036,185 bytes | 公開条件・既知容量・pairingを確認した全modalityの候補 |
| 取得完了 | 5,908 files／21,313,810,836 bytes | 2026-11-22 19:07:50 JSTの37台live DBと実ファイル |
| 取得途中 | 3,228,565,504 bytes | 同観測時点のpart，完了量へ非加算 |
| 完了ファイルの欠損／サイズ不一致 | 0／0 | 上記snapshotで検査した範囲 |
| 初期学習source | 8 files／138,420,368 bytes | Zenodo 806023，checksum・元監査SHAと波形payloadを照合 |
| 準備済みtrain | 659窓／1,977秒＝0.5491666667時間 | 非重複3秒窓，4記録 |
| 準備済みvalidation | 327窓／981秒 | 2記録，学習には非投入 |
| 実train入力の固有量 | 659窓／1,977秒 | モデルへ実際に提示したIDと区間を記録 |
| 反復提示 | 6,590窓 | 10epochs，固有量へ重複加算しない |

取得完了bytesはEEGのみの量や独立EEG時間ではありません．候補metadataが未解決の784件，provider/path衝突13件，容量不明42,028 groups，EEG scope未確認2,924件は保留理由を記録しており，取得済みと数えていません．カウント単位は各台帳に従い，異なる単位を合算しません．

過去のworkstation取得760 unique／49.54GB／連続EEG255.33hは履歴です．ユーザー指示により773ローカルファイル／50,476,724,659 bytesを削除済みで，新しいS Macの保持量へ重複加算しません．既存モデル・評価・台帳は保持しています．

## 学習条件と実測

| 項目 | 値 |
| --- | --- |
| run | smac806023_ssl_20261122_v1 |
| 実機 | smac01，arm64，64GiB RAM，MPS native |
| モデル | 公式CBraMod，upstream commit b9e961003214326972c567eff390e75b0287e32a |
| 入力 | float32，16ch，3patches × 200samples，200Hz，V_x_1e4 |
| 最適化 | AdamW，LR 1e-4，batch64，10epochs／110steps |
| 実capacity probe | batch1／8／32／64，全shapeのbackward・有限非ゼロgradient・memory guardを通過 |
| 初期validation | subject/group平均masked MSE 0.0021402554759976803 |
| 選択checkpoint validation | 0.0019222948831695787，epoch10／step110 |
| fit timer | 18.61261525005102秒 |
| sampled memory | 868,352,000 bytes，40GiBの登録枠内 |
| test入力／behavior label | 0／0 |

小さい初期corpusでの波形再構成validationです．この低下は行動分類，EEG→LLM，未知sourceへの汎化，自由意図推定や実OS操作の改善を実証するものではありません．MPS memoryはサンプル観測で，true peakではありません．

公式checkpointから開始し，probe後はprobe weightsを継承せず公式初期値へ戻して本学習を行いました．旧phase3の小型encoderとは別lineageです．旧phase3選択モデルの入力union29.985888889hを，このCBraModの入力時間へ加算しません．

## 分離・品質・因果性

既存metadataのtrain/validation/testを維持し，test2記録は波形を開かずdenyへ保持しました．元file SHA，physical-V payload SHA，ordered channels，native sample rate，校正，重複，既存review/test denyを確認します．固定100µV peak／0.01µV channel標準偏差のwindow QCを適用しました．

追加フィルタと512→200Hz処理はcausalですが，配布FIFの過去のoffline処理の因果性は未確認です．従ってoffline SSLへ限定し，online harnessへの適格性を認めていません．

source・manifest・artifact・admission・protocolのhashを固定し，別sourceの追加は新runへ分けます．同じvalidation/testを後続trainへ移しません．次段階の2392979は8chの別bucketとし，既存16ch validationを保持します．そのvalidationは806023ドメインの指標であり，2392979汎化の検証には使いません．QCで適格窓がなくなった場合は明示的に保留し，単位倍率や閾値を調整して通過させません．

## 保存と運用

raw／DB／download logは各nodeの `/Users/yasufumi/pj-xavier/smacNN/` 配下です．初期corpusとrunはsmac01の `prepared/smac806023_ssl_v1` と `runs/smac806023_ssl_20261122_v1/run` にあります．controllerはsmac31の `/Users/yasufumi/pj-xavier/control_20261122/` です．

取得workerは各node1stream，100GiB空きreserve，600秒cycle，起動から48時間の監視枠です．429のRetry-After/backoff，アクセス条件，TLS検証を維持します．取得失敗・partialをcompleteへ変更しません．同じDB/manifestを保持して安全に継続し，二重workerは起動しません．毎時の既存automationもS Macの実jobと台帳を引き継ぎます．

smac01の `upstream/CBraMod_pinned` は.git/shallowを含む検証済みcheckoutです．初回capacity attemptはgit metadata欠落でforward前に停止し，sourceチェックを緩めず実checkoutを転送してattempt2で成功しました．失敗証拠は保存しています．

## 実装

- `model/cbramod_ssl_smac.py`：8/14/16/27/32/64chのbucket，MPS probe，固定validation，checkpoint/optimizer/RNG，parent-run lineage，実入力interval計数．旧sealed trainerは無変更．
- `next_multichannel/prepare_smac_fif.py`：初期16ch sourceのSHA・単位・split・QC照合とcausal shard．
- `next_multichannel/prepare_smac_ssvep.py`：must-link／重複alias／既存test denyを継承する8ch追加．
- `fleet/fleet.py`：storage guardとpolicy/registryをrepo-relative配置でbundleへ同梱．
- `acquisition/acquire.py`：存在しない0-byte partの早期verifyを防ぐ修正．

## 主要SHA256

| 対象 | SHA256 |
| --- | --- |
| training protocol | 51a5c1b15f407405da0e7d9eb58f60f23978c9008e21ff4b2bf649de42048f0b |
| trainer | ad0d8c27352411db52640c077af4b4bb38c6b1652ea20f3e6e965af2c172ca8a |
| official weights | 0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178 |
| selected best.pt | 3450e90e0c2cdf8b0b7f2fa28d39ae5037f7d9b70fea95cc5d93a885d98e6d10 |
| prepared manifest | 58d1b9f31572b224cabeda91942aedaba382d5b5f63c9c980db2d18ae95f2d7b |
| prepared artifacts | 6dbda123e7c9ed1c174a8b984ca6e9aa97903ce3bb78cf5b1cc0874a4130b2cf |
| source admission | 000530f58f55d6c17bb239a50e8bb55a1cd170704131bcbc9e15bd52f22f0233 |
| full fleet plan | 0c448af11bbfcedf3e29c79b0d13222ea0de1ed518439a7378c7f3a817cb230a |

原EEG，個人別ID，weight本体，認証情報は公開reportに含めません．最新の継続状態は[HANDOFF](../../HANDOFF.md)とprivate coordinatorを参照してください．

## 独立検証

保存checkpointをread-onlyで公式weightsと比較し，33項目すべてを通過しました．210個すべてのtrainable tensor，12encoder層，入力・spectral・位置・出力projectionに更新を確認しています．4,923,800 trainable要素を持つmodelの全AdamW momentは有限非ゼロで，stepは110です．意図的に凍結されたmask encodingは不変でした．

固定val327窓だけをeval／no_gradで再計算し，初期とbestのMSEを保存値と一致させました．相対低下率は10.1838586689％です．新training step0，test window0，best checkpointの変更0です．次段階の実read_parent契約も受理されました．個人別結果を含めない集計は [independent_verification.json](independent_verification.json) にあります．

焦点テストは取得34，fleet17，trainer9，初期prep6，追加prep7の計73件をrootが再実行し，全件PASSしました．実MPS／実dataの検証は上記の別receiptです．
