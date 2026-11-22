# Figshare 31578517 v2：固定 split による S Mac 追加 SSL

## 対象と完了条件

対象は [Design Emotion Dataset v2](https://figshare.com/articles/dataset/Design_Emotion_Dataset/31578517/2)（DOI: 10.6084/m9.figshare.31578517.v2，CC BY 4.0）の配布 EDF です．25 participant group の metadata を先に固定し，17 train の全候補を検証します．適格な窓を，完了・検証済み BCI8089820 stage の累積 corpus とモデルへ追加します．

本書は実行契約です．実行結果は該当 stage の preparation / training / verification receipt で判定します．metadata の native 時間，QC 後の独立 EEG 時間，実際の gradient exposure 時間を区別します．

## Source contract

公開 landing description には 64 channels / 200 Hz とありますが，同じ配布物の README は **32 channels / approximately 1024 Hz** と訂正しています．全 25 EDF の header と実サイズは後者に一致します．version 2 の provider metadata と，header-only 証拠を固定して採用しています．

| 項目 | 固定した契約 |
|---|---|
| EDF / participant metadata group | 25 / 25，1 EDF ごとに distinct participant token |
| 実 EEG channels | 32，配布 header の順序を保持 |
| 実 sample rate | samples per record / record duration を有理数として保持 |
| rate の metadata 範囲 | 1024.0～1024.000835071 Hz |
| native 時間，全 25 EDF | 8.8587458333 h，QC 前の metadata 値 |
| EDF physical unit | uV，digital / physical endpoint を使って float64 の V へ変換 |
| reference | publisher が REST / average と記載．独立検証や追加 rereference は行わない |
| source の過去 preprocessing | 独立検証していない |
| 利用範囲 | offline masked SSL |
| 使用しないもの | 質問票，心理指標，events / outcome labels，publisher の実行 script |

「独立 group」は publisher の 25 complete participants 記述と，25 unique filename token / source SHA / provider MD5 に基づきます．生物学的な独立性や，別 dataset との未記載 participant linkage を独立検証したとは主張しません．個別 token，file path，questionnaire 値は公開集計へ含めません．

## 信号読込前の metadata freeze

freeze_smac_design31578517_metadata.py は provider metadata と全 header の seal・版・アクセス条件・ライセンス・サイズ・校正・一意性を確認します．EDF payload，質問票，event table は開きません．

source version と participant token の SHA-256 を辞書順に並べ，先頭 17 group を train，次の 5 group を val，残る 3 group を test とします．全件を含む immutable plan と freeze receipt を作成します．

- 固定日時：**2026-11-23 05:23:16.199517 JST**
- plan SHA-256：42e0c2470d1415661a2aa6a0a493375788349388095cacd6b79c3ddc3e5d73d7
- provider metadata SHA-256：5e68ee6e2b712b037282989dba7723f80ef30b2c1990a78fd521f312555a0822
- header / README evidence SHA-256：70290db281b300fadb9cf790d510c69482b330adede35d78fe8cc18ac7ad1185
- freeze implementation SHA-256：d4d1b03daa54878d4064df263d2107d445401a16c69d3d52497a33deae4d5f6a

元の subject token と EDF path を含む plan は private 管理です．val 5 / test 3 の信号は今回の prep / fit で開きません．

## Train-only EDF preparation

prepare_smac_design31578517.py は immutable plan と post-BCI corpus を要求します．BCI source が含まれない古い corpus からの分岐を拒否します．

各 train EDF について，全 source SHA-256，provider MD5，header SHA，size / channel / calibration を再検証します．16-bit digital 値の差分演算に入る前に float64 へ変換し，宣言された microvolt を V に変換します．全 32 channels の先頭・中央・末尾の 3 区間を MNE の独立読取と照合します．最大差の許容値は 1e-11 V です．full integer payload と sample-major float64 V payload の SHA-256 も記録します．

追加処理は既存と同じ固定条件です．

1. 各 original record の初期状態をゼロとし，forward-only Butterworth 4 次 SOS の 3–40 Hz bandpass を適用します．
2. 80 Hz lowpass 6 次 SOS を追加します．
3. 元 header の正確な rate 比により cutoff index を求め，1 native sample 遅延した補間で 200 Hz にします．未来の native sample を使いません．
4. 2 秒の warmup 後，重なりのない 3 秒窓を作ります．出力形状は [N,32,3,200]，単位は V × 10^4 です．
5. native peak 0.1 V，変換後 peak 100 µV，channel ごとの minimum standard deviation 0.01 µV を固定判定します．unit の推測修正，clipping，閾値の緩和は行いません．

この追加処理の因果性は，publisher が既に実施した preprocessing の因果性を保証しません．online 利用可否は一貫して false です．

適格な追加窓が 0 件の場合は，具体的な QC 理由を receipt に保存します．trainable completion や base copy を作らず，その stage の学習を開始しません．技術エラーや source / seal 不一致を QC 全落ちとして扱うことはありません．

## 累積 corpus とモデルの継承

親 corpus の completion と必須 metadata に加え，存在する certificates / split manifest / parent provenance の SHA を検証します．親 admission，protocol，completion，旧 partition / certificate 証拠は新しい parent_provenance.json に snapshot し，その SHA を completion に結び付けます．

旧 raw / payload deny を保持し，旧 source の original integer / scalar identity も重複照合へ含めます．excluded_source_groups と excluded_subject_groups を統合し，祖先の val / test / review group を保持します．明示 train の exact-duplicate alias や一部 QC hold は，その participant 全体を heldout group に昇格させません．

既存 806023 validation は値と所属を保持します．今回の source val / test を model selection に使いません．新 source 全体の一般化性能を，既存 806023 validation から推定したとは主張しません．

実行時には以下を確認してから，親 corpus と best.pt 一式を S Mac 内で転送します．

- BCI run の status=completed，非ゼロ step，finite / nonzero gradient，登録 bucket の全観測．
- checkpoint / membership / corpus completion の SHA が一致．
- 独立 model verification が status=passed，すべての checks が true．
- 独立証拠の before / after checkpoint と corpus completion の binding が，選択した親へ一致．

checkpoint と prepared array は S Mac 間のメモリ経由で転送し，全 source / destination SHA を照合します．workstation に payload を戻しません．

## 実行資源と probe

task process の FD soft limit は hard limit の範囲内で 4096 以上にします．同じ node の既存 train_gpu.lock を取得し，別 trainer / probe が動いていないことを確認します．

すべての montage / native-rate / channel bucket に対して，batch 1 / 8 / 32 / 64 の native MPS backward probes を実施します．probe の source / protocol seal，finite / nonzero gradient，全 bucket coverage を確認します．正規の memory_within_budget=false capacity failure だけを候補落ちとして扱い，技術エラーでは停止します．

合格した capacity report を使い，**新しい最大 3600 秒・10 epochs・40 GiB memory budget** の追加 stage を登録します．probe の更新済み weights は fit の初期化へ使いません．fit は検証済み親 best.pt から開始します．この stage の登録 seed は recipe に 20261122 と明記します．親子の指標比較では実際の各 protocol の seed を読み，評価 mask と条件を揃えて再計算します．異なる seed の過去 summary を直接比較しません．旧 DB・raw・既存 stage を再作成しません．

## 2026-11-23 の実判定結果

**17 train EDF の全件審査は完了しましたが，固定 QC を通過した窓は 0 件でした．この source を追加した学習は開始していません．** 最終 stage は 2026-11-23 06:25:50.200240 JST に blocked_no_eligible_new_source_windows で終了しています．結果の数値と SHA は [公開集計 receipt](../reports/smac_design31578517_qc_20261123_v1.json) に保存しています．

| 対象 | 最終結果 |
|---|---:|
| frozen train の信号読込 | 17 / 17 records |
| frozen val / test の信号読込 | 0 / 8 records |
| 審査した 3 秒窓 | 7,130 |
| 変換後 peak > 100 µV | 7,062 窓 |
| channel std < 0.01 µV | 68 窓 |
| 採用窓・追加学習時間 | 0 窓・0 h |
| probe / fit | 0 / 0 |
| trainable completion / prepared waveform | 作成 0 / 0 |

新 corpus 側へ親 waveform を複製せず，親 BCI checkpoint と corpus は保持しました．metadata-only の固定 plan，親 provenance，25 件の判定 certificate，非適格の理由は保存しています．raw の変更，質問票や心理 label の読込，val / test の信号読込は行っていません．

### 数値条件の切分け

追加の読取診断は frozen train 17 件だけを対象とし，元 source / integer / physical payload の SHA と，7,062 / 68 の QC 判定を再現しました．17 件 × 3 区間 × 32 channels の manual physical-V 値と MNE の最大差は **3.47 × 10^-18 V** でした．この検査で parser / 宣言単位換算の不一致は見つかっていません．アンプ校正そのものを独立検証した意味ではありません．

元 integer samples の **693,197,509 / 703,004,992（98.6049%）** が，EDF digital endpoint −32768 または 32767 と一致していました．file ごとの割合は 94.4675～99.9856% です．端点への強い集中は確認できましたが，収録・export 等のどの段階で生じたかは未確定です．

| 数値 | 全 train の範囲 |
|---|---:|
| native physical min / max | −0.00983 / +0.00983 V |
| file ごとの native std | 0.000140458～0.002697667 V |
| 固定処理後 min / max | −0.019382369 / +0.018473034 V |
| file ごとの固定処理後 std | 0.000038433～0.000454258 V |
| model scale（処理後 V × 10^4）の min / max | −193.823693 / +184.730340 |
| file ごとの model scale std | 0.384327～4.542584 |

min / max は全 17 records の最小・最大です．std の範囲は各 file の全 channel・全時刻に対する母標準偏差の最小・最大であり，17 files を連結した pooled std ではありません．model scale は数値比較のために計算したもので，採用 waveform として保存していません．

配布 README は ±150 µV を超える epoch を除外したと記載していますが，今回の delivered EDF 値とその記載の整合は確認できません．同梱 pre.py には MNE 読込，filter，average reference，ICA，200 Hz resampling，EDF export があり，追加の unit correction / normalization はありません．実 header は約 1024 Hz であり，同 script が配布データへ実行された証拠にもなりません．同 script は実行していません．根拠のない単位変更，振幅縮小，clipping，固定 QC の緩和は行わず，この source の現版は非適格として保持します．

別実装による metadata-only の [独立判定監査](../reports/smac_training_20261122_v1/Design31578517_blocked_independent_verification_aggregate_v1.json) も **42 / 42 checks PASS** となり，元判定・全件計数・17 / 5 / 3 の固定・heldout 未読込・probe / fit 0 が一致しました．この独立監査で raw を追加読込していません．

数値診断は 06:29:32.158074 JST に完了しました．診断 result SHA-256 は ebe92454a8602f4e669556c6b5d9a372d75b7d685284468b429593dfaa95d25d，preparation result は 924cd5dc02b498e9fe79f5bcf9741ea18275472951a094fd82bc1fd8af2d0f76，certificates は 5376f04b155ada1c2d65a4e2eb5e8941bd4f569045288677c738056090ac8f30 です．

### 実運用の保存場所と失敗履歴

最終 controller は smac31 の ~/pj-xavier/control_20261122/design31578517_attempt4，子 stage は smac10 の ~/pj-xavier/smac10/runs/smac_BCI_plusDesign31578517_ssl_20261123_v2 です．審査結果は同 node の prepared/smac_BCI_plusDesign31578517_ssl_20261123_v2 にあります．いずれも科学的停止を記録して終了済みです．code snapshot は training_code_design31578517_v2，private runner は tools/design_node_runner_v2.py です．stage_config SHA-256 は 50c2933bcff5c9e919234192b61da173f598eec4487fb81c732e1c7461742726 です．

過去の attempt2 / attempt3 と child stage v1 は保持しています．前者は起動用 Python の改行 escape による parse failure，後者は deny JSON 末尾の serialization error で，どちらも新 source の信号読込・fit 開始前に停止しました．修正後は新しい namespace を使用し，元の BCI corpus 616 files，親 run 8 files，独立証拠 1 file を両 node の fresh SHA inventory で再照合して再利用しました．追加転送は 0 bytes です．source adapter，metadata split，親 checkpoint，seed，QC は変更していません．

## 検証

~~~sh
.venv-xavier/bin/python -m unittest discover \
  -s experiments/pj_xavier/next_multichannel \
  -p test_freeze_smac_design31578517_metadata.py -v

.venv-xavier/bin/python -m unittest discover \
  -s experiments/pj_xavier/next_multichannel \
  -p test_prepare_smac_design31578517.py -v
~~~

metadata freeze 5 件と EDF adapter 12 件，計 **17 synthetic tests** が PASS しています．主な確認対象は，metadata 順序に依存しない固定 17 / 5 / 3，実 rate の保持，元 hash / unit / geometry 改変の拒否，3 区間 MNE 校正，未来 sample の非参照，固定 QC，全 train 準備，val / test 8 files が存在しない状態での非読込，旧 validation の不変，旧 group deny / scalar identity / provenance の保持と改変拒否です．

private controller / runner / deploy helper には別途 13 件の tests があります．不完全な親，独立 verification binding 不一致，technical probe error，旧 validation を誤って deny に入れる projection に加え，全 JSON / PID writer の実ファイル roundtrip，state JSONL，埋込 Python の compile，実生成 launcher のダミー子起動と重複拒否を確認しています．実 source の QC 成否と学習成果は，これら synthetic tests とは別の実行 receipt で確認します．
