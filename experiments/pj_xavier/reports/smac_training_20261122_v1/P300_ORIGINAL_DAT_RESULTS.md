# S Mac P300 original DAT追加・独立検証（2026-11-22）

PJ-XavierのCBraMod系offline masked EEG再構成SSLに，OpenNeuro `ds005028`の**オリジナルBCI2000 DAT**を新規追加しました．既存のphase1～3，CNT，InMIDの封印済みrunは再実行していません．今回の結果は脳波から自由意図やOS行動を解読した結果ではありません．

## 実行結果

| 項目 | 実測 |
| --- | ---: |
| 元配布ファイル | 216件，1,563,795,662 B（smac05にて完了確認） |
| 元DATと派生EDFの区別 | オリジナルDAT105記録を監査．同じ波形のEDFを時間に加算しない |
| 凍結メタデータsplit | train77，val9，test19 |
| 著者の分析除外 | train中の12記録（sub-05）を保留 |
| 今回読込・学習した新しいDAT | 65記録，32 EEG ch，native 256 Hz |
| 固定QC合格の新規窓 | **4,997×3秒＝14,991秒＝4.164167時間** |
| 累積CBraMod系統の固有train | **25,876窓／77,628秒＝21.563333時間** |
| 新runの学習 | Apple MPS float32，10 epochs，4,070 optimizer steps，batch64 |
| 学習提示数 | 258,760窓（反復量，固有時間に非加算） |
| 固定806023 validation327窓・最良MSE | **0.0015065842255285976** |
| 直前InMID親の同validation MSE | 0.0016149027567592268 |
| 今回P300の独立val/test | **未評価・信号読込なし** |
| 新sourceの自由意図／OS作業／LLM接続精度 | **未実証** |

上記のvalidationは既存のZenodo806023被験者群だけです．P300への汎化，統計的有意差，オンライン運用での性能向上を主張しません．著者が分析に使用しなかったsub-05も今回の学習から除外しました．

## 複数S Mac間の実実行

smac05に取得済みの配布物から，**train候補のオリジナルDAT65件だけ**をsmac35へcluster内ストリーム転送しました．送受信の元ファイルを全SHA256で照合し，workstationにはrawを中継・保存していません．val9／test19／著者除外12のDATは開いていません．最初の転送controllerのcode-install処理は文字列置換誤りで停止しましたが，raw転送のSHA一致を確認したうえで**未完了のcode-install工程だけ**を再実施し，新しいimmutable code snapshotを構成しました．失敗証拠もprivate receiptに保持しています．

使用した新たな前処理・適格性コードは [`prepare_smac_p300.py`](../../next_multichannel/prepare_smac_p300.py) です．凍結台帳の元DATと32ch montage，オリジナルraw EEG payload SHA，BCI2000 header software calibration，元データの全SHAを再照合し，フィルタ前の上限0.1V，追加の因果3–40Hz＋80Hzローパス，200Hzへの因果的な遅延補間，3秒窓の固定100µV／0.01µV QCに従いました．元保存state／oracle／decoder出力／未来のTextResultはSSL入力に使用しません．BCI2000 source block時刻と実刺激遅延，装置側の校正は独立に実証していません．元の処理が非因果でないことも保証しないため，**offline SSL限定**です．

workstationの関連CPU回帰51件が成功，smac35上で新モジュールの6件が成功しました．容量probeはbatch1／8／32／64のすべてでMPS実backward成功．学習・全probeの終了コードは0で，model 210 trainable tensors／12 encoder層，MPS実勾配有限，40GiB枠内（サンプル最大2,643,640,320B）でした．別実行のread-only検証で，全25,876 train窓＋327 val窓・161 arrayを再hashしてmanifestと照合し，best checkpoint SHA，optimizer，RNG，seen-ID，親と封印済みsplitを照合しました．

最良checkpoint SHA256：`9448e34ea1549d406a14b1a3e3eae5603a2ff7882008156d0937e4edb482cc8c`．新runは `smac35:/Users/yasufumi/pj-xavier/smac35/runs/smac806023_CNT_InMID_plusP300_ssl_20261122_v1/run`，実際の合格train配列は同node `prepared/smac806023_CNT_InMID_plusP300_ssl_v1` にあり，重み・個別EEG・予測・被験者IDはGitに含めていません．

## 同じ37台の全量取得

2026-11-22 20:30:55 JSTにsmac31から37取得nodeを観測しました．**完了60,732ファイル／395,625,524,078 B，partial70,262,345,104 B**，alive37，完了ファイルの欠損・サイズ不一致0，観測error0です．約23.07TB／598,556 filesは割当候補で，今回の学習量ではありません．archiveのみが進行中のBCI8089820はまだcompleteとは計数しません．SSVEP9970619の64ch processed exportは単位未確定なので，根拠ない倍率補正で学習へ混ぜていません．

## 次の工程

既存download supervisorを二重起動せず継続し，BCI8089820，SSVEP9970619，P300新val/testの出典・単位・元波形時刻・cross-ID must-link・新しい未使用heldoutを別protocolで監査します．次の適格sourceを新runとして登録し，親checkpointの選択，実MPS backward probe，固定QC，累積seen-ID union，独立checkpoint評価を継承します．EEG＋判定前環境→X/YやLLM行動接続，実機成功率は別の固定benchmarkと安全なハーネスで評価するまで未実証です．
