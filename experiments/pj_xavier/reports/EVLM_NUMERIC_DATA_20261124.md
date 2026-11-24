# EVLM 数値 EEG データ準備記録 — 2026-11-24

## 完了した範囲

2026-11-24 05:55 JST 時点で，certified SSL corpus の適格 train **79,493 元窓**を，最大 6 channel の数値 EEG 入力 **358,260 例**へ変換し，smac05 から smac35 への転送と SHA 検証を完了しました．元 validation **327 窓**は **981 例**になりました．

この完了は **データ準備・転送の完了**を示します．準備した全例の学習，学習効果，EEG に基づく意図推定，画像理解，computer use の成功を示すものではありません．本報告には個別 EEG 値，個別 row，被験者識別子を含めていません．

| 項目 | 確定値・条件 |
|---|---|
| 対象 | 元の certified corpus に admitted train として収録された全 6 source |
| 元 train 窓 | 79,493 窓，1 窓 3 秒，101 source group |
| 6 ch train 提示例 | 358,260 例，sample cap なし |
| 元 validation | zenodo:806023 の 2 group，327 窓 |
| 6 ch validation 提示例 | 981 例 |
| 入力形状 | 最大 6 ch × 32 点，最後の時間 slot は 24 点 |
| 時間軸 | prepared 200 Hz，cycle 0 |
| 量子化 | 1 µV，最近接整数，ties-to-even |
| 振幅 clipping・無言 truncate | いずれも 0 |
| 登録した token 上限 | 801 |
| 実 tokenizer による全例の最大 | 710 tokens，completion 最大 15 tokens |
| 検証した信号 artifact | 607，すべて SHA と shape／dtype を確認 |
| 全 visible 値がゼロの提示例 | 0 |
| test／review 信号の読取り | 0 |
| CPU prepare 所要時間 | 302.273 秒 |
| CPU tests | 19 件 PASS，smac05 preparation venv で再実行済み |

元 corpus の非重複 train 窓をすべて使用しました．channel block への分割によって提示例は増えますが，独立した記録時間や被験者数が増えるわけではありません．本集計はこの certified SSL corpus の範囲であり，別の direct EEG 行動分類パイロットの件数は合算していません．

## 1．Source ごとの元窓と提示例

各元窓の全 channel を，元の順序と channel 名を保持して，重複・欠落なく最大 6 ch の block に割り当てました．最終 block に channel を補充する処理は行っていません．例えば 14 ch は 6＋6＋2，27 ch は 6＋6＋6＋6＋3 です．

| Source | 元 channel 数 | 元 train 窓 | 旧 8 ch 例数 | 新 6 ch 例数 | 新版最大 tokens |
|---|---:|---:|---:|---:|---:|
| zenodo:806023 | 16 | 659 | 1,318 | 1,977 | 683 |
| figshare:2061654 | 14 | 3,385 | 6,770 | 10,155 | 703 |
| mendeley:rjx76wd5v6 | 14 | 16,835 | 33,670 | 50,505 | 675 |
| openneuro:ds005028 | 32 | 4,997 | 19,988 | 29,982 | 709 |
| figshare:29606906 | 16 | 1,222 | 2,444 | 3,666 | 614 |
| zenodo:8089820 | 27 | 52,395 | 209,580 | 261,975 | 710 |
| **合計** | — | **79,493** | **273,770** | **358,260** | **710** |

train の増分は **84,490 提示例**です．validation は同じ 327 元窓を 16 ch から 3 block に分割するため，654 例から 981 例へ増えています．

全 channel は提示例群として保持しています．単一 prompt に同時に入るのは最大 6 ch のため，別 block の channel 間関係を一つの入力で直接観測できる設計ではありません．

## 2．66.244 時間と cycle 0 の 3.487 時間の意味

元 train 窓が表す総時間は，次の計算で **238,479 秒＝66.2441667 時間**です．

```text
79,493 元窓 × 3 秒 = 238,479 秒
```

今回準備した cycle 0 は，各 3 秒窓から 32 点，または最後の slot の 24 点を切り出した入力です．その時間 coverage は **12,553.96 秒＝3.4872111 時間**です．これは準備済み入力が表す時間であり，モデルが既に学習した時間ではありません．

| Source | 元 train 窓の時間 | Cycle 0 が表す時間 |
|---|---:|---:|
| zenodo:806023 | 1,977 秒 | 104.00 秒 |
| figshare:2061654 | 10,155 秒 | 533.92 秒 |
| mendeley:rjx76wd5v6 | 50,505 秒 | 2,658.24 秒 |
| openneuro:ds005028 | 14,991 秒 | 789.28 秒 |
| figshare:29606906 | 3,666 秒 | 193.24 秒 |
| zenodo:8089820 | 157,185 秒 | 8,275.28 秒 |
| **合計** | **238,479 秒** | **12,553.96 秒** |

同じ元窓の channel block は時間を共有するため，block 数を掛けて時間を水増ししていません．8 ch 版と 6 ch 版で，元 train／validation 窓数と cycle 0 の時間 coverage が一致することを確認しました．validation の元時間は 981 秒，cycle 0 の時間 coverage は 51.68 秒です．

3 秒窓は prepared 200 Hz の 600 点です．時間軸を 32 点の 18 slot と 24 点の 1 slot，合計 19 slot に分けています．元窓 ID と固定 seed 20261124 の hash で初期 offset を決め，cycle を進めると別 slot を選ぶ方式です．**19 cycle を揃えると，元 600 点を各 channel について一度ずつ覆います．今回作成済みなのは cycle 0 のみです．** 全 3 秒の点をすでに入力化・学習したとは報告しません．将来の cycle には，その cycle に対する新しい token 検証と receipt が必要です．

## 3．信号表現・単位・自己教師タスク

Serialization 識別子は `eeg-raw32-uv1-channel-major-cycle19-v1` です．manifest には `max_channels_per_training_block=6` と `max_seq_length=801` を追加し，geometry と上限を結合しています．

識別子中の raw32 は，既存の certified 波形から時間平均を追加せずに sample を切り出すことを指します．入力 corpus は **既に 3–40 Hz の処理を受けた prepared EEG**です．センサーの未加工信号そのものとは区別します．200 Hz は prepared sample rate であり，元の native sample rate は source ごとの upstream provenance に保持されています．

保存配列は float32 の `[channel,3,200]` です．3 の軸は周波数帯域ではなく，連続する時間 patch です．元 prepare 実装で，連続 600 点を `reshape(channel,3,200)` していることを確認しました．

単位は保存済み `V_x_1e4` を利用します．µV への変換は **保存値 ×100**です．今回の処理に正規化・再スケーリング・時間平均は追加していません．その後の 1 µV 量子化は `np.rint` により行い，振幅を clip して収める操作は行っていません．

各 prompt は実際の channel 名，形状，prepared sample rate，量子幅，crop 開始 sample を含みます．channel-major の整数列から，block 内の 1 channel の連続 4 点を隠し，その実測量子化値を同じ順序の JSON 整数配列として予測する **`quantized_eeg_reconstruction`** タスクです．隠す span の両側には観測値を残しています．

EEG-only 素材に行動・意図・画面・画像の疑似ラベルを付けていません．外付けの学習済み EEG encoder は使用していません．学習する対象は，この入力契約に従う数値 EEG の欠損点再構成です．

## 4．QC・分割・情報漏れの確認

元 corpus の completion digest，7 種類の metadata digest，positive source certificate と各 artifact digest を確認してから信号を読みました．元の source／payload／group deny と train／validation の役割を継承し，分割の作り直しは行っていません．train は元 certificate の `fit_allowed=True`，validation は `fit_allowed=False` かつ validation 許可を条件としています．

processed 3 秒窓に対する固定 QC は，有限値，**絶対振幅 100 µV 以下**，**各 channel の標準偏差 0.01 µV 以上**です．この閾値を変更せず，stored 波形を物理 µV に戻した後にも確認しました．異常値を補間・clip して通過させる処理はありません．

元窓と同じ派生例は同じ split に留まります．train／validation の group と payload の重複を拒否しています．test／review の信号は今回の変換・検証に使用していません．

baseline は zero と，同一 channel の可視の左右端点だけから計算する線形補間です．マスク位置の正解値を変更しても，prompt，observed 値，baseline が変わらないことを CPU test で確認しています．別 channel を横断する補間や，正解値に依存した mask の選択は行いません．

追加した CPU 検証には，以下を含みます．

- 単位変換が ×100 であることと，未知単位・非有限値・flatline の拒否．
- 19 cycle が元 600 点を過不足なく覆い，最後の slot に人工 padding を加えないこと．
- 14／16／27／32 ch の全 channel と全 crop sample が，6 ch block 群から欠落なく復元できること．
- 6 ch 再分割でも crop offset と時間 coverage が変わらないこと．
- 登録した 801 token 上限を超える場合の拒否と，既存 CLI default 8 ch／1024 の維持．
- probe と prepare の seed，geometry，code／helper identity を結合し，異なる設定で probe を使い回さないこと．

全実データの token 検証は，既存 base の固定 tokenizer を local files only で読み，no-thinking の train／eval prefix 一致と assistant completion の復元一致を確認しました．全 359,241 例で最大 710 tokens となり，801 超過はありませんでした．この値は cycle 0 の実測です．

## 5．Validation と MSE 比較の限界

すべての source manifest が共有する validation は，元 corpus に存在した **zenodo:806023 の 2 group／327 窓／981 派生例**です．他の 5 source ごとの validation group を新しく確保したものではありません．

したがって，共通 dev での改善は，この 806023 側の入力分布における再構成の改善を示します．各 source への汎化，異なる montage への一般化，未知被験者への広い汎化まで示すものではありません．981 の派生例も，981 個の独立した人や独立した記録を意味しません．元窓・group 内の相関を考慮する必要があります．

8 ch から 6 ch に変更しても，crop hash，seed，mask hash の関数は維持しています．ただし block の channel 数と境界が変わるため，選ばれる実 channel と mask の時間位置は変わり得ます．この変更を新 manifest に記録しました．

**8 ch 版と 6 ch 版の MSE は，そのまま同一の正解集合に対する改善量として比較しません．** 新版の評価には，新版の target に対して計算した baseline を用います．

| Validation の記述的 baseline | 旧 8 ch | 新 6 ch |
|---|---:|---:|
| Zero MSE | 26.288991 µV² | 31.413863 µV² |
| 同一 channel の局所線形補間 MSE | 22.741437 µV² | 24.869725 µV² |

これらはデータ上で算出した baseline であり，学習済みモデルの性能ではありません．モデルの評価では valid output rate とともに，同じ dev 行・同じ target に対する学習前後の MSE を比較します．数値再構成だけから，意図推定・画像理解・computer use の能力を評価しません．

## 6．保存場所と完了証拠

6 ch 版の smac35 data root：

```text
/Users/yasufumi/pj-xavier/smac35/prepared/evlm_numeric_6ch_20261124_v1
```

`source_queue.json` が全 6 source の現地 manifest と例数を列挙します．各 source の `transfer_receipt.json` に転送元と転送先の照合結果を保存しました．数値 JSONL と QC receipt の内容は転送前後で同一であり，manifest のローカル path のみ移しています．元 manifest も `source_manifest.json` として保持しています．EEG payload を workstation 経由では転送していません．

集計資料は `preparation_summary.json`，全体の QC 証拠は `qc_receipt.json`，変換時の code 一式の identity は `code_snapshot_receipt.json` です．workstation 側には，個別 row を含まない集計 receipt を次の project-private path に保存しました．

```text
private/xavier/evlm_numeric_6ch_20261124_v1/preparation_summary_mcp.json
```

| 集約証拠 | SHA256 |
|---|---|
| 元 certified corpus の completion | `a76b870cc24fa543ab8b9310eea2a063872b37cc567f26104fa78334ba85209a` |
| 6 ch 全体 QC receipt | `09c2b418a346eb0c7729a5bc192f913a4b106430925c7ff39efdd2f9c9a6115d` |
| smac35 の 6 ch 全 source queue | `8218c7d821e4afbd4a6e841129df47236db3d759711ff8eec66991a7a0e94510` |
| 6 ch code snapshot receipt | `1270e687473d1f5b60212821dd7b0a8ced450ed91b40787a070306b2bb01a620` |
| smac35 の 6 ch preparation summary | `d5bd9ab746160c48e5c019d195de9f4930ad535a73b751a1052988d4cdb3564d` |
| workstation の集計 receipt | `1d8c496b91f455cb35094d370d980f311977dc2bd1364070f040a0a9a635d1fc` |

変換元の smac05 namespace は `prepared/evlm_numeric_6ch_20261124_v1` です．旧 8 ch namespace `prepared/evlm_numeric_20261124_v1` は変更せず保持しています．6 ch の CPU prepare は **05:50:12 JST**，全 source の smac35 転送は **05:52:59 JST** に完了しました．

## 7．コードと今後の解釈

実装は [evlm_prepare_numeric.py](../model/evlm_prepare_numeric.py)，CPU tests は [test_evlm_prepare_numeric.py](../model/test_evlm_prepare_numeric.py) です．新 CLI は `--block-channels 6 --max-tokens 801` を明示して実行しました．指定しない場合の default は従来どおり 8 ch／1024 です．

6 ch prepare で使用した converter SHA は `6c1472e5b7fe94dc5978b760ed68ec627522f5a7b580268081d21f80af7ffb89`，test SHA は `3f45ecdc12f41aa918e93f4acf8a18f80f2f9e0a926c58a1082ae7ac961c5148` です．変換時の prompt／validator は固定済み helper snapshot から読み，その SHA を preparation identity に含めています．実学習 runner の版・checkpoint・optimizer 更新数は，それぞれの学習 run receipt を参照します．

元の 3–40 Hz 処理済み信号を，1 µV，短い crop，最大 6 ch の prompt として追加学習できるデータ基盤まで準備しました．全準備例が学習キューに存在することと，全例の勾配更新が完了したことは別に追跡します．現在の cycle 0 は長い時間構造や全 channel の同時関係を一つの入力に含めないため，その制約を踏まえて再構成 baseline と比較し，別途 EEG 行動課題・視覚保持・computer use を評価する必要があります．
