# S Mac：Figshare 29606906 BDF追加の自己教師学習（2026-11-23 01:21 JST）

## 実行結果（S Mac側のcheckpoint・台帳を読戻し検証）

| 対象 | 実測・確認値 |
| --- | ---: |
| 原配布BDFアーカイブ | 598,663,902 B，公開ソース：Figshare 29606906 |
| 元BDF20記録の凍結分割 | train 14，val 5，test 1 |
| BDF新train記録 | 14記録，16 EEG ch，native 500Hz |
| 追加された固有train窓 | 1,222窓 × 3秒 = 3,666秒 = **1.018333時間** |
| 現CBraMod系統・累積の固有train | **27,098窓 × 3秒 = 81,294秒 = 22.581667時間** |
| MPS学習 | batch64，10 epochs，4,270 optimizer steps |
| 反復提示 | 270,980窓（固有時間には加算しない） |
| 初期の固定806023 validation MSE | 0.001504635860 |
| 新run最良の固定806023 validation MSE | **0.001426294970** |
| 直前のP300学習モデルの最良806023 validation MSE | 0.001506584226 |
| 新しいBDFの独立validation/test評価 | 未実施（BDFのval5/test1の信号未読込） |
| EEG→LLM行動／自由意図／実OS成功 | 未実証 |

新run `smac35:/Users/yasufumi/pj-xavier/smac35/runs/smac806023_CNT_InMID_P300_plusBDF29606906_ssl_20261123_v1/run` の `run_summary.json` と `best.pt` のSHAを実機で再照合しました．終了状態は `completed`，`test_windows=0`，`behavior_labels_used=0`，`memory_within_budget=true`，`all_gradients_finite_nonzero=true`，`all_registered_buckets_observed=true` です．モデルcheckpoint SHA256は `f2864fc4dba584f6ccda232a43b000f51d15c5124d7cb22a6ca9f95a0b6e8503` でした．

新BDFに対する分割は波形読込前に確定し，20記録中14件をtrain用に展開しました．`source_certificates.json` と `split_manifest.json`，新しいcorpusの27,098 train／327 validation窓を別の読み取り専用実装で検査しました．BDFのval/test信号を開いておらず，追加14群と既存親群の重複は0でした．原配布の物理単位はBDFヘッダーのµV校正値とMNEのV変換を照合し，固定閾値を緩めず，3–40Hzの因果的フィルタ処理を使用しています．ただし原収録パイプライン自体のオンライン因果性は未検証なので，**offline SSL限定**です．

今回の固定806023 validation MSEは直前モデルの最良値から約5.33％減少しましたが，これは新しいBDFの未見被験者への汎化を示しません．同一評価セットで複数の段階を比較しているため，統計的有意性も主張しません．親に選んだP300モデルのcheckpoint SHA256は `9448e34ea1549d406a14b1a3e3eae5603a2ff7882008156d0937e4edb482cc8c` です．今回の学習系統に，旧phase3の別系統29.985888889時間は加算しません．

## 37台の取得継続（2026-11-23 01:18 JST）

S Mac管理側から37台の取得DBと実ファイルを再照合したsnapshotで，取得完了 **168,898ファイル／1,484,540,481,925 B**，partial **407,571,866,481 B**，生存supervisor37，欠損0，サイズ不一致0，snapshot error0を確認しました．この容量はEEG以外の付随データ等を含み，学習時間とは区別します．計画済み約23.07TBは取得候補であり，全件取得・学習完了ではありません．

今後の追加学習は新しいsourceと分割・単位の検証を通すごとに別runを作成します．smac19は除外します．raw／preparedデータやモデル重みをworkstation・GitHubへ移さず，既存checkpoint・optimizer・RNG・学習台帳をS Mac側で保持します．[P300の別被験者validation](P300_HELDOUT_VALIDATION_RESULTS.md)および既存のPhase 1～3はそのまま保持しています．
