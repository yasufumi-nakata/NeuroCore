# ITU 外部評価集合の準備・全件検証

**25 記録から，外部評価専用の 630 窓を作成し，全窓を元 EEG と照合して workstation に回収しました．** 学習集合と validation 集合は 0 件です．この準備・検証工程ではモデルの学習・推論・checkpoint 選択を行っていません．

## 準備結果

| 項目 | 確認した値 |
| --- | ---: |
| 原記録の全件監査 | 26 記録 |
| 固定 clock 条件の合格記録を処理 | 25 記録，すべて処理 |
| Clock fail を保持して除外 | 1 記録 |
| 外部評価窓 | 630 |
| AI request 送信の実イベント | 126 窓 |
| Keyboard | 252 窓 |
| Mouse click | 252 窓 |
| train／validation | 0／0 |
| EEG 配列 | 25 files，9,034,880 B |
| 1 窓の形状 | 8 channels × 448 samples，float32 |
| サンプリング周波数・窓幅 | 256 Hz，1.75 秒 |
| 原記録全体の時間 | 21.30617947048611 時間 |
| Clock-pass 記録の時間 | 20.516737196180557 時間 |
| 選択した非重複入力窓の時間 | 0.30625 時間＝18.375 分 |
| CPU 準備 | 39.81 秒，exit 0，最大 2 threads |
| 独立 CPU 検証 | 39.45 秒，exit 0 |

元記録時間と選択窓時間は，同じ原 EEG の異なる集計です．加算しません．元の ITU archive も既に取得済みのデータなので，この派生配列を新たな独立原記録時間として数えません．

Clock fail の記録は，凍結済み audit の最大誤差が約 4,302 ms で，既存の 100 ms 条件を満たさなかったため除外しました．条件変更，offset の手調整，phase scale の変更はしていません．合格した 25 記録について，窓数・label 数を見る前に固定した選択規則をそのまま適用しました．

## 既存 v3 と同じ入力規則

共通 channel は CP3，C3，F5，PO3，PO4，F6，C4，CP4 です．EEG 終端は対象イベントの 4.5 秒＋clock guard 前，窓幅は 1.75 秒です．既存 recipe に基づく仮定上の filter lookahead 4.125 秒を維持し，それより前に特徴が使えるとは扱いません．追加 filter，resample，rereference は行っていません．

Case は実際に記録された `GPTSentPrompt`，control は既存の keyboard／mouse-click の規則です．入力窓は同一記録内で非重複です．過去履歴は EEG 開始前と guarded decision 前の 2 系統で，cutoff と同時刻・未来のイベント，GPT イベント，キー内容，座標，duration を含めません．

原 binary と公開前処理 recipe の完全な同一性，元 XDF の clock，実オンラインの feature availability は未確認です．この制約は合格に書き換えていません．操作分類の正解は観測された action であり，潜在意図の正解を作ったものではありません．

## 全件の独立検証

準備プロセスとは別の CPU validator で，以下を確認しました．

- 25 配列すべての SHA-256，dtype，shape，有限性．
- **630 窓すべて**を元 BrainVision から再読込みし，全サンプルが保存済み配列と一致．
- **630 label すべて**を元イベント行から再導出し，同じ case/control 選択になったこと．
- **1,260 組の過去履歴特徴**を再計算し，保存値と全件一致．
- 使用した原ファイル **204 個**の再 SHA-256 検証．全 26 の EEG binary，header，marker，phase と，合格記録の行動ログを含みます．
- 同一記録内の窓の非重複，clock fail 記録の使用 0，train／validation 行の混入 0．

Fixture 8 件も workstation で PASS しました．ローカル scratch は MNE 未導入のため，7 件 PASS と実 BrainVision fixture 1 件 skip です．実データの検証は MNE 1.13.2／NumPy 2.4.6 の Colab CPU で行いました．

## Site と人物 ID の扱い

UniBa v3 の全 split が参照する **32 原波形 SHA-256**と，ITU 全 **26 原波形 SHA-256**を比較し，完全に同一の binary hash は 0 件でした．一方，site を付けない participant label は 21 個が共通しています．

そこで `itu:<participant label>` を観測 ID として分離しました．同名だから同一人物とは断定せず，site が違うから必ず別人とも断定していません．`human_identity_disjointness_verified` は false のままです．これは外部 site の評価集合として準備できた結果であり，人物の独立性まで確認できた結果ではありません．

## 固定した来歴

| 成果物 | SHA-256 |
| --- | --- |
| ITU 外部 protocol | `6dae42f26833eab36028b886907a4cfcf9047f3ad76af182c497129cd885eca0` |
| 凍結 UniBa v3 protocol | `d4e6c5f685dce53e73b3aa88d12ed07c88a8ac08a27fb78392dbd99aed0a6396` |
| ITU clock audit | `b7d2bd664c661c2328f5394a7f3a2c24d8177e154974b21ef86cf4788c7ce916` |
| Preparer | `d3b3dc3b7daf8c1eba55d7ee8ae4ddf3eeac4807d16f9e6ef3c0bb1e9ef76256` |
| Independent validator | `2f60e0240d58b2ca0ee5532526d1d0af2c1706682defac7ccf44c1c82ab50dd9` |
| 固定 external manifest | `cfd93863b8f8e263174e6ab59395496efbd397e821542fa3a1f0c1e5faa503b0` |
| History features | `1326aef4ea7b0a3010f21c13cff56f7e476ff6da948673dac99dd8974ab853c9` |
| Labels | `59933900a4998579bcf3d0484d0b07dc128859b2a8cbb023e9362db3ddab5d41` |
| EEG artifact inventory | `87d5e0b944df33ac27f4101308022a9aa769c1080f043332a2d06feb0d2e3f02` |
| External membership | `66351f4e59cf84f6e51eff0032139ab685327a2afbf6916ec5dcb9bdb52f3762` |

## Workstation への回収

Colab 操作はすべて workstation CLI を経由しました．8,901,045 B の回収 archive の SHA-256 は `7379269fe23ef3138c1d6db65daad78ac1ac28d0a91841cb755a4a6af9633a13` です．回収後に **49 files の全サイズと SHA-256**を検証しました．

Workstation の保存先は `private/xavier/prepared/ai_coding_itu_external_v3_natural_casecontrol_v1` です．回収 archive と実行・終了・回収の記録は `private/xavier/itu_external_recovery_v1` に保存しています．元 manifest と既存 model／checkpoint は変更していません．公開 report には集約値・規則・来歴 hash のみを掲載し，EEG 配列と個別イベントは private に保持しています．

この固定集合へのモデル推論は，別途固定する外部評価 protocol に従います．ITU 上で fit，checkpoint 選択，閾値調整，window の再選択を行わず，既存 UniBa 評価と分けて結果を記録する設計です．
