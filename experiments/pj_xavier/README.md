# PJ-Xavier：EEG と LLM の学習・評価

NeuroCore の EEG デコーダと LLM 接続部を開発する実験コードです．目標は，判断以前の環境情報と EEG から操作を推定し，実ハーネスでタスク完了まで評価することです．現在の実装では，学習可能な EEG encoder／prefix を既存 LLM の `inputs_embeds` に渡し，出力トークンの損失から逆伝播します．

**実 EEG による 1.7B・8B モデルの学習，checkpoint 保存，held-out 評価，workstation での推論，NeuroCore router への dry-run を実行しました．目的の自律エージェント能力と EEG の追加寄与は，まだ実証できていません．** コードと公開集計は [PR #70](https://github.com/yasufumi-nakata/NeuroCore/pull/70) に保存しています．最新の実行場所と再開手順は [HANDOFF.md](HANDOFF.md) を参照してください．

## データ量：候補・取得済み・検証済みを区別

2026-11-22 日本時間の観測です．GB／TB は十進，GiB は二進です．

| 範囲 | 確認した量 | 意味 |
| --- | ---: | --- |
| EEG-dataset の目録 | 5,165 行 | 全 ID と取得先解決結果を照合し，不足・余剰 0 |
| workstation の取得キュー | 609 releases，468,040 file jobs | EEG を含む候補の release 全体．認証・規約操作・metadata の保留を含む |
| 上記キューの既知の容量 | 15,821,047,310,466 B，約 15.82 TB | 動画・補助データ・版の重複を含む候補容量．38 files はサイズ未確定 |
| 取得完了し SHA-256 を再検証した内容 | 39 files，14,569,523,189 B，約 14.57 GB | 10:08〜10:09 の workstation＋Colab 観測．同一内容の複製を二重加算しない |
| 元 EEG の header／binary を数量検証した範囲 | 118 記録，87.90746536 時間 | チャンネル数を掛けた時間ではなく，記録の長さの合計 |
| 上記の物理 EEG 値の非圧縮容量 | 10,924,347,024 B，約 10.92 GB | 補助チャンネル・動画・header・窓配列・モデル重みを除く．元の保存形式のまま |

取得状況の根拠は [acquired_progress_002.json](reports/acquired_progress_002.json)，全キュー更新は [fresh_queue_append_20261122_01](reports/fresh_queue_append_20261122_01/README.md)，数量の定義と hash は [verified_eeg_quantities_20261122.json](reports/verified_eeg_quantities_20261122.json) にあります．未完了の partial は取得済み容量に加算しません．取得候補全体の EEG 時間は未確定です．ZIP 容量から EEG 時間への一律換算はしません．

| 元データ | 記録数 | EEG 時間 | 物理 EEG チャンネル |
| --- | ---: | ---: | ---: |
| EEGlass | 1 | 1.08882222 h | 3 |
| AI-assisted coding：UniBa | 34 | 30.61540256 h | 8 |
| AI-assisted coding：ITU | 26 | 21.30617947 h | 64 |
| Portable EEG systems comparison | 57 | 34.89706111 h | 32 |
| 合計 | 118 | 87.90746536 h | 記録ごとに異なる |

raw と filtered export，同じ archive の保全コピー，原波形から切り出した学習窓を新しい独立記録として数えません．記録時間の検証は，信号品質・意図の正解・オンライン処理の因果性の証明とは別です．異なる corpus 間の同一参加者照合も未実施です．

Colab の優先取得は EEGlass と AI-assisted coding の全 23 files／11,221,153,522 B が完了しています．AI-assisted coding は EEG に加えて gaze，screen，behavioral，補助情報も取得しています．モデルへ実際に入力しているのは EEG と判定前の page／AI availability です．画面動画を vision encoder で学習した実験ではありません．

## 実際に学習したモデル

全て LLM 本体を凍結し，EEG 接続部を学習した実測です．基盤重みは immutable revision で固定しています．時間は原則として学習・検証 loop の範囲で，ダウンロードや環境構築を含みません．

| モデル／cohort | 実機 | 完了した学習 | 実測時間 | メモリの観測 |
| --- | --- | ---: | ---: | --- |
| SmolLM2-1.7B／EEGlass | Colab L4 | 3 epochs，171 steps | 35.01 秒 | CUDA 最大予約 3.90 GB |
| Qwen3-8B／EEGlass | Colab L4 | 3 epochs，675 steps | 227.35 秒 | CUDA 最大予約 16.82 GB |
| Qwen3-8B／UniBa v2 | Colab L4 | 3 epochs，1,635 steps | 再開後 1,332.96 秒 | CUDA 最大予約 20.63 GB |
| SmolLM2-1.7B／UniBa v3 | M1／16 GiB | 3 epochs，600 steps | 1,541.72 秒 | MPS driver の観測最大 4.44 GB，allocator quota 8 GiB |

MPS のメモリ値はサンプリング観測で，瞬間最大値の測定ではありません．実際の M1 上で学習が完走した事実と，true peak の未測定を分けて記録します．UniBa v3 の Smol は 166,656 parameters，v2 の Qwen は 299,776 parameters を学習しました．frozen backbone はそれぞれ約 1.71B／8.19B parameters です．

Qwen v2 は初回 step 24 で予約メモリが 21 GiB の設定を超えたため停止し，optimizer／RNG／cursor を含む checkpoint を保存しました．同じ実験を `PYTORCH_ALLOC_CONF=expandable_segments:True` で再開し，完走しました．短い容量 probe の成功だけを完走証拠にはしていません．

### 完了した held-out 評価

EEGlass は 1 人の時間分割です．両モデルとも全 78 test 例をクリックと予測しました．accuracy 83.3% はクラス比率によるもので，balanced accuracy は 33.3%，EEG shuffle による差は 0 です．

Qwen v2 は別の 8 人／1,956 test 窓で，同じ凍結 checkpoint の入力を変えて比較しました．

| 入力条件 | Forced-choice balanced accuracy | Forced-choice macro F1 |
| --- | ---: | ---: |
| EEG＋判定前の環境 | 39.54% | 0.3737 |
| EEG を除く入力 ablation | 39.31% | 0.3502 |
| 環境を除く入力 ablation | 33.33% | 0.2077 |
| 同じ記録内の EEG shuffle | 39.42% | 0.3720 |
| 原記録から実抽出した 10 秒前の EEG | 38.68% | 0.3622 |

fused minus shuffle の macro F1 差は +0.00171，参加者単位の補助 bootstrap 95% 区間は −0.00700〜+0.00946 でした．EEG の時点に固有の追加寄与は確立していません．固定 confidence 0.8 に達した fused 予測は 0 件で，採用率 0% です．forced-choice の値は採用された操作の成功率ではありません．この補助 bootstrap は partial score 閲覧後の記述解析で，test による再学習・選択には使っていません．[完全な結果と制約](benchmark/QWEN8B_UNIBA_V2_RESULTS.md) を保存しています．

## v3：AI への送信イベントを保持する評価

v2 の静かな窓条件では AI 送信が train 2／val 2／test 1 例しか残らないため，別の事前プロトコルを作りました．v3 は実ログの 150 AI 送信を保持し，過去の AI availability を合わせた keyboard／click の対照を取ります．参加者 split は v2 と同一です．

| 分割 | 参加者 | 全窓 | AI 送信 | keyboard | click |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 18 | 400 | 80 | 160 | 160 |
| validation | 6 | 155 | 31 | 62 | 62 |
| test | 8 | 203 | 39 | 82 | 82 |

758 窓の入力長は 0.36847222 時間です．元記録 30.6154 時間と，選択した窓を区別します．v2 と v3 は同じ原記録に由来し，合算して新しいデータ時間を作りません．AI 送信を増やした case-control 標本のため，通常利用時の precision や誤作動／分は推定できません．

**EEG を使わない強い対照も実測しました．** 判定前の key／mouse の回数・経過時間だけの CPU モデルでは，AI 送信対その他の balanced accuracy が 83.11%，さらに粗い環境を加えると 86.59% でした．後者は AI 39／39 を拾う一方，その他 164 例中 44 例を誤検出し，この濃縮標本内の precision は 46.99% です．意図を解読した成績ではありません．[history baseline](benchmark/HISTORY_BASELINES.md) を EEG モデルの比較対象にします．

Smol v3 は M1 上で学習を完了し，凍結 best の 5 条件評価を実行中です．Qwen v3 は別途固定した 30 epochs または 3,600 秒の予算で Colab 上の学習を実行中です．Qwen と Smol の学習予算は異なるため，成績差をモデルサイズだけの効果とみなしません．[Qwen v3 の事前プロトコル](benchmark/protocol.qwen3_8b_uniba_v3_30e.json) と [自然作業 cohort の仕様](model/NATURAL_COHORT_V3.md) を参照してください．

## ラベリングと拡張可能性

元データの gold，観測された操作イベント，AI の弱教師ラベルを別々に保持します．workstation の Ollama qwen3.5:2b で train の 15 種類のイベント表記を正規化し，225 窓へ出典付きで伝播しました．teacher digest／prompt hash／元ラベルを保存し，held-out のラベル作成には流用していません．これは新しい EEG や潜在意図の正解を増やしたという意味ではありません．

UniBa は 34 記録中 32 記録，28.58953451 時間が時刻同期検査に通過しました．ITU は 26 記録中 25 記録，20.51673720 時間が同じ基準に通過し，独立したサイト・装置の外部 test 候補として予約しています．[ITU 監査](model/ITU_EXTERNAL_AUDIT.md) では学習・モデル選択・性能評価を行っていません．

事前学習済み EEG encoder の候補として，公式 CBraMod の source／weights を固定し，M1 の CPU 上で実 train EEG 3 窓を通しました．4,883,800 parameters，約 19.8 MB の weights，2 CPU threads の 3 例 batch が 0.033 秒で完了しました．これは encoder の互換性・資源確認です．LLM との結合学習や解読性能は未測定です．[実測 JSON](reports/cbramod_cpu_capacity_probe.json) を次の独立実験の出発点にします．

## オンライン利用と実機の条件

AI-assisted coding の公開 EEG は offline filter 済みです．公開 export recipe を再構成すると未来側の support は条件付きで 4.125 秒です．窓終端からイベントまで 4.5 秒＋時計誤差を確保しても，feature の可用時点からイベントまでは最低 0.375 秒です．配布 binary と recipe の同一性，元 XDF，online causality は未確認です．M1 の validation candidate scoring は約 1 秒であり，この名目 lead を満たすオンライン動作は実証できていません．

Colab は workstation の既存 CLI を経由しています．SFC VPN は接続でき，smac19 は DNS／SSH 到達まで確認しましたが，smac19／smac31 の SSH 認証は未成立です．今回 Mac cluster への計算投入はできていません．学習・推論が実証できた実機は workstation M1 と Colab L4 です．

workstation の取得は空き容量 35 GiB を残して進めます．Colab からの大きな保全コピーは，CLI の一括メモリ使用を避けるため，64 MiB ごとの SHA 検証・再開可能な転送にしました．保全コピーは新しい取得データ量へ加算しません．Dataverse の必須 Guestbook 応答がある 342 files は `access_action_required` として保持しています．

## 再現と試用

リポジトリのルートで Python 3.12 を用います．実験用依存は core runtime の標準依存と分離しています．

```bash
uv venv --python 3.12 .venv-xavier
uv pip install --python .venv-xavier/bin/python -e '.[server,dev]'
uv pip install --python .venv-xavier/bin/python -r experiments/pj_xavier/model/requirements.txt
```

以下は実行確認済みの，保存した EEGlass train 3 窓を使う推論例です．精度評価とは別の runtime probe です．

```bash
.venv-xavier/bin/python experiments/pj_xavier/model/eeg_llm.py predict \
  --checkpoint private/xavier/colab_snapshot/runs/smollm2_1_7b_eeglass/best_pinned.pt \
  --manifest private/xavier/local_runtime_probe_manifest.jsonl \
  --output private/xavier/local_m1_predictions.jsonl \
  --device mps --memory-budget-gib 8 --candidate-batch-size 1 \
  --dtype bfloat16 --max-examples 3
```

NeuroCore の実 router と dry-run sink への接続も実行確認済みです．不明な keyboard 文字や scroll 方向を補完せず，明示した mouse click proxy だけを `select` に対応させます．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/replay_to_neurocore.py \
  --predictions private/xavier/local_m1_predictions.jsonl \
  --output private/xavier/neurocore_dry_run.jsonl \
  --allow-mouse-click-proxy --human-armed
```

## 構成

| ディレクトリ | 用途 |
| --- | --- |
| `acquisition/` | 取得先解決，公開データの再開可能な取得，checksum，容量・時間監査 |
| `model/` | EEG prefix＋frozen LLM，前処理，学習・推論，公式 EEG encoder の容量 probe |
| `benchmark/` | split・時刻・波形の漏洩検査，負の対照，history baseline，実機・ハーネスの判定 |
| `labeling/` | train のみを使う AI 補助ラベル，teacher／出典／元ラベルの保持 |
| `fleet/` | 認証済み node の preflight，容量を考慮した割当 |
| `baselines/` | EEG spectrum／環境／結合の CPU baseline |
| `controller/` | ジョブ監視，raw と成果物の検証付き回収，終了条件 |
| `integration/` | モデル出力と NeuroCore router の dry-run |
| `reports/` | 実測値，失敗・未達事項，成果物 hash |

生 EEG，checkpoint，詳細ログ，イベント，ジョブ台帳は git 除外の `private/xavier/` に保持します．継続予約と実行中ジョブは [HANDOFF.md](HANDOFF.md) に記録しています．

## 一次資料

- [EEG-dataset 目録](https://github.com/AoyamaLab/EEG-dataset)
- [EEGlass 固定 release](https://zenodo.org/records/4453396)
- [AI-assisted coding 固定 release](https://zenodo.org/records/20801101)
- [Portable EEG systems comparison 固定 release](https://zenodo.org/records/15698441)
- [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [CBraMod 公式実装](https://github.com/wjq-learning/CBraMod)
- [UniMind 公式実装](https://github.com/kaleidoyao/neuroverse)
