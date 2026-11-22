# PJ-Xavier：EEG と LLM の学習・評価

NeuroCore の外部デコーダを研究開発する実験コードです．目標は，判定より前に得た環境情報と EEG から操作の意図を推定し，ハーネス上で結果を評価することです．EEG の埋め込みを学習可能な prefix として既存 LLM の `inputs_embeds` に渡し，LLM の出力損失から EEG 接続部へ逆伝播します．

**2026-11-22 の時点で，実データによる 1.7B・8B モデルの EEG 接続部の学習，checkpoint 保存，保留データでの評価を実行しました．目的の自律エージェント能力は未達・未検証です．** 最初の EEGlass 試験では両モデルとも全試行をクリックと予測し，クラス別に均等評価した正解率は 33.3%，EEG を入れ替えたときの差は 0 でした．単純正解率 83.3% はクリックが多いデータ構成によるもので，EEG 解読の成功として扱いません．

## 構成

| ディレクトリ | 用途 |
| --- | --- |
| `acquisition/` | 目録の取得先解決，公開ファイルのストリーミング取得，Range 再開，checksum と台帳 |
| `model/` | EEG prefix と frozen LLM，任意の LoRA，学習・推論，EEGlass / AI-assisted coding の前処理 |
| `benchmark/` | 被験者・セッション・時刻・波形の漏洩検査，負の対照，実機資源とハーネス評価の判定 |
| `labeling/` | train データだけを対象とする Ollama 補助ラベル，出典・teacher・元ラベルの保持 |
| `fleet/` | 認証済みノードの事前確認，容量を考慮した割当，node ごとの実行 |
| `baselines/` | EEG・事前環境・結合の CPU 分類器，validation のみで設定選択 |
| `controller/` | 実行中ジョブの監視，成果物と raw の SHA 付き回収，停止条件 |
| `integration/` | モデル出力を NeuroCore の実際の router に渡す dry-run |
| `reports/` | 実測の集計，資源使用量，失敗・未達事項，成果物の hash |
| `HANDOFF.md` | 継続作業の場所，実行中ジョブの確認方法，次に行う作業 |

生データ，波形配列，モデル checkpoint，詳細イベントとジョブ台帳は `private/xavier/` に保持します．基盤 LLM は指定した immutable revision から取得し，接続部の checkpoint と組み合わせます．重みや生データをこの git ツリーへ同梱しません．

## 実測した最初のモデル

対象は EEGlass の 1 人・1 記録の時間分割です．学習 225，検証 76，最終テスト 78 の非重複窓を使っています．同一人物の時間分割による動作確認であり，新しい利用者への汎化評価ではありません．

| 項目 | SmolLM2-1.7B-Instruct | Qwen3-8B |
| --- | ---: | ---: |
| frozen LLM のパラメータ | 1,711,376,384 | 8,190,735,360 |
| 学習した EEG 接続部のパラメータ | 163,776 | 296,896 |
| epoch / optimizer step | 3 / 171 | 3 / 675 |
| 学習・検証・最終評価の実測時間 | 35.01 秒 | 227.35 秒 |
| CUDA 最大予約メモリ | 3,900,702,720 B | 16,821,256,192 B |
| テスト balanced accuracy | 0.3333 | 0.3333 |
| テスト macro F1 | 0.3030 | 0.3030 |
| EEG 入替による balanced accuracy の差 | 0 | 0 |
| 候補ラベル scoring の p95 | 38.83 ms | 262.97 ms |

実機は Colab の NVIDIA L4，GPU 23,659,151,360 B，host RAM 56,863,301,632 B です．**時間にはモデル・データのダウンロードを含みません．latency には EEG 読込・前処理・ハーネス操作を含みません．LLM 本体を全パラメータ学習した時間でもありません．** Colab の操作は workstation の既存 Colab CLI を経由しています．

workstation（M1 / 16 GiB）でも，保存した SmolLM2-1.7B の接続部を用いて実 EEG 3 窓の推論に成功しました．GPU driver の観測値は 4,310,319,104 B，候補 scoring の中央値は 776.4 ms です．最初の呼出を含む 3 件だけの実機確認であり，定常 latency や held-out 精度の評価ではありません．Mac GPU が未対応の非整数比 adaptive pooling は，同じ区間平均と勾配を保つ実装に置き換えました．

通常の判別値だけでなく，クラス不均衡，shuffle，環境情報のみの baseline，時刻ずらし，誤作動を観測する idle 期間，実際のタスク完了を別々に測ります．取得できなかった正解・環境情報・idle 期間の値は未測定として残します．

## データ量の読み方

最新の EEG-dataset 目録は **5,165 行**です．全行の ID をキューと照合して不足・余剰がないことを確認しました．数値 ID の最大値は行数ではありません．

- 解決済み公開ファイル全体は延べ **50,696 entries / 10,431,903,294,358 B**です．EEG と確認できない資料，付随モダリティ，重複の可能性を含む台帳上の値です．
- EEG を含むと仮分類した解決済み候補は **405 releases / 29,229 files / 4,673,956,432,245 B**です．このうち優先 2 releases を Colab に，残り 403 releases を workstation の取得キューに割り当てました．全件取得済みの値ではありません．
- source file ID の異なる同名ファイルを区別し，全キューの **2,201 entries / 32 releases** を別 namespace にしました．ファイルを間引いて数を合わせていません．sidecar の対応が曖昧な release は対応確認まで学習を保留します．
- 生 EEG の記録時間は header の sampling rate と sample 数から測ります．ZIP の容量を一律の係数で EEG 時間へ換算しません．重複版，channel 数，付随動画は記録時間に加算しません．
- AI-assisted coding の UniBa 側 EEG は，配布 archive の MD5/SHA-256 と実 header を照合し，34 人・34 記録，合計 **30.61540256 時間**と確認しました．32 人が EEG と行動ログの時計対応の検査に通過し，2 人を学習から保留しています．圧縮取得量は 881,786,830 B，展開 archive 全体は 1,129,850,210 B です．
- EEGlass の検証済み元記録は **3,919.76 秒，約 1.0888 時間**です．379 窓で実際に入力した長さは合計 **664.008 秒**です．元記録全体と学習・評価に採用した窓の時間を区別します．

取得状況は実行中に変わります．`private/xavier/*downloads.sqlite` と controller の現在の状態を確認してください．制限付き・不明な取得先は理由を保持し，ログインや利用制限を回避しません．

## ラベルの扱い

元データの gold，観測された操作イベント，AI の弱教師ラベルを別々に保持します．Ollama による実測では，225 個の train 窓に含まれる **15 種類のイベント表記**を正規化しました．teacher は qwen3.5:2b，実体は 2.3B / Q8_0 で，model digest と prompt hash を保存しています．

これは表記・メタデータの整理です．EEG 波形や独立した意図の正解を 15 件・225 件新しく獲得したという意味ではありません．validation/test のラベル作成には流用せず，元の正解を AI 出力で上書きしません．

## 再現と試用

リポジトリのルートから実行してください．使用する Python は 3.12 を想定します．

```bash
uv venv --python 3.12 .venv-xavier
uv pip install --python .venv-xavier/bin/python -e '.[server,dev]'
uv pip install --python .venv-xavier/bin/python -r experiments/pj_xavier/model/requirements.txt
```

記録済みの EEG を 3 窓だけ推論する例です．`local_runtime_probe_manifest.jsonl` は実際の train 窓から切り出した実機確認用の manifest で，評価スコアの計算には使いません．

```bash
.venv-xavier/bin/python experiments/pj_xavier/model/eeg_llm.py predict \
  --checkpoint private/xavier/colab_snapshot/runs/smollm2_1_7b_eeglass/best_pinned.pt \
  --manifest private/xavier/local_runtime_probe_manifest.jsonl \
  --output private/xavier/local_m1_predictions.jsonl \
  --device mps --memory-budget-gib 8 --candidate-batch-size 1 \
  --dtype bfloat16 --max-examples 3
```

NeuroCore との接続は次の dry-run で確認できます．クリックという観測行動を `select` に対応させることを明示し，一時的な router だけを armed にします．不明なキーボード文字やスクロール方向は作りません．このコマンドは OS 操作も cloud agent 呼出も行いません．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/replay_to_neurocore.py \
  --predictions private/xavier/local_m1_predictions.jsonl \
  --output private/xavier/neurocore_dry_run.jsonl \
  --allow-mouse-click-proxy --human-armed
```

データ取得，ラベリング，split と評価の詳細は各ディレクトリの README を参照してください．研究用実験コードは core runtime の既存 API と独立しており，実験用依存は標準インストールへ追加していません．

## 未解決の条件

AI-assisted coding の公開 EEG は既に offline filtering が適用されています．公開された MNE 1.10.2 の export recipe を同じ設定で再構成すると，有限フィルタの片側 support は 4.125 秒になりました．実際の配布 binary がその recipe と一致するという条件付きの検証です．action の 0.25 秒前で窓を切るだけでは将来の信号の混入を除外できません．同期監査と前処理の可用時刻を確認し，オンライン意図解読が検証できたとは表示しません．

SFC Mac 群については，この実行では SSH 認証が成立していません．38 台の履歴上の合計 RAM を単一モデルの利用可能 RAM と見なさず，認証と実機 preflight が成立した node だけを割り当てます．現時点の学習・評価の実測根拠は workstation と Colab です．

## 一次資料

- [EEG-dataset 目録](https://github.com/AoyamaLab/EEG-dataset)
- [EEGlass 固定 release](https://zenodo.org/records/4453396)
- [AI-assisted coding 固定 release](https://zenodo.org/records/20801101)
- [SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)

研究上の性能と実機測定の根拠はこのディレクトリの `reports/` に，完全な実行ログと checksum は private の台帳に保存します．
