# 録画済み EEG の一括試用結果

**M1 上の SmolLM2 1.7B + EEG connector から NeuroCore の dry-run まで，1 回のコマンドで正常終了しました．** 学習集合から固定選択した 3 例を推論し，3 例とも固定閾値 0.8 に達しなかったため棄権し，router が保留しました．OS 操作とクラウド agent 呼出しは 0 件です．

これは記録済みデータの入出力経路を試す実行結果です．学習集合の 3 例を使っているため，この試用から汎化精度や意図推定の成功率は推定しません．被験者を分離した 5 条件評価の終了を確認してから，model 担当が 1 回だけ実行しました．

## 3 例の結果

確率はモデルの 3 クラス間の値です．校正済みの「本人の意図が正しい確率」ではありません．

| 学習集合の記録ラベル | AI prompt submit | Keyboard | Mouse click | 最大確率クラス | 棄権 | Router |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `ai_prompt_submit` | 15.9165％ | 39.4503％ | 44.6332％ | `mouse_click` | はい | 保留 |
| `keyboard` | 16.0374％ | 39.3848％ | 44.5778％ | `mouse_click` | はい | 保留 |
| `mouse_click` | 13.1874％ | 32.7814％ | 54.0312％ | `mouse_click` | はい | 保留 |

3 例の保留理由は `model_abstained_or_abstention_flag_missing` です．実際の prediction には `abstain: true` が記録されており，閾値未満による保留を確認しました．動作を見せるための閾値引下げ，mouse-click proxy の明示 mapping，human-armed の設定は行っていません．

## 実行したもの

```bash
cd /Users/yasufumi/Documents/GitHub/NeuroCore-xavier-training
.venv-xavier/bin/python experiments/pj_xavier/integration/try_recorded_eeg.py \
  --execute-prepared private/xavier/demos/m1_natural_v3_train_examples_v1
```

既存 `eeg_llm.py predict` と `replay_to_neurocore.py` を使用しました．MPS，bfloat16，メモリ予算 8 GiB，候補バッチ 2，棄権閾値 0.8 を保持しています．checkpoint と backbone は変更していません．実行開始記録の attempt は 1，wrapper の開始から完了までは約 16.54 秒でした．

準備時と実行時に，checkpoint を `weights_only=True` で CPU 読込みし，全 32 配列の SHA-256 とサイズ，元 manifest，全 train／validation／test ID と training membership，モデル設定を照合しました．各クラスの train レコードを ID の辞書順で 1 件ずつ選んでおり，予測結果による例の選び直しはありません．Heldout を試用に使った件数は 0 です．実行後も元ファイルの不変性を検証しました．

## 保存した証拠

| 証拠 | SHA-256 |
| --- | --- |
| Frozen checkpoint | `281716da86293d2b700ec003c967ca2c141515cc565b136a12137c34b73a3475` |
| 元 manifest | `030e8550bd6f4c79005722eb617e9b64f5f80e87b77866b7f17bec4e7e2a1093` |
| 準備済み plan | `65150113bbc0760552d76b7c52ad389832562c9495101b7e9688ed8365db67e5` |
| Prediction JSONL | `9581778ba5608b084b27df85793d3219f9c2a15f74116ec98a67da23741c88ec` |
| Dry-run audit JSONL | `18dc57b92cc40e78d01beef1a314a366d3dcc1f347fb68bb4195d6887767f3eb` |
| Private 完了 summary | `2975e744870eb69b876157f5b22d610c2b29c814551d155a22e6a262b3c18211` |
| 試用 wrapper | `8387729dedf3bacba2c68d496db489079b03709d7e209f3f9b24bfd6cfdbc7d6` |

個別のレコード ID と予測・監査ファイルは private の試用ディレクトリに保存しています．公開 `summary.json` には，確率・棄権・理由・実行条件・来歴ハッシュを集計しました．EEG 配列は公開していません．

## 再び試す場合

完了済みの出力ディレクトリは上書き・再実行しません．新しい出力先で，同じ固定モデルの試用コマンドを実行できます．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/try_recorded_eeg.py \
  --output-dir private/xavier/demos/m1_natural_v3_train_examples_local_02
```

元モデル，配列，分割，コマンド，閾値が一致する必要があります．詳しい手順は `integration/README_TRY_RECORDED_EEG.md` を参照してください．
