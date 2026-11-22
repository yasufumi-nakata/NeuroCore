# Controller の v3 gate と大容量回収の修正

**v3 の5条件評価が未完了なら Colab を停止しない gate を追加し，元の絶対終了時刻を保持して controller を再開しました．** 同時に，大容量 raw の回収を64 MiB以下に分け，workstation の RAM に原本全体を載せないようにしました．

観測時点：2026-11-22T01:08:35.318495+00:00

| 項目 | 確認結果 |
|---|---|
| 旧 controller／転送子 | PID12764／22431，argv・開始時刻で同定して終了 |
| 旧 lock | daemon／tick の両方で解放確認 |
| 停止前後の既存 state | SHA-256 が同一 |
| 新 apply daemon | PID29527 |
| 元の絶対終了 epoch | 1795324658.879168，延長なし |
| v3 gate | live probe で未達を確認 |
| fixture | 36件 PASS，scratch と workstation |
| GPU job の操作 | 0 |
| Colab 新規 session | 0 |

## 遅延の原因と修正

installed Colab CLI v0.7.4 の download 実装を確認したところ，HTTP 応答を JSON/base64 として全量受け取った後で保存する方式でした．ITU EEG の原本4,650,970,671 bytes を直接回収していた子は約13分稼働し，local partial はまだ作られていませんでした．CLI 本体は変更していません．

controller 専用 protocol では，完成済み raw 台帳の source SHA・size を照合し，read-only の原本から最大64 MiBの chunk を作ります．各 chunk の offset・size・SHA と source identity を照合した後，workstation で1 MiBずつ追記します．confirmed offset と pending chunk の receipt を保存するため，中断後も確認済み prefix を再検証して続行できます．append と receipt commit の間で中断した場合も，tail を照合して残りだけを追記します．

ACK 中断で sidecar だけが残った場合は，要求と残存 sidecar の完全一致を確認したときだけ cleanup を完了します．source 原本は削除・変更せず，最終的な原本全体の SHA-256 が一致した後にだけ preservation complete とします．

## 実転送の確認

観測時点で 6 chunks・402,653,184 bytes の検証と local 追記を完了し，残りを転送中です．原本全体の preservation はまだ完了していません．

- 原本：bids_eeg_itu.tar.gz，4,650,970,671 bytes．
- 原本 SHA-256：`1e192b5e65931b9a44858acd987d540d3436de3fddf99454170b93434c2c99b5`．
- chunk 上限：67,108,864 bytes．
- local stream buffer：1,048,576 bytes．

これは取得済み Colab データを workstation へ保全する転送です．独立 EEG データが増えたものとして corpus 容量に加算しません．進捗は controller state の `raw_recovery_progress` と各原本の `.chunks.json` に残ります．

## v3 の停止 gate

`uniba_natural_v3_five_variant_evaluation` は，run `qwen3_8b_aicoding_uniba_natural_v3_fused_30e_v1` の `variant_evaluation_completion.json` と `best.pt` を照合します．設定ファイルの保存だけではなく，新 daemon 起動後の実 probe にこの gate が現れたことを確認しています．benchmark 担当へ反映完了を連絡済みです．

元の launch receipt と state の backup，停止・lock 解放の証跡，read-only tick，新 launch，実転送検証は workstation private に保存されています．[公開集計・source SHA一覧](summary.json) と [実装手順](../../controller/README.md) に再現情報があります．

新 daemon の最初の cycle は起動時に読み込んだ remote source を使います．その後の ACK 中断修正は，次 cycle で再読込みする形で配備しており，進行中の回収を二重起動していません．
