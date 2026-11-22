# ITU：独立したサイト・装置での検証候補

## 確認できた範囲

**EEG 全26記録は21.3061794705時間です．固定した時刻同期検査には25記録，20.5167371962時間が合格しました．** ITU の参加者は全員，外部サイトの test 候補として扱います．この監査では，モデルの学習・モデル選択・予測・性能測定を行っていません．

| 項目 | 実測・検証結果 |
|---|---:|
| 完全取得した EEG archive | 4,650,970,671 bytes |
| 原波形 binary | 5,183,844,600 bytes，補助チャンネル込み |
| 全 raw 記録 | 26，重複 binary 0，数量検査の保留 0 |
| 収録時間 | 21.30617947048611 h |
| 各記録のチャンネル | EEG 64＋補助 2 |
| 各記録のサンプリング | 256 Hz |
| UniBa と共通の8 EEG／otree_phase | 26/26 記録に存在 |
| 同期基準を満たす記録 | 25/26，20.516737196180557 h |
| 合格記録における最大同期誤差の範囲 | 0.9844〜14.7128 ms |
| 不合格1記録の最大 leave-one-anchor-out 誤差 | 4,302.1625 ms |

公開元は [AI-assisted coding replication package，Zenodo 20801101](https://zenodo.org/records/20801101) です．数量は，完全な archive を再検証し，ヘッダのサンプル形式・チャンネル数と binary の実 byte 数から算出しました．EEG と補助信号の同じ時間を重複加算していません．全 binary の SHA-256 を取り，BIDS のチャンネル順・種類・ヘッダとの整合も確認しました．元サンプルの有限性やアーチファクト品質を全量検査した値ではありません．

## Export recipe と時刻同期

公開 `scripts/01-b_load.ipynb` の `save_eeg_data` は両サイトで共有され，`filter(3, 40)` に続いて50・100 Hzのnotchを適用しています．実ITUヘッダの256 Hzと，公開依存関係に合わせたMNE 1.10.2での再構成では，423 tapsと1,691 taps，合成2,113 tapsのゼロ位相FIRとなり，未来側のsupportは4.125秒です．根拠のhashと確認箇所は `itu_recipe_evidence.json`，係数と再構成結果は `filter_support_evidence.json` にあります．

これは**公開binaryがそのrecipeと依存関係から作られた場合の条件付きの結果**です．原XDFは公開物に含まれず，binaryとrecipeの完全同一性，online causalityは確認していません．ITUはA2を物理参照とする64ch装置，UniBaは8chのdry electrodeであり，装置間の分布が同等であるとも判断していません．

同期検査では，凍結済み `prepare_ai_coding.py` の関数をそのまま使用しました．一意に対応するphase anchorを最低4個，最低120秒の範囲から取り，affine時計変換を求め，全anchorの残差とleave-one-anchor-out誤差を検査します．基準はUniBaと同じ100 ms，5,000 ppm，phase scale 1です．ITUの結果を見て基準や補正量を変更していません．MNE 1.13.2で読み出したfs／n_timesが，独立したbinary数量検査に一致することも確認しました．

この合格が示すのは，公開EEG内のphase channelと公開behavioral phase timestampの整合性です．原XDFや独立したハードウェア時計との同期を証明するものではありません．

## ファイルと再実行

- 数量監査：`audit_brainvision_archive.py`．CPUの回帰試験9件が通過しています．
- 同期監査：`audit_itu_external_site.py`．公開EEGとbehavior archiveを再ハッシュし，専用rootへ安全に展開します．共通チャンネル・元波形SHA・MNEの長さ・同期を検査します．
- 公開集計：`../reports/ai_coding_itu_audit.aggregate.json` と `../reports/ai_coding_itu_external_clock.aggregate.json`．
- workstationの全記録監査：`private/xavier/ai_coding_itu_quantity_audit_v1/` と `private/xavier/ai_coding_itu_external_clock_v1/`．archiveを回収した後，すべての監査ファイルのSHAを再照合しています．
- Colabの検証済みraw root：`/content/xavier/verified_ai_coding_itu_v1/`．モデル入力窓は作成していません．

実際に実行したargv，開始・終了receiptは各監査のprivateディレクトリに保存しています．出力を保存したまま再実行する場合は新しい監査出力ディレクトリを指定します．Colabへのアクセスは引き続きworkstationのColab CLI経由で行います．

```text
EEG archive SHA-256:
1e192b5e65931b9a44858acd987d540d3436de3fddf99454170b93434c2c99b5

Quantity full audit SHA-256:
328aa2edc840a6acfe8c4e0f5a336e0266e62d562a1db51bd7afce9acff58d45

Clock full audit SHA-256:
b7d2bd664c661c2328f5394a7f3a2c24d8177e154974b21ef86cf4788c7ce916
```
