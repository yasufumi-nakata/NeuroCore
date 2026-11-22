# 第2回：公開EEG候補の取得キュー追加

2026年11月22日13:02 JSTに，最新の固定metadataから **174 releases・182,443 file jobs** を既存の取得台帳へ追加しました．smac19を計算計画から外して進める新フェーズの一部です．この処理は取得候補を追加したもので，本体の取得完了や学習入力への採用とは別です．

| 範囲 | 件数・容量 |
| --- | ---: |
| 全catalogと照合したID | 5,165，不足・余剰0 |
| 今回追加したrelease | 174 |
| 今回追加したfile jobs | 182,443 |
| 追加分の公開サイズ既知の合計 | 8,351,215,406,047 B |
| 追加分のサイズ未確定 | 42,360 files |
| 追加後のrelease | 783 |
| 追加後のfile jobs | 650,483 |
| 追加後の公開サイズ既知の合計 | 24,172,262,716,513 B，約24.17 TB |
| 追加後のサイズ未確定 | 42,398 files |

これらの容量は，複数版・動画・補助情報などを含むrelease全体の候補です．元EEGの重複なし時間，純EEG bytes，取得済み容量への換算はしていません．このmetadata追加によって取得済み容量や検証済みEEG時間に加えた値は0です．EEGMMIDBは独立した取得台帳で管理し，このキューへ重複追加していません．

## 実行と独立検証

734,280,014 Bのfresh metadataを固定snapshotへ保存し，既存の追加専用実装でplanを作成しました．applyは19.35秒で正常終了しています．追加後に別のread-only処理で **既存468,040行と新規182,443行のURL・size・checksum・provider identityなどの固定metadata** を全件照合し，不一致0を確認しました．

既存download行の更新・削除は行わず，取得worker PID8814のlaunch receiptと実commandも保持されています．元deadline1795322496.632661，元転送量予算64,062,645,375 Bを変えていません．今回のmetadata追加ではworkerの起動・停止，rawの移動・削除・全量再hashは0です．

サイズ未確定42,398件はmetadata_blocked，必須Guestbook342件はaccess_action_requiredとして保持します．catalog000352の保存namespaceがfilesystem component上限を超える問題も，元情報を保持したまま保留です．未確認のscopeを学習用データと認定していません．学習入力への採用は，個別の波形品質・ラベル・時刻・分割の検証から決めます．

workstationで35 GiBを残す方針を維持し，今回のsnapshotとappendには，学習準備の増分に余裕を持たせる39 GiBのreserveを指定しました．候補全体を現ストレージへ保存できた，という結果ではありません．

## 根拠

[集計とSHA](summary.json)に全件照合の結果，snapshot／plan／apply receiptのSHAを保存しています．詳細なURL台帳とreceiptはworkstationのprivate領域です．使用した[append_public_queue.py](../../acquisition/append_public_queue.py)は既存の検証済みsourceを変更せず実行しました．今回の実データでの独立照合を追加し，無関係な学習・評価を再実行していません．
