# 追加の連続 EEG 学習：実行前の固定と検証

科学設定は phase2 の test 結果を見る前に固定しました．新規 encoder のマスク自己教師学習，運動想起の教師あり学習，凍結 SmolLM2 への接続を順に行います．この記録は学習前の検証です．

workstation の Python 3.12.13／PyTorch 2.15.0 で focused tests 17 件が通過しました．同じ seed で作った phase2 ScratchDecoder.encoder と新規 MaskedEEGModel.encoder の全 parameter・BatchNorm buffer の一致，固定マスク，勾配，元モデルの凍結，データ境界，予定時間停止だけを引き継ぐ処理を検証しています．

CPU の小型 Llama fixture で，A/B の二行だけを評価する出力層と元の全語彙出力を比較しました．logits・loss・encoder／projector の勾配を事前に固定した許容差で比較しています．これは実データに対する性能検証ではありません．実際の L4／BF16／固定 SmolLM2 でも同じ種類の検査に合格した場合のみ最適化を採用し，不合格なら元の全語彙経路を使います．bit 単位の同一性は主張しません．

データ追加の量と独立した区間会計は [前処理結果](../eegmmidb_phase3_preparation_20261122_v1/README.md)，固定した科学設定は [protocol](../../eegmmidb_phase3/protocol.scientific.v1.json) にあります．GPU 学習・test 評価・実機試用の完了は，この検証結果だけからは主張しません．
