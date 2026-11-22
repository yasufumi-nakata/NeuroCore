# EEGMMIDB phase3：回収・Mac試用・停止までの完了記録

**追加EEG学習，固定した2モデルの評価，全成果の回収，M1での1コマンド試用，所有Colabの停止まで完了しました．** 学習と実行経路の成立を確認しましたが，自由な意図の読取りや日常の自律操作は実証していません．

## 今回の到達点

- SSLで実際に観測した独立なEEG区間は **29.8875時間**です．選択SSL重みから教師ありMI・LLM融合までの入力区間を重複除外すると，**29.985888889時間**です．原記録の時間やepoch反復は加算していません．
- 20epochのSSLと60epochのMI教師学習を完走し，LLM融合は登録した1,800秒で正常partial終了しました．選択fusedは完全なvalidationに基づくepoch11のbestです．
- 新2モデルを同じ未学習17名・1,489窓で，各条件1回評価しました．全5モデルの比較は[固定ベンチマーク結果](../../eegmmidb_phase3/RESULTS.md)へ記録しています．
- 全476ファイル，489,554,060 bytesの必須成果を回収し，元byteのSHAを照合しました．全checkpoint，optimizer/RNG，validation履歴，test予測，probe，失敗証拠，ソース，metadataを保持しています．
- M1でtrain3例と12回のfresh LLM routingが完走しました．**固定0.8閾値では12例すべて棄権**し，採用操作0，OS操作0でした．動作確認の完了と，操作成功件数を区別しています．
- 2026-11-22 **06:18:59 UTC**に所有Colabを停止しました．停止後は残高405.38 CU，usage rate0.00/hr，active assignments0です．

## すぐに試す

このworktreeのルートで実行します．固定モデル・固定train3例を使い，結果を新しい日時付きフォルダへ保存します．

```bash
python3 scripts/try_pj_xavier_recorded.py
```

[実機試用の条件・SHA・結果](../../integration/EEGMMIDB_PHASE3_RECORDED_TRIAL.md)に，実行契約と使う入力形式を記載しています．試用モデルはEEG側72,416パラメーターを更新し，1,711,376,384パラメーターのSmolLM2を凍結しています．自己教師学習・分類器の内訳は[学習モデルカード](../eegmmidb_phase3_training_20261122_v1/MODEL_CARD.md)と[パラメーターのCPU実測](parameter_counts.json)を参照してください．

## M1の実測と範囲

モデルとroutingを含む内側の実行は18.745秒，子プロセス全体は21.194秒でした．EEG3例のbatch呼出しは1,667.972msで，1例あたりの償却値は555.991msです．EEG取得・前処理・全harnessを含む連続運用の遅延ではありません．

MPSの観測値はdriver4,381,786,112 bytes，allocated3,423,507,712 bytesでした．これらはsampled値で，実際の全期間peakを保証しません．MPS allocatorの予算は8GiBです．同じworkstationで軽いCPU処理も行われており，無負荷の隔離測定とは扱いません．

## 停止時の運用追補

最初のSTOP helperは，M1試用後の空き容量が35GiB未満だったため，Colabへstopを送る前に止まりました．回収済み全476ファイルのhashは通過していました．元helperと失敗ログを保全し，**全回収byte・所有job終端・GPU空を再検証するSTOP専用処理**で所有sessionの停止を完了しました．新規学習・取得・回収の容量制限は緩めていません．

回収後・M1前の空き39,349,374,976 bytesに対し，STOP専用処理の開始時は37,028,466,688 bytesでした．差は−2,320,908,288 bytesです．同期間の変化という観測だけを記録し，M1処理や他の処理が原因であるとは断定していません．

根拠の全hash，選択モデルの学習経路，資源観測，停止receiptは[完了集計JSON](summary.json)にあります．この文書の作成による追加学習・推論は0回です．
