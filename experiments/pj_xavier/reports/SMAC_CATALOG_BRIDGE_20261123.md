# PJ-Xavier：S Mac ソース差分導入と ds005841 公式manifestの取得

観測日：2026-11-23 JST．対象：既存のS Mac学習・取得を中断しない，出典管理とsource候補のCPU/metadata作業．本作業は実EEG取得や学習ではありません．

## 実施結果

- **公式ソースを読み取る独立bridgeの実機導入・動作確認済み**：smac31の `/Users/yasufumi/pj-xavier/catalog-bridge-v1/` へ，認証済み接続でCSV 2件・stdioのみの比較コード・公式OpenNeuro object-listをコピーしました．全ファイルで転送後SHA256を照合しました．実行は `--check-only` と，既存領域へ触れない独立reportの双方が成功しました．さらに公開用の`public_20261123`２コードをそのまま実機へ導入，実行0終了・件数一致を確認しました．
- **ソース数の二重計上を排除**：青山研の高信頼5,165行は既存S Macカタログ5,165行に全件対応し，強いキーで5,162件，題名と正規化URLの完全一致aliasで3件でした．全件に公開sourceのraw/学習適格性があるという意味ではありません．広域発見22,327行では8,535行が索引と一致，未対応13,792行は探索metadataであり，そのまま新しいdownload対象やEEG時間にできません．未対応かつ`immediate`・`eeg_text_match=True`は0行でした．
- **既存source ds005841のfiles0を解消する一次metadata**：公式のOpenNeuro S3 ListObjectsV2（公開）と`dataset_description.json`を取得・ページ分割検証し，`EEG Experiment measuring ERPs in VR`，CC0，DOI `10.18112/openneuro.ds005841.v1.0.0`，**872 objects，288 BDF，7,889,279,183 bytes**を確認しました．公開object listは元カタログ `003254` に対応し，**新しいカタログ行やrawを作成していません**．フルmanifestのSHA256は `ae581279f3320129a57274943d33f540c0ade1ed94b590bd13d7262046326c17`．S3 ETagを誤って検証済みMD5としていません．
- **研究資源の隔離**：37台の既存acquisitionのGuardian (`pid=87453`，直近実観測 `2026-11-23 07:20:54 JST`，37 window active) を確認し，変更・再起動しませんでした．本runの新規ダウンロード0B，preparedデータ0，MPS fit0step，評価用の新規ヒト意図label 0．
- **公開可能なプログラムを固定**：`experiments/pj_xavier/catalog_bridge/` にstdlib-only source-key正規化・独立差分CLI・公式S3 metadata probeと，人工fixtureのみの5件のunittestを追加．smac31で公開用2コードのactual read-only実行も一致しました．レポート・コードは公開可能なaggregateのみで，EEG payload・参加者ID・重み・秘密の接続情報・個々のS3 file名を含みません．

## 取得・学習の次の安全な段階

1. ds005841のsource version（既存entryと公式DOI）を照合し，入手条件と差分files，チェックサム方式，companion，EEG montage・unit，must-link/split，固定train/val/test denyを調べます．既存カタログの元行をそのまま保ち，承認済み固定planを非破壊のversioned amendmentで更新します．**7.889GBを今の取得済み容量へ加算しないでください**．
2. 前処理・QCで適格trainが確認されたsourceのみ，データが存在するS Mac workerでGPUの別jobとの衝突を検査し，MPS forward/backward memory probe → source stage fit → 同じseedの固定validationの独立比較へ進みます．Design Emotion31578517は17train全てが固定QCで適格窓0のため，再fitしません．
3. NeuroCore向けPJ-Xavier **B案**として，EEGと操作前画面／履歴が同期した人間labelの有無を調査し，強い非EEG対照と内部融合の対応付き下流比較を実施します．波形MSE低下は意図推定改善の代替になりません．

## 根拠と成果物

- S Mac実機コード・未公開集計：`~/pj-xavier/catalog-bridge-v1/README.md`，`reports/catalog_diff_20261122T222228Z.json`，`ds005841_official_s3_manifest_metadata.json`．
- 公開再実行コード：`experiments/pj_xavier/catalog_bridge/README.md`，`catalog_identity.py`，`catalog_diff_standalone.py`，`s3_manifest_probe.py`．
- 公式Dataset metadata：`https://s3.amazonaws.com/openneuro.org/ds005841/dataset_description.json`；公式Source: `https://openneuro.org/datasets/ds005841`．
- Google Doc実行タブ（既存追記済み・readback済み）：`https://docs.google.com/document/d/1kGlNqvZEg_AzSYKGTiGF3Z0HcVIxNGNimNwpR6w1680/edit?tab=t.7cvfsge8pwil`．

現時点の全体取得は完了していません．2026-11-23 07:02 JSTの最後の既知の37台fleet snapshotでは，complete 242,699files／2,699,118,609,941B，partial 832,965,755,329B，alive37／37，missing/size不一致0でした．この実測より後に本runで37台のfull snapshotを取り直したという意味ではありません．
