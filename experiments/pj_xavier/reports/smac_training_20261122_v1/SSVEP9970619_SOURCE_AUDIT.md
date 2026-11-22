# Figshare 9970619：S Mac取得済みSSVEPの学習適格性監査（2026-11-22 22:25 JST）

## 結論

**44記録のヘッダーを全件確認しましたが，新しいgradient学習は開始しませんでした．** 物理Vへの校正証拠，固定被験者単位split，元波形の前処理由来・因果性が未確定です．取得済みの波形時間と学習適格時間を混同しません．元のQCを成果に合わせて緩めたり，EEGLABの慣習から単位倍率を仮定したりしません．

S Macの取得ノードsmac02のFigshare 9970619 v1は**88ファイル／1,708,501,504B取得完了**です．smac35へは，原データをworkstationに中継せず，44個の`.set`ヘッダー（25,588,704B）と代表1個の`.fdt`（38,602,400B）のみcluster内で直接SHA256照合転送しました．残る43個の`.fdt`はsmac02に保持したままで，今回の監査では読み込んでいません．転送時の最初の空ファイル作成エラーは記録し，作成した自分の空試行ファイル0Bのみ削除後，完全な転送に成功しました．取得済み全88ファイルを本監査で再hashした意味ではありません．

## `.set` 44件から観測した構造

| 項目 | 観測値 | 解釈 |
| --- | ---: | --- |
| 連続 `.set` | 44件 | 11種類のファイル名被験者token×4実験；11 tokenは本人確認済み実人数ではない |
| 各ファイルのch | 73 | 先頭64はEEG，末尾はEXG1–EXG8／Status（EEGに加算しない） |
| サンプリングレート | 200 Hz | 配布された**後処理済み**ファイルの値 |
| 記録時間の合計 | 28,817秒（約8.005時間） | 44ファイルの名目記録時間であり，unique学習時間ではない |
| montage/reference | 全44件で共通／`common` | 配布ヘッダーの文字列のみ；元のreference処理の再現は未認証 |
| `.set`の実event | 0件 | cueや認識応答の時刻を推測でgoldへ変換しない |
| 明示した電圧unit field | 0件 | `V_x_1e4`への変換不可，学習投入を保留 |
| `.fdt`理論値 | 1,682,912,800 B | 44記録をfloat32×73chで仮定した期待値．本実行では他43件の実検査なし |
| S Mac新gradient時間 | **0時間** | preparation/metadata監査を実学習と呼ばない |

代表1件の`.fdt` sampled native値は有限でしたが，先頭64chの絶対値中央値約4438.12，99%点約17963.78，最大約18343.29で，**数値単位・reference/offsetを確定できません**．大振幅を理由に任意の再倍率・clipを施していません．この数値は全44記録の品質合格を意味しません．

## 元研究とオンライン境界

一次論文（Minami，Azuma，Nakauchi，PLOS ONE，2020）では，BioSemi ActiveTwoで64 EEGチャネルを**512Hzで収録**し，EEGLABで連続データを**200Hzへdownsample**したと記載されています．また，元論文の解析では80μVを超えたepochの除外や1–20Hzの10次Butterworth等が説明されていますが，これは我々の新学習プロトコル/QCに自動移植する値ではありません．**200Hzの配布ファイルを後から因果filterへ通しても，元のoffline resamplingによる将来情報依存を消せません**．本資料から自由意図・任意OS操作の教師goldも生成できません．

- 元論文：[Steady-state visually evoked potential is modulated by the difference of recognition condition](https://doi.org/10.1371/journal.pone.0235309)（2020-07-02）
- 公開データ：[Figshare 9970619 v1](https://doi.org/10.6084/m9.figshare.9970619.v1)

## 再現性と次段階

新しい`next_multichannel/audit_smac_ssvep9970619.py` は実行hostをsmac workerに限定し，smac19／workstationを拒否，44 `.set` の正確な件数，チャンネルと補助信号境界，200Hz／73ch／1連続trial，元 `.fdt`参照名を検査します．`.fdt`信号は開かず，個別データとファイル名を含むmanifestはS Mac privateに保持します．

- 実smac35出力：`~/pj-xavier/smac35/inspection/ssvep9970619_20261122/header_audit_full_v1.json`
- 出力SHA256：`6aae94f6df6a319014cd8eed296afa6e0f355289c93182e6e1e2a508b7f32137`
- 実行コードSHA256：`f1515609d4938354ced0b4075424c0959954eeec0c18c8283883a229bca66325`
- 状態：`hold_missing_unit_evidence_and_frozen_split`，`fit_allowed=false`．

次は原著と配布者の物理量・reference/元処理の定義，原波形に対する被験者単位・must-link/payload重複，train-only QCと新holdoutを別runで確定し，条件を満たした場合にのみ別checkpointでoffline SSLを学習します．今回の状態を既存P300 train，P300 validation，Phase1–3の結果に遡及適用しません．

## 同時に確認した取得状況

2026-11-22 **22:20 JST**，smac31から全37台の実DBと実ファイルを照合し，取得完了 **99,527ファイル／839,585,508,076B**，partial **203,790,046,364B**，37 supervisor alive，complete欠損0，サイズ不一致0，観測エラー0でした．この値は全modalityの保存量であり，EEG固有時間でも学習量でもありません．BCI8089820（27,475,466,178B）は部分取得 **5,241,833,568B** で未完了のため，追加学習対象へ入れません．
