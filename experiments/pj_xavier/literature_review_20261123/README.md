# PJ-Xavier：一次文献取得・Scopus照合記録

更新日：2026-11-23（Asia/Tokyo）．最終Scopus UI確認：07:02:14 JST．

## 完了したこと

B案「EEG＋判定前の画面＋操作履歴を統合し，EEG表現をLLM/VLM内部へ接続して意図・次操作を推定する」に関係する**15件の独立した論文**を選び，全文PDFを取得しました．合計は**108,187,805 bytes，344ページ**です．既存の正規取得済みコピー3件を再利用し，残り12件は出版社・会議公式サイト・arXiv・著者の大学サイトから取得しました．

全15件でPDFシグネチャ，サイズ，SHA-256，ページ数，テキスト抽出，タイトル・版を確認しました．これは**15件すべての全文精読や研究再現の完了を意味しません**．重点4件（Shiinoki，UniMind，Chiossi，SENSE-42）は方法本文まで照合し，その他は書誌・一次資料の要旨または必要な方法箇所を確認しています．全文通読の件数は本取得台帳では認定していません．

Scopusの実検索では，選定15件のうち**Documentsに13件，Preprintsのみで2件**が一致しました．個別のScopusレコードURL・数字IDを直接確認できたものは5件です．**明示的なEID文字列の確認は0件**で，数字IDやDOIからEIDを推測して補完していません．一括エクスポートはScopusアカウントを要求されたため実施しておらず，新規の認証・登録は行っていません．

対象15件のPDF未取得は0件です．途中でPMCのPDF候補がHTMLを返した件とIEEE PDFがHTTP 502を返した件は履歴に残し，前者はNature公式PDF，後者は著者の大学サイトにある最終版で解決しました．PDF本文と保存先の詳細は私的な取得証跡で管理し，この公開ディレクトリには書誌・比較・検証メタデータだけを置いています．

## ファイル

| ファイル | 用途 |
|---|---|
| [REVIEW_REPORT_JA.md](REVIEW_REPORT_JA.md) | 統合査読報告．主要指摘10件，章別修正，必要実験，投稿方針 |
| [references.bib](references.bib) | 15件の引用用BibTeX．出版版・プレプリント版，著者順を固定 |
| [reference_manifest.json](reference_manifest.json) | DOI，公式URL，版，取得・照合状態，PDFのSHA-256・サイズ，Scopus状態 |
| [PRIMARY_PAPER_COMPARISON.md](PRIMARY_PAPER_COMPARISON.md) | B案に対する先行研究の重なり，必要な比較，過大主張を防ぐ条件 |
| [INDEPENDENT_METHODS_REVIEW.md](INDEPENDENT_METHODS_REVIEW.md) | 別担当による実原稿・実験方法の独立レビュー |

## 検索方法と実測結果

これは主題に沿った**目的選定の文献レビュー**です．網羅的な系統的レビュー，文献数による分野全体の推定，引用数による品質順位付けではありません．検索はworkstationの既存のScopus閲覧環境で実行しました．公式一次資料と実PDFを併用し，Scopus検索成功とPDF取得成功を別々に記録しています．

### Q1：直接先行研究

Title / Abstract / Keywords欄に以下の引用符付き句を入力しました．

```text
"Human intention inference with a large language model"
```

実測はDocuments **0件**，独立したPreprints（Beta）コレクション **1件**です．ShiinokiらのbioRxiv論文に一致しました．これは「論文が存在しない」ことや「査読済み論文である」ことを意味しません．本調査では同タイトルの査読済み出版版は確認できていません．

### Q2：中心論文の複合タイトル検索

```text
TITLE("NeuroLM" OR "UniMind" OR "CBraMod" OR "EEGPT" OR "DeWave" OR "COFETT" OR "Large Brain Model" OR "BENDR" OR "Evaluating EEG-to-text models" OR "The Perils and Pitfalls of Block Design" OR "Anticipation Before Action" OR "A multimodal human-computer interaction dataset" OR "OSWorld" OR "SeeClick" OR "Human intention inference")
```

Documentsは**26件**でした．表示件数を50件に変更し，26件すべてを確認したうえで，選定対象の**12件**についてタイトル・著者・掲載先を照合しました．別主題の意図推定，別のEEGPT応用，単語の偶然一致は今回の引用集合へ追加していません．初期の広い略語検索727件は探索用であり，採用件数やレビュー母集団には使っていません．

### Q3・Q4：COFETTとUniMindの補足

```text
TITLE("Is EEG-to-Text Feasible" OR "UniMind")
```

Documentsは**1件**で，ACL 2026のCOFETT論文に一致しました．Preprintsへ切り替えた際のUIは収録対象期間として `AND PUBYEAR > 2016` を付加し，**2件**（COFETT，UniMind）を表示しました．COFETTはタイトルに略語がないため補足検索が必要でした．出版版とプレプリントが両方見つかっても独立研究2件とは数えません．

検索日・コレクション・実測件数・観測済みレコードURLは[manifest](reference_manifest.json)に記録しています．Scopusの可変な検索IDは検索式の代わりには使っていません．

## 版と書誌の注意

- **UniMind**はarXiv 2506.18962 **v2（2026-05-11）**を採用しています．著者は9名で，v1の著者一覧を流用していません．要旨ページとPDF/HTMLの平均改善率には12%と11%の差があるため，その数値を本レビューの根拠にしません．
- **Shiinoki**はbioRxiv **v2（2025-06-10）**です．プレプリントとして引用します．
- **Joら**はScientific Reports **16:350（2026）**の出版版を採用します．オンライン公開日は2025-12-01です．関連プレプリント *Are EEG-to-Text Models Working?* は同系列の版として管理し，独立論文数を増やしません．
- **DeWave**は会議の書誌タイトルとPDF／Scopusのタイトル表記が異なります．BibTeXは公式会議ページのタイトルを使い，別表記をmanifestに記録しました．
- **LaBraM**のLi-Ming Zhaoと**SeeClick**のYantao Liは実PDFの著者表記を優先しました．**EEGPT**のGuagnyu Wangは公式PDF・書誌の綴りを保ち，黙って別表記へ直していません．

## 原稿の章別の使い方

| 原稿の論点 | 中心文献 |
|---|---|
| EEGをLLM内部へ接続する既存方式 | UniMind，NeuroLM，DeWave |
| 文脈を使う意図推定と判定前EEG | Shiinoki，Chiossi |
| EEG encoderと継続SSLの比較 | CBraMod，LaBraM，EEGPT，BENDR |
| EEG追加情報の識別と評価の妥当性 | Jo，COFETT，Liらのblock-design研究 |
| 強い画面・操作系baselineと行動評価 | SeeClick，OSWorld |
| 実生活に近い多モダリティ計測の参考 | SENSE-42 |

各論文は異なる課題を扱います．別データセットの正解率や再構成誤差を横並びにしてB案の優越性を主張しません．B案の新規性・実証条件は[比較表](PRIMARY_PAPER_COMPARISON.md)で分けて示しています．
