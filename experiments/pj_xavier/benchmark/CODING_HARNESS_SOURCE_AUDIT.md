# AI-assisted coding 公開資料から構成できる harness と残る不足

## 結論

**課題仕様を根拠にした独立のコード実行 harness は作れますが，取得済みデータだけで「EEG を使うエージェントが課題を達成した」とは検証できません．** 公開された Connection 課題の説明・UML・初期 API から新しい 8 ケースを作り，手製の Java fixture 5 種で動作確認まで完了しました．これは synthetic／non-EEG の工学検証です．

既存 EEG モデルの学習，checkpoint 選択，閾値，公開済み ITU 外部評価は変更していません．本 harness は外部評価結果を見た後に追加した探索的な基盤整備です．参加者の最終解答を参照してテストや fixture を作っていません．

## 調査対象と根拠

取得済み [Zenodo 20801101](https://zenodo.org/records/20801101) の resources，scripts，両 site の code／behavioral／metadata，README の 9 ファイルを SHA-256 で再計算し，controller の取得・保存台帳と一致することを確認しました．根拠となる公開 member の個別 hash，archive 内の件数，行動ログの列名は [`coding_harness_source_audit.json`](coding_harness_source_audit.json) にあります．再検証用の [`audit_coding_harness_sources.py`](audit_coding_harness_sources.py) は archive を展開・実行せず，参加者の Java 本文，GPT prompt，キー内容を読みません．

| 構成要素 | 実際に確認した内容 | 評価への扱い |
|---|---|---|
| 課題の仕様 | Connection と Shopping の説明，UML 図，表示用画像 | 公開された課題入力として利用できます．この版のテストは Connection の明確な規則だけを使います． |
| 初期コード | ConnectionAPI，DeviceAPI，StockAPI の抽象 API | API の形と呼び出し規則を確認できます．完全な解答ではありません． |
| 実験用 UI | oTree／Flask，エディタと AI チャットへの接続，実験開始・終了などの処理 | 元実験の環境構成を理解できます．EEG による操作 policy は含まれていません． |
| 自動テストの呼び出し口 | `run-tests.sh` が `/testing/tests/*.java` と `tests.Main` を参照 | 呼び出されるテスト本体が検査対象の release にありません． |
| テスト実装の所在 | resources 129 regular files，その中の英語・イタリア語 app archive 各 52 files を検査 | Java テスト source と Java build manifest はいずれも 0 件です．元テストが世界中に存在しないと断定しているわけではありません． |
| 手動評価 | rubric workbook と site 別のコード評価／提出物 | 人手の採点と実行可能な自動 oracle は分ける必要があります． |
| 行動データ | キー・マウス・oTree・GPT イベントの時刻，UniBa では prompt text も記録 | 対応する時刻付きエディタ全文，ファイル差分，実行済みテストの結果を保証する schema ではありません． |

`TaskDescription1.html` は両方使える場合の WiFi 優先を明示します．Connection の UML activity は利用可能性を DeviceAPI，接続成功を ConnectionAPI に分け，失敗時の次候補と候補切れの例外を示します．これらを新しいテストの根拠としました．元の UI はテスト関連領域を CSS で非表示にしていますが，その事実から参加者がどのコードをどの時刻に実行したかまでは導けません．

## EEG と課題達成をつなぐために不足しているもの

1. **判定時点以前の正確なコード状態**がありません．課題の starter API はありますが，作業途中のファイル，カーソルや選択，未保存の編集内容，clipboard，補完結果などを完全に再現できる時刻付き snapshot／patch 履歴は，検査した structured log にありません．動画の OCR から正確な状態を復元できるかは未検証です．最終提出コードや将来の GPT prompt を事前環境として渡すと未来情報が混ざります．
2. **実行可能な元のテスト oracle**が release 内にありません．手動点数を自動テストの合否と同一視できません．新しくテストを作る場合は，独立実装した oracle と明記する必要があります．
3. **エージェントが実行する意味のある指示の正解**がありません．現在の AI 送信・キー・クリックのラベルは観測イベントです．何を直すか，どの提案を採用するか，作業を止めるかという潜在的な意図の正解に改名できません．
4. **新しいエージェント操作による結果**が記録されていません．過去の人間による最終成果を，新しい EEG policy の反実仮想の成功率として使うことはできません．
5. **リアルタイムの入力可用性**が未実証です．既存 EEG は offline filter の未来参照を含み得るため，公開資料に基づく仮定の遅延と実証済みの causal preprocessing を区別したままにします．

## 実装した独立の動作確認

[`connection_harness/`](connection_harness/) に，公開入力，新しい oracle，手製 fixture，実行器を分けて保存しました．workstation の既存 Corretto JDK 11.0.25 を利用し，別々の一時作業領域で実行しました．

| 手製 fixture | 実測結果 | 意味 |
|---|---:|---|
| 公開仕様に沿う実装 | 8/8 通過 | 実行経路と正の fixture が成立しています． |
| MobileData を先に選ぶ実装 | 7/8 通過 | 両方成功する条件の WiFi 優先違反を検出しました． |
| 常に WiFi 成功を返す実装 | 2/8 通過 | 利用可能性，失敗，fallback の違反を検出しました． |
| 偶発的な NullPointerException を起こす実装 | 0/8 通過 | 失敗が期待される条件でも，偶発的な crash は成功と数えません． |
| 構文エラーを含む実装 | コンパイル失敗 | 実行できなかったケースを得点と混同せず記録しました． |

8 ケースは，WiFi 優先，WiFi のみ，MobileData のみ，WiFi 失敗後の fallback，候補なし，全候補失敗，利用不能な各 type を fallback に使わないことを確認します．`null`，未知の type，再接続など，明示されていない動作には規則を追加していません．

合否を数える前に，各ケースが 1 回ずつ報告されることを検査します．失敗時は exact `java.lang.RuntimeException` を明示的に投げることを独立 oracle の規約に置き，偶発的な subclass を成功として数えません．この明示規則を元研究の未取得 oracle の規則とは主張しません．重複／欠落，非 boolean 判定，任意の実行対象 path，候補入力への oracle 混入に加え，失敗 case の集合と NPE の 0 点を確認する 6 つの Python fixture も通過しています．コンパイル 15 秒，Java 実行 5 秒，JVM heap 128 MiB に制限しています．実行器は同梱 fixture 専用であり，敵対的な任意コードを実行する sandbox としては使用できません．

先行する v1 実行には RuntimeException の subclass も成功と数える弱点がありました．v1 の source・receipt を private snapshot に保全し，独立レビューを受けた v2 を新しい出力先で再検証しました．EEG・LLM 推論は再実行していません．公開集計は v2 を示し，旧実測の hash と訂正理由を保持します．

実測は [`connection_harness_plumbing_aggregate.json`](connection_harness_plumbing_aggregate.json) にまとめました．EEG 入力 0，モデル推論 0，参加者解答の読込 0，ネットワーク呼び出し 0，デスクトップ操作 0 です．この 8/8 を EEG task completion として集計しません．

## 次に実装できる最小の実課題 harness

最初の実課題は，同じ公開仕様で**新規に開始するセッション**に限定すると，途中コードの履歴がない問題を避けられます．初期ファイルと課題仕様を hash で固定し，エージェントには公開入力だけを渡します．テスト oracle は別領域に置き，生成された patch はネットワーク遮断・資源制限のある実行環境で適用・検査します．合成 fixture の成功と，モデルが新しく生成した patch の成功を別々に記録します．

EEG と結びつける段階では，少なくとも `task_id`，`decision_id`，時刻付きの事前コード snapshot/hash，EEG の原信号 hash・区間・feature availability，提示した環境の cutoff，候補操作，実行された操作，実行後だけに得られるテスト結果を保存します．実行後の情報はその decision の入力に戻しません．人が実際に選択した指示と，AI が補助生成した弱教師ラベルも別にします．

まずコード生成のみの context-only 基準を凍結し，同じ課題・同じテスト・同じ計算予算で EEG 付き，EEG のみ，記録内 shuffle，時刻ずらしを比較します．EEG 学習と評価で参加者・session・課題の分離を固定し，未記録の idle 区間から誤作動頻度を推測しません．追加の real EEG と操作結果が揃うまでは，課題完了率の EEG 増分を `blocked` のままにします．

新しいセッションの記録器と隔離した patch 実行環境が次の実装対象です．今回の小さな工学検証は既存 workstation だけで完了し，GPU を使用していません．それ以降の自律操作性能は，実際の収集と評価が終わるまで未実証です．
