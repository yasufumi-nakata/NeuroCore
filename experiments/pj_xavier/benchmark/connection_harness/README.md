# Connection 課題の独立 plumbing 検証

**公開仕様から新しく作ったテストを，手製の Java fixture で動作確認するための小さなベンチマークです．EEG，参加者の解答，LLM の推論を使いません．元研究の自動テストを復元したものでもありません．**

`public_input/task.json` は課題の公開入力です．根拠となる HTML，UML，API stub の member 名と SHA-256 を記録しています．WiFi の優先，利用可能な接続の試行，失敗時の次候補，成功時の `connection` 設定，候補が尽きたときの `RuntimeException` は公開説明と UML に対応します．名前は公開 API の小文字・空白なしの規則に従います．不明なネットワーク種別，入力 `null`，既存接続の再利用など，仕様が明らかでない動作はテストに追加していません．

出典は [AI-assisted coding dataset，Zenodo 20801101](https://zenodo.org/records/20801101) の resources archive です．ここにある `ConnectionAPI.java` と `DeviceAPI.java` は，公開された class／method signatures に合わせて独立に書いた最小 skeleton です．元ファイルの Javadoc，著作権表記，実装をコピーしていません．公開元のコード専用 reuse license は本監査では確認しておらず，元コードをこちらのライセンスへ変更したものとは扱いません．原 API member の内容 hash と所在は `task.json` に保持し，課題の設計は公開元に帰属します．

`oracle/` は新規の 8 ケースです．`fixtures/` は正解例，優先順位の誤り，無条件の成功，偶発的な NullPointerException，構文エラーという 5 つの手製例です．参加者の提出コードは読みません．将来のモデルに渡す入力は `public_input/` だけとし，oracle と fixture の実装は渡さない構造です．失敗時は exact `java.lang.RuntimeException` を明示的に投げることをこの独立テストの規約とし，偶発的な例外の subclass は成功として数えません．これは新しいテスト側で明示した規則で，元研究の未取得 oracle の規則だとは主張しません．

```bash
python experiments/pj_xavier/benchmark/connection_harness/run_plumbing.py \
  --output private/xavier/connection_harness_v2/result.json
python -m unittest discover \
  -s experiments/pj_xavier/benchmark/connection_harness \
  -p test_run_plumbing.py -v
```

既存の `java` と `javac` が必要です．欠けていれば `blocked` で終了し，自動インストールしません．各 fixture は別の一時ディレクトリでコンパイル・実行し，終了後に除去します．既存結果の上書きは拒否します．JVM heap は 128 MiB，コンパイルは 15 秒，実行は 5 秒に制限しています．heap 上限はプロセス全体の実測ピークメモリを意味しません．

**この実行器は，同梱の確認済み fixture だけを扱います．任意の投稿コードを安全に実行する sandbox ではありません．** 外部コードの実行には，ネットワーク遮断，CPU・メモリ・ファイル制限のある別の隔離環境が必要です．この版はクラウド API，デスクトップ操作，参加者データへのアクセスを行いません．

workstation の Corretto JDK 11.0.25 で，正解例 8/8，優先順位の誤り 7/8，無条件成功 2/8，偶発的な NPE は 0/8，構文エラーはコンパイル失敗という期待どおりの結果でした．単なる `test_failure` ではなく，失敗した case の集合まで期待値と照合します．v1 は RuntimeException の subclass も期待する失敗として数える弱点があったため，実行済み source と receipt を保全し，この v2 を新しい出力先で検証しました．成功を報告できるのはこの plumbing と誤り検出です．EEG からの意図推定，エージェントの自律操作，課題達成率は未評価です．外部 EEG 評価を見た後の追加工学検証として位置づけ，既存モデル・閾値・分割は変更していません．

出典監査と残る不足項目は [`../CODING_HARNESS_SOURCE_AUDIT.md`](../CODING_HARNESS_SOURCE_AUDIT.md)，集計は [`../connection_harness_plumbing_aggregate.json`](../connection_harness_plumbing_aggregate.json) を参照してください．
