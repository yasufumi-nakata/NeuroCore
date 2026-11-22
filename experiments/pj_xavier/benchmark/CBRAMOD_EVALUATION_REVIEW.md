# CBraMod 接続方式の評価 API レビュー

**固定した新 module に，既存の 5 条件評価を妨げる重大な API 不整合や，正解が prompt に入る問題は確認しませんでした．このレビューはモデル性能の合格判定ではありません．**

公式 EEG weights と実 train 3 窓を含む CPU 検証記録は 13/13 合格で，出典 SHA を照合しました．LLM の勾配 fixture は小型の random GPT2 であり，pretrained SmolLM の性能測定ではありません．

CPU 固定 encoder が親 model の device 変換で移動しないこと，encoder／LLM を凍結して projection へ勾配が流れること，A/B/C の単一 token で学習と推論の式が一致すること，protocol と runtime identity が照合されることを確認しました．共有 evaluator と旧 EEG model は変更されていません．

元配列は 1.75 秒の 8×448 を保持し，入力に使うのはその最後の 1 秒です．同じ変換を通常・shuffle・実 raw の 10 秒前の窓へ適用します．source／array／checkpoint hash と元 test ID の検証は既存の契約を使用します．

v3 は既に結果を見た集合なので，今回の比較は探索的です．EEG encoder と token scoring を同時に変えており，改善があっても片方だけの効果とは判定できません．offline filtering の因果性，潜在意図，実ハーネス完了，日常使用時の calibration は引き続き未検証です．

詳細と固定した source SHA は cbramod_evaluation_compatibility_audit.json に保存しています．
