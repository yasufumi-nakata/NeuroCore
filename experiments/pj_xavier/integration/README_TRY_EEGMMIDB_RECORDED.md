# EEGMMIDB Phase 2：Macでの記録済みEEG試用

`try_eegmmidb_recorded.py` は，回収・検証済みの fused model を M1 の MPS で動かすための独立wrapperです．既存のUniBa／ITU／CBraModの試用コードと，凍結済みのphase2モデル・評価コードは変更しません．

## この試用で確認すること

固定manifestのtrain IDを，seed 20261122とIDだけのSHA-256で並べ，先頭3例を使います．正解classや成績を見て選びません．64 channel×600試料，200Hz，物理単位Vの実配列を，同じval-selected checkpointへ入力します．

3例それぞれで4種類の均衡した環境mappingを作り，凍結Smolへ**実際に12回のgreedy routing推論**を行います．以前の16件のrouting cacheは再利用しません．環境と予測されたcontrolだけをpromptへ入れ，gold，participant，trial ID，時刻は渡しません．X/Y以外の返答と，固定threshold 0.8未満の予測は状態を変更しません．厳密にX/Yと解釈できる返答はメモリ内の環境状態を更新し，別のoracleで選択先を照合します．棄権した例でも12回の実応答は収集し，実行採用とは別に数えます．

この3例は**学習例の動作確認**です．held-out精度，自発的な意図，実OSでの課題達成を測るものではありません．0.8は固定した実験上の採用基準であり，校正済みの意図確率ではありません．

今回の実測は `private/xavier/eegmmidb_phase2_v1/recorded_trial_m1_v1` に保存済みです．20.53秒，exit 0，12 fresh callsで完了し，3例すべて0.8未満のため12episodeとも棄権しました．選択操作・成功episode・追加test forward・実OS操作・cloud API呼出しは0です．summary SHAは `faea2ee2641012602a333d069d1bf04bcec1286290a2f7973cfda8e9d78d243c`，completion SHAは `52f18e7e857f7059179d320ecb85db52962c577085f6f80de2cc8b8218a3e86b` です．これは実行経路の確認であり，学習例の成功率を性能と呼ぶものではありません．

## 実行前に必要な証拠

`prepare` は次をhashで固定し，model forwardを行いません．選択済みepochの全validation予測からprimary subject平均BAを再計算し，記録された最初の最大値とbest checkpointのSHAが一致することも確認します．

- 元manifest，main配列台帳，metadataだけで選んだtrain3 IDと対応する実配列．
- fused runのbest checkpoint，run config，training membership，validation選択summary．
- 全5条件の評価完了，実測CUDAメモリ上限，予測identity audit，保存予測のSHA．
- 回収した全memberのbytesとSHAを再照合できるrecovery receipt．
- wrapper自身，既存phase2の科学source，benchmark execution seal．

recovery receiptは `status=completed`，`all_member_hashes_verified=true`，`member_count`，`files={phase相対path:{sha256,bytes,local_path}}`，`checkpoint_sha256={scratch:...,fused:...}` を持ちます．phaseの基点は `private/xavier/eegmmidb_phase2_v1` です．必要なrun/evaluationのmemberが回収台帳から欠落している場合は進みません．

## コマンド

repository rootをcwdにします．以下は今回の実データを使って確定したprepare／runです．prepare済み契約のSHAは `dba168a89a622d83112b80725a10c08f6814a619e8ed2209397a0f73b4a1f5c2` です．既存の契約や出力は上書きできません．再試用には同じ契約と，新しい未使用output directoryを指定してください．

```bash
.venv-xavier/bin/python experiments/pj_xavier/integration/try_eegmmidb_recorded.py prepare \
  --repo . \
  --evaluation private/xavier/eegmmidb_phase2_v1/evaluations/eegnet_smollm_fused_v1 \
  --recovery private/xavier/eegmmidb_phase2_v1/phase2_recovery_receipt_v1.json \
  --output private/xavier/eegmmidb_phase2_v1/recorded_trial_contract_v1.json

.venv-xavier/bin/python experiments/pj_xavier/integration/try_eegmmidb_recorded.py run \
  --repo . \
  --contract private/xavier/eegmmidb_phase2_v1/recorded_trial_contract_v1.json \
  --expected-contract-sha256 dba168a89a622d83112b80725a10c08f6814a619e8ed2209397a0f73b4a1f5c2 \
  --output private/xavier/eegmmidb_phase2_v1/recorded_trial_m1_v1 \
  --max-seconds 300
```

通常の全epoch完走に加えて，予定stage時間または元phase2予算のfit cutoff（最終期限から2,100秒前）で正常停止したrunも，完了済みvalidation，選択checkpoint，stop receipt，実起動引数と終了receiptが一致する場合だけ受け付けます．固定trainerはpartialを保存した正常な予算停止でもexit 2を返すため，この場合はその値，timeoutなし，同一PIDとcommand，予定された停止理由を同時に要求します．partialに対するexit 0やその他のcodeだけを成功とは扱いません．OOM，nonfinite，任意に早めたdeadline，途中のvalidationから選ばれたcheckpointは受け付けません．

出力先は未使用のdirectoryだけを受け付けます．MPS allocatorの上限は8GiB，CPU threadsは2，生成は最大8 tokensです．180〜300秒のwall-time上限を親processが監視します．失敗・時間超過もlogとcompletion receiptに残し，成功として扱いません．

Hugging Faceの既存local cacheだけを使い，offline環境設定と `local_files_only=true` を併用します．新しいcloud APIやOS操作を呼びません．Macに必要なモデルcacheが無い場合は，明示的に失敗します．trialを通すためにtestを再推論したり，閾値，重み，prompt，データを変更したりしません．

## 出力の読み方

`summary.json` はtrain3例，12実LLM呼出し，12合成環境episode，成功・棄権・不正操作，メモリ観測，実行時間を記録します．`trace.jsonl` はprivateに保存する個別状態遷移です．`completion.json` のexit codeとsummary SHAが一致することを確認します．MPSメモリはsampled allocated/driver値で，実GPUピークの計測保証とは区別しています．

## 限定したfixture検証

`test_try_eegmmidb_recorded.py` は18試験で，train-onlyの選択，ラベルや成績によらないID順，12 fresh calls，held-out入力拒否，棄権／不正返答のno-op，期限切れ拒否，frozen parameter契約，回収SHA／path改変拒否，予定時間停止のpartial exit 2を含む実trainer契約を確認します．fixtureの模擬モデル結果を実脳波の性能には算入しません．
