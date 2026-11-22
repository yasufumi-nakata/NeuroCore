#!/usr/bin/env python3
"""Run the pinned PJ-Xavier recorded-EEG trial on the local M1, offline."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

CONTRACT = "private/xavier/eegmmidb_phase3_v1/recorded_trial_contract_v1.json"
EXPECTED_CONTRACT_SHA256 = "3ae5f2dfce28994b733adeb93eb4be419a8820455813d4d695c746ab74e34d39"
WRAPPER = "experiments/pj_xavier/integration/try_eegmmidb_phase3_recorded.py"
EXPECTED_WRAPPER_SHA256 = "9893da260cc1c01b2e47bd711ce493583ec1baecda278c018df56ee6888fc5aa"
OUTPUT_ROOT = "private/xavier/eegmmidb_phase3_v1/recorded_trials"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def verify(repo):
    require(len(EXPECTED_CONTRACT_SHA256) == 64, "実機試用の契約がまだ固定されていません．")
    require(sha(repo / CONTRACT) == EXPECTED_CONTRACT_SHA256, "固定された試用契約と一致しません．")
    require(sha(repo / WRAPPER) == EXPECTED_WRAPPER_SHA256, "試用コードが固定版と一致しません．")
    value = json.loads((repo / CONTRACT).read_text())
    require(value["schema"] == "xavier-phase3-SSL-train-three-M1-contract-v1", "別の実験の契約です．")
    require(value["partition"] == "train" and len(value["selected_ids"]) == 3
            and value["run"] == "private/xavier/eegmmidb_phase3_v1/runs/eegnet_ssl_smollm_fused_v1",
            "モデルまたは3件の学習例の指定が異なります．")
    require(value["device"] == "mps" and value["memory_budget_gib"] == 8
            and value["acceptance_threshold"] == .8 and value["route_calls"] == 12
            and value["max_new_tokens"] == 8 and value["test_forward_calls"] == 0
            and value["routing_cache_reuse"] is False
            and value["OS_actions_allowed"] is False and value["cloud_calls_allowed"] is False,
            "固定した試用条件が変わっています．")
    require((repo / ".venv-xavier/bin/python").is_file(), "準備済みの実行環境が見つかりません．")
    return value


def summarize(result, contract):
    require(result["status"] == "completed" and result["contract_sha256"] == EXPECTED_CONTRACT_SHA256
            and result["checkpoint_sha256"] == contract["checkpoint_sha256"], "結果のモデル識別が異なります．")
    require(result["train_records"] == 3 and result["actual_fresh_LLM_routing_calls"] == 12
            and result["test_forward_calls"] == result["OS_actions"] == result["cloud_API_calls"] == 0
            and result["local_files_only"] is True, "実際の処理が契約範囲と異なります．")
    return {
        "status": "completed", "training_examples": 3, "fresh_routing_calls": 12,
        "abstained_episodes": result["abstained_episodes"],
        "successful_synthetic_episodes": result["successful_synthetic_episodes"],
        "invalid_actions": result["invalid_actions"],
        "incorrect_target_actions": result["incorrect_target_actions"],
        "heldout_performance": False, "autonomous_goal_validated": False,
    }


def run(repo):
    repo = Path(repo).resolve()
    contract = verify(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = repo / OUTPUT_ROOT / (stamp + "_" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "result"
    command = [str(repo / ".venv-xavier/bin/python"), str(repo / WRAPPER), "run",
               "--repo", str(repo), "--contract", str(repo / CONTRACT),
               "--expected-contract-sha256", EXPECTED_CONTRACT_SHA256,
               "--output", str(output), "--max-seconds", "300"]
    save(directory / "launcher_plan.json", {
        "command": command, "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "wrapper_sha256": EXPECTED_WRAPPER_SHA256, "launcher_sha256": sha(__file__),
        "created_epoch": time.time(), "output_is_new": True,
    })
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with (directory / "launcher.log").open("x") as log:
        child = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=330)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            save(directory / "launcher_completion.json", {"status": "stopped", "exit_code": child.returncode})
            raise
    if code != 0:
        save(directory / "launcher_completion.json", {"status": "failed", "exit_code": code})
        raise RuntimeError("試用が完了しませんでした．ログを確認してください：" + str(directory / "launcher.log"))
    result = summarize(json.loads((output / "summary.json").read_text()), contract)
    result.update(output=str(output.relative_to(repo)), summary_sha256=sha(output / "summary.json"))
    save(directory / "launcher_completion.json", result)
    print("実機試用が完了しました．学習例3件，LLM処理12回，棄権" + str(result["abstained_episodes"]) + "件です．")
    print("これは保存済み学習例の動作確認です．未知データの精度や実環境での自律操作の証明には使いません．")
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    require(len(sys.argv) == 1, "このコマンドは固定済みモデル専用です．追加の引数は不要です．")
    run(Path(__file__).resolve().parents[1])
