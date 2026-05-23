from __future__ import annotations

import json
import subprocess
import sys

from neurocore import DatasetInventory, load_eeg_dataset_inventory
from neurocore.cli import main


INVENTORY_HEADER = (
    "ID,旧ID,データセット名,公開URL,DOI,提供元ドメイン,アクセス区分,評価点,サイズ,人数,時間,刺激の種類,"
    "観測機器,実験条件,公開年,説明（日本語）,判定根拠,検索ソース\n"
)


def write_inventory(path) -> None:
    path.write_text(
        "\ufeff"
        + INVENTORY_HEADER
        + (
            "1,435,SSVEP sample,https://example.test/ds,10.1/example,zenodo.org,すぐに使える,5,"
            "sub-001_eeg.xdf,10名,60秒,視覚,g.USBamp XDF EEG,BCI,2026,"
            "XDF と EEGLAB .set の説明,CSV metadata; xdf evidence,https://example.test/a; manual\n"
        )
        + (
            "2,476,Spreadsheet sample,https://example.test/table,,zenodo.org,要登録,3,"
            "database.xlsx,59名,,臨床,spreadsheet,clinical,2020,"
            "Excel workbook only,workbook evidence,zenodo\n"
        ),
        encoding="utf-8",
    )


def test_eeg_dataset_inventory_loader_handles_japanese_bom_csv(tmp_path) -> None:
    path = tmp_path / "eeg_dataset_summary_ja.csv"
    write_inventory(path)

    inventory = load_eeg_dataset_inventory(path)

    assert isinstance(inventory, DatasetInventory)
    assert len(inventory) == 2
    assert inventory.records[0].score == 5
    assert inventory.records[0].year == 2026
    assert inventory.records[0].search_sources == ("https://example.test/a", "manual")
    assert inventory.access_status_counts() == {"すぐに使える": 1, "要登録": 1}
    assert inventory.format_mentions()["xdf"] == 1
    assert inventory.format_mentions()["eeglab_set"] == 1
    assert inventory.format_mentions()["xlsx"] == 1
    assert inventory.loader_coverage()["records_with_supported_signal_hint"] == 1
    assert inventory.validate() == []


def test_eeg_dataset_inventory_loader_accepts_english_columns(tmp_path) -> None:
    path = tmp_path / "inventory.csv"
    path.write_text(
        "id,dataset_name,url,doi,source_domain,access_status,score,description\n"
        "a1,English sample,https://example.test,,openneuro.org,open,4,EDF and BrainVision .vhdr dataset\n",
        encoding="utf-8",
    )

    inventory = load_eeg_dataset_inventory(path)

    assert len(inventory) == 1
    assert inventory.records[0].name == "English sample"
    assert inventory.format_mentions()["edf"] == 1
    assert inventory.format_mentions()["vhdr"] == 1
    assert inventory.validate() == []


def test_dataset_inventory_cli_reports_summary(tmp_path, capsys) -> None:
    path = tmp_path / "eeg_dataset_summary_ja.csv"
    write_inventory(path)

    exit_code = main(["dataset-inventory", str(path), "--limit", "1", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["record_count"] == 2
    assert payload["loader_coverage"]["records_with_supported_signal_hint"] == 1
    assert ".edf" in payload["supported_loader_extensions"]
    assert len(payload["sample_records"]) == 1
    assert payload["sample_records"][0]["name"] == "SSVEP sample"


def test_dataset_loading_verifier_writes_private_report(tmp_path) -> None:
    inventory = tmp_path / "eeg_dataset_summary_ja.csv"
    write_inventory(inventory)
    report = tmp_path / "dataset_report.json"
    summary = tmp_path / "dataset_report.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_dataset_loading.py",
            "--inventory",
            str(inventory),
            "--dataset-root",
            str(tmp_path),
            "--signal",
            "samples/synthetic_eeg.csv",
            "--skip-local-scan",
            "--output",
            str(report),
            "--markdown-output",
            str(summary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["confidentiality"].startswith("private dataset loading verification")
    assert payload["metrics"]["inventory"]["summary"]["record_count"] == 2
    assert payload["metrics"]["signal_csv"]["failed_checks"] == 0
    assert payload["overall"]["failed_checks"] == 0
    assert "Do not copy this report into public docs" in summary.read_text(encoding="utf-8")
