from __future__ import annotations

import json
import subprocess
import sys

from neurocore.acquisition import (
    detect_provider,
    local_readiness_for_record,
    plan_acquisition,
    summarize_acquisition_plans,
)
from neurocore.cli import main
from neurocore.datasets import DatasetRecord, DatasetInventory
from neurocore.materialization import RemoteFileResolution


def record(**overrides) -> DatasetRecord:
    values = {
        "record_id": "1",
        "legacy_id": "",
        "name": "sample",
        "url": "https://zenodo.org/records/123456",
        "doi": "10.5281/zenodo.123456",
        "source_domain": "zenodo.org",
        "access_status": "すぐに使える",
        "score": 5,
        "size": "sample.zip",
        "participants": "",
        "duration": "",
        "stimulus": "",
        "equipment": "EDF",
        "conditions": "",
        "year": 2026,
        "description": "raw EEG .edf file",
        "evidence": "",
        "search_sources": (),
    }
    values.update(overrides)
    return DatasetRecord(**values)


def test_plan_acquisition_detects_public_api_and_account_paths() -> None:
    zenodo = plan_acquisition(record())
    kaggle = plan_acquisition(
        record(
            record_id="2",
            url="https://www.kaggle.com/datasets/example/eeg",
            doi="",
            source_domain="kaggle.com",
            access_status="すぐに使える",
        )
    )
    unusable = plan_acquisition(record(record_id="3", access_status="利用困難"))

    assert zenodo.provider == "zenodo"
    assert zenodo.automation_status == "direct_api"
    assert zenodo.candidates[0].method == "zenodo_api"
    assert kaggle.provider == "kaggle"
    assert kaggle.automation_status == "direct_api"
    assert kaggle.candidates[0].method == "kaggle_api"
    assert kaggle.candidates[0].role == "file_listing"
    assert unusable.automation_status == "unusable"


def test_plan_acquisition_understands_english_access_statuses() -> None:
    restricted = plan_acquisition(
        record(
            record_id="4",
            url="https://zenodo.org/records/123456",
            doi="10.5281/zenodo.123456",
            source_domain="zenodo.org",
            access_status="restricted; request access",
        )
    )
    unavailable = plan_acquisition(record(record_id="5", access_status="metadata only; no raw data available"))

    assert restricted.automation_status == "account_required"
    assert unavailable.automation_status == "unusable"


def test_plan_acquisition_treats_generic_doi_as_automatable() -> None:
    plan = plan_acquisition(
        record(
            record_id="generic-doi",
            url="https://doi.org/10.9999/example-dataset",
            doi="10.9999/example-dataset",
            source_domain="doi.org",
            access_status="すぐに使える",
        )
    )

    assert plan.provider == "doi"
    assert plan.automation_status == "direct_api"
    assert plan.candidates[0].method == "doi_resolver"


def test_plan_acquisition_treats_public_unknown_hosts_as_web_landing() -> None:
    plan = plan_acquisition(
        record(
            record_id="web",
            url="https://lab.example.test/eeg-dataset",
            doi="",
            source_domain="lab.example.test",
            access_status="すぐに使える",
        )
    )

    assert plan.provider == "web_landing"
    assert plan.automation_status == "direct_api"
    assert plan.candidates[0].method == "http_landing"


def test_plan_acquisition_treats_public_sciencedb_as_resolvable_api() -> None:
    plan = plan_acquisition(
        record(
            record_id="6",
            url="https://www.scidb.cn/en/detail?dataSetId=9cacad83bdaa45d08a264c7f2d21a222",
            doi="10.57760/sciencedb.23155",
            source_domain="www.scidb.cn",
            access_status="すぐに使える",
        )
    )

    assert plan.provider == "scidb"
    assert plan.automation_status == "direct_api"
    assert plan.candidates[0].method == "scidb_api"
    assert plan.candidates[0].requires_auth is False


def test_plan_acquisition_resolves_mendeley_dataset_id_from_doi() -> None:
    plan = plan_acquisition(
        record(
            record_id="mendeley-doi",
            url="https://doi.org/10.17632/v6346g59xh",
            doi="10.17632/v6346g59xh",
            source_domain="doi.org",
            access_status="すぐに使える",
        )
    )

    assert plan.provider == "mendeley"
    assert plan.automation_status == "direct_api"
    assert plan.candidates[0].method == "mendeley_api"
    assert plan.candidates[0].url == "https://data.mendeley.com/public-api/datasets/v6346g59xh"


def test_plan_acquisition_detects_figshare_compatible_repositories() -> None:
    plan = plan_acquisition(
        record(
            record_id="7",
            url="https://data.4tu.nl/articles/_/12707438/1",
            doi="10.4121/uuid:8e8cfaf2-ab00-45b2-90a0-623fabf75ca9",
            source_domain="data.4tu.nl",
            access_status="すぐに使える",
        )
    )

    assert plan.provider == "figshare"
    assert plan.automation_status == "direct_api"
    assert plan.candidates[0].method == "figshare_api"
    assert plan.candidates[0].url == "https://api.figshare.com/v2/articles/12707438"


def test_plan_acquisition_detects_dataverse_compatible_repositories() -> None:
    borealis = plan_acquisition(
        record(
            record_id="8",
            url="https://doi.org/10.5683/SP3/JJ2YZZ",
            doi="10.5683/SP3/JJ2YZZ",
            source_domain="doi.org",
            access_status="すぐに使える",
        )
    )
    ntu = plan_acquisition(
        record(
            record_id="9",
            url="https://researchdata.ntu.edu.sg/citation?persistentId=doi%3A10.21979%2FN9%2FTITSXU",
            doi="10.21979/n9/titsxu",
            source_domain="researchdata.ntu.edu.sg",
            access_status="すぐに使える",
        )
    )

    assert borealis.provider == "dataverse"
    assert borealis.automation_status == "direct_api"
    assert borealis.candidates[0].method == "dataverse_api"
    assert borealis.candidates[0].url.startswith("https://borealisdata.ca/api/datasets/:persistentId/")
    assert ntu.provider == "dataverse"
    assert ntu.candidates[0].url == (
        "https://researchdata.ntu.edu.sg/api/datasets/:persistentId/"
        "?persistentId=doi%3A10.21979%2FN9%2FTITSXU"
    )


def test_plan_acquisition_detects_stanford_and_data_ru_repositories() -> None:
    stanford = plan_acquisition(
        record(
            record_id="10",
            url="https://purl.stanford.edu/pp371jh5722",
            doi="10.25740/pp371jh5722",
            source_domain="purl.stanford.edu",
            access_status="すぐに使える",
        )
    )
    data_ru = plan_acquisition(
        record(
            record_id="11",
            url="https://data.ru.nl/collections/di/dcc/DSC_2022.00139_820",
            doi="10.34973/6dw9-0924",
            source_domain="data.ru.nl",
            access_status="すぐに使える",
        )
    )

    assert stanford.provider == "stanford_sdr"
    assert stanford.automation_status == "direct_api"
    assert stanford.candidates[0].method == "stanford_purl_json"
    assert stanford.candidates[0].url == "https://purl.stanford.edu/pp371jh5722.json"
    assert data_ru.provider == "data_ru"
    assert data_ru.automation_status == "direct_api"
    assert data_ru.candidates[0].method == "data_ru_landing"


def test_plan_acquisition_detects_bnci_and_repository_html_sources() -> None:
    bnci = plan_acquisition(
        record(
            record_id="12",
            url="https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf",
            doi="10.3389/fnins.2012.00055",
            source_domain="bnci-horizon-2020.eu",
            access_status="すぐに使える",
        )
    )
    datashare = plan_acquisition(
        record(
            record_id="13",
            url="https://datashare.ed.ac.uk/handle/10283/2100",
            doi="10.7488/ds/1478",
            source_domain="datashare.ed.ac.uk",
            access_status="すぐに使える",
        )
    )

    assert bnci.provider == "bnci"
    assert bnci.automation_status == "direct_api"
    assert bnci.candidates[0].method == "bnci_index"
    assert datashare.provider == "repository_html"
    assert datashare.automation_status == "direct_api"
    assert datashare.candidates[0].method == "http_landing"


def test_plan_acquisition_treats_public_file_listing_providers_as_direct_api() -> None:
    cases = [
        ("openneuro", "https://openneuro.org/datasets/ds000001", "openneuro.org", "openneuro_api"),
        ("github", "https://github.com/example/eeg-data", "github.com", "github_tree_api"),
        ("huggingface", "https://huggingface.co/datasets/example/eeg-set", "huggingface.co", "huggingface_api"),
        ("physionet", "https://physionet.org/content/auditory-eeg/1.0.0", "physionet.org", "physionet_index"),
        ("dandi", "https://dandiarchive.org/dandiset/000055", "dandiarchive.org", "dandi_api"),
        ("gin", "https://gin.g-node.org/doi/example-dataset", "gin.g-node.org", "gin_index"),
        ("nemar", "https://nemar.org/dataexplorer/detail?dataset_id=nm000113", "nemar.org", "nemar_index"),
    ]

    for provider, url, domain, method in cases:
        plan = plan_acquisition(record(record_id=provider, url=url, doi="", source_domain=domain))

        assert plan.provider == provider
        assert plan.automation_status == "direct_api"
        assert plan.candidates[0].method == method
        assert plan.candidates[0].role == "file_listing"


def test_plan_inventory_acquisition_summary_counts() -> None:
    inventory = DatasetInventory(
        source="inventory.csv",
        header=("id",),
        records=(
            record(record_id="1"),
            record(record_id="2", url="https://openneuro.org/datasets/ds000001", doi="", source_domain="openneuro.org"),
            record(record_id="3", url="https://unknown.example.test", doi="", source_domain="unknown.example.test"),
        ),
    )
    plans = tuple(plan_acquisition(item) for item in inventory.records)
    summary = summarize_acquisition_plans(plans)

    assert detect_provider(inventory.records[1]) == "openneuro"
    assert detect_provider(inventory.records[2]) == "web_landing"
    assert summary["automation_status_counts"]["direct_api"] == 3
    assert summary["automation_status_counts"].get("tooling_required", 0) == 0
    assert summary["automation_status_counts"].get("manual_review", 0) == 0


def test_local_readiness_finds_cached_signal_file(tmp_path) -> None:
    cached = tmp_path / "1"
    cached.mkdir()
    signal = cached / "subject.edf"
    signal.write_text("", encoding="utf-8")

    readiness = local_readiness_for_record(record(), tmp_path)

    assert readiness["local_signal_file_count"] == 1
    assert readiness["local_signal_files"] == [str(signal)]


def test_dataset_acquisition_plan_cli_reports_summary(tmp_path, capsys) -> None:
    csv_path = tmp_path / "inventory.csv"
    csv_path.write_text(
        "id,dataset_name,url,doi,source_domain,access_status,score,description\n"
        "1,sample,https://zenodo.org/records/123456,10.5281/zenodo.123456,zenodo.org,すぐに使える,5,raw EEG .edf file\n",
        encoding="utf-8",
    )

    exit_code = main(["dataset-acquisition-plan", str(csv_path), "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["record_count"] == 1
    assert payload["automation_status_counts"]["direct_api"] == 1
    assert payload["sample_plans"][0]["candidates"][0]["method"] == "zenodo_api"


def test_dataset_resolve_files_cli_reports_summary_without_network(tmp_path, capsys, monkeypatch) -> None:
    csv_path = tmp_path / "inventory.csv"
    csv_path.write_text(
        "id,dataset_name,url,doi,source_domain,access_status,score,description\n"
        "1,sample,https://zenodo.org/records/123456,10.5281/zenodo.123456,zenodo.org,すぐに使える,5,raw EEG .edf file\n",
        encoding="utf-8",
    )

    captured_kwargs = {}

    def fake_resolve(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return (RemoteFileResolution("1", "sample", "zenodo", "resolved", ()),)

    monkeypatch.setattr("neurocore.cli.resolve_inventory_remote_files", fake_resolve)

    exit_code = main(["dataset-resolve-files", str(csv_path), "--offset", "10", "--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["records_resolved"] == 1
    assert payload["sample_resolutions"][0]["provider"] == "zenodo"
    assert captured_kwargs["offset"] == 10


def test_dataset_acquisition_verifier_writes_private_report(tmp_path) -> None:
    csv_path = tmp_path / "inventory.csv"
    csv_path.write_text(
        "id,dataset_name,url,doi,source_domain,access_status,score,description\n"
        "1,sample,https://zenodo.org/records/123456,10.5281/zenodo.123456,zenodo.org,すぐに使える,5,raw EEG .edf file\n",
        encoding="utf-8",
    )
    report = tmp_path / "acquisition.json"
    summary = tmp_path / "acquisition.md"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_dataset_acquisition.py",
            "--inventory",
            str(csv_path),
            "--cache-root",
            str(tmp_path),
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
    assert payload["confidentiality"].startswith("private dataset acquisition verification")
    assert payload["plan_summary"]["automation_status_counts"]["direct_api"] == 1
    assert payload["dataset_exercise_records"][0]["record_id"] == "1"
    assert payload["remote_file_resolution_records"] == []
    assert payload["local_readiness_summary"]["records_checked"] == 1
    assert "Do not copy this report into public docs" in summary.read_text(encoding="utf-8")
