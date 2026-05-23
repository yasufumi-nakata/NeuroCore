from __future__ import annotations

from neurocore.acquisition import plan_acquisition
from neurocore.datasets import DatasetRecord
from neurocore.exercise import exercise_dataset_records, summarize_dataset_exercise
from neurocore.materialization import RemoteFileCandidate, RemoteFileResolution


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
        "size": "",
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


def test_dataset_exercise_marks_direct_remote_raw_candidates() -> None:
    plan = plan_acquisition(record())
    resolution = RemoteFileResolution(
        record_id="1",
        dataset_name="sample",
        provider="zenodo",
        status="resolved",
        files=(
            RemoteFileCandidate(
                record_id="1",
                dataset_name="sample",
                provider="zenodo",
                name="sub-01_eeg.edf",
                url="https://example.test/sub-01_eeg.edf",
                source_url="https://zenodo.org/records/123456",
                directly_loadable=True,
                materialization_action="download_then_load",
            ),
        ),
    )

    exercise = exercise_dataset_records((plan,), remote_resolutions=(resolution,))
    summary = summarize_dataset_exercise(exercise)

    assert exercise[0].state == "remote_direct_raw_ready"
    assert exercise[0].required_action == "download direct raw files and load with NeuroCore loaders"
    assert summary["records_with_actionable_raw_path"] == 1
    assert summary["records_with_direct_raw_candidates"] == 1


def test_dataset_exercise_separates_external_blockers_and_unexercised_rows() -> None:
    account = plan_acquisition(
        record(
            record_id="2",
            url="https://discover.pennsieve.io/datasets/1",
            doi="",
            source_domain="discover.pennsieve.io",
            access_status="要アカウント/利用登録",
        )
    )
    unusable = plan_acquisition(record(record_id="3", access_status="利用困難"))
    planned = plan_acquisition(record(record_id="4"))

    exercise = exercise_dataset_records((account, unusable, planned))
    by_id = {item.record_id: item for item in exercise}
    summary = summarize_dataset_exercise(exercise)

    assert by_id["2"].state == "blocked_account_required"
    assert by_id["3"].state == "blocked_unusable"
    assert by_id["4"].state == "remote_not_exercised"
    assert summary["records_with_external_blocker"] == 2
    assert summary["records_not_yet_remotely_exercised"] == 1
