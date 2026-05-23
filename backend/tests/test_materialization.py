from __future__ import annotations

import zipfile

from neurocore.acquisition import plan_acquisition
from neurocore.datasets import DatasetRecord
from neurocore.materialization import (
    extract_supported_signal_files_from_archive,
    materialize_remote_files,
    resolve_remote_files,
    summarize_remote_file_resolutions,
)


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


def test_zenodo_resolution_classifies_direct_and_archive_files() -> None:
    plan = plan_acquisition(record())

    def fetch_json(url: str):
        assert url == "https://zenodo.org/api/records/123456"
        return {
            "files": [
                {
                    "key": "sub-01_task-test_eeg.edf",
                    "size": 123,
                    "checksum": "md5:abc",
                    "links": {"self": "https://zenodo.org/api/files/sub-01.edf"},
                },
                {
                    "key": "dataset.zip",
                    "size": 456,
                    "links": {"self": "https://zenodo.org/api/files/dataset.zip"},
                },
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    summary = summarize_remote_file_resolutions((resolution,))

    assert resolution.status == "resolved"
    assert resolution.files[0].directly_loadable is True
    assert resolution.files[0].materialization_action == "download_then_load"
    assert resolution.files[1].archive is True
    assert summary["directly_loadable_file_count"] == 1
    assert summary["archive_file_count"] == 1


def test_remote_resolution_does_not_treat_metadata_csv_as_raw_signal() -> None:
    plan = plan_acquisition(record())

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "statistical_learning_group_and_iq.csv",
                    "size": 123,
                    "links": {"self": "https://example.test/group.csv"},
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is False
    assert resolution.files[0].materialization_action == "metadata_or_manual_review"


def test_dataverse_resolution_builds_access_datafile_urls() -> None:
    plan = plan_acquisition(
        record(
            url="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ABC",
            doi="10.7910/DVN/ABC",
            source_domain="dataverse.harvard.edu",
        )
    )

    def fetch_json(url: str):
        assert "api/datasets/:persistentId/" in url
        return {
            "data": {
                "latestVersion": {
                    "files": [
                        {
                            "dataFile": {
                                "id": 42,
                                "filename": "subject01.vhdr",
                                "filesize": 789,
                                "md5": "deadbeef",
                                "contentType": "text/plain",
                            }
                        }
                    ]
                }
            }
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].url == "https://dataverse.harvard.edu/api/access/datafile/42"
    assert resolution.files[0].directly_loadable is True


def test_materialize_remote_files_downloads_direct_files_only(tmp_path) -> None:
    plan = plan_acquisition(record())

    def fetch_json(_url: str):
        return {
            "files": [
                {"key": "subject.edf", "size": 4, "links": {"self": "https://example.test/subject.edf"}},
                {"key": "archive.zip", "size": 4, "links": {"self": "https://example.test/archive.zip"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    def fetch_bytes(url: str, max_bytes: int) -> bytes:
        assert url == "https://example.test/subject.edf"
        assert max_bytes == 10
        return b"EEG!"

    results = materialize_remote_files(resolution.files, tmp_path, fetch_bytes=fetch_bytes, max_bytes=10)

    assert [result.status for result in results] == ["downloaded", "skipped"]
    assert (tmp_path / "1" / "subject.edf").read_bytes() == b"EEG!"


def test_archive_extraction_finds_supported_signal_files(tmp_path) -> None:
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("sub-01/eeg/sub-01_task-test_eeg.csv", "Fz,Cz\n1,2\n")
        handle.writestr("README.txt", "metadata")

    result = extract_supported_signal_files_from_archive(archive)

    assert result.status == "extracted"
    assert result.extracted_member_count == 2
    assert result.signal_file_count == 1
    assert result.signal_files[0].endswith("sub-01_task-test_eeg.csv")
