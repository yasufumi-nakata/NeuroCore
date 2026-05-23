from __future__ import annotations

import zipfile
from urllib.parse import unquote

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


def test_openneuro_resolution_uses_latest_snapshot_and_urls() -> None:
    plan = plan_acquisition(
        record(
            url="https://openneuro.org/datasets/ds005280",
            doi="",
            source_domain="openneuro.org",
            description="BIDS EEG .vhdr files",
        )
    )

    def fetch_json(url: str):
        decoded = unquote(url)
        if "datasetInfo" in decoded:
            return {"data": {"dataset": {"snapshots": [{"tag": "1.0.0", "created": "2024-01-01"}]}}}
        if "snapshotFiles" in decoded:
            return {
                "data": {
                    "snapshot": {
                        "files": [
                            {
                                "filename": "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
                                "size": 20,
                                "directory": False,
                                "annexed": True,
                                "id": "checksum",
                                "urls": ["https://s3.amazonaws.com/openneuro.org/ds005280/sub-01/eeg/file.vhdr"],
                            }
                        ]
                    }
                }
            }
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].directly_loadable is True
    assert resolution.files[0].source_url.endswith("/versions/1.0.0")


def test_doi_resolution_delegates_to_figshare_landing() -> None:
    plan = plan_acquisition(
        record(
            url="https://doi.org/10.11583/dtu.30589397",
            doi="10.11583/dtu.30589397",
            source_domain="doi.org",
            description="BDF EEG archive",
        )
    )

    def fetch_json(url: str):
        if url.startswith("doi+resolve://"):
            return {"url": "https://data.dtu.dk/articles/dataset/example/30589397"}
        assert url == "https://api.figshare.com/v2/articles/30589397"
        return {
            "files": [
                {
                    "name": "bdf_NH.zip",
                    "size": 100,
                    "download_url": "https://ndownloader.figshare.com/files/1",
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.provider == "doi"
    assert resolution.files[0].provider == "figshare"
    assert resolution.files[0].archive is True


def test_doi_resolution_falls_back_to_invenio_record_landing() -> None:
    plan = plan_acquisition(
        record(
            url="https://doi.org/10.15161/oar.it/cx0v8-k7w40",
            doi="10.15161/oar.it/cx0v8-k7w40",
            source_domain="doi.org",
            description="raw EEG archive",
        )
    )

    def fetch_json(url: str):
        if url.startswith("doi+resolve://"):
            return {"url": "https://www.openaccessrepository.it/records/cx0v8-k7w40"}
        assert url == "https://www.openaccessrepository.it/api/records/cx0v8-k7w40"
        return {
            "files": {
                "entries": {
                    "NeuroConn.zip": {
                        "key": "NeuroConn.zip",
                        "size": 123,
                        "checksum": "md5:abc",
                        "mimetype": "application/zip",
                        "links": {"content": "https://example.test/api/records/cx0v8-k7w40/files/NeuroConn.zip/content"},
                    }
                }
            }
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].name == "NeuroConn.zip"
    assert resolution.files[0].archive is True


def test_http_landing_resolves_invenio_record_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://www.openaccessrepository.it/records/cx0v8-k7w40",
            doi="10.15161/oar.it/cx0v8-k7w40",
            source_domain="www.openaccessrepository.it",
            description="raw EEG archive",
        )
    )

    def fetch_json(url: str):
        assert url == "https://www.openaccessrepository.it/api/records/cx0v8-k7w40"
        return {
            "files": {
                "entries": {
                    "NeuroConn.zip": {
                        "key": "NeuroConn.zip",
                        "links": {"content": "https://example.test/NeuroConn.zip"},
                        "size": 123,
                    }
                }
            }
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].url == "https://example.test/NeuroConn.zip"
    assert resolution.files[0].archive is True


def test_http_landing_scrapes_direct_file_links() -> None:
    plan = plan_acquisition(
        record(
            url="https://example.test/dataset",
            doi="",
            source_domain="example.test",
            description="EDF data",
        )
    )

    html = '<a href="/files/sub-01_eeg.edf?download=1">download</a><a href="/login">login</a>'
    resolution = resolve_remote_files(plan, fetch_json=lambda url: html if url.startswith("html+landing://") else {})

    assert resolution.files[0].name == "sub-01_eeg.edf"
    assert resolution.files[0].directly_loadable is True


def test_github_resolution_uses_tree_api_and_raw_urls() -> None:
    plan = plan_acquisition(
        record(
            url="https://github.com/example/eeg-data.git",
            doi="",
            source_domain="github.com",
            description="MAT EEG files",
        )
    )

    def fetch_json(url: str):
        assert url == "https://api.github.com/repos/example/eeg-data/git/trees/HEAD?recursive=1"
        return {"tree": [{"path": "data/eeg.mat", "type": "blob", "size": 10, "sha": "abc"}]}

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].url == "https://raw.githubusercontent.com/example/eeg-data/HEAD/data/eeg.mat"
    assert resolution.files[0].directly_loadable is True


def test_huggingface_resolution_lists_dataset_tree() -> None:
    plan = plan_acquisition(
        record(
            url="https://huggingface.co/datasets/example/eeg-set",
            doi="",
            source_domain="huggingface.co",
            description="preprocessed_eeg signals",
        )
    )

    def fetch_json(url: str):
        if url == "https://huggingface.co/api/datasets/example/eeg-set/tree/main?recursive=1":
            return [{"type": "directory", "path": "preprocessed_eeg"}]
        if url == "https://huggingface.co/api/datasets/example/eeg-set/tree/main/preprocessed_eeg":
            return [{"type": "file", "path": "preprocessed_eeg/sub-01/eeg.npy", "size": 10, "oid": "abc"}]
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].url.endswith("/resolve/main/preprocessed_eeg/sub-01/eeg.npy")
    assert resolution.files[0].directly_loadable is True


def test_dandi_resolution_marks_nwb_assets_loadable() -> None:
    plan = plan_acquisition(
        record(
            url="https://dandiarchive.org/dandiset/000055",
            doi="",
            source_domain="dandiarchive.org",
            description="NWB ecephys",
        )
    )

    def fetch_json(url: str):
        assert url.startswith("https://api.dandiarchive.org/api/dandisets/000055/versions/draft/assets/")
        return {
            "next": None,
            "results": [
                {
                    "asset_id": "asset-1",
                    "blob": "blob-1",
                    "path": "sub-01/sub-01_ses-1_behavior+ecephys.nwb",
                    "size": 123,
                }
            ],
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].url.endswith("/assets/asset-1/download/")
    assert resolution.files[0].directly_loadable is True


def test_physionet_resolution_walks_html_index() -> None:
    plan = plan_acquisition(
        record(
            url="https://physionet.org/content/auditory-eeg/1.0.0",
            doi="",
            source_domain="physionet.org",
            description="EDF recordings",
        )
    )

    def fetch_json(url: str):
        if url.endswith("/1.0.0/"):
            return '<a href="sub-01/">sub-01/</a>'
        if url.endswith("/sub-01/"):
            return '<a href="../">../</a><a href="sub-01_eeg.edf">sub-01_eeg.edf</a>'
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].name == "sub-01/sub-01_eeg.edf"
    assert resolution.files[0].directly_loadable is True


def test_gin_resolution_provides_extractable_archives() -> None:
    plan = plan_acquisition(
        record(
            url="https://gin.g-node.org/doi/example-dataset",
            doi="",
            source_domain="gin.g-node.org",
            description="GIN repository",
        )
    )

    resolution = resolve_remote_files(plan, fetch_json=lambda _url: "")

    assert resolution.files[0].archive is True
    assert resolution.files[0].url == "https://gin.g-node.org/doi/example-dataset/archive/master.zip"


def test_kaggle_resolution_lists_archived_public_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://www.kaggle.com/datasets/example/eeg-dataset",
            doi="",
            source_domain="kaggle.com",
            description="CSV EEG files",
        )
    )

    def fetch_json(url: str):
        if url == "https://www.kaggle.com/api/v1/datasets/list/example/eeg-dataset":
            return {
                "datasetFiles": [{"name": "sub-01/eeg.csv", "totalBytes": 42}],
                "nextPageToken": "next",
            }
        if url.endswith("pageToken=next"):
            return {"datasetFiles": [{"name": "metadata.txt", "totalBytes": 4}]}
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].name == "sub-01/eeg.csv.zip"
    assert "file_name=sub-01%2Feeg.csv" in resolution.files[0].url
    assert resolution.files[0].archive is True


def test_nemar_resolution_extracts_download_links_from_detail_page() -> None:
    plan = plan_acquisition(
        record(
            url="https://nemar.org/dataexplorer/detail?dataset_id=nm000113",
            doi="10.82901/nemar.nm000113",
            source_domain="nemar.org",
            description="EDF BIDS dataset",
        )
    )

    html = """
    <a href="/dataexplorer/download?filepath=/data/nemar/openneuro//zip_files/nm000113.zip">zip</a>
    <script>
    download_file('\\/dataexplorer\\/download?filepath=\\/data\\/nemar\\/openneuro\\/\\/nm000113\\/sub-01\\/eeg\\/sub-01_eeg.edf');
    </script>
    """

    resolution = resolve_remote_files(plan, fetch_json=lambda _url: html)

    assert [file.name for file in resolution.files] == ["nm000113.zip", "sub-01/eeg/sub-01_eeg.edf"]
    assert resolution.files[0].archive is True
    assert resolution.files[1].directly_loadable is True


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


def test_materialize_remote_files_max_files_ignores_unselected_skips(tmp_path) -> None:
    plan = plan_acquisition(record(description="BrainVision EEG .vhdr files"))

    def fetch_json(_url: str):
        return {
            "files": [
                {"key": "README.md", "size": 4, "links": {"self": "https://example.test/README.md"}},
                {"key": "sub-01_task-test_eeg.vhdr", "size": 4, "links": {"self": "https://example.test/sub.vhdr"}},
                {"key": "sub-02_task-test_eeg.vhdr", "size": 4, "links": {"self": "https://example.test/sub2.vhdr"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    downloaded: list[str] = []

    def fetch_bytes(url: str, _max_bytes: int) -> bytes:
        downloaded.append(url)
        return b"Brain Vision Data Exchange Header File Version 1.0"

    results = materialize_remote_files(
        resolution.files,
        tmp_path,
        fetch_bytes=fetch_bytes,
        max_files=1,
        max_bytes=100,
    )

    assert [result.status for result in results] == ["skipped", "downloaded"]
    assert downloaded == ["https://example.test/sub.vhdr"]


def test_materialize_remote_files_includes_loader_companions(tmp_path) -> None:
    plan = plan_acquisition(record(description="BrainVision EEG .vhdr files"))

    def fetch_json(_url: str):
        return {
            "files": [
                {"key": "sub-01_task-test_eeg.vhdr", "size": 4, "links": {"self": "https://example.test/sub.vhdr"}},
                {"key": "sub-01_task-test_eeg.eeg", "size": 4, "links": {"self": "https://example.test/sub.eeg"}},
                {"key": "sub-01_task-test_eeg.vmrk", "size": 4, "links": {"self": "https://example.test/sub.vmrk"}},
                {"key": "participants.tsv", "size": 4, "links": {"self": "https://example.test/participants.tsv"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    downloaded: list[str] = []

    def fetch_bytes(url: str, _max_bytes: int) -> bytes:
        downloaded.append(url)
        return b"sidecar"

    results = materialize_remote_files(resolution.files, tmp_path, fetch_bytes=fetch_bytes, max_bytes=10)

    assert [result.status for result in results] == ["downloaded", "downloaded", "downloaded", "skipped"]
    assert downloaded == [
        "https://example.test/sub.vhdr",
        "https://example.test/sub.eeg",
        "https://example.test/sub.vmrk",
    ]


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
