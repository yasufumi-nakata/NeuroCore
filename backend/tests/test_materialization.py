from __future__ import annotations

import zipfile
from urllib.parse import unquote

from neurocore.acquisition import plan_acquisition
from neurocore.datasets import DatasetRecord
from neurocore.materialization import (
    extract_supported_signal_files_from_archive,
    materialize_remote_files,
    resolve_inventory_remote_files,
    resolve_remote_files,
    summarize_remote_file_resolutions,
    _html_fetch_target,
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


def test_inventory_remote_resolution_supports_offsets_for_batch_exercise() -> None:
    plans = [
        plan_acquisition(
            record(
                record_id=str(index),
                url=f"https://zenodo.org/records/{1000 + index}",
                doi=f"10.5281/zenodo.{1000 + index}",
            )
        )
        for index in range(1, 5)
    ]
    queried: list[str] = []

    def fetch_json(url: str):
        queried.append(url)
        record_id = url.rsplit("/", 1)[-1]
        return {
            "files": [
                {
                    "key": f"sub-{record_id}_eeg.edf",
                    "links": {"self": f"https://example.test/{record_id}.edf"},
                }
            ]
        }

    resolutions = resolve_inventory_remote_files(plans, limit=2, offset=1, fetch_json=fetch_json)

    assert [resolution.record_id for resolution in resolutions] == ["2", "3"]
    assert queried == [
        "https://zenodo.org/api/records/1002",
        "https://zenodo.org/api/records/1003",
    ]
    assert all(resolution.files[0].directly_loadable for resolution in resolutions)


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


def test_openneuro_resolution_chooses_url_matching_filename() -> None:
    plan = plan_acquisition(
        record(
            url="https://openneuro.org/datasets/ds004147",
            doi="",
            source_domain="openneuro.org",
            description="BIDS BrainVision EEG",
        )
    )

    def fetch_json(url: str):
        decoded = unquote(url)
        if "datasetInfo" in decoded:
            return {"data": {"dataset": {"snapshots": [{"tag": "1.0.2", "created": "2024-01-01"}]}}}
        if "snapshotFiles" in decoded:
            return {
                "data": {
                    "snapshot": {
                        "files": [
                            {
                                "filename": "sub-36/eeg/sub-36_task-casinos_eeg.vhdr",
                                "size": 6140,
                                "directory": False,
                                "annexed": True,
                                "id": "checksum",
                                "urls": [
                                    "https://s3.amazonaws.com/openneuro.org/ds004147/sub-38/eeg/sub-38_task-casinos_eeg.vhdr",
                                    "https://s3.amazonaws.com/openneuro.org/ds004147/sub-36/eeg/sub-36_task-casinos_eeg.vhdr",
                                ],
                            }
                        ]
                    }
                }
            }
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].name == "sub-36/eeg/sub-36_task-casinos_eeg.vhdr"
    assert "/sub-36/eeg/sub-36_task-casinos_eeg.vhdr" in resolution.files[0].url
    assert resolution.files[0].directly_loadable is True


def test_osf_resolution_preserves_materialized_path_for_raw_detection() -> None:
    plan = plan_acquisition(
        record(
            url="https://osf.io/d5yrf/",
            doi="10.17605/osf.io/d5yrf",
            source_domain="osf.io",
            description="EEG CSV recordings",
        )
    )

    def fetch_json(_url: str):
        return {
            "data": [
                {
                    "attributes": {
                        "kind": "file",
                        "name": "eeg01.csv",
                        "materialized_path": "/EEG_CSV/eeg01.csv",
                        "size": 10,
                    },
                    "links": {"download": "https://osf.io/download/eeg01/"},
                },
                {
                    "attributes": {
                        "kind": "file",
                        "name": "stimuli_eeg01.txt",
                        "materialized_path": "/EEG_stimuli_TXT/stimuli_eeg01.txt",
                        "size": 10,
                    },
                    "links": {"download": "https://osf.io/download/stimuli/"},
                },
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].name == "EEG_CSV/eeg01.csv"
    assert resolution.files[0].directly_loadable is True
    assert resolution.files[1].directly_loadable is False


def test_doi_resolution_delegates_to_figshare_landing() -> None:
    plan = plan_acquisition(
        record(
            url="https://doi.org/10.9999/example-landing",
            doi="10.9999/example-landing",
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


def test_doi_resolution_delegates_to_figshare_compatible_landing() -> None:
    plan = plan_acquisition(
        record(
            url="https://doi.org/10.4121/uuid:8e8cfaf2-ab00-45b2-90a0-623fabf75ca9",
            doi="10.4121/uuid:8e8cfaf2-ab00-45b2-90a0-623fabf75ca9",
            source_domain="doi.org",
            description="EEG archive",
        )
    )

    def fetch_json(url: str):
        if url.startswith("doi+resolve://"):
            return {"url": "https://data.4tu.nl/articles/_/12707438/1"}
        assert url == "https://api.figshare.com/v2/articles/12707438"
        return {
            "files": [
                {
                    "name": "eeg-data.zip",
                    "size": 123,
                    "download_url": "https://ndownloader.figshare.com/files/24061748",
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].provider == "figshare"
    assert resolution.files[0].archive is True


def test_doi_resolution_delegates_to_dataverse_compatible_landing() -> None:
    plan = plan_acquisition(
        record(
            url="https://doi.org/10.5683/SP3/JJ2YZZ",
            doi="10.5683/SP3/JJ2YZZ",
            source_domain="doi.org",
            description="BDF EEG archive",
        )
    )

    def fetch_json(url: str):
        if url.startswith("doi+resolve://"):
            return {"url": "https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/JJ2YZZ"}
        assert url == "https://borealisdata.ca/api/datasets/:persistentId/?persistentId=doi%3A10.5683%2FSP3%2FJJ2YZZ"
        return {
            "data": {
                "latestVersion": {
                    "files": [
                        {
                            "dataFile": {
                                "id": 99,
                                "filename": "sub-01_task-drive_eeg.bdf",
                                "filesize": 456,
                                "contentType": "application/octet-stream",
                            }
                        }
                    ]
                }
            }
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].provider == "dataverse"
    assert resolution.files[0].directly_loadable is True


def test_mendeley_resolution_uses_content_detail_download_urls() -> None:
    plan = plan_acquisition(
        record(
            url="https://data.mendeley.com/datasets/rckc8c7mh9/1",
            doi="10.17632/rckc8c7mh9.1",
            source_domain="data.mendeley.com",
            description="EEG archive",
        )
    )

    def fetch_json(url: str):
        assert url == "https://data.mendeley.com/public-api/datasets/rckc8c7mh9"
        return {
            "files": [
                {
                    "filename": "ADMCI.zip",
                    "size": 4509699,
                    "content_details": {
                        "download_url": "https://data.mendeley.com/public-files/datasets/rckc8c7mh9/files/file/file_downloaded",
                        "sha256_hash": "abc",
                        "content_type": "application/zip",
                    },
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].archive is True
    assert resolution.files[0].checksum == "sha256:abc"


def test_stanford_purl_resolution_reads_cocina_file_manifest() -> None:
    plan = plan_acquisition(
        record(
            url="https://purl.stanford.edu/pp371jh5722",
            doi="10.25740/pp371jh5722",
            source_domain="purl.stanford.edu",
            description="MATLAB RawEEG files",
        )
    )

    def fetch_json(url: str):
        assert url == "https://purl.stanford.edu/pp371jh5722.json"
        return {
            "purl": "https://purl.stanford.edu/pp371jh5722",
            "structural": {
                "contains": [
                    {
                        "structural": {
                            "contains": [
                                {
                                    "type": "https://cocina.sul.stanford.edu/models/file",
                                    "filename": "CleanEEG_stim01.mat",
                                    "size": 123,
                                    "hasMimeType": "application/octet-stream",
                                    "hasMessageDigests": [{"type": "md5", "digest": "abc"}],
                                    "access": {"download": "world"},
                                },
                                {
                                    "type": "https://cocina.sul.stanford.edu/models/file",
                                    "filename": "restricted.mat",
                                    "access": {"download": "none"},
                                },
                            ]
                        }
                    }
                ]
            },
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].provider == "stanford_sdr"
    assert resolution.files[0].url == "https://stacks.stanford.edu/file/druid:pp371jh5722/CleanEEG_stim01.mat"
    assert resolution.files[0].directly_loadable is True


def test_data_ru_resolution_uses_json_ld_webdav_manifest() -> None:
    plan = plan_acquisition(
        record(
            url="https://data.ru.nl/collections/di/dcc/DSC_2022.00139_820",
            doi="10.34973/6dw9-0924",
            source_domain="data.ru.nl",
            description="BrainVision EEG data",
        )
    )

    def fetch_json(url: str):
        if url.startswith("html+landing://") and "MANIFEST.txt" not in url:
            return """
            <script type="application/ld+json">
            {"@type":"Dataset","distribution":{"@type":"DataDownload","contentUrl":"https://webdav.data.ru.nl/dcc/DSC_2022.00139_820_v1"}}
            </script>
            """
        if url.startswith("html+landing://") and "MANIFEST.txt" in url:
            return """
            sha1 README.md
            sha2 sourcedata/sub-01/eeg/sub-01_task-rest_eeg.vhdr
            sha3 sourcedata/sub-01/eeg/sub-01_task-rest_eeg.eeg
            sha4 sourcedata/sub-01/eeg/sub-01_task-rest_eeg.vmrk
            """
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[1].provider == "data_ru"
    assert resolution.files[1].directly_loadable is True
    assert resolution.files[1].url.endswith("sourcedata/sub-01/eeg/sub-01_task-rest_eeg.vhdr")


def test_generic_mat_detection_avoids_behavior_only_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://data.ru.nl/collections/di/dcc/DSC_2022.00139_820",
            doi="10.34973/6dw9-0924",
            source_domain="data.ru.nl",
            description="MATLAB EEG files",
        )
    )

    def fetch_json(url: str):
        if url.startswith("html+landing://") and "MANIFEST.txt" not in url:
            return """
            <script type="application/ld+json">
            {"distribution":{"contentUrl":"https://webdav.data.ru.nl/dcc/example_v1"}}
            </script>
            """
        if url.startswith("html+landing://") and "MANIFEST.txt" in url:
            return """
            sha1 data/behavior/S01_behresults.mat
            sha2 data/eeg/S01_eeg.mat
            sha3 data/clean/CleanEEG_stim01.mat
            """
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    by_name = {file.name: file for file in resolution.files}
    assert by_name["data/behavior/S01_behresults.mat"].directly_loadable is False
    assert by_name["data/eeg/S01_eeg.mat"].directly_loadable is True
    assert by_name["data/clean/CleanEEG_stim01.mat"].directly_loadable is True


def test_generic_csv_detection_uses_raw_task_paths_without_treating_stimuli_as_raw() -> None:
    plan = plan_acquisition(
        record(
            url="https://osf.io/5m24j/",
            doi="10.17605/osf.io/5m24j",
            source_domain="osf.io",
            description="raw Muse EEG CSV recordings",
        )
    )

    def fetch_json(_url: str):
        return {
            "data": [
                {
                    "attributes": {
                        "kind": "file",
                        "materialized_path": "/Raw Muse Data/KL_018_postinside_oddball.csv",
                    },
                    "links": {"download": "https://osf.io/download/raw/"},
                },
                {
                    "attributes": {
                        "kind": "file",
                        "materialized_path": "/EEG_stimuli_TXT/stimuli_eeg01.txt",
                    },
                    "links": {"download": "https://osf.io/download/stimuli/"},
                },
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is True
    assert resolution.files[1].directly_loadable is False


def test_bnci_resolution_filters_dataset_links_from_index() -> None:
    plan = plan_acquisition(
        record(
            url="https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf",
            doi="10.3389/fnins.2012.00055",
            source_domain="bnci-horizon-2020.eu",
            description="MAT EEG files",
        )
    )

    def fetch_json(url: str):
        assert url.startswith("html+landing://")
        return """
        <a href="/database/data-sets/001-2014/A01T.mat">A01T.mat</a>
        <a href="/database/data-sets/001-2014/A01E.mat">A01E.mat</a>
        <a href="/database/data-sets/002-2014/S01T.mat">S01T.mat</a>
        <a href="/database/data-sets/001-2014/description.pdf">description.pdf</a>
        """

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert [file.name for file in resolution.files] == ["A01T.mat", "A01E.mat"]
    assert all(file.directly_loadable for file in resolution.files)


def test_repository_html_resolution_uses_public_bitstream_links() -> None:
    plan = plan_acquisition(
        record(
            url="https://datashare.ed.ac.uk/handle/10283/2100",
            doi="10.7488/ds/1478",
            source_domain="datashare.ed.ac.uk",
            description="BDF EEG data",
        )
    )

    def fetch_json(url: str):
        assert url.startswith("html+landing://")
        return """
        <a href="/download/DS_10283_2100.zip">Download all</a>
        <a href="/bitstream/handle/10283/2100/d1.bdf?sequence=37&isAllowed=y">d1.bdf</a>
        <a href="/bitstream/handle/10283/2100/readme.txt?sequence=1&isAllowed=y">readme.txt</a>
        """

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].archive is True
    assert resolution.files[1].directly_loadable is True


def test_http_landing_uses_link_text_when_download_url_has_no_extension() -> None:
    plan = plan_acquisition(
        record(
            url="https://deepblue.lib.umich.edu/data/concern/data_sets/bg257f92t",
            doi="10.7302/z29c6vnh",
            source_domain="deepblue.lib.umich.edu",
            description="EEG dataset archive",
        )
    )

    def fetch_json(url: str):
        assert url.startswith("html+landing://")
        return """
        <a href="/data/downloads/vm40xs661">README.txt</a>
        <a href="/data/downloads/t435gf09p">alice_eeg.zip</a>
        """

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].name == "alice_eeg.zip"
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


def test_http_landing_uses_plain_http_for_known_expired_cert_hosts() -> None:
    assert (
        _html_fetch_target("https://archive.ics.uci.edu/dataset/457/eeg")
        == "http://archive.ics.uci.edu/dataset/457/eeg"
    )


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


def test_scidb_resolution_uses_public_zip_endpoint() -> None:
    plan = plan_acquisition(
        record(
            url="https://www.scidb.cn/en/detail?dataSetId=9cacad83bdaa45d08a264c7f2d21a222",
            doi="10.57760/sciencedb.23155",
            source_domain="www.scidb.cn",
            description="EEG archive",
        )
    )

    def fetch_json(url: str):
        if url.startswith("head+metadata://"):
            return {"status": 200, "content_length": "42647254461"}
        assert url == "https://www.scidb.cn/api/sdb-openapi-service/json?doi=10.57760%2Fsciencedb.23155"
        return {
            "@id": "https://doi.org/10.57760/sciencedb.23155",
            "conditionsOfAccess": "unrestricted",
            "isAccessibleForFree": True,
            "version": "V2",
            "size": {"value": 42647254461, "unitText": "bytes"},
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].name == "9cacad83bdaa45d08a264c7f2d21a222_V2.zip"
    assert resolution.files[0].url.endswith("dataSetId=9cacad83bdaa45d08a264c7f2d21a222&version=V2")
    assert resolution.files[0].size_bytes == 42647254461
    assert resolution.files[0].archive is True


def test_scidb_resolution_can_resolve_dataset_id_from_doi_landing() -> None:
    plan = plan_acquisition(
        record(
            url="https://doi.org/10.57760/sciencedb.23155",
            doi="10.57760/sciencedb.23155",
            source_domain="doi.org",
            description="EEG archive",
        )
    )

    def fetch_json(url: str):
        if url == "https://www.scidb.cn/api/sdb-openapi-service/json?doi=10.57760%2Fsciencedb.23155":
            return {"conditionsOfAccess": "PUBLIC", "version": "2.0.0"}
        if url.startswith("doi+resolve://"):
            return {"url": "https://www.scidb.cn/en/detail?dataSetId=9cacad83bdaa45d08a264c7f2d21a222"}
        if url.startswith("head+metadata://"):
            return {"status": 200}
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "resolved"
    assert resolution.files[0].name.endswith("_V2.zip")
    assert resolution.files[0].provider == "scidb"


def test_scidb_resolution_skips_archives_that_require_auth() -> None:
    plan = plan_acquisition(
        record(
            url="https://www.scidb.cn/detail?dataSetId=de4b079329404152919b0f26fd9996ff",
            doi="10.57760/sciencedb.psych.00751",
            source_domain="www.scidb.cn",
            description="EEG archive",
        )
    )

    def fetch_json(url: str):
        if url == "https://www.scidb.cn/api/sdb-openapi-service/json?doi=10.57760%2Fsciencedb.psych.00751":
            return {"conditionsOfAccess": "PUBLIC", "version": "V1"}
        if url.startswith("head+metadata://"):
            return {"status": 401}
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.status == "empty"
    assert resolution.files == ()


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


def test_materialize_remote_files_preserves_relative_paths_for_sidecars(tmp_path) -> None:
    plan = plan_acquisition(record(description="BrainVision EEG .vhdr files"))

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "sourcedata/sub-01/eeg/sub-01_task-test_eeg.vhdr",
                    "size": 4,
                    "links": {"self": "https://example.test/sub.vhdr"},
                },
                {
                    "key": "sourcedata/sub-01/eeg/sub-01_task-test_eeg.eeg",
                    "size": 4,
                    "links": {"self": "https://example.test/sub.eeg"},
                },
                {
                    "key": "sourcedata/sub-01/eeg/sub-01_task-test_eeg.vmrk",
                    "size": 4,
                    "links": {"self": "https://example.test/sub.vmrk"},
                },
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    materialize_remote_files(resolution.files, tmp_path, fetch_bytes=lambda _url, _max_bytes: b"sidecar", max_bytes=10)

    assert (tmp_path / "1" / "sourcedata" / "sub-01" / "eeg" / "sub-01_task-test_eeg.vhdr").read_bytes() == b"sidecar"
    assert (tmp_path / "1" / "sourcedata" / "sub-01" / "eeg" / "sub-01_task-test_eeg.eeg").read_bytes() == b"sidecar"
    assert (tmp_path / "1" / "sourcedata" / "sub-01" / "eeg" / "sub-01_task-test_eeg.vmrk").read_bytes() == b"sidecar"


def test_materialize_remote_files_with_archives_skips_metadata_only_files(tmp_path) -> None:
    plan = plan_acquisition(record(description="EEG archive"))

    def fetch_json(_url: str):
        return {
            "files": [
                {"key": "README.md", "size": 4, "links": {"self": "https://example.test/README.md"}},
                {"key": "dataset.zip", "size": 4, "links": {"self": "https://example.test/dataset.zip"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    downloaded: list[str] = []

    def fetch_bytes(url: str, _max_bytes: int) -> bytes:
        downloaded.append(url)
        return b"PK"

    results = materialize_remote_files(
        resolution.files,
        tmp_path,
        fetch_bytes=fetch_bytes,
        max_bytes=10,
        direct_only=False,
    )

    assert [result.status for result in results] == ["skipped", "downloaded"]
    assert downloaded == ["https://example.test/dataset.zip"]


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
