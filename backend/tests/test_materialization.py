from __future__ import annotations

import zipfile
from urllib.error import HTTPError
from urllib.parse import unquote

from neurocore.acquisition import plan_acquisition
from neurocore.datasets import DatasetRecord
from neurocore.materialization import (
    _doi_resolution_url,
    _fetch_json,
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


def test_mendeley_short_condition_csv_names_can_be_raw_eeg() -> None:
    plan = plan_acquisition(
        record(
            url="https://data.mendeley.com/datasets/4r8hp2hmb4",
            doi="10.17632/4r8hp2hmb4.1",
            source_domain="data.mendeley.com",
            name="Electroencephalogram ( EEG ) dataset with rest and executive function task",
            description="Raw EEG CSV files for eyes-closed, eyes-open, and task conditions.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "filename": "EC.csv",
                    "content_details": {"download_url": "https://example.test/ec.csv"},
                },
                {
                    "filename": "Participants.xlsx",
                    "content_details": {"download_url": "https://example.test/participants.xlsx"},
                },
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is True
    assert resolution.files[0].materialization_action == "download_then_load"
    assert resolution.files[1].directly_loadable is False


def test_device_named_csv_files_can_be_raw_eeg_when_dataset_is_eeg() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/5879794",
            doi="10.5281/zenodo.5879794",
            source_domain="zenodo.org",
            name="Motor Cortex EEG from OpenBCI and g.tec devices",
            description="CSV EEG recordings from OpenBCI and g.tec Unicorn devices.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "M_Gtec_20211214_130448-walk.csv",
                    "links": {"self": "https://example.test/M_Gtec_20211214_130448-walk.csv"},
                },
                {
                    "key": "A_openbci_20211218_140648-walk.csv",
                    "links": {"self": "https://example.test/A_openbci_20211218_140648-walk.csv"},
                },
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert [file.materialization_action for file in resolution.files] == ["download_then_load", "download_then_load"]


def test_subject_and_experiment_named_mat_files_can_be_raw_eeg() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/123456",
            doi="10.5281/zenodo.123456",
            source_domain="zenodo.org",
            name="Wireless EEG recordings and object category EEG dataset",
            description="MAT raw and preprocessed EEG files.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {"key": "ACSub1.mat", "links": {"self": "https://example.test/ACSub1.mat"}},
                {"key": "S10_a1.mat", "links": {"self": "https://example.test/S10_a1.mat"}},
                {"key": "P101C_BABBLE_AR.mat", "links": {"self": "https://example.test/P101C_BABBLE_AR.mat"}},
                {"key": "tFUS_rest_WT_m1.mat", "links": {"self": "https://example.test/tFUS_rest_WT_m1.mat"}},
                {"key": "opto_base_PV_r1.mat", "links": {"self": "https://example.test/opto_base_PV_r1.mat"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert all(file.directly_loadable for file in resolution.files)


def test_adhd_group_mat_files_can_be_raw_eeg() -> None:
    plan = plan_acquisition(
        record(
            url="https://data.mendeley.com/datasets/example",
            doi="10.17632/example.1",
            source_domain="data.mendeley.com",
            name="A Dataset of EEG Signals from Adults with ADHD and Healthy Controls",
            description="Resting state, cognitive challenge, and auditory stimulus raw EEG MAT files.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {"filename": "FADHD.mat", "content_details": {"download_url": "https://example.test/FADHD.mat"}},
                {"filename": "FC.mat", "content_details": {"download_url": "https://example.test/FC.mat"}},
                {"filename": "MADHD.mat", "content_details": {"download_url": "https://example.test/MADHD.mat"}},
                {"filename": "MC.mat", "content_details": {"download_url": "https://example.test/MC.mat"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert all(file.directly_loadable for file in resolution.files)
    assert {file.materialization_action for file in resolution.files} == {"download_then_load"}


def test_subject_preprocessed_mat_files_can_be_loader_material() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/11058711",
            doi="10.5281/zenodo.11058711",
            source_domain="zenodo.org",
            name="Audiovisual, Gaze-controlled Auditory Attention Decoding Dataset KU Leuven",
            equipment="MAT EEG",
            description="Subject-level preprocessed EEG MAT signals.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "2024-AV-GC-AAD-sub15_preprocessed.mat",
                    "links": {"self": "https://example.test/sub15_preprocessed.mat"},
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is True
    assert resolution.files[0].materialization_action == "download_then_load"


def test_hdf5_eeg_session_files_can_be_loader_material() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/123456",
            doi="10.12751/g-node.d76994",
            source_domain="zenodo.org",
            name="Simultaneous scalp EEG and intracranial EEG during verbal working memory",
            equipment="NIX HDF5",
            description="Data_Subject_*_Session_*.h5 files with scalp EEG and iEEG signals.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "data_nix/Data_Subject_01_Session_01.h5",
                    "links": {"self": "https://example.test/data_nix/Data_Subject_01_Session_01.h5"},
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is True
    assert resolution.files[0].materialization_action == "download_then_load"


def test_huggingface_parquet_eeg_splits_can_be_loader_material() -> None:
    plan = plan_acquisition(
        record(
            url="https://huggingface.co/datasets/JuniorThap/EEG-relaxation-concentration",
            doi="",
            source_domain="huggingface.co",
            name="JuniorThap/EEG-relaxation-concentration",
            equipment="Parquet EEG",
            description="Hugging Face dataset split into train and test Parquet EEG tables.",
        )
    )

    def fetch_json(_url: str):
        return [
            {"path": "data/train-00000-of-00002.parquet", "type": "file"},
            {"path": "data/test-00000-of-00001.parquet", "type": "file"},
            {"path": "README.md", "type": "file"},
        ]

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    by_name = {file.name: file for file in resolution.files}
    assert by_name["data/train-00000-of-00002.parquet"].directly_loadable is True
    assert by_name["data/test-00000-of-00001.parquet"].materialization_action == "download_then_load"
    assert by_name["README.md"].directly_loadable is False


def test_huggingface_torch_batch_files_can_be_loader_material() -> None:
    plan = plan_acquisition(
        record(
            url="https://huggingface.co/datasets/conorhassan/fast-autoregressive-inference-eeg",
            doi="",
            source_domain="huggingface.co",
            name="conorhassan/fast-autoregressive-inference-eeg",
            equipment="PyTorch EEG",
            description="Hugging Face train batch PT files for EEG inference.",
        )
    )

    def fetch_json(_url: str):
        return [
            {"path": "train/batch_000000.pt", "type": "file"},
            {"path": "train/batch_000001.pt", "type": "file"},
            {"path": ".gitattributes", "type": "file"},
        ]

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    by_name = {file.name: file for file in resolution.files}
    assert by_name["train/batch_000000.pt"].directly_loadable is True
    assert by_name["train/batch_000001.pt"].materialization_action == "download_then_load"
    assert by_name[".gitattributes"].directly_loadable is False


def test_huggingface_numpy_sentence_files_can_be_loader_material() -> None:
    plan = plan_acquisition(
        record(
            url="https://huggingface.co/datasets/PromiseZ5Q2SQ/EEG-Hallucination",
            doi="",
            source_domain="huggingface.co",
            name="PromiseZ5Q2SQ/EEG-Hallucination",
            equipment="NumPy EEG",
            description="EEG sentence-level NPY arrays with paired JSON labels.",
        )
    )

    def fetch_json(_url: str):
        return [
            {"path": "dataset/0_sen.npy", "type": "file"},
            {"path": "dataset/0_sen.json", "type": "file"},
            {"path": "img/AMBER_103.jpg", "type": "file"},
        ]

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    by_name = {file.name: file for file in resolution.files}
    assert by_name["dataset/0_sen.npy"].directly_loadable is True
    assert by_name["dataset/0_sen.npy"].materialization_action == "download_then_load"
    assert by_name["dataset/0_sen.json"].directly_loadable is False


def test_music_eeg_subject_episode_mat_files_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://purl.stanford.edu/rz763kn3821",
            doi="10.25740/rz763kn3821",
            source_domain="purl.stanford.edu",
            name="Naturalistic Music EEG Dataset - Rhythm Pilot",
            equipment="EGI MAT",
            description="EEG MAT recordings by subject and experiment episode.",
        )
    )

    def fetch_json(_url: str):
        return {
            "externalIdentifier": "druid:rz763kn3821",
            "structural": {
                "contains": [
                    {
                        "structural": {
                            "contains": [
                                {
                                    "type": "https://cocina.sul.stanford.edu/models/file",
                                    "filename": "S01_E01.mat",
                                    "access": {"download": "world"},
                                }
                            ]
                        }
                    }
                ]
            },
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is True
    assert resolution.files[0].materialization_action == "download_then_load"


def test_seeg_segment_mat_paths_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://osf.io/b7n5c",
            doi="10.17605/osf.io/b7n5c",
            source_domain="osf.io",
            name="Functional Mapping of Movement and Speech Using Task-Based Electrophysiological Changes in Stereoelectroencephalography",
            equipment="MAT",
            description="Task-based sEEG MATLAB data.",
        )
    )

    def fetch_json(_url: str):
        return {
            "data": [
                {
                    "attributes": {
                        "kind": "file",
                        "materialized_path": "/data/AML/AML_language_noun_seg.mat",
                    },
                    "links": {"download": "https://osf.io/download/seg/"},
                },
                {
                    "attributes": {
                        "kind": "file",
                        "materialized_path": "/data/AML/AML_stim_lang_elecs.mat",
                    },
                    "links": {"download": "https://osf.io/download/stim/"},
                },
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert by_name["data/AML/AML_language_noun_seg.mat"].directly_loadable is True
    assert by_name["data/AML/AML_stim_lang_elecs.mat"].directly_loadable is False


def test_numeric_session_mat_files_can_be_raw_eeg_when_dataset_context_is_neural() -> None:
    plan = plan_acquisition(
        record(
            url="https://data.mendeley.com/datasets/7r4z3p3g4m",
            doi="10.17632/7r4z3p3g4m",
            source_domain="data.mendeley.com",
            name="N&C-TEC: ECG and EEG files",
            equipment="MAT",
            description="EEG and ECG MATLAB session files.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {"filename": "01_01.mat", "content_details": {"download_url": "https://example.test/01_01.mat"}},
                {"filename": "02-05_07.mat", "content_details": {"download_url": "https://example.test/02-05_07.mat"}},
                {"filename": "participants.xlsx", "content_details": {"download_url": "https://example.test/participants.xlsx"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert by_name["01_01.mat"].directly_loadable is True
    assert by_name["02-05_07.mat"].directly_loadable is True
    assert by_name["participants.xlsx"].directly_loadable is False


def test_bci_x_named_mat_files_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://figshare.com/articles/dataset/example/14721297",
            doi="10.6084/m9.figshare.14721297",
            source_domain="figshare.com",
            name="BCI classification task",
            equipment="MAT",
            description="Brain-computer interface EEG classification matrices.",
        )
    )

    def fetch_json(_url: str):
        return {"files": [{"name": "X7.mat", "download_url": "https://example.test/X7.mat"}]}

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is True


def test_rds_and_ecog_mat_names_can_be_raw_signal_files() -> None:
    r_plan = plan_acquisition(
        record(
            url="https://figshare.com/articles/dataset/example/4210863",
            doi="10.6084/m9.figshare.4210863",
            source_domain="figshare.com",
            name="Early integration of conceptual modality in word recognition: ERP evidence",
            equipment="BrainVision RDS",
            description="ERP EEG R data exports.",
        )
    )
    mat_plan = plan_acquisition(
        record(
            url="https://data.mendeley.com/datasets/w68hwtb98d",
            doi="10.17632/w68hwtb98d",
            source_domain="data.mendeley.com",
            name="Stable decoding continuous hand movement trajectory from ECoG signals",
            equipment="MAT",
            description="ECoG MATLAB signal file plus code.",
        )
    )

    def fetch_r(_url: str):
        return {
            "files": [
                {"name": "EEG.rds", "download_url": "https://example.test/EEG.rds"},
                {"name": "EEG.window1.rds", "download_url": "https://example.test/EEG.window1.rds"},
            ]
        }

    def fetch_mat(_url: str):
        return {
            "files": [
                {"filename": "ECoG.mat", "content_details": {"download_url": "https://example.test/ECoG.mat"}},
                {"filename": "Motion.mat", "content_details": {"download_url": "https://example.test/Motion.mat"}},
            ]
        }

    r_resolution = resolve_remote_files(r_plan, fetch_json=fetch_r)
    mat_resolution = resolve_remote_files(mat_plan, fetch_json=fetch_mat)
    r_by_name = {file.name: file for file in r_resolution.files}
    mat_by_name = {file.name: file for file in mat_resolution.files}

    assert r_by_name["EEG.rds"].directly_loadable is True
    assert r_by_name["EEG.window1.rds"].directly_loadable is True
    assert mat_by_name["ECoG.mat"].directly_loadable is True
    assert mat_by_name["Motion.mat"].directly_loadable is False


def test_named_eeg_text_file_can_be_raw_signal_without_inventory_text_hint() -> None:
    plan = plan_acquisition(
        record(
            url="https://figshare.com/articles/dataset/example/14199467",
            doi="10.6084/m9.figshare.14199467",
            source_domain="figshare.com",
            name="A Novel Brain-Computer Interfaces System Design Based on Combining of fNIRS and EEG Signals",
            description="Signal dataset.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {"name": "dataset_eeg_fnirs.txt", "download_url": "https://example.test/dataset_eeg_fnirs.txt"},
                {"name": "readme.txt", "download_url": "https://example.test/readme.txt"},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert by_name["dataset_eeg_fnirs.txt"].directly_loadable is True
    assert by_name["readme.txt"].directly_loadable is False


def test_resting_state_session_and_dataset_mat_files_can_be_raw_signal_files() -> None:
    session_plan = plan_acquisition(
        record(
            url="https://figshare.com/articles/dataset/example/21644429",
            doi="10.6084/m9.figshare.21644429",
            source_domain="figshare.com",
            name="EEG Dataset for short-term meditation and sensorimotor rhythm BCI performance",
            description="MAT EEG subject intervention, resting state, and session files.",
        )
    )
    driver_plan = plan_acquisition(
        record(
            url="https://figshare.com/articles/dataset/example/14273687",
            doi="10.6084/m9.figshare.14273687",
            source_domain="figshare.com",
            name="EEG driver drowsiness dataset",
            equipment="MAT",
            description="MAT EEG driver drowsiness dataset.",
        )
    )

    def fetch_session(_url: str):
        return {
            "files": [
                {"name": "S01_Intervention1.mat", "download_url": "https://example.test/S01_Intervention1.mat"},
                {"name": "S01_RestingState.mat", "download_url": "https://example.test/S01_RestingState.mat"},
                {"name": "S01_Session2.mat", "download_url": "https://example.test/S01_Session2.mat"},
                {"name": "IntervalData.xlsx", "download_url": "https://example.test/IntervalData.xlsx"},
            ]
        }

    def fetch_driver(_url: str):
        return {
            "files": [
                {"name": "dataset.mat", "download_url": "https://example.test/dataset.mat"},
                {"name": "unbalanced_dataset.mat", "download_url": "https://example.test/unbalanced_dataset.mat"},
            ]
        }

    session_resolution = resolve_remote_files(session_plan, fetch_json=fetch_session)
    driver_resolution = resolve_remote_files(driver_plan, fetch_json=fetch_driver)
    session_by_name = {file.name: file for file in session_resolution.files}

    assert session_by_name["S01_Intervention1.mat"].directly_loadable is True
    assert session_by_name["S01_RestingState.mat"].directly_loadable is True
    assert session_by_name["S01_Session2.mat"].directly_loadable is True
    assert session_by_name["IntervalData.xlsx"].directly_loadable is False
    assert all(file.directly_loadable for file in driver_resolution.files)


def test_group_named_mat_and_numeric_csv_raw_exports_can_be_signal_files() -> None:
    mat_plan = plan_acquisition(
        record(
            url="https://figshare.com/articles/dataset/example/31403512",
            doi="10.6084/m9.figshare.31403512",
            source_domain="figshare.com",
            name="EEG p-adic quantum potential accurately identifies depression and schizophrenia",
            description="MAT EEG disease cohort data.",
        )
    )
    csv_plan = plan_acquisition(
        record(
            url="https://figshare.com/articles/dataset/example/31403511",
            doi="10.6084/m9.figshare.31403511",
            source_domain="figshare.com",
            name="EEG RAW data",
            description="Raw EEG CSV exports.",
        )
    )

    def fetch_mat(_url: str):
        return {
            "files": [
                {"name": "alz_c1_new.mat", "download_url": "https://example.test/alz_c1_new.mat"},
                {"name": "controls_c1_new.mat", "download_url": "https://example.test/controls_c1_new.mat"},
                {"name": "AUTHOR_DATASET_SHOR_BENNINGER.txt", "download_url": "https://example.test/readme.txt"},
            ]
        }

    def fetch_csv(_url: str):
        return {
            "files": [
                {"name": "260114-2-8.CSV", "download_url": "https://example.test/260114-2-8.CSV"},
                {"name": "README.txt", "download_url": "https://example.test/README.txt"},
            ]
        }

    mat_resolution = resolve_remote_files(mat_plan, fetch_json=fetch_mat)
    csv_resolution = resolve_remote_files(csv_plan, fetch_json=fetch_csv)
    mat_by_name = {file.name: file for file in mat_resolution.files}
    csv_by_name = {file.name: file for file in csv_resolution.files}

    assert mat_by_name["alz_c1_new.mat"].directly_loadable is True
    assert mat_by_name["controls_c1_new.mat"].directly_loadable is True
    assert mat_by_name["AUTHOR_DATASET_SHOR_BENNINGER.txt"].directly_loadable is False
    assert csv_by_name["260114-2-8.CSV"].directly_loadable is True
    assert csv_by_name["README.txt"].directly_loadable is False


def test_pickle_ecog_container_can_be_raw_signal_file() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/15220273",
            doi="10.5281/zenodo.15220273",
            source_domain="zenodo.org",
            name="ECoG Data of 8 Subjects Listening to a Podcast",
            description="Python pickle container with ECoG signal data.",
        )
    )

    def fetch_json(_url: str):
        return {"files": [{"key": "all_data.pkl", "links": {"self": "https://example.test/all_data.pkl"}}]}

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is True
    assert resolution.files[0].materialization_action == "download_then_load"


def test_eeg_named_spreadsheets_can_be_signal_files_without_marking_tables() -> None:
    plan = plan_acquisition(
        record(
            url="https://data.mendeley.com/datasets/sbyj5f6c3k",
            doi="10.17632/sbyj5f6c3k",
            source_domain="data.mendeley.com",
            name="EEG data: anxiety patients & control group",
            description="Spreadsheet EEG data.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {"filename": "EEG data.xlsx", "content_details": {"download_url": "https://example.test/EEG data.xlsx"}},
                {"filename": "Table_1.xls", "content_details": {"download_url": "https://example.test/Table_1.xls"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert by_name["EEG data.xlsx"].directly_loadable is True
    assert by_name["Table_1.xls"].directly_loadable is False


def test_extensionless_rawdata_parts_are_treated_as_archives() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/15760254",
            doi="10.5281/zenodo.15760254",
            source_domain="zenodo.org",
            name="MaskedFacePerception",
            description="Raw EEG dataset split into RawData_part files.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "RawData_part_1",
                    "links": {"self": "https://example.test/RawData_part_1"},
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].archive is True
    assert resolution.files[0].materialization_action == "download_extract_then_scan"


def test_extensionless_lettered_dataset_parts_are_treated_as_archives() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/123456",
            doi="10.5281/zenodo.123456",
            source_domain="zenodo.org",
            name="ERP Differences in Processing Canonical and Noncanonical Finger-Numeral Configurations",
            description="Compressed EEG dataset split into lettered part files.",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "Soylu_2019_DataversePublicData_part_a",
                    "links": {"self": "https://example.test/Soylu_2019_DataversePublicData_part_a"},
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].archive is True
    assert resolution.files[0].materialization_action == "download_extract_then_scan"


def test_ssvep_trial_csv_and_text_names_can_be_raw_signal_files() -> None:
    csv_plan = plan_acquisition(
        record(
            url="https://data.mendeley.com/datasets/f8v96skxj3",
            doi="10.17632/f8v96skxj3.1",
            source_domain="data.mendeley.com",
            name="RAW signal of EEG using SSVEP paradigm",
            description="CSV raw EEG files.",
            equipment="",
        )
    )
    text_plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/7758424",
            doi="10.5281/zenodo.7758424",
            source_domain="zenodo.org",
            name="SSVEP database elicited by four visual stimuli types",
            description=".txt EEG SSVEP database.",
            equipment="",
        )
    )

    def fetch_csv(_url: str):
        return {
            "files": [
                {"filename": "L_12_S1.csv", "content_details": {"download_url": "https://example.test/L_12_S1.csv"}}
            ]
        }

    def fetch_text(_url: str):
        return {"files": [{"key": "S06-mOO.txt", "links": {"self": "https://example.test/S06-mOO.txt"}}]}

    csv_resolution = resolve_remote_files(csv_plan, fetch_json=fetch_csv)
    text_resolution = resolve_remote_files(text_plan, fetch_json=fetch_text)

    assert csv_resolution.files[0].directly_loadable is True
    assert text_resolution.files[0].directly_loadable is True


def test_ssvep_numeric_numpy_files_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://github.com/hosseinhamidi92/SSVEP-Dataset",
            doi="",
            source_domain="github.com",
            name="hosseinhamidi92/SSVEP-Dataset",
            description="SSVEP EEG NumPy arrays by subject.",
            equipment="NumPy",
        )
    )

    def fetch_json(_url: str):
        return {
            "tree": [
                {"path": "1.npy", "type": "blob", "size": 10, "sha": "abc"},
                {"path": "README.md", "type": "blob", "size": 10, "sha": "def"},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert by_name["1.npy"].directly_loadable is True
    assert by_name["1.npy"].materialization_action == "download_then_load"
    assert by_name["README.md"].directly_loadable is False


def test_bci_and_example_data_mat_names_can_be_raw_signal_files() -> None:
    bci_plan = plan_acquisition(
        record(
            url="https://github.com/example/bci-challenge",
            doi="",
            source_domain="github.com",
            name="Clinical Brain Computer Interfaces Challenge WCCI 2020 Glasgow",
            description="BCI MATLAB training and evaluation files.",
            equipment="MAT",
        )
    )
    example_plan = plan_acquisition(
        record(
            url="https://purl.stanford.edu/dg856vy8753",
            doi="10.25740/dg856vy8753",
            source_domain="purl.stanford.edu",
            name="Example Data for SENSI EEG PREPROC - Bad Channel Detection Module",
            description="Example data MAT files for EEG preprocessing.",
            equipment="MAT",
        )
    )
    data_plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/123456",
            doi="10.5281/zenodo.123456",
            source_domain="zenodo.org",
            name="High density EEG measurement",
            description="High density EEG MATLAB data.",
            equipment="MAT",
        )
    )

    bci_resolution = resolve_remote_files(
        bci_plan,
        fetch_json=lambda _url: {
            "tree": [
                {"path": "parsed_P01T.mat", "type": "blob", "size": 10, "sha": "abc"},
                {"path": "eeglab_chan12_mod.locs", "type": "blob", "size": 10, "sha": "def"},
            ]
        },
    )
    example_resolution = resolve_remote_files(
        example_plan,
        fetch_json=lambda _url: {
            "externalIdentifier": "druid:dg856vy8753",
            "structural": {
                "contains": [
                    {
                        "type": "https://cocina.sul.stanford.edu/models/file",
                        "filename": "data1_W.mat",
                        "access": {"download": "world"},
                    }
                ]
            },
        },
    )
    data_resolution = resolve_remote_files(
        data_plan,
        fetch_json=lambda _url: {
            "files": [
                {"key": "data/S1.mat", "links": {"self": "https://example.test/data/S1.mat"}},
                {"key": "Scripts/head256.loc", "links": {"self": "https://example.test/Scripts/head256.loc"}},
            ]
        },
    )

    assert bci_resolution.files[0].directly_loadable is True
    assert bci_resolution.files[1].directly_loadable is False
    assert example_resolution.files[0].directly_loadable is True
    assert data_resolution.files[0].directly_loadable is True
    assert data_resolution.files[1].directly_loadable is False


def test_epoch_cohort_and_sleep_numpy_names_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/17955369",
            doi="10.5281/zenodo.17955369",
            source_domain="zenodo.org",
            name="Resting-State EEG Dataset for Depression and Healthy Controls",
            description=".mat and .npz EEG recordings.",
            equipment="",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {"key": "VarekaGTNEpochs.mat", "links": {"self": "https://example.test/VarekaGTNEpochs.mat"}},
                {"key": "depec22.mat", "links": {"self": "https://example.test/depec22.mat"}},
                {"key": "hcec04.mat", "links": {"self": "https://example.test/hcec04.mat"}},
                {
                    "key": "SensoryStimulationData_BlockDesign.mat",
                    "links": {"self": "https://example.test/SensoryStimulationData_BlockDesign.mat"},
                },
                {"key": "shhs1-200010.npz", "links": {"self": "https://example.test/shhs1-200010.npz"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert all(file.directly_loadable for file in resolution.files)


def test_macaque_ecog_combined_mat_names_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/123456",
            doi="10.5281/zenodo.123456",
            source_domain="zenodo.org",
            name="Local-global Macaque ECoG",
            description="Raw MAT ECoG data files.",
            equipment="",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "Data_fixBB_corrected_screen_combined_Pre500_Post1700_Qu.mat",
                    "links": {"self": "https://example.test/Data_fixBB_corrected_screen_combined_Pre500_Post1700_Qu.mat"},
                },
                {"key": "elPosition_Qu.mat", "links": {"self": "https://example.test/elPosition_Qu.mat"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert by_name["Data_fixBB_corrected_screen_combined_Pre500_Post1700_Qu.mat"].directly_loadable is True
    assert by_name["elPosition_Qu.mat"].directly_loadable is False


def test_osf_resolution_prioritizes_raw_eeg_folders_with_small_page_limit() -> None:
    plan = plan_acquisition(
        record(
            url="https://osf.io/nech6",
            doi="",
            source_domain="osf.io",
            description="Raw EEGLAB recordings.",
        )
    )

    def fetch_json(url: str):
        if url == "https://api.osf.io/v2/nodes/nech6/files/":
            return {
                "data": [
                    {
                        "type": "files",
                        "attributes": {"kind": "folder", "name": "Behavioural Data"},
                        "relationships": {"files": {"links": {"related": {"href": "https://api.osf.io/v2/folders/behav/"}}}},
                    },
                    {
                        "type": "files",
                        "attributes": {"kind": "folder", "name": "Raw EEG"},
                        "relationships": {"files": {"links": {"related": {"href": "https://api.osf.io/v2/folders/raw/"}}}},
                    },
                ],
                "links": {"next": None},
            }
        if url == "https://api.osf.io/v2/folders/raw/":
            return {
                "data": [
                    {
                        "type": "files",
                        "attributes": {"kind": "file", "materialized_path": "/Raw EEG/sub-01.set", "size": 10},
                        "links": {"download": "https://osf.io/download/sub-01.set"},
                    }
                ],
                "links": {"next": None},
            }
        if url == "https://api.osf.io/v2/nodes/nech6/children/?page[size]=100":
            return {"data": [], "links": {"next": None}}
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json, max_pages=2)

    assert resolution.files[0].name == "Raw EEG/sub-01.set"
    assert resolution.files[0].directly_loadable is True


def test_sleep_stage_torch_samples_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/18725150",
            doi="10.5281/zenodo.18725150",
            source_domain="zenodo.org",
            name="Small sample dataset for preclinical sleep stage classification",
            description="PyTorch .pth EEG sample tensors.",
            equipment="",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "11_saline_sleep_annotated_sample1907.pth",
                    "links": {"self": "https://example.test/11_saline_sleep_annotated_sample1907.pth"},
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is True
    assert resolution.files[0].materialization_action == "download_then_load"


def test_sleep_edf_split_torch_files_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/TITSXU",
            doi="10.21979/N9/TITSXU",
            source_domain="researchdata.ntu.edu.sg",
            name="Preprocessed SLeep-EDF dataset",
            description="single-channel EEG Fpz-Cz sleep epochs in public train.pt, val.pt, and test.pt files",
            equipment="EEG Fpz-Cz",
        )
    )

    def fetch_json(_url: str):
        return {
            "data": {
                "latestVersion": {
                    "files": [
                        {"dataFile": {"id": 1, "filename": "train.pt"}},
                        {"dataFile": {"id": 2, "filename": "val.pt"}},
                        {"dataFile": {"id": 3, "filename": "test.pt"}},
                    ]
                }
            }
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert all(file.directly_loadable for file in resolution.files)


def test_sleep_edf_npz_recordings_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/MA1AVG",
            doi="10.21979/N9/MA1AVG",
            source_domain="researchdata.ntu.edu.sg",
            name="Preprocessed Sleep-EDF-20 dataset",
            description="Sleep-EDF EEG Fpz-Cz 100Hz recordings distributed as SC4001E0.npz files.",
            equipment="EEG Fpz-Cz",
        )
    )

    def fetch_json(_url: str):
        return {
            "data": {
                "latestVersion": {
                    "files": [
                        {"dataFile": {"id": 1, "filename": "SC4001E0.npz"}},
                        {"dataFile": {"id": 2, "filename": "SC4192E0.npz"}},
                    ]
                }
            }
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert all(file.directly_loadable for file in resolution.files)


def test_motor_imagery_session_mat_names_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/251NOW",
            doi="10.7910/DVN/251NOW",
            source_domain="dataverse.harvard.edu",
            name="A cross-session motor imagery EEG dataset",
            description="Subject-session MATLAB EEG files S01D1.mat through S14D2.mat.",
            equipment="Neuroscan SynAmps2 EEG",
        )
    )

    def fetch_json(_url: str):
        return {
            "data": {
                "latestVersion": {
                    "files": [
                        {"dataFile": {"id": 1, "filename": "S01D1.mat"}},
                        {"dataFile": {"id": 2, "filename": "S14D2.mat"}},
                    ]
                }
            }
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert all(file.directly_loadable for file in resolution.files)


def test_time_series_classification_ts_files_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://zenodo.org/records/18956117",
            doi="10.5281/zenodo.18956117",
            source_domain="zenodo.org",
            name="EpilepticSeizures",
            description="single-channel EEG seizure benchmark distributed as Time Series Classification .ts files",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "EpilepticSeizures_TRAIN.ts",
                    "links": {"self": "https://example.test/EpilepticSeizures_TRAIN.ts"},
                },
                {
                    "key": "EpilepticSeizures_TEST.ts",
                    "links": {"self": "https://example.test/EpilepticSeizures_TEST.ts"},
                },
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert all(file.directly_loadable for file in resolution.files)


def test_eeg_binary_bodies_are_flagged_as_companion_metadata_required() -> None:
    plan = plan_acquisition(
        record(
            url="https://osf.io/qm37x",
            doi="10.17605/osf.io/qm37x",
            source_domain="osf.io",
            name="Learning through socio-emotional feedback and age differences: An ERP study",
            description="EEG_data folders expose .eeg raw body files without BrainVision headers.",
        )
    )

    def fetch_json(_url: str):
        return {
            "data": [
                {
                    "attributes": {
                        "kind": "file",
                        "materialized_path": "/EEG_data/Older_Adults/369.eeg",
                    },
                    "links": {"download": "https://osf.io/download/369/"},
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is False
    assert resolution.files[0].materialization_action == "download_with_companion_metadata"


def test_rds_ecog_files_can_be_raw_signal_files() -> None:
    plan = plan_acquisition(
        record(
            url="https://osf.io/4vdyu",
            doi="",
            source_domain="osf.io",
            name="HUP (peri-seizure ECoG RDS release)",
            description="iEEG and ECoG peri-seizure data distributed as RDS files.",
        )
    )

    def fetch_json(_url: str):
        return {
            "data": [
                {
                    "attributes": {
                        "kind": "file",
                        "materialized_path": "/HUPData_HUP082_5.rds",
                    },
                    "links": {"download": "https://osf.io/download/HUPData_HUP082_5/"},
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].directly_loadable is True
    assert resolution.files[0].materialization_action == "download_then_load"


def test_nitrc_frs_resolution_extracts_raw_eeg_archives_from_download_links() -> None:
    plan = plan_acquisition(
        record(
            url="https://www.nitrc.org/frs/?group_id=1223",
            doi="",
            source_domain="nitrc.org",
            name="VEP EEG raw data",
            description="EEGLAB ESS raw archive",
        )
    )

    def fetch_json(url: str):
        assert url.startswith("html+landing://")
        return """
        <a href="/frs/download.php/10376/README.md" title="README.md">README.md</a>
        <a href="/frs/downloadlink.php/10377">VEP raw EEG containerized</a>
        <a href="/frs/downloadlink.php/10378">VEP EEG cleaned with MARA</a>
        """

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert [file.name for file in resolution.files] == [
        "VEP raw EEG containerized.zip",
        "VEP EEG cleaned with MARA.zip",
    ]
    assert all(file.archive for file in resolution.files)


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


def test_doi_resolution_fetcher_falls_back_to_range_get_when_head_is_forbidden(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        status = 206
        headers = {"Content-Type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return "https://neurodata.riken.jp/id/20240220-001"

    def fake_urlopen(request, timeout):
        calls.append((request.get_method(), dict(request.header_items()), timeout))
        if len(calls) == 1:
            raise HTTPError(request.full_url, 403, "Forbidden", {}, None)
        return FakeResponse()

    monkeypatch.setattr("neurocore.materialization.urlopen", fake_urlopen)

    payload = _fetch_json(_doi_resolution_url("https://doi.org/10.60178/cbs.20240220-001"), timeout=7)

    assert payload["url"] == "https://neurodata.riken.jp/id/20240220-001"
    assert calls[0][0] == "HEAD"
    assert calls[1][1]["Range"] == "bytes=0-0"


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


def test_generic_mat_detection_accepts_subject_data_names_when_plan_hints_mat() -> None:
    plan = plan_acquisition(
        record(
            url="https://figshare.com/articles/dataset/example/23641017",
            doi="10.6084/m9.figshare.23641017",
            source_domain="figshare.com",
            description="SSVEP EEG MATLAB dataset",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {"name": "data_s1_64.mat", "download_url": "https://example.test/data_s1_64.mat"},
                {"name": "Sub_score.mat", "download_url": "https://example.test/Sub_score.mat"},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert by_name["data_s1_64.mat"].directly_loadable is True
    assert by_name["Sub_score.mat"].directly_loadable is False


def test_figshare_collection_resolution_expands_member_articles() -> None:
    plan = plan_acquisition(
        record(
            url="https://springernature.figshare.com/collections/example/5769449",
            doi="10.6084/m9.figshare.c.5769449",
            source_domain="springernature.figshare.com",
            description="BIDS MATLAB EEG collection",
        )
    )

    def fetch_json(url: str):
        if url == "https://api.figshare.com/v2/collections/5769449/articles?page_size=100":
            return [{"id": 17701079, "url_public_api": "https://api.figshare.com/v2/articles/17701079"}]
        if url == "https://api.figshare.com/v2/articles/17701079":
            return {
                "files": [
                    {"name": "s04.mat", "download_url": "https://example.test/s04.mat"},
                    {"name": "README.txt", "download_url": "https://example.test/README.txt"},
                ]
            }
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert resolution.status == "resolved"
    assert by_name["s04.mat"].directly_loadable is True
    assert by_name["README.txt"].directly_loadable is False


def test_generic_mat_detection_accepts_session_and_group_names_but_skips_derived_tables() -> None:
    plan = plan_acquisition(
        record(
            url="https://data.mendeley.com/datasets/skw8hhmjnx/1",
            doi="10.17632/skw8hhmjnx.1",
            source_domain="data.mendeley.com",
            description="MATLAB EEG calibration and gameplay recordings",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {"filename": "1_calibration.mat", "content_details": {"download_url": "https://example.test/1_calibration.mat"}},
                {
                    "filename": "10_singleplayer.mat",
                    "content_details": {"download_url": "https://example.test/10_singleplayer.mat"},
                },
                {"filename": "AD.mat", "content_details": {"download_url": "https://example.test/AD.mat"}},
                {
                    "filename": "Data_Design_Sub_1.mat",
                    "content_details": {"download_url": "https://example.test/Data_Design_Sub_1.mat"},
                },
                {"filename": "dataica.mat", "content_details": {"download_url": "https://example.test/dataica.mat"}},
                {"filename": "Subj_1_rest.mat", "content_details": {"download_url": "https://example.test/Subj_1_rest.mat"}},
                {"filename": "Subj_1_TMS.mat", "content_details": {"download_url": "https://example.test/Subj_1_TMS.mat"}},
                {"filename": "CLASS_A.mat", "content_details": {"download_url": "https://example.test/CLASS_A.mat"}},
                {"filename": "DIFF6.mat", "content_details": {"download_url": "https://example.test/DIFF6.mat"}},
                {
                    "filename": "Central_alpha_fft_rest.mat",
                    "content_details": {"download_url": "https://example.test/Central_alpha_fft_rest.mat"},
                },
                {"filename": "EEGSTARTTIME.mat", "content_details": {"download_url": "https://example.test/EEGSTARTTIME.mat"}},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert by_name["1_calibration.mat"].directly_loadable is True
    assert by_name["10_singleplayer.mat"].directly_loadable is True
    assert by_name["AD.mat"].directly_loadable is True
    assert by_name["Data_Design_Sub_1.mat"].directly_loadable is True
    assert by_name["dataica.mat"].directly_loadable is True
    assert by_name["Subj_1_rest.mat"].directly_loadable is True
    assert by_name["Subj_1_TMS.mat"].directly_loadable is True
    assert by_name["CLASS_A.mat"].directly_loadable is True
    assert by_name["DIFF6.mat"].directly_loadable is True
    assert by_name["Central_alpha_fft_rest.mat"].directly_loadable is False
    assert by_name["EEGSTARTTIME.mat"].directly_loadable is False


def test_dryad_resolution_follows_version_files_and_marks_archive() -> None:
    plan = plan_acquisition(
        record(
            url="https://doi.org/10.5061/dryad.46786",
            doi="10.5061/dryad.46786",
            source_domain="doi.org",
            description="EGI and MAT EEG archive",
        )
    )

    def fetch_json(url: str):
        if url == "https://datadryad.org/api/v2/datasets/doi%3A10.5061%2Fdryad.46786":
            return {"_links": {"stash:version": {"href": "/api/v2/versions/19362"}}}
        if url == "https://datadryad.org/api/v2/versions/19362":
            return {"_links": {"stash:files": {"href": "/api/v2/versions/19362/files"}}}
        if url == "https://datadryad.org/api/v2/versions/19362/files":
            return {
                "_embedded": {
                    "stash:files": [
                        {
                            "path": "ftonsets.zip",
                            "size": 10600732436,
                            "mimeType": "application/zip",
                            "digest": "4b953232184a1821313fe420ebf11162",
                            "digestType": "md5",
                            "_links": {"stash:download": {"href": "/api/v2/files/65462/download"}},
                        },
                        {
                            "path": "README_for_ftonsets.txt",
                            "size": 3718,
                            "mimeType": "text/plain",
                            "_links": {"stash:download": {"href": "/api/v2/files/65463/download"}},
                        },
                    ]
                }
            }
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert resolution.status == "resolved"
    assert by_name["ftonsets.zip"].archive is True
    assert by_name["ftonsets.zip"].materialization_action == "download_extract_then_scan"
    assert by_name["ftonsets.zip"].checksum == "md5:4b953232184a1821313fe420ebf11162"
    assert by_name["README_for_ftonsets.txt"].materialization_action == "metadata_or_manual_review"


def test_dspace_resolution_follows_item_bundles_and_bitstreams_from_doi() -> None:
    plan = plan_acquisition(
        record(
            url="https://doi.org/10.3929/ethz-b-000458693",
            doi="10.3929/ethz-b-000458693",
            source_domain="doi.org",
            description="CYBATHLON EEG ZIP archive",
        )
    )

    def fetch_json(url: str):
        if url.startswith("doi+resolve://"):
            return {"url": "https://www.research-collection.ethz.ch/handle/20.500.11850/458693"}
        if url.startswith("https://www.research-collection.ethz.ch/server/api/discover/search/objects?"):
            return {
                "_embedded": {
                    "searchResult": {
                        "_embedded": {
                            "objects": [
                                {
                                    "_links": {
                                        "indexableObject": {
                                            "href": "https://www.research-collection.ethz.ch/server/api/core/items/item-1"
                                        }
                                    },
                                    "_embedded": {
                                        "indexableObject": {"handle": "20.500.11850/458693"}
                                    },
                                }
                            ]
                        }
                    }
                }
            }
        if url == "https://www.research-collection.ethz.ch/server/api/core/items/item-1":
            return {"_links": {"bundles": {"href": "https://www.research-collection.ethz.ch/server/api/core/items/item-1/bundles"}}}
        if url == "https://www.research-collection.ethz.ch/server/api/core/items/item-1/bundles":
            return {
                "_embedded": {
                    "bundles": [
                        {
                            "name": "ORIGINAL",
                            "_links": {
                                "bitstreams": {
                                    "href": "https://www.research-collection.ethz.ch/server/api/core/bundles/original/bitstreams"
                                }
                            },
                        },
                        {
                            "name": "LICENSE",
                            "_links": {
                                "bitstreams": {
                                    "href": "https://www.research-collection.ethz.ch/server/api/core/bundles/license/bitstreams"
                                }
                            },
                        },
                    ]
                }
            }
        if url == "https://www.research-collection.ethz.ch/server/api/core/bundles/original/bitstreams":
            return {
                "_embedded": {
                    "bitstreams": [
                        {
                            "name": "ReadMe.docx",
                            "sizeBytes": 14683,
                            "_links": {"content": {"href": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/readme/content"}},
                        },
                        {
                            "name": "Cybathlon_Data.zip",
                            "sizeBytes": 3151378269,
                            "checkSum": {"checkSumAlgorithm": "MD5", "value": "9edd0803863bb5a3448cf0063a4b9f9c"},
                            "_links": {"content": {"href": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/data/content"}},
                        },
                    ]
                }
            }
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert resolution.status == "resolved"
    assert by_name["Cybathlon_Data.zip"].archive is True
    assert by_name["Cybathlon_Data.zip"].checksum == "md5:9edd0803863bb5a3448cf0063a4b9f9c"
    assert by_name["ReadMe.docx"].materialization_action == "metadata_or_manual_review"


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


def test_generic_csv_detection_accepts_numeric_raw_files_but_not_marker_or_metadata_csvs() -> None:
    plan = plan_acquisition(
        record(
            url="https://figshare.com/articles/dataset/example/31964610",
            doi="10.6084/m9.figshare.31964610",
            source_domain="figshare.com",
            description="raw EEG participant CSV recordings",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {"name": "235745.csv", "download_url": "https://example.test/235745.csv"},
                {"name": "235745_intervalMarker.csv", "download_url": "https://example.test/235745_intervalMarker.csv"},
                {"name": "participant_metadata.csv", "download_url": "https://example.test/participant_metadata.csv"},
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)
    by_name = {file.name: file for file in resolution.files}

    assert by_name["235745.csv"].directly_loadable is True
    assert by_name["235745_intervalMarker.csv"].directly_loadable is False
    assert by_name["participant_metadata.csv"].directly_loadable is False


def test_generic_csv_and_text_detection_accepts_kmi_raw_exports() -> None:
    csv_plan = plan_acquisition(
        record(
            url="https://data.mendeley.com/datasets/msgzn862ns",
            doi="10.17632/msgzn862ns",
            source_domain="data.mendeley.com",
            description="raw EEG CSV files for kinesthetic motor imagery",
        )
    )
    text_plan = plan_acquisition(
        record(
            url="https://figshare.com/articles/dataset/example/25773342",
            doi="10.6084/m9.figshare.25773342",
            source_domain="figshare.com",
            description="raw EEG text files for kinesthetic motor imagery",
        )
    )

    def fetch_csv(_url: str):
        return {
            "files": [
                {"filename": "S01_10_KMI.csv", "content_details": {"download_url": "https://example.test/S01_10_KMI.csv"}},
                {"filename": "info_subjects.xlsx", "content_details": {"download_url": "https://example.test/info_subjects.xlsx"}},
            ]
        }

    def fetch_text(_url: str):
        return {
            "files": [
                {"name": "user001_10_1.txt", "download_url": "https://example.test/user001_10_1.txt"},
                {"name": "readme.txt", "download_url": "https://example.test/readme.txt"},
            ]
        }

    csv_resolution = resolve_remote_files(csv_plan, fetch_json=fetch_csv)
    text_resolution = resolve_remote_files(text_plan, fetch_json=fetch_text)

    csv_by_name = {file.name: file for file in csv_resolution.files}
    text_by_name = {file.name: file for file in text_resolution.files}
    assert csv_by_name["S01_10_KMI.csv"].directly_loadable is True
    assert csv_by_name["info_subjects.xlsx"].directly_loadable is False
    assert text_by_name["user001_10_1.txt"].directly_loadable is True
    assert text_by_name["readme.txt"].directly_loadable is False


def test_osf_resolution_follows_child_nodes_and_prioritizes_raw_folders() -> None:
    plan = plan_acquisition(
        record(
            url="https://osf.io/5jz9d",
            doi="",
            source_domain="osf.io",
            description="BrainVision EEG RawData",
        )
    )
    queried: list[str] = []

    def fetch_json(url: str):
        queried.append(url)
        if url == "https://api.osf.io/v2/nodes/5jz9d/files/":
            return {
                "data": [
                    {
                        "type": "files",
                        "attributes": {"kind": "folder", "name": "osfstorage"},
                        "relationships": {
                            "files": {
                                "links": {"related": {"href": "https://api.osf.io/v2/nodes/5jz9d/files/osfstorage/"}}
                            }
                        },
                    }
                ]
            }
        if url == "https://api.osf.io/v2/nodes/5jz9d/files/osfstorage/":
            return {
                "data": [
                    {
                        "type": "nodes",
                        "id": "other",
                        "attributes": {"title": "MeanERP"},
                        "relationships": {"children": {"links": {"related": {"href": "https://api.osf.io/v2/nodes/other/children/"}}}},
                    },
                    {"type": "nodes", "id": "raw1", "attributes": {"title": "RawData"}},
                ]
            }
        if url == "https://api.osf.io/v2/nodes/raw1/files/":
            return {
                "data": [
                    {
                        "type": "files",
                        "attributes": {
                            "kind": "file",
                            "materialized_path": "/sub-01/eeg/sub-01_task-test_eeg.vhdr",
                            "size": 123,
                        },
                        "links": {"download": "https://files.osf.io/sub-01_task-test_eeg.vhdr"},
                    }
                ]
            }
        return {"data": []}

    resolution = resolve_remote_files(plan, fetch_json=fetch_json, max_pages=5)

    assert queried.index("https://api.osf.io/v2/nodes/raw1/files/") < len(queried)
    assert resolution.files[0].directly_loadable is True


def test_xz_tarballs_are_classified_as_extractable_archives() -> None:
    plan = plan_acquisition(record(description="EEG FIF archive"))

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "key": "EEG.tar.xz",
                    "size": 123,
                    "links": {"self": "https://example.test/EEG.tar.xz"},
                }
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].archive is True
    assert resolution.files[0].materialization_action == "download_extract_then_scan"


def test_split_7z_parts_are_classified_as_archive_candidates() -> None:
    plan = plan_acquisition(
        record(
            url="https://data.mendeley.com/datasets/zwcx948yjc",
            doi="10.17632/zwcx948yjc",
            source_domain="data.mendeley.com",
            equipment="EDF",
            description="telemetry EEG split 7z archives",
        )
    )

    def fetch_json(_url: str):
        return {
            "files": [
                {
                    "filename": "A05_GD_MDZ3.7z.001",
                    "content_details": {"download_url": "https://example.test/A05_GD_MDZ3.7z.001"},
                },
                {
                    "filename": "A05_GD_MDZ3.7z.002",
                    "content_details": {"download_url": "https://example.test/A05_GD_MDZ3.7z.002"},
                },
            ]
        }

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert all(file.archive for file in resolution.files)
    assert {file.materialization_action for file in resolution.files} == {"download_extract_then_scan"}


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


def test_http_landing_uses_aria_label_when_download_url_has_no_extension() -> None:
    plan = plan_acquisition(
        record(
            url="https://depositonce.tu-berlin.de/handle/11303/10934.2",
            doi="10.14279/depositonce-9827.2",
            source_domain="depositonce.tu-berlin.de",
            description="MAT motor imagery EEG files",
        )
    )

    html = '<a href="/bitstreams/abc/download" aria-label="Download pp1.mat"></a>'
    resolution = resolve_remote_files(plan, fetch_json=lambda url: html if url.startswith("html+landing://") else {})

    assert resolution.files[0].name == "pp1.mat"
    assert resolution.files[0].url == "https://depositonce.tu-berlin.de/bitstreams/abc/download"
    assert resolution.files[0].directly_loadable is True


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


def test_http_landing_delegates_arxiv_dataset_doi_to_zenodo() -> None:
    plan = plan_acquisition(
        record(
            url="https://arxiv.org/abs/1904.09111",
            doi="10.48550/arXiv.1904.09111",
            source_domain="arxiv.org",
            description="P300 BCI EEG dataset in MAT and CSV formats.",
        )
    )
    queried: list[str] = []

    def fetch_json(url: str):
        queried.append(url)
        if url.startswith("html+landing://"):
            return '<a href="https://doi.org/10.5281/zenodo.1494163">dataset DOI</a>'
        assert url == "https://zenodo.org/api/records/1494163"
        return {"files": [{"key": "subject01.mat", "links": {"self": "https://example.test/subject01.mat"}}]}

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert queried == [
        "html+landing://?url=https%3A%2F%2Farxiv.org%2Fabs%2F1904.09111",
        "https://zenodo.org/api/records/1494163",
    ]
    assert resolution.files[0].provider == "zenodo"
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


def test_github_resolution_accepts_api_repo_urls() -> None:
    plan = plan_acquisition(
        record(
            url="https://api.github.com/repos/bdsp-core/Hypothermia-EEG/git/trees/main?recursive=1",
            doi="",
            source_domain="github.com",
            description="MAT EEG files",
        )
    )

    def fetch_json(url: str):
        assert url == "https://api.github.com/repos/bdsp-core/Hypothermia-EEG/git/trees/HEAD?recursive=1"
        return {"tree": [{"path": "data/raw_subject01.mat", "type": "blob", "size": 10, "sha": "abc"}]}

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].url == "https://raw.githubusercontent.com/bdsp-core/Hypothermia-EEG/HEAD/data/raw_subject01.mat"
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
        if url.endswith("/SHA256SUMS.txt"):
            return ""
        if url.endswith("/1.0.0/"):
            return '<a href="sub-01/">sub-01/</a>'
        if url.endswith("/sub-01/"):
            return '<a href="../">../</a><a href="sub-01_eeg.edf">sub-01_eeg.edf</a>'
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].name == "sub-01/sub-01_eeg.edf"
    assert resolution.files[0].directly_loadable is True


def test_physionet_resolution_discovers_version_from_project_landing() -> None:
    plan = plan_acquisition(
        record(
            url="https://physionet.org/content/bigp3bci/",
            doi="",
            source_domain="physionet.org",
            description="EDF recordings",
        )
    )

    def fetch_json(url: str):
        if url.startswith("html+landing://"):
            return '<title>bigP3BCI v1.0.0</title><a href="/content/bigp3bci/1.0.0/">version</a>'
        if url.endswith("/SHA256SUMS.txt"):
            return ""
        if url.endswith("/1.0.0/"):
            return '<a href="sub-01/">sub-01/</a>'
        if url.endswith("/sub-01/"):
            return '<a href="../">../</a><a href="sub-01_eeg.edf">sub-01_eeg.edf</a>'
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json)

    assert resolution.files[0].name == "sub-01/sub-01_eeg.edf"
    assert resolution.files[0].directly_loadable is True


def test_physionet_resolution_prefers_sha256_manifest_when_available() -> None:
    plan = plan_acquisition(
        record(
            url="https://physionet.org/content/bigp3bci/",
            doi="",
            source_domain="physionet.org",
            description="EDF recordings",
        )
    )

    def fetch_json(url: str):
        if url.startswith("html+landing://"):
            return '<a href="/content/bigp3bci/1.0.0/">version</a>'
        if url.endswith("/SHA256SUMS.txt"):
            return "abc123 bigP3BCI-data/StudyA/A_01/SE001/Test/CB/A_01_SE001_CB_Test06.edf\n"
        raise AssertionError(url)

    resolution = resolve_remote_files(plan, fetch_json=fetch_json, max_pages=1)

    assert resolution.files[0].name == "bigP3BCI-data/StudyA/A_01/SE001/Test/CB/A_01_SE001_CB_Test06.edf"
    assert resolution.files[0].checksum == "sha256:abc123"
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
