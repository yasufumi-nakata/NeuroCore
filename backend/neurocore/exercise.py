from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .acquisition import AcquisitionPlan
from .materialization import RemoteFileResolution


@dataclass(frozen=True)
class DatasetExerciseRecord:
    record_id: str
    name: str
    provider: str
    access_status: str
    automation_status: str
    state: str
    required_action: str
    candidate_count: int
    remote_status: str | None = None
    remote_file_count: int = 0
    directly_loadable_file_count: int = 0
    archive_file_count: int = 0
    local_signal_file_count: int = 0
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "name": self.name,
            "provider": self.provider,
            "access_status": self.access_status,
            "automation_status": self.automation_status,
            "state": self.state,
            "required_action": self.required_action,
            "candidate_count": self.candidate_count,
            "remote_status": self.remote_status,
            "remote_file_count": self.remote_file_count,
            "directly_loadable_file_count": self.directly_loadable_file_count,
            "archive_file_count": self.archive_file_count,
            "local_signal_file_count": self.local_signal_file_count,
            "errors": list(self.errors),
        }


def exercise_dataset_records(
    plans: Iterable[AcquisitionPlan],
    *,
    remote_resolutions: Iterable[RemoteFileResolution] = (),
    local_readiness: Iterable[dict[str, Any]] = (),
) -> tuple[DatasetExerciseRecord, ...]:
    resolution_by_id = {item.record_id: item for item in remote_resolutions}
    readiness_by_id = {str(item.get("record_id", "")): item for item in local_readiness}
    return tuple(_exercise_plan(plan, resolution_by_id.get(plan.record_id), readiness_by_id.get(plan.record_id)) for plan in plans)


def summarize_dataset_exercise(records: Iterable[DatasetExerciseRecord]) -> dict[str, Any]:
    items = tuple(records)
    state_counts = Counter(item.state for item in items)
    provider_counts = Counter(item.provider for item in items)
    automation_counts = Counter(item.automation_status for item in items)
    actionable_states = {"local_raw_ready", "remote_direct_raw_ready", "remote_archive_requires_extraction"}
    external_blocker_states = {"blocked_account_required", "blocked_unusable"}
    needs_resolver_states = {"remote_resolution_failed", "remote_no_files_found", "remote_metadata_only", "no_acquisition_candidate"}
    return {
        "record_count": len(items),
        "state_counts": dict(state_counts.most_common()),
        "automation_status_counts": dict(automation_counts.most_common()),
        "provider_counts": dict(provider_counts.most_common(30)),
        "records_with_actionable_raw_path": sum(item.state in actionable_states for item in items),
        "records_with_external_blocker": sum(item.state in external_blocker_states for item in items),
        "records_needing_resolver_work": sum(item.state in needs_resolver_states for item in items),
        "records_not_yet_remotely_exercised": state_counts.get("remote_not_exercised", 0),
        "records_with_remote_file_candidates": sum(item.remote_file_count > 0 for item in items),
        "records_with_direct_raw_candidates": sum(item.directly_loadable_file_count > 0 for item in items),
        "records_with_archive_candidates": sum(item.archive_file_count > 0 for item in items),
        "records_with_local_signal_files": sum(item.local_signal_file_count > 0 for item in items),
        "remote_file_count": sum(item.remote_file_count for item in items),
        "directly_loadable_file_count": sum(item.directly_loadable_file_count for item in items),
        "archive_file_count": sum(item.archive_file_count for item in items),
        "error_count": sum(len(item.errors) for item in items),
    }


def _exercise_plan(
    plan: AcquisitionPlan,
    resolution: RemoteFileResolution | None,
    readiness: dict[str, Any] | None,
) -> DatasetExerciseRecord:
    local_signal_file_count = int(readiness.get("local_signal_file_count", 0)) if readiness else 0
    if local_signal_file_count:
        state = "local_raw_ready"
        required_action = "load cached raw signal files"
    elif plan.automation_status == "account_required":
        state = "blocked_account_required"
        required_action = "obtain provider credentials or approval before raw access"
    elif plan.automation_status == "unusable":
        state = "blocked_unusable"
        required_action = "replace or repair the inventory source; raw data is marked unavailable"
    elif not plan.candidates:
        state = "no_acquisition_candidate"
        required_action = "add a provider resolver or source URL"
    elif resolution is None:
        state = "remote_not_exercised"
        required_action = "run remote file resolution for this provider"
    else:
        state, required_action = _state_from_resolution(resolution)
    return DatasetExerciseRecord(
        record_id=plan.record_id,
        name=plan.name,
        provider=plan.provider,
        access_status=plan.access_status,
        automation_status=plan.automation_status,
        state=state,
        required_action=required_action,
        candidate_count=len(plan.candidates),
        remote_status=resolution.status if resolution else None,
        remote_file_count=len(resolution.files) if resolution else 0,
        directly_loadable_file_count=sum(file.directly_loadable for file in resolution.files) if resolution else 0,
        archive_file_count=sum(file.archive for file in resolution.files) if resolution else 0,
        local_signal_file_count=local_signal_file_count,
        errors=resolution.errors if resolution else (),
    )


def _state_from_resolution(resolution: RemoteFileResolution) -> tuple[str, str]:
    directly_loadable = sum(file.directly_loadable for file in resolution.files)
    archives = sum(file.archive for file in resolution.files)
    if resolution.status == "failed":
        return "remote_resolution_failed", "inspect provider response and update resolver"
    if directly_loadable:
        return "remote_direct_raw_ready", "download direct raw files and load with NeuroCore loaders"
    if archives:
        return "remote_archive_requires_extraction", "download archive, extract safely, scan for raw signal files"
    if any(file.materialization_action == "download_with_companion_metadata" for file in resolution.files):
        return "remote_raw_body_requires_metadata", "download raw body files and locate matching header metadata before loading"
    if resolution.files:
        return "remote_metadata_only", "inspect remote files and add format-specific raw detection"
    if resolution.status == "skipped":
        return "remote_skipped", "remove auth gate or install provider tooling before retrying"
    return "remote_no_files_found", "inspect landing page or provider API and extend resolver"
