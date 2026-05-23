from __future__ import annotations

import json
import re
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from .acquisition import AcquisitionCandidate, AcquisitionPlan
from .loaders import can_load_extension, find_supported_signal_files


JsonFetcher = Callable[[str], Any]
ByteFetcher = Callable[[str, int], bytes]

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".7z", ".rar")
CONFIDENT_SIGNAL_SUFFIXES = {
    ".bdf",
    ".cnt",
    ".data",
    ".edf",
    ".egi",
    ".fif",
    ".gdf",
    ".lay",
    ".mefd",
    ".mff",
    ".nxe",
    ".set",
    ".vhdr",
    ".xdf",
}
GENERIC_NUMERIC_SUFFIXES = {".csv", ".mat", ".npy", ".npz"}
RESOLVABLE_METHODS = {
    "dataverse_api",
    "dryad_api",
    "figshare_api",
    "mendeley_api",
    "osf_api",
    "zenodo_api",
}


@dataclass(frozen=True)
class RemoteFileCandidate:
    record_id: str
    dataset_name: str
    provider: str
    name: str
    url: str
    source_url: str
    size_bytes: int | None = None
    checksum: str = ""
    media_type: str = ""
    directly_loadable: bool = False
    archive: bool = False
    materialization_action: str = "review"

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "dataset_name": self.dataset_name,
            "provider": self.provider,
            "name": self.name,
            "url": self.url,
            "source_url": self.source_url,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "media_type": self.media_type,
            "directly_loadable": self.directly_loadable,
            "archive": self.archive,
            "materialization_action": self.materialization_action,
        }


@dataclass(frozen=True)
class RemoteFileResolution:
    record_id: str
    dataset_name: str
    provider: str
    status: str
    files: tuple[RemoteFileCandidate, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "dataset_name": self.dataset_name,
            "provider": self.provider,
            "status": self.status,
            "file_count": len(self.files),
            "directly_loadable_count": sum(file.directly_loadable for file in self.files),
            "archive_count": sum(file.archive for file in self.files),
            "files": [file.to_dict() for file in self.files],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class MaterializationResult:
    record_id: str
    name: str
    url: str
    status: str
    path: str | None = None
    bytes_written: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "name": self.name,
            "url": self.url,
            "status": self.status,
            "path": self.path,
            "bytes_written": self.bytes_written,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ArchiveExtractionResult:
    archive_path: str
    status: str
    output_dir: str | None = None
    extracted_member_count: int = 0
    signal_file_count: int = 0
    signal_files: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_path": self.archive_path,
            "status": self.status,
            "output_dir": self.output_dir,
            "extracted_member_count": self.extracted_member_count,
            "signal_file_count": self.signal_file_count,
            "signal_files": list(self.signal_files),
            "reason": self.reason,
        }


def resolve_remote_files(
    plan: AcquisitionPlan,
    *,
    fetch_json: JsonFetcher | None = None,
    timeout: float = 20.0,
    max_pages: int = 30,
) -> RemoteFileResolution:
    fetcher = fetch_json or (lambda url: _fetch_json(url, timeout=timeout))
    files: list[RemoteFileCandidate] = []
    errors: list[str] = []
    skipped = 0
    for candidate in plan.candidates:
        if candidate.requires_auth or candidate.method not in RESOLVABLE_METHODS:
            skipped += 1
            continue
        try:
            files.extend(_resolve_candidate(plan, candidate, fetcher=fetcher, max_pages=max_pages))
        except Exception as exc:  # pragma: no cover - defensive path for changing provider APIs
            errors.append(f"{candidate.method} {candidate.url}: {type(exc).__name__}: {exc}")
    unique = _dedupe_files(files)
    if unique:
        status = "resolved"
    elif errors:
        status = "failed"
    elif skipped:
        status = "skipped"
    else:
        status = "empty"
    return RemoteFileResolution(
        record_id=plan.record_id,
        dataset_name=plan.name,
        provider=plan.provider,
        status=status,
        files=tuple(unique),
        errors=tuple(errors),
    )


def resolve_inventory_remote_files(
    plans: Iterable[AcquisitionPlan],
    *,
    limit: int | None = None,
    providers: set[str] | None = None,
    automation_statuses: set[str] | None = None,
    fetch_json: JsonFetcher | None = None,
    timeout: float = 20.0,
    max_pages: int = 30,
) -> tuple[RemoteFileResolution, ...]:
    selected: list[RemoteFileResolution] = []
    for plan in plans:
        if providers and plan.provider not in providers:
            continue
        if automation_statuses and plan.automation_status not in automation_statuses:
            continue
        if not any(candidate.method in RESOLVABLE_METHODS and not candidate.requires_auth for candidate in plan.candidates):
            continue
        selected.append(resolve_remote_files(plan, fetch_json=fetch_json, timeout=timeout, max_pages=max_pages))
        if limit is not None and len(selected) >= limit:
            break
    return tuple(selected)


def summarize_remote_file_resolutions(resolutions: Iterable[RemoteFileResolution]) -> dict[str, Any]:
    items = tuple(resolutions)
    status_counts = Counter(item.status for item in items)
    provider_counts = Counter(item.provider for item in items)
    remote_files = [file for item in items for file in item.files]
    action_counts = Counter(file.materialization_action for file in remote_files)
    return {
        "records_resolved": len(items),
        "resolution_status_counts": dict(status_counts.most_common()),
        "provider_counts": dict(provider_counts.most_common(30)),
        "remote_file_count": len(remote_files),
        "directly_loadable_file_count": sum(file.directly_loadable for file in remote_files),
        "archive_file_count": sum(file.archive for file in remote_files),
        "materialization_action_counts": dict(action_counts.most_common()),
        "error_count": sum(len(item.errors) for item in items),
    }


def materialize_remote_file(
    file: RemoteFileCandidate,
    cache_root: str | Path,
    *,
    fetch_bytes: ByteFetcher | None = None,
    max_bytes: int = 100_000_000,
    overwrite: bool = False,
    direct_only: bool = True,
) -> MaterializationResult:
    if direct_only and not file.directly_loadable:
        return MaterializationResult(
            record_id=file.record_id,
            name=file.name,
            url=file.url,
            status="skipped",
            reason="remote file is not directly loadable by NeuroCore",
        )
    if file.size_bytes is not None and file.size_bytes > max_bytes:
        return MaterializationResult(
            record_id=file.record_id,
            name=file.name,
            url=file.url,
            status="skipped",
            reason=f"remote file is larger than max_bytes ({file.size_bytes} > {max_bytes})",
        )
    target = Path(cache_root).expanduser() / _safe_path_part(file.record_id) / _safe_filename(file.name, file.url)
    if target.exists() and not overwrite:
        return MaterializationResult(
            record_id=file.record_id,
            name=file.name,
            url=file.url,
            status="cached",
            path=str(target),
            bytes_written=target.stat().st_size,
        )
    fetcher = fetch_bytes or _fetch_bytes
    payload = fetcher(file.url, max_bytes)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return MaterializationResult(
        record_id=file.record_id,
        name=file.name,
        url=file.url,
        status="downloaded",
        path=str(target),
        bytes_written=len(payload),
    )


def materialize_remote_files(
    files: Iterable[RemoteFileCandidate],
    cache_root: str | Path,
    *,
    max_files: int | None = None,
    max_bytes: int = 100_000_000,
    overwrite: bool = False,
    direct_only: bool = True,
    fetch_bytes: ByteFetcher | None = None,
) -> tuple[MaterializationResult, ...]:
    results: list[MaterializationResult] = []
    for file in files:
        if max_files is not None and len(results) >= max_files:
            break
        try:
            results.append(
                materialize_remote_file(
                    file,
                    cache_root,
                    fetch_bytes=fetch_bytes,
                    max_bytes=max_bytes,
                    overwrite=overwrite,
                    direct_only=direct_only,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive path for network and filesystem errors
            results.append(
                MaterializationResult(
                    record_id=file.record_id,
                    name=file.name,
                    url=file.url,
                    status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return tuple(results)


def extract_supported_signal_files_from_archive(
    archive_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_members: int = 20_000,
    max_member_bytes: int = 2_000_000_000,
) -> ArchiveExtractionResult:
    archive = Path(archive_path).expanduser()
    if not archive.is_file():
        return ArchiveExtractionResult(archive_path=str(archive), status="failed", reason="archive does not exist")
    if not _is_archive_name(archive.name):
        return ArchiveExtractionResult(archive_path=str(archive), status="skipped", reason="not a supported archive name")
    target = Path(output_dir).expanduser() if output_dir else archive.parent / f"{_archive_stem(archive)}_extracted"
    try:
        if zipfile.is_zipfile(archive):
            extracted = _extract_zip(archive, target, max_members=max_members, max_member_bytes=max_member_bytes)
        elif tarfile.is_tarfile(archive):
            extracted = _extract_tar(archive, target, max_members=max_members, max_member_bytes=max_member_bytes)
        else:
            return ArchiveExtractionResult(
                archive_path=str(archive), status="skipped", reason="archive compression is not zip or tar"
            )
    except Exception as exc:
        return ArchiveExtractionResult(archive_path=str(archive), status="failed", reason=f"{type(exc).__name__}: {exc}")
    signal_files = find_supported_signal_files(target)
    return ArchiveExtractionResult(
        archive_path=str(archive),
        status="extracted",
        output_dir=str(target),
        extracted_member_count=extracted,
        signal_file_count=len(signal_files),
        signal_files=tuple(str(path) for path in signal_files[:100]),
    )


def _resolve_candidate(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    *,
    fetcher: JsonFetcher,
    max_pages: int,
) -> list[RemoteFileCandidate]:
    if candidate.method == "zenodo_api":
        return _resolve_zenodo(plan, candidate, fetcher)
    if candidate.method == "figshare_api":
        return _resolve_figshare(plan, candidate, fetcher)
    if candidate.method == "osf_api":
        return _resolve_osf(plan, candidate, fetcher, max_pages=max_pages)
    if candidate.method == "dataverse_api":
        return _resolve_dataverse(plan, candidate, fetcher)
    if candidate.method == "dryad_api":
        return _resolve_generic_following_file_links(plan, candidate, fetcher, max_pages=max_pages)
    if candidate.method == "mendeley_api":
        return _resolve_generic_following_file_links(plan, candidate, fetcher, max_pages=max_pages)
    return []


def _resolve_zenodo(plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher) -> list[RemoteFileCandidate]:
    payload = fetcher(candidate.url)
    files = []
    for item in _list(payload.get("files")):
        name = _string(item.get("key") or item.get("filename") or item.get("name"))
        links = _dict(item.get("links"))
        url = _string(links.get("download") or links.get("content") or links.get("self") or item.get("download_url"))
        files.append(
            _remote_file(
                plan,
                candidate,
                name=name,
                url=url,
                size_bytes=_int_or_none(item.get("size") or item.get("filesize")),
                checksum=_string(item.get("checksum")),
                media_type=_string(item.get("type")),
            )
        )
    return [file for file in files if file.url and file.name]


def _resolve_figshare(
    plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher
) -> list[RemoteFileCandidate]:
    payload = fetcher(candidate.url)
    files = []
    for item in _list(payload.get("files")):
        files.append(
            _remote_file(
                plan,
                candidate,
                name=_string(item.get("name")),
                url=_string(item.get("download_url")),
                size_bytes=_int_or_none(item.get("size")),
                checksum=_string(item.get("computed_md5") or item.get("supplied_md5")),
            )
        )
    return [file for file in files if file.url and file.name]


def _resolve_osf(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
    *,
    max_pages: int,
) -> list[RemoteFileCandidate]:
    files: list[RemoteFileCandidate] = []
    queue = [candidate.url]
    seen: set[str] = set()
    pages = 0
    while queue and pages < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        payload = fetcher(url)
        pages += 1
        for item in _json_api_data(payload):
            attributes = _dict(item.get("attributes"))
            links = _dict(item.get("links"))
            name = _string(attributes.get("name") or attributes.get("materialized_path")).strip("/")
            download_url = _string(links.get("download") or attributes.get("download_url"))
            if attributes.get("kind") == "file" and download_url:
                files.append(
                    _remote_file(
                        plan,
                        candidate,
                        name=Path(name).name,
                        url=download_url,
                        size_bytes=_int_or_none(attributes.get("size")),
                        media_type=_string(attributes.get("contentType")),
                    )
                )
            related = _relationship_href(item, "files")
            if related:
                queue.append(related)
        next_url = _string(_dict(payload.get("links")).get("next"))
        if next_url:
            queue.append(next_url)
    return files


def _resolve_dataverse(
    plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher
) -> list[RemoteFileCandidate]:
    payload = fetcher(candidate.url)
    dataset = _dict(payload.get("data"))
    version = _dict(dataset.get("latestVersion") or dataset.get("version"))
    host = f"{urlparse(candidate.url).scheme}://{urlparse(candidate.url).netloc}"
    files = []
    for item in _list(version.get("files")):
        data_file = _dict(item.get("dataFile"))
        file_id = data_file.get("id")
        if not file_id:
            continue
        files.append(
            _remote_file(
                plan,
                candidate,
                name=_string(data_file.get("filename") or item.get("label")),
                url=f"{host}/api/access/datafile/{file_id}",
                size_bytes=_int_or_none(data_file.get("filesize")),
                checksum=_string(data_file.get("md5")),
                media_type=_string(data_file.get("contentType")),
            )
        )
    return [file for file in files if file.name]


def _resolve_generic_following_file_links(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
    *,
    max_pages: int,
) -> list[RemoteFileCandidate]:
    queue = [candidate.url]
    seen: set[str] = set()
    files: list[RemoteFileCandidate] = []
    pages = 0
    while queue and pages < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        payload = fetcher(url)
        pages += 1
        files.extend(_generic_remote_files(plan, candidate, payload, source_url=url))
        for href in _file_link_hrefs(payload, base_url=url):
            if href not in seen:
                queue.append(href)
        next_url = _string(_dict(payload.get("links")).get("next"))
        if next_url:
            queue.append(urljoin(url, next_url))
    return _dedupe_files(files)


def _generic_remote_files(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    value: Any,
    *,
    source_url: str,
) -> list[RemoteFileCandidate]:
    files = []
    for item in _file_like_objects(value):
        name = _string(
            item.get("name")
            or item.get("filename")
            or item.get("path")
            or item.get("displayName")
            or item.get("label")
        )
        url = _string(
            item.get("downloadUrl")
            or item.get("downloadURL")
            or item.get("download_url")
            or item.get("url")
            or item.get("href")
        )
        if not name or not url:
            continue
        files.append(
            _remote_file(
                plan,
                candidate,
                name=Path(name).name,
                url=urljoin(source_url, url),
                size_bytes=_int_or_none(item.get("size") or item.get("sizeBytes") or item.get("filesize")),
                checksum=_string(item.get("md5") or item.get("checksum")),
                media_type=_string(item.get("mimeType") or item.get("contentType")),
                source_url=source_url,
            )
        )
    return files


def _remote_file(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    *,
    name: str,
    url: str,
    size_bytes: int | None = None,
    checksum: str = "",
    media_type: str = "",
    source_url: str | None = None,
) -> RemoteFileCandidate:
    archive = _is_archive_name(name)
    directly_loadable = _is_directly_loadable_signal_name(name)
    if directly_loadable:
        action = "download_then_load"
    elif archive:
        action = "download_extract_then_scan"
    else:
        action = "metadata_or_manual_review"
    return RemoteFileCandidate(
        record_id=plan.record_id,
        dataset_name=plan.name,
        provider=candidate.provider,
        name=name,
        url=url,
        source_url=source_url or candidate.url,
        size_bytes=size_bytes,
        checksum=checksum,
        media_type=media_type,
        directly_loadable=directly_loadable,
        archive=archive,
        materialization_action=action,
    )


def _fetch_json(url: str, *, timeout: float) -> Any:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "NeuroCore/0.1 dataset-resolver"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_bytes(url: str, max_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "NeuroCore/0.1 dataset-materializer"})
    with urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
        if length and int(length) > max_bytes:
            raise ValueError(f"remote file is larger than max_bytes ({length} > {max_bytes})")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"remote file exceeded max_bytes ({total} > {max_bytes})")
            chunks.append(chunk)
        return b"".join(chunks)


def _extract_zip(archive: Path, target: Path, *, max_members: int, max_member_bytes: int) -> int:
    extracted = 0
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        infos = [info for info in handle.infolist() if not info.is_dir()]
        if len(infos) > max_members:
            raise ValueError(f"archive has too many members ({len(infos)} > {max_members})")
        for info in infos:
            if info.file_size > max_member_bytes:
                raise ValueError(f"archive member is too large ({info.filename})")
            destination = _safe_extract_path(target, info.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source:
                destination.write_bytes(source.read())
            extracted += 1
    return extracted


def _extract_tar(archive: Path, target: Path, *, max_members: int, max_member_bytes: int) -> int:
    extracted = 0
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        members = [member for member in handle.getmembers() if member.isfile()]
        if len(members) > max_members:
            raise ValueError(f"archive has too many members ({len(members)} > {max_members})")
        for member in members:
            if member.size > max_member_bytes:
                raise ValueError(f"archive member is too large ({member.name})")
            source = handle.extractfile(member)
            if source is None:
                continue
            destination = _safe_extract_path(target, member.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read())
            extracted += 1
    return extracted


def _safe_extract_path(target: Path, member_name: str) -> Path:
    destination = (target / member_name).resolve()
    root = target.resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"archive member escapes output directory: {member_name}")
    return destination


def _file_like_objects(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        has_name = any(key in value for key in ("name", "filename", "path", "displayName", "label"))
        has_url = any(key in value for key in ("downloadUrl", "downloadURL", "download_url", "url", "href"))
        if has_name and has_url:
            found.append(value)
        for item in value.values():
            found.extend(_file_like_objects(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_file_like_objects(item))
    return found


def _file_link_hrefs(value: Any, *, base_url: str) -> list[str]:
    hrefs: list[str] = []
    links = _dict(value.get("_links")) if isinstance(value, dict) else {}
    for key, link in links.items():
        if "file" not in str(key).lower():
            continue
        if isinstance(link, dict) and link.get("href"):
            hrefs.append(urljoin(base_url, str(link["href"])))
        elif isinstance(link, list):
            for item in link:
                if isinstance(item, dict) and item.get("href"):
                    hrefs.append(urljoin(base_url, str(item["href"])))
    return hrefs


def _json_api_data(payload: Any) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _relationship_href(item: dict[str, Any], name: str) -> str:
    relationships = _dict(item.get("relationships"))
    relation = _dict(relationships.get(name))
    links = _dict(relation.get("links"))
    related = links.get("related")
    if isinstance(related, dict):
        return _string(related.get("href"))
    return _string(related)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_archive_name(name: str) -> bool:
    normalized = name.lower()
    return any(normalized.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _is_directly_loadable_signal_name(name: str) -> bool:
    if not can_load_extension(name):
        return False
    suffix = Path(name.lower()).suffix
    if name.lower().endswith(".fif.gz"):
        suffix = ".fif"
    if suffix in CONFIDENT_SIGNAL_SUFFIXES:
        return True
    if suffix in GENERIC_NUMERIC_SUFFIXES:
        return bool(re.search(r"(^|[_./ -])(eeg|raw|signal|signals|recording|subject|sub-[a-z0-9]+|ses-[a-z0-9]+|task-[a-z0-9]+)", name, re.IGNORECASE))
    return False


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "dataset"


def _safe_filename(name: str, url: str) -> str:
    candidate = Path(urlparse(url).path).name if not name else name
    cleaned = re.sub(r"[^A-Za-z0-9_.() -]+", "_", candidate).strip(" ._")
    return cleaned or "remote-file"


def _archive_stem(path: Path) -> str:
    name = path.name
    for suffix in (".tar.gz", ".tgz", ".zip", ".tar", ".gz", ".7z", ".rar"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _dedupe_files(files: Iterable[RemoteFileCandidate]) -> list[RemoteFileCandidate]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[RemoteFileCandidate] = []
    for file in files:
        key = (file.record_id, file.name, file.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(file)
    return unique
