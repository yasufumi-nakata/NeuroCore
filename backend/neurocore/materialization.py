from __future__ import annotations

import json
import re
import tarfile
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from .acquisition import AcquisitionCandidate, AcquisitionPlan
from .loaders import can_load_extension, find_supported_signal_files


JsonFetcher = Callable[[str], Any]
ByteFetcher = Callable[[str, int], bytes]

ARCHIVE_SUFFIXES = (
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.xz",
    ".txz",
    ".tar.bz2",
    ".tbz2",
    ".gz",
    ".7z",
    ".rar",
)
SPLIT_ARCHIVE_PATTERN = re.compile(
    r"\.(?:7z|zip|rar|tar|tar\.gz|tgz|tar\.xz|txz|tar\.bz2|tbz2)\.\d{3}$",
    re.IGNORECASE,
)
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
    ".nwb",
    ".nxe",
    ".set",
    ".vhdr",
    ".xdf",
}
GENERIC_NUMERIC_SUFFIXES = {".csv", ".mat", ".npy", ".npz", ".pt", ".pth", ".tab", ".tsv"}
TEXT_SIGNAL_SUFFIXES = {".txt", ".tab", ".tsv"}
RESOLVABLE_METHODS = {
    "bnci_index",
    "dataverse_api",
    "data_ru_landing",
    "dandi_api",
    "dandi_client",
    "doi_resolver",
    "dspace_api",
    "dryad_api",
    "figshare_api",
    "figshare_collection_api",
    "gin_index",
    "gin_client",
    "github_tree_api",
    "git_clone",
    "huggingface_api",
    "huggingface_client",
    "http_landing",
    "kaggle_api",
    "kaggle_client",
    "mendeley_api",
    "nemar_index",
    "nemar_client",
    "nitrc_frs",
    "openneuro_api",
    "openneuro_cli",
    "osf_api",
    "physionet_index",
    "physionet_client",
    "scidb_api",
    "stanford_purl_json",
    "zenodo_api",
}
FIGSHARE_COMPATIBLE_HOSTS = {
    "data.4tu.nl",
    "data.dtu.dk",
    "bridges.monash.edu",
    "drum.um.edu.mt",
}
DATAVERSE_COMPATIBLE_HOSTS = {
    "borealisdata.ca",
    "dataverse.harvard.edu",
    "entrepot.recherche.data.gouv.fr",
    "rdr.kuleuven.be",
    "repo.researchdata.hu",
    "researchdata.ntu.edu.sg",
    "researchdata.lib.cityu.edu.hk",
    "redu.unicamp.br",
}
REPOSITORY_HTML_HOSTS = {
    "datashare.ed.ac.uk",
    "deepblue.lib.umich.edu",
}
DATA_RU_HOSTS = {
    "data.ru.nl",
    "webdav.data.ru.nl",
}
STANFORD_SDR_HOSTS = {
    "purl.stanford.edu",
    "stacks.stanford.edu",
}
HTML_HTTP_FALLBACK_HOSTS = {
    "archive.ics.uci.edu",
    "www.archive.ics.uci.edu",
}
BNCI_DATASETS_URL = "https://bnci-horizon-2020.eu/database/data-sets"


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

    def to_dict(self, *, file_limit: int | None = None) -> dict[str, Any]:
        files = self.files if file_limit is None else self.files[:file_limit]
        return {
            "record_id": self.record_id,
            "dataset_name": self.dataset_name,
            "provider": self.provider,
            "status": self.status,
            "file_count": len(self.files),
            "directly_loadable_count": sum(file.directly_loadable for file in self.files),
            "archive_count": sum(file.archive for file in self.files),
            "files": [file.to_dict() for file in files],
            "files_truncated": file_limit is not None and len(self.files) > file_limit,
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
    offset: int = 0,
    providers: set[str] | None = None,
    automation_statuses: set[str] | None = None,
    fetch_json: JsonFetcher | None = None,
    timeout: float = 20.0,
    max_pages: int = 30,
) -> tuple[RemoteFileResolution, ...]:
    selected: list[RemoteFileResolution] = []
    eligible_seen = 0
    for plan in plans:
        if providers and plan.provider not in providers:
            continue
        if automation_statuses and plan.automation_status not in automation_statuses:
            continue
        if not any(candidate.method in RESOLVABLE_METHODS and not candidate.requires_auth for candidate in plan.candidates):
            continue
        if eligible_seen < offset:
            eligible_seen += 1
            continue
        eligible_seen += 1
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
        "loader_materialization_file_count": sum(len(_loader_materialization_keys(item.files)) for item in items),
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
    target = Path(cache_root).expanduser() / _safe_path_part(file.record_id) / _safe_relative_path(file.name, file.url)
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
    file_tuple = tuple(files)
    selected_keys = _actionable_materialization_keys(file_tuple, include_archives=not direct_only)
    selected_attempts = 0
    for file in file_tuple:
        if _file_key(file) not in selected_keys:
            results.append(
                MaterializationResult(
                    record_id=file.record_id,
                    name=file.name,
                    url=file.url,
                    status="skipped",
                    reason="remote file is not needed for loader materialization",
                )
            )
            continue
        if max_files is not None and selected_attempts >= max_files:
            break
        selected_attempts += 1
        try:
            results.append(
                materialize_remote_file(
                    file,
                    cache_root,
                    fetch_bytes=fetch_bytes,
                    max_bytes=max_bytes,
                    overwrite=overwrite,
                    direct_only=False,
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
    if candidate.method == "figshare_collection_api":
        return _resolve_figshare_collection(plan, candidate, fetcher)
    if candidate.method == "osf_api":
        return _resolve_osf(plan, candidate, fetcher, max_pages=max_pages)
    if candidate.method == "dataverse_api":
        return _resolve_dataverse(plan, candidate, fetcher)
    if candidate.method == "data_ru_landing":
        return _resolve_data_ru(plan, candidate, fetcher)
    if candidate.method == "dspace_api":
        return _resolve_dspace(plan, candidate, fetcher)
    if candidate.method == "dryad_api":
        return _resolve_dryad(plan, candidate, fetcher)
    if candidate.method == "mendeley_api":
        return _resolve_mendeley(plan, candidate, fetcher)
    if candidate.method == "doi_resolver":
        return _resolve_doi(plan, candidate, fetcher, max_pages=max_pages)
    if candidate.method in {"openneuro_api", "openneuro_cli"}:
        return _resolve_openneuro(plan, candidate, fetcher)
    if candidate.method in {"github_tree_api", "git_clone"}:
        return _resolve_github(plan, candidate, fetcher)
    if candidate.method in {"huggingface_api", "huggingface_client"}:
        return _resolve_huggingface(plan, candidate, fetcher, max_pages=max_pages)
    if candidate.method in {"physionet_index", "physionet_client"}:
        return _resolve_physionet(plan, candidate, fetcher, max_pages=max_pages)
    if candidate.method in {"dandi_api", "dandi_client"}:
        return _resolve_dandi(plan, candidate, fetcher, max_pages=max_pages)
    if candidate.method in {"gin_index", "gin_client"}:
        return _resolve_gin(plan, candidate, fetcher)
    if candidate.method in {"kaggle_api", "kaggle_client"}:
        return _resolve_kaggle(plan, candidate, fetcher, max_pages=max_pages)
    if candidate.method in {"nemar_index", "nemar_client"}:
        return _resolve_nemar(plan, candidate, fetcher)
    if candidate.method == "nitrc_frs":
        return _resolve_nitrc_frs(plan, candidate, fetcher)
    if candidate.method == "scidb_api":
        return _resolve_scidb(plan, candidate, fetcher)
    if candidate.method == "stanford_purl_json":
        return _resolve_stanford_purl(plan, candidate, fetcher)
    if candidate.method == "bnci_index":
        return _resolve_bnci(plan, candidate, fetcher)
    if candidate.method == "http_landing":
        return _resolve_http_landing(plan, candidate, fetcher, max_pages=max_pages)
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


def _resolve_figshare_collection(
    plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher
) -> list[RemoteFileCandidate]:
    payload = fetcher(candidate.url)
    files = []
    for item in _list(payload):
        article_url = _string(item.get("url_public_api") or item.get("url") or item.get("url_api"))
        article_id = _int_or_none(item.get("id"))
        if not article_url and article_id:
            article_url = f"https://api.figshare.com/v2/articles/{article_id}"
        if not article_url:
            continue
        article = AcquisitionCandidate("figshare", "figshare_api", article_url, "file_listing")
        files.extend(_resolve_figshare(replace(plan, provider="figshare"), article, fetcher))
    return _dedupe_files(files)


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
        related_links: list[tuple[str, str]] = []
        for item in _json_api_data(payload):
            attributes = _dict(item.get("attributes"))
            links = _dict(item.get("links"))
            name = _string(attributes.get("materialized_path") or attributes.get("name") or attributes.get("title")).strip("/")
            download_url = _string(links.get("download") or attributes.get("download_url"))
            if attributes.get("kind") == "file" and download_url:
                files.append(
                    _remote_file(
                        plan,
                        candidate,
                        name=name,
                        url=download_url,
                        size_bytes=_int_or_none(attributes.get("size")),
                        media_type=_string(attributes.get("contentType")),
                    )
                )
            related = _relationship_href(item, "files")
            if related:
                related_links.append((name, related))
            if item.get("type") == "nodes":
                node_id = _string(item.get("id"))
                if node_id:
                    related_links.append((name, f"https://api.osf.io/v2/nodes/{node_id}/files/"))
                    related_links.append((name, f"https://api.osf.io/v2/nodes/{node_id}/children/?page[size]=100"))
            children = _relationship_href(item, "children")
            if children:
                related_links.append((name, children))
        for _, related in sorted(related_links, key=lambda entry: _osf_queue_priority(*entry)):
            if related not in seen:
                queue.append(related)
        next_url = _string(_dict(payload.get("links")).get("next"))
        if next_url:
            queue.append(next_url)
    return _dedupe_files(files)


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


def _resolve_mendeley(
    plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher
) -> list[RemoteFileCandidate]:
    payload = _dict(fetcher(candidate.url))
    files = []
    for item in _list(payload.get("files")):
        content = _dict(item.get("content_details"))
        files.append(
            _remote_file(
                plan,
                candidate,
                name=_string(item.get("filename")),
                url=_string(content.get("download_url")),
                size_bytes=_int_or_none(item.get("size") or content.get("size")),
                checksum=f"sha256:{content['sha256_hash']}" if content.get("sha256_hash") else "",
                media_type=_string(content.get("content_type")),
            )
        )
    return [file for file in files if file.name and file.url]


def _resolve_openneuro(
    plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher
) -> list[RemoteFileCandidate]:
    dataset_id = _first_match(r"(ds\d{6,})", candidate.url)
    if not dataset_id:
        return []
    tag = _first_match(r"/versions/([A-Za-z0-9_.-]+)", candidate.url)
    if not tag:
        metadata = fetcher(_openneuro_graphql_url(_openneuro_dataset_query(dataset_id)))
        snapshots = _list(_dict(_dict(metadata.get("data")).get("dataset")).get("snapshots"))
        tagged = [item for item in snapshots if item.get("tag")]
        if tagged:
            tagged.sort(key=lambda item: str(item.get("created") or item.get("tag")))
            tag = str(tagged[-1]["tag"])
    if not tag:
        tag = "1.0.0"
    payload = fetcher(_openneuro_graphql_url(_openneuro_files_query(dataset_id, tag)))
    snapshot = _dict(_dict(payload.get("data")).get("snapshot"))
    files = []
    for item in _list(snapshot.get("files")):
        if item.get("directory"):
            continue
        urls = _list(item.get("urls"))
        name = _string(item.get("filename"))
        files.append(
            _remote_file(
                plan,
                candidate,
                name=name,
                url=_openneuro_download_url(urls, name),
                size_bytes=_int_or_none(item.get("size")),
                checksum=_string(item.get("id")),
                media_type="annexed" if item.get("annexed") else "git-object",
                source_url=f"https://openneuro.org/datasets/{dataset_id}/versions/{tag}",
            )
        )
    return [file for file in files if file.name and file.url]


def _resolve_doi(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
    *,
    max_pages: int,
) -> list[RemoteFileCandidate]:
    payload = _dict(fetcher(_doi_resolution_url(candidate.url)))
    target = _string(payload.get("url") or candidate.url)
    if not target or target == candidate.url:
        return []
    return _resolve_delegated_url(plan, target, fetcher, max_pages=max_pages)


def _resolve_delegated_url(
    plan: AcquisitionPlan,
    target: str,
    fetcher: JsonFetcher,
    *,
    max_pages: int,
) -> list[RemoteFileCandidate]:
    host = urlparse(target).netloc.lower()
    if "zenodo.org" in host:
        record_id = _first_match(r"zenodo\.(\d+)|records/(\d+)", target)
        if record_id:
            delegated_plan = replace(plan, provider="zenodo")
            delegated = AcquisitionCandidate(
                "zenodo", "zenodo_api", f"https://zenodo.org/api/records/{record_id}", "file_listing"
            )
            return _resolve_zenodo(delegated_plan, delegated, fetcher)
    if _is_figshare_host(host):
        article_id = _figshare_article_id(target)
        if article_id:
            delegated_plan = replace(plan, provider="figshare")
            delegated = AcquisitionCandidate(
                "figshare", "figshare_api", f"https://api.figshare.com/v2/articles/{article_id}", "file_listing"
            )
            return _resolve_figshare(delegated_plan, delegated, fetcher)
    if _is_dataverse_host(host):
        api_url = _dataverse_api_url_from_landing(target)
        if api_url:
            delegated_plan = replace(plan, provider="dataverse")
            delegated = AcquisitionCandidate("dataverse", "dataverse_api", api_url, "file_listing")
            return _resolve_dataverse(delegated_plan, delegated, fetcher)
    if _is_data_ru_host(host):
        delegated_plan = replace(plan, provider="data_ru")
        delegated = AcquisitionCandidate("data_ru", "data_ru_landing", target, "file_listing")
        return _resolve_data_ru(delegated_plan, delegated, fetcher)
    if _is_stanford_sdr_host(host):
        druid = _stanford_druid(target)
        if druid:
            delegated_plan = replace(plan, provider="stanford_sdr")
            delegated = AcquisitionCandidate(
                "stanford_sdr",
                "stanford_purl_json",
                f"https://purl.stanford.edu/{druid}.json",
                "file_listing",
            )
            return _resolve_stanford_purl(delegated_plan, delegated, fetcher)
    if "osf.io" in host:
        node_id = _first_match(r"osf\.io/([a-z0-9]{4,8})", target)
        if node_id:
            delegated_plan = replace(plan, provider="osf")
            delegated = AcquisitionCandidate("osf", "osf_api", f"https://api.osf.io/v2/nodes/{node_id}/files/", "file_listing")
            return _resolve_osf(delegated_plan, delegated, fetcher, max_pages=max_pages)
    if "openneuro.org" in host:
        delegated_plan = replace(plan, provider="openneuro")
        delegated = AcquisitionCandidate("openneuro", "openneuro_api", target, "file_listing")
        return _resolve_openneuro(delegated_plan, delegated, fetcher)
    if "github.com" in host:
        delegated_plan = replace(plan, provider="github")
        delegated = AcquisitionCandidate("github", "github_tree_api", target, "file_listing")
        return _resolve_github(delegated_plan, delegated, fetcher)
    if "physionet.org" in host:
        delegated_plan = replace(plan, provider="physionet")
        delegated = AcquisitionCandidate("physionet", "physionet_index", target, "file_listing")
        return _resolve_physionet(delegated_plan, delegated, fetcher, max_pages=max_pages)
    if "dandiarchive.org" in host:
        delegated_plan = replace(plan, provider="dandi")
        delegated = AcquisitionCandidate("dandi", "dandi_api", target, "file_listing")
        return _resolve_dandi(delegated_plan, delegated, fetcher, max_pages=max_pages)
    if "huggingface.co" in host:
        delegated_plan = replace(plan, provider="huggingface")
        delegated = AcquisitionCandidate("huggingface", "huggingface_api", target, "file_listing")
        return _resolve_huggingface(delegated_plan, delegated, fetcher, max_pages=max_pages)
    if "kaggle.com" in host:
        delegated_plan = replace(plan, provider="kaggle")
        delegated = AcquisitionCandidate("kaggle", "kaggle_api", target, "file_listing")
        return _resolve_kaggle(delegated_plan, delegated, fetcher, max_pages=max_pages)
    if "nemar.org" in host:
        delegated_plan = replace(plan, provider="nemar")
        delegated = AcquisitionCandidate("nemar", "nemar_index", target, "file_listing")
        return _resolve_nemar(delegated_plan, delegated, fetcher)
    if "scidb.cn" in host:
        delegated_plan = replace(plan, provider="scidb")
        delegated = AcquisitionCandidate("scidb", "scidb_api", _scidb_candidate_url(target=target), "file_listing")
        return _resolve_scidb(delegated_plan, delegated, fetcher)
    if "bnci-horizon-2020.eu" in host:
        delegated_plan = replace(plan, provider="bnci")
        delegated = AcquisitionCandidate("bnci", "bnci_index", target, "file_listing")
        return _resolve_bnci(delegated_plan, delegated, fetcher)
    if "research-collection.ethz.ch" in host:
        delegated_plan = replace(plan, provider="dspace")
        delegated = AcquisitionCandidate("dspace", "dspace_api", target, "file_listing")
        return _resolve_dspace(delegated_plan, delegated, fetcher)
    if _is_repository_html_host(host):
        delegated_plan = replace(plan, provider="repository_html")
        delegated = AcquisitionCandidate("repository_html", "http_landing", target, "file_listing")
        return _resolve_http_landing(delegated_plan, delegated, fetcher, max_pages=max_pages)
    delegated = AcquisitionCandidate(plan.provider, "http_landing", target, "landing")
    return _resolve_http_landing(plan, delegated, fetcher, max_pages=max_pages)


def _resolve_github(
    plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher
) -> list[RemoteFileCandidate]:
    repo = _github_repo(candidate.url)
    if not repo:
        return []
    payload = fetcher(f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1")
    files = []
    for item in _list(payload.get("tree")):
        if item.get("type") != "blob":
            continue
        path = _string(item.get("path"))
        files.append(
            _remote_file(
                plan,
                candidate,
                name=path,
                url=f"https://raw.githubusercontent.com/{repo}/HEAD/{quote(path, safe='/')}",
                size_bytes=_int_or_none(item.get("size")),
                checksum=_string(item.get("sha")),
                media_type="git-blob",
                source_url=f"https://github.com/{repo}",
            )
        )
    return [file for file in files if file.name]


def _resolve_huggingface(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
    *,
    max_pages: int = 30,
) -> list[RemoteFileCandidate]:
    repo = _first_match(r"huggingface\.co/datasets/([^/\s?#]+/[^/\s?#]+)", candidate.url)
    if not repo:
        return []
    queue = [f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=1"]
    seen: set[str] = set()
    files = []
    pages = 0
    while queue and pages < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        payload = fetcher(url)
        pages += 1
        for item in _list(payload):
            path = _string(item.get("path"))
            if not path:
                continue
            if item.get("type") == "directory":
                queue.append(f"https://huggingface.co/api/datasets/{repo}/tree/main/{quote(path, safe='/')}")
                continue
            if item.get("type") not in {"file", "blob"}:
                continue
            files.append(
                _remote_file(
                    plan,
                    candidate,
                    name=path,
                    url=f"https://huggingface.co/datasets/{repo}/resolve/main/{quote(path, safe='/')}",
                    size_bytes=_int_or_none(item.get("size")),
                    checksum=_string(item.get("oid")),
                    media_type="huggingface-file",
                    source_url=f"https://huggingface.co/datasets/{repo}",
                )
            )
    return [file for file in files if file.name]


def _resolve_physionet(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
    *,
    max_pages: int,
) -> list[RemoteFileCandidate]:
    parsed = urlparse(candidate.url)
    match = re.search(r"/content/([^/]+)/([^/]+)/?", parsed.path)
    if not match:
        project = _first_match(r"/content/([^/]+)/?", parsed.path)
        if not project:
            return []
        html = _string(fetcher(_html_landing_url(candidate.url)))
        version = _physionet_version_from_landing(html, project)
        if not version:
            return []
    else:
        project, version = match.groups()
    root = f"https://physionet.org/files/{project}/{version}/"
    manifest_files = _resolve_physionet_sha256_manifest(plan, candidate, fetcher, root)
    if manifest_files:
        return manifest_files
    return _resolve_html_index(plan, candidate, fetcher, root, max_pages=max_pages)


def _resolve_dandi(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
    *,
    max_pages: int,
) -> list[RemoteFileCandidate]:
    dandiset_id = _first_match(r"dandiset/(\d{6})|DANDI:(\d{6})", candidate.url)
    if not dandiset_id:
        return []
    version = _first_match(r"dandiset/\d{6}/([^/\s?#]+)", candidate.url) or "draft"
    api_root = "https://api.dandiarchive.org/api"
    next_url = f"{api_root}/dandisets/{dandiset_id}/versions/{version}/assets/?page_size=1000"
    files: list[RemoteFileCandidate] = []
    pages = 0
    while next_url and pages < max_pages:
        payload = fetcher(next_url)
        pages += 1
        for item in _list(payload.get("results")):
            path = _string(item.get("path"))
            asset_id = _string(item.get("asset_id"))
            if not path or not asset_id:
                continue
            files.append(
                _remote_file(
                    plan,
                    candidate,
                    name=path,
                    url=f"{api_root}/dandisets/{dandiset_id}/versions/{version}/assets/{asset_id}/download/",
                    size_bytes=_int_or_none(item.get("size")),
                    checksum=_string(item.get("blob") or item.get("zarr") or asset_id),
                    media_type="dandi-asset",
                    source_url=f"https://dandiarchive.org/dandiset/{dandiset_id}/{version}",
                )
            )
        next_url = _string(payload.get("next"))
    return files


def _resolve_gin(plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher) -> list[RemoteFileCandidate]:
    parsed = urlparse(candidate.url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return []
    repo_path = "/".join(parts[:2])
    base = f"{parsed.scheme}://{parsed.netloc}/{repo_path}"
    files = [
        _remote_file(plan, candidate, name="master.zip", url=f"{base}/archive/master.zip", source_url=base),
        _remote_file(plan, candidate, name="master.tar.gz", url=f"{base}/archive/master.tar.gz", source_url=base),
        _remote_file(plan, candidate, name="master.gin.zip", url=f"{base}/archive/master.gin.zip", source_url=base),
    ]
    try:
        payload = fetcher(candidate.url)
    except Exception:
        return files
    for href in _html_links(_string(payload)):
        if "/raw/" not in href:
            continue
        name = href.rsplit("/", 1)[-1]
        files.append(_remote_file(plan, candidate, name=name, url=urljoin(base, href), source_url=base))
    return _dedupe_files(files)


def _resolve_kaggle(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
    *,
    max_pages: int,
) -> list[RemoteFileCandidate]:
    ref = _first_match(r"kaggle\.com/(?:datasets/)?([^/\s?#]+/[^/\s?#]+)", candidate.url)
    if not ref:
        return []
    url = f"https://www.kaggle.com/api/v1/datasets/list/{ref}"
    files: list[RemoteFileCandidate] = []
    pages = 0
    while url and pages < max_pages:
        payload = fetcher(url)
        pages += 1
        for item in _list(_dict(payload).get("datasetFiles")):
            name = _string(item.get("name") or item.get("nameNullable"))
            if not name:
                continue
            archive_name = name if _is_archive_name(name) else f"{name}.zip"
            files.append(
                _remote_file(
                    plan,
                    candidate,
                    name=archive_name,
                    url=f"https://www.kaggle.com/api/v1/datasets/download/{ref}?{urlencode({'file_name': name})}",
                    size_bytes=_int_or_none(item.get("totalBytes")),
                    media_type="kaggle-file",
                    source_url=f"https://www.kaggle.com/datasets/{ref}",
                )
            )
        token = _string(_dict(payload).get("nextPageToken"))
        url = f"https://www.kaggle.com/api/v1/datasets/list/{ref}?{urlencode({'pageToken': token})}" if token else ""
    return _dedupe_files(files)


def _resolve_nemar(plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher) -> list[RemoteFileCandidate]:
    html = _string(fetcher(candidate.url))
    files = []
    for link in _nemar_download_links(html):
        filepath = _string(parse_qs(urlparse(link).query).get("filepath", [""])[0])
        name = _nemar_file_name(filepath) or Path(urlparse(link).path).name
        files.append(
            _remote_file(
                plan,
                candidate,
                name=name,
                url=urljoin("https://nemar.org", link),
                source_url=candidate.url,
            )
        )
    return _dedupe_files(files)


def _resolve_nitrc_frs(
    plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher
) -> list[RemoteFileCandidate]:
    html = _string(fetcher(_html_landing_url(candidate.url)))
    files: list[RemoteFileCandidate] = []
    for href, label in _html_link_items(html):
        target = urljoin(candidate.url, unescape(href).replace("\\/", "/"))
        path = urlparse(target).path.lower()
        if "/frs/download.php/" in path:
            name = _download_name(target)
        elif "/frs/downloadlink.php/" in path:
            name = _download_name_from_label(label)
            if name and "." not in Path(name).name:
                name = f"{name}.zip"
        else:
            continue
        if not name or not _is_resolvable_download_name(name, plan):
            continue
        files.append(_remote_file(plan, candidate, name=name, url=target, source_url=candidate.url))
    return _dedupe_files(files)


def _resolve_scidb(plan: AcquisitionPlan, candidate: AcquisitionCandidate, fetcher: JsonFetcher) -> list[RemoteFileCandidate]:
    params = parse_qs(urlparse(candidate.url).query)
    landing = _string(params.get("landing", [""])[0])
    data_set_id = _string(params.get("dataSetId", [""])[0]) or _scidb_dataset_id(landing)
    doi = _string(params.get("doi", [""])[0])
    metadata: dict[str, Any] = {}
    if doi:
        metadata = _dict(fetcher(_scidb_openapi_json_url(doi)))
    if not data_set_id and doi:
        resolved = _dict(fetcher(_doi_resolution_url(f"https://doi.org/{doi}")))
        target = _string(resolved.get("url"))
        data_set_id = _scidb_dataset_id(target)
        landing = landing or target
    if not data_set_id:
        data_set_id = _scidb_dataset_id(_string(metadata.get("url") or metadata.get("@id") or metadata.get("identifier")))
    if not data_set_id or not _scidb_publicly_accessible(metadata):
        return []
    version = _scidb_download_version(_string(metadata.get("version")))
    url = _scidb_zip_url(data_set_id, version)
    head = _dict(fetcher(_head_metadata_url(url)))
    if _int_or_none(head.get("status")) not in range(200, 300):
        return []
    size = _scidb_size_bytes(metadata) or _int_or_none(head.get("content_length"))
    name = f"{data_set_id}_{version}.zip"
    return [
        _remote_file(
            plan,
            candidate,
            name=name,
            url=url,
            size_bytes=size,
            media_type="application/zip",
            source_url=landing or _string(metadata.get("url") or metadata.get("@id")) or candidate.url,
        )
    ]


def _resolve_stanford_purl(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
) -> list[RemoteFileCandidate]:
    payload = _dict(fetcher(candidate.url))
    druid = _stanford_druid(candidate.url) or _stanford_druid(_string(payload.get("purl")))
    if not druid:
        return []
    files = []
    for item in _cocina_files(payload):
        if _string(_dict(item.get("access")).get("download")) not in {"world", "stanford"}:
            continue
        filename = _string(item.get("filename"))
        if not filename:
            continue
        digests = _list(item.get("hasMessageDigests"))
        checksum = ""
        if digests:
            first_digest = _dict(digests[0])
            checksum = ":".join(part for part in (_string(first_digest.get("type")), _string(first_digest.get("digest"))) if part)
        files.append(
            _remote_file(
                replace(plan, provider="stanford_sdr"),
                candidate,
                name=filename,
                url=f"https://stacks.stanford.edu/file/druid:{druid}/{quote(filename, safe='/')}",
                size_bytes=_int_or_none(item.get("size")),
                checksum=checksum,
                media_type=_string(item.get("hasMimeType")),
                source_url=f"https://purl.stanford.edu/{druid}",
            )
        )
    return _dedupe_files(files)


def _resolve_data_ru(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
) -> list[RemoteFileCandidate]:
    landing_html = _string(fetcher(_html_landing_url(candidate.url)))
    content_url = _data_ru_content_url(landing_html)
    if not content_url:
        return []
    root = content_url.rstrip("/")
    manifest = _string(fetcher(_html_landing_url(f"{root}/MANIFEST.txt")))
    files = []
    for line in manifest.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        checksum, _, relative_path = line.partition(" ")
        relative_path = relative_path.strip()
        if not relative_path:
            continue
        files.append(
            _remote_file(
                replace(plan, provider="data_ru"),
                candidate,
                name=relative_path,
                url=f"{root}/{quote(relative_path, safe='/')}",
                checksum=f"sha256:{checksum}" if checksum else "",
                source_url=candidate.url,
            )
        )
    return _dedupe_files(files)


def _resolve_dryad(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
) -> list[RemoteFileCandidate]:
    dataset = _dict(fetcher(candidate.url))
    version_url = _link_href(dataset, "stash:version")
    version = _dict(fetcher(urljoin(candidate.url, version_url))) if version_url else dataset
    files_url = _link_href(version, "stash:files")
    if not files_url:
        return []
    payload = _dict(fetcher(urljoin(candidate.url, files_url)))
    items = _dict(payload.get("_embedded")).get("stash:files") or []
    files = []
    iterable = items if isinstance(items, list) else []
    for item in iterable:
        if not isinstance(item, dict):
            continue
        name = _string(item.get("path") or item.get("name") or item.get("filename"))
        download = _link_href(item, "stash:download")
        if not name or not download:
            continue
        digest = _string(item.get("digest"))
        digest_type = _string(item.get("digestType")).lower()
        checksum = f"{digest_type}:{digest}" if digest and digest_type else digest
        files.append(
            _remote_file(
                replace(plan, provider="dryad"),
                candidate,
                name=name,
                url=urljoin(candidate.url, download),
                size_bytes=_int_or_none(item.get("size")),
                checksum=checksum,
                media_type=_string(item.get("mimeType")),
                source_url=candidate.url,
            )
        )
    return _dedupe_files(files)


def _resolve_dspace(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
) -> list[RemoteFileCandidate]:
    parsed = urlparse(candidate.url)
    base = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
    if not base and "10.3929/ethz-b" in candidate.url.lower():
        base = "https://www.research-collection.ethz.ch"
    if not base:
        return []
    item_url = _dspace_item_url(plan, candidate.url, base, fetcher)
    if not item_url:
        return []
    item = _dict(fetcher(item_url))
    bundles_url = _link_href(item, "bundles") or f"{item_url.rstrip('/')}/bundles"
    bundles_payload = _dict(fetcher(urljoin(item_url, bundles_url)))
    bundles = _embedded_list(bundles_payload, "bundles")
    files = []
    for bundle in bundles:
        if _string(bundle.get("name")).upper() != "ORIGINAL":
            continue
        bitstreams_url = _link_href(bundle, "bitstreams")
        if not bitstreams_url:
            continue
        bitstreams_payload = _dict(fetcher(urljoin(item_url, bitstreams_url)))
        for bitstream in _embedded_list(bitstreams_payload, "bitstreams"):
            name = _string(bitstream.get("name"))
            content = _link_href(bitstream, "content")
            if not name or not content:
                continue
            checksum_payload = _dict(bitstream.get("checkSum"))
            algorithm = _string(checksum_payload.get("checkSumAlgorithm")).lower()
            digest = _string(checksum_payload.get("value"))
            checksum = f"{algorithm}:{digest}" if algorithm and digest else digest
            files.append(
                _remote_file(
                    replace(plan, provider="dspace"),
                    candidate,
                    name=name,
                    url=urljoin(item_url, content),
                    size_bytes=_int_or_none(bitstream.get("sizeBytes")),
                    checksum=checksum,
                    source_url=item_url,
                )
            )
    return _dedupe_files(files)


def _resolve_bnci(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
) -> list[RemoteFileCandidate]:
    dataset_id = _bnci_dataset_id(candidate.url) or _bnci_dataset_id_from_name(plan.name)
    if not dataset_id:
        return []
    html = _string(fetcher(_html_landing_url(BNCI_DATASETS_URL)))
    files = []
    for href in _html_links(html):
        url = urljoin(BNCI_DATASETS_URL, unescape(href).replace("\\/", "/"))
        parsed = urlparse(url)
        if f"/database/data-sets/{dataset_id}/" not in parsed.path:
            continue
        name = _download_name(url)
        if not name or not _is_resolvable_download_name(name, replace(plan, provider="bnci")):
            continue
        files.append(
            _remote_file(
                replace(plan, provider="bnci"),
                candidate,
                name=name,
                url=url,
                source_url=f"{BNCI_DATASETS_URL}/{dataset_id}",
            )
        )
    return _dedupe_files(files)


def _resolve_http_landing(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
    *,
    max_pages: int,
) -> list[RemoteFileCandidate]:
    if _is_resolvable_direct_url(candidate.url, plan):
        return [_remote_file(plan, candidate, name=_download_name(candidate.url), url=candidate.url, source_url=candidate.url)]

    files: list[RemoteFileCandidate] = []
    api_url = _invenio_api_record_url(candidate.url)
    if api_url:
        try:
            files.extend(_resolve_invenio_record(plan, candidate, fetcher(api_url), source_url=candidate.url))
        except Exception:
            pass
    if files:
        return _dedupe_files(files)

    payload = fetcher(_html_landing_url(candidate.url))
    if isinstance(payload, dict):
        files.extend(_resolve_invenio_record(plan, candidate, payload, source_url=candidate.url))
        return _dedupe_files(files)

    html = _string(payload)
    for href, label in _html_link_items(html):
        url = urljoin(candidate.url, unescape(href).replace("\\/", "/"))
        if not url.startswith(("http://", "https://")):
            continue
        if _looks_like_non_download_url(url):
            continue
        if "/files-archive" in urlparse(url).path:
            name = "files-archive.zip"
        else:
            name = _download_name(url)
        if not name or not _is_resolvable_download_name(name, plan):
            name = _download_name_from_label(label)
        if not name or not _is_resolvable_download_name(name, plan):
            continue
        files.append(_remote_file(plan, candidate, name=name, url=url, source_url=candidate.url))
        if max_pages is not None and len(files) >= max_pages * 1000:
            break
    return _dedupe_files(files)


def _resolve_invenio_record(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    payload: Any,
    *,
    source_url: str,
) -> list[RemoteFileCandidate]:
    data = _dict(payload)
    files = _dict(_dict(data.get("files")).get("entries"))
    resolved = []
    for key, item in files.items():
        entry = _dict(item)
        links = _dict(entry.get("links"))
        name = _string(entry.get("key") or key)
        url = _string(links.get("content") or links.get("download") or links.get("self"))
        if not name or not url:
            continue
        resolved.append(
            _remote_file(
                plan,
                candidate,
                name=name,
                url=url,
                size_bytes=_int_or_none(entry.get("size")),
                checksum=_string(entry.get("checksum")),
                media_type=_string(entry.get("mimetype")),
                source_url=source_url,
            )
        )
    archive = _string(_dict(data.get("links")).get("archive"))
    if archive:
        resolved.append(_remote_file(plan, candidate, name="files-archive.zip", url=archive, source_url=source_url))
    return resolved


def _resolve_html_index(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
    root: str,
    *,
    max_pages: int,
) -> list[RemoteFileCandidate]:
    queue = [root]
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
        for href in _html_links(_string(payload)):
            if href.startswith(("#", "?")) or href in {"../", "./"}:
                continue
            target = urljoin(url, href)
            if not target.startswith(root):
                continue
            if href.endswith("/"):
                queue.append(target)
                continue
            name = target.removeprefix(root)
            files.append(_remote_file(plan, candidate, name=name, url=target, source_url=root))
    return _dedupe_files(files)


def _resolve_physionet_sha256_manifest(
    plan: AcquisitionPlan,
    candidate: AcquisitionCandidate,
    fetcher: JsonFetcher,
    root: str,
) -> list[RemoteFileCandidate]:
    try:
        manifest = _string(fetcher(f"{root}SHA256SUMS.txt"))
    except Exception:
        return []
    files = []
    for line in manifest.splitlines():
        checksum, _, name = line.strip().partition(" ")
        name = name.strip()
        if not checksum or not name:
            continue
        files.append(
            _remote_file(
                plan,
                candidate,
                name=name,
                url=f"{root}{quote(name, safe='/')}",
                checksum=f"sha256:{checksum}",
                source_url=root,
            )
        )
    return _dedupe_files(files)


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
                name=name.strip("/"),
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
    directly_loadable = _is_directly_loadable_signal_name(name) or _is_generic_signal_supported_by_plan(name, plan)
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
    if url.startswith("doi+resolve://"):
        target = parse_qs(urlparse(url).query).get("url", [""])[0]
        request = Request(target, method="HEAD", headers={"Accept": "*/*", "User-Agent": "NeuroCore/0.1 dataset-resolver"})
        with urlopen(request, timeout=timeout) as response:
            return {"url": response.geturl(), "content_type": response.headers.get("Content-Type", "")}
    if url.startswith("head+metadata://"):
        target = parse_qs(urlparse(url).query).get("url", [""])[0]
        request = Request(target, method="HEAD", headers={"Accept": "*/*", "User-Agent": "NeuroCore/0.1 dataset-resolver"})
        try:
            with urlopen(request, timeout=timeout) as response:
                return {
                    "url": response.geturl(),
                    "status": response.status,
                    "content_length": response.headers.get("Content-Length", ""),
                    "content_type": response.headers.get("Content-Type", ""),
                }
        except HTTPError as exc:
            return {
                "url": target,
                "status": exc.code,
                "content_length": exc.headers.get("Content-Length", ""),
                "content_type": exc.headers.get("Content-Type", ""),
            }
    if url.startswith("openneuro+graphql://"):
        query = parse_qs(urlparse(url).query).get("query", [""])[0]
        payload = json.dumps({"query": query}).encode("utf-8")
        request = Request(
            "https://openneuro.org/crn/graphql",
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "NeuroCore/0.1 dataset-resolver",
            },
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    if url.startswith("html+landing://"):
        target = _html_fetch_target(parse_qs(urlparse(url).query).get("url", [""])[0])
        request = Request(target, headers={"Accept": "text/html,*/*", "User-Agent": "NeuroCore/0.1 dataset-resolver"})
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "ignore")
    if url.startswith("https://physionet.org/files/") or "gin.g-node.org" in url or "nemar.org/dataexplorer" in url:
        request = Request(url, headers={"Accept": "text/html", "User-Agent": "NeuroCore/0.1 dataset-resolver"})
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "ignore")
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


def _dspace_item_url(
    plan: AcquisitionPlan,
    target: str,
    base: str,
    fetcher: JsonFetcher,
) -> str:
    handle = _first_match(r"/handle/([^?#]+)", target)
    search_queries = [query for query in (handle, plan.name) if query]
    for query in search_queries:
        payload = _dict(fetcher(f"{base}/server/api/discover/search/objects?{urlencode({'query': query})}"))
        for item in _dspace_search_objects(payload):
            embedded = _dict(_dict(item.get("_embedded")).get("indexableObject"))
            if handle and _string(embedded.get("handle")) and _string(embedded.get("handle")) != handle:
                continue
            href = _link_href(item, "indexableObject") or _link_href(embedded, "self")
            if href:
                return urljoin(base, href)
    return ""


def _dspace_search_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    search_result = _dict(_dict(_dict(payload.get("_embedded")).get("searchResult")).get("_embedded"))
    objects = search_result.get("objects") or []
    return [item for item in objects if isinstance(item, dict)]


def _link_href(value: Any, key: str) -> str:
    links = _dict(_dict(value).get("_links"))
    link = links.get(key)
    if isinstance(link, dict):
        return _string(link.get("href"))
    if isinstance(link, list):
        for item in link:
            href = _string(_dict(item).get("href"))
            if href:
                return href
    return ""


def _embedded_list(value: dict[str, Any], key: str) -> list[dict[str, Any]]:
    items = _dict(value.get("_embedded")).get(key) or []
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _cocina_files(value: Any) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if _string(value.get("type")).endswith("/file") and value.get("filename"):
            files.append(value)
        for item in value.values():
            files.extend(_cocina_files(item))
    elif isinstance(value, list):
        for item in value:
            files.extend(_cocina_files(item))
    return files


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._active_links: list[tuple[str, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)
                self._active_links.append((value, []))
                return

    def handle_data(self, data: str) -> None:
        if self._active_links:
            self._active_links[-1][1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._active_links:
            href, parts = self._active_links.pop()
            self.links.append((href, " ".join(part.strip() for part in parts if part.strip())))


def _html_links(html: str) -> list[str]:
    parser = _HrefParser()
    parser.feed(html)
    return parser.hrefs


def _html_link_items(html: str) -> list[tuple[str, str]]:
    parser = _HrefParser()
    parser.feed(html)
    if parser.links:
        return parser.links
    return [(href, "") for href in parser.hrefs]


def _json_ld_objects(html: str) -> list[dict[str, Any]]:
    objects = []
    for match in re.finditer(
        r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        try:
            payload = json.loads(unescape(match.group(1)).strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            objects.append(payload)
        elif isinstance(payload, list):
            objects.extend(item for item in payload if isinstance(item, dict))
    return objects


def _data_ru_content_url(html: str) -> str:
    for item in _json_ld_objects(html):
        for value in _recursive_values_for_key(item, "contentUrl"):
            text = _string(value)
            if "webdav.data.ru.nl" in text:
                return text
    return _first_match(r'"contentUrl"\s*:\s*"([^"]*webdav\.data\.ru\.nl[^"]+)"', html) or ""


def _recursive_values_for_key(value: Any, key: str) -> list[Any]:
    found = []
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if item_key == key:
                found.append(item_value)
            found.extend(_recursive_values_for_key(item_value, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_recursive_values_for_key(item, key))
    return found


def _nemar_download_links(html: str) -> list[str]:
    links = []
    for href in _html_links(html):
        if "/dataexplorer/download?" in href:
            links.append(href)
    for match in re.finditer(r"download_file\('([^']+)'\)", html):
        links.append(match.group(1))
    cleaned = []
    for link in links:
        value = unescape(link).replace("\\/", "/")
        if "/dataexplorer/download?" in value:
            cleaned.append(value)
    return list(dict.fromkeys(cleaned))


def _nemar_file_name(filepath: str) -> str:
    if not filepath:
        return ""
    normalized = filepath.replace("\\", "/")
    match = re.search(r"/openneuro/+([^/]+)/(.+)$", normalized)
    if match:
        return match.group(2)
    return normalized.rsplit("/", 1)[-1]


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


def _osf_queue_priority(label: str, url: str) -> tuple[int, str]:
    text = f"{label} {url}".casefold()
    if any(
        token in text
        for token in ("rawdata", "raw data", "raw eeg", "epoched eeg", "/raw", "eegdata", "eeg data", "eeg_data", "/eeg")
    ):
        return (0, text)
    if any(token in text for token in ("stage 2", "stage2", "data", "osfstorage")):
        return (1, text)
    if any(token in text for token in ("snapshot", "stage 1", "stage1", "script", "supplement", "manuscript")):
        return (3, text)
    return (2, text)


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


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    for group in match.groups():
        if group:
            return group
    return match.group(0)


def _openneuro_graphql_url(query: str) -> str:
    return "openneuro+graphql://?" + urlencode({"query": query})


def _doi_resolution_url(url: str) -> str:
    return "doi+resolve://?" + urlencode({"url": url})


def _head_metadata_url(url: str) -> str:
    return "head+metadata://?" + urlencode({"url": url})


def _scidb_candidate_url(*, target: str) -> str:
    params = {"landing": target}
    data_set_id = _scidb_dataset_id(target)
    if data_set_id:
        params["dataSetId"] = data_set_id
    return "scidb+resolve://?" + urlencode(params)


def _scidb_openapi_json_url(doi: str) -> str:
    return "https://www.scidb.cn/api/sdb-openapi-service/json?" + urlencode({"doi": doi})


def _scidb_zip_url(data_set_id: str, version: str) -> str:
    return "https://china.scidb.cn/getZipFile?" + urlencode({"dataSetId": data_set_id, "version": version})


def _scidb_dataset_id(value: str) -> str:
    return _first_match(r"dataSetId=([A-Za-z0-9]+)", value) or ""


def _scidb_download_version(value: str) -> str:
    text = value.strip()
    if not text:
        return "V1"
    if re.fullmatch(r"V\d+", text, flags=re.IGNORECASE):
        return text.upper()
    match = re.match(r"(\d+)(?:\.\d+)*", text)
    if match:
        return f"V{match.group(1)}"
    return text


def _scidb_size_bytes(metadata: dict[str, Any]) -> int | None:
    size = metadata.get("size")
    if isinstance(size, dict):
        return _int_or_none(size.get("value"))
    return _int_or_none(size)


def _scidb_publicly_accessible(metadata: dict[str, Any]) -> bool:
    if not metadata:
        return True
    text = " ".join(
        _string(value)
        for value in (
            metadata.get("conditionsOfAccess"),
            metadata.get("accessRights"),
            metadata.get("isAccessibleForFree"),
        )
    ).casefold()
    restricted_text = text.replace("unrestricted", "")
    return "restricted" not in restricted_text and "protected" not in restricted_text and "false" not in restricted_text


def _html_landing_url(url: str) -> str:
    return "html+landing://?" + urlencode({"url": url})


def _html_fetch_target(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.netloc.lower().removeprefix("www.") in HTML_HTTP_FALLBACK_HOSTS:
        return parsed._replace(scheme="http").geturl()
    return url


def _is_figshare_host(host: str) -> bool:
    normalized = host.lower().removeprefix("www.")
    return "figshare.com" in normalized or normalized in FIGSHARE_COMPATIBLE_HOSTS


def _is_dataverse_host(host: str) -> bool:
    normalized = host.lower().removeprefix("www.")
    return "dataverse" in normalized or normalized in DATAVERSE_COMPATIBLE_HOSTS


def _is_data_ru_host(host: str) -> bool:
    return host.lower().removeprefix("www.") in DATA_RU_HOSTS


def _is_stanford_sdr_host(host: str) -> bool:
    return host.lower().removeprefix("www.") in STANFORD_SDR_HOSTS


def _github_repo(value: str) -> str | None:
    match = re.search(r"api\.github\.com/repos/([^/\s?#]+/[^/\s?#]+)", value, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"github\.com[:/]([^/\s]+/[^/\s?#]+)", value, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).removesuffix(".git")


def _is_repository_html_host(host: str) -> bool:
    return host.lower().removeprefix("www.") in REPOSITORY_HTML_HOSTS


def _figshare_article_id(value: str) -> str | None:
    return _first_match(r"articles/(?:dataset/)?[^/]+/(\d+)|articles/(\d+)", value)


def _dataverse_api_url_from_landing(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    persistent_id = _string(parse_qs(parsed.query).get("persistentId", [""])[0])
    if not persistent_id:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/api/datasets/:persistentId/?{urlencode({'persistentId': persistent_id})}"


def _invenio_api_record_url(url: str) -> str:
    parsed = urlparse(url)
    match = re.search(r"/records/([^/?#]+)", parsed.path)
    if not match or not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}/api/records/{match.group(1)}"


def _download_name(url: str) -> str:
    parsed = urlparse(url)
    path = unescape(parsed.path).rstrip("/")
    if not path:
        return ""
    return path.rsplit("/", 1)[-1]


def _download_name_from_label(label: str) -> str:
    text = " ".join(unescape(label).replace("\n", " ").split())
    if not text:
        return ""
    suffixes = sorted(
        (*ARCHIVE_SUFFIXES, *CONFIDENT_SIGNAL_SUFFIXES, *GENERIC_NUMERIC_SUFFIXES, *TEXT_SIGNAL_SUFFIXES),
        key=len,
        reverse=True,
    )
    suffix_pattern = "|".join(re.escape(suffix) for suffix in suffixes)
    match = re.search(rf"([^\s()<>\"']+?(?:{suffix_pattern}))\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" ,;:")
    return text.strip(" ,;:")


def _bnci_dataset_id(url: str) -> str:
    return _first_match(r"/data-sets/(\d{3}-\d{4})(?:/|$)", urlparse(url).path) or ""


def _bnci_dataset_id_from_name(name: str) -> str:
    text = name.casefold()
    if "bcic iv 2a" in text:
        return "001-2014"
    if "bcic iv 2b" in text:
        return "004-2014"
    return _first_match(r"\((\d{3}-\d{4})\)", name) or ""


def _stanford_druid(value: str) -> str:
    return _first_match(r"(?:purl\.stanford\.edu/|10\.25740/)([a-z]{2}\d{3}[a-z]{2}\d{4})", value) or ""


def _is_resolvable_direct_url(url: str, plan: AcquisitionPlan) -> bool:
    return _is_resolvable_download_name(_download_name(url), plan)


def _is_resolvable_download_name(name: str, plan: AcquisitionPlan) -> bool:
    return (
        _is_archive_name(name)
        or _is_directly_loadable_signal_name(name)
        or _is_generic_signal_supported_by_plan(name, plan)
    )


def _looks_like_non_download_url(url: str) -> bool:
    parsed = urlparse(url)
    lowered = parsed.path.lower()
    if parsed.scheme not in {"http", "https"}:
        return True
    return any(
        marker in lowered
        for marker in (
            "/login",
            "/signup",
            "/search",
            "/preview/",
            "/help",
            "/communities",
            "/statistics",
        )
    )


def _openneuro_dataset_query(dataset_id: str) -> str:
    return (
        "query datasetInfo { "
        f'dataset(id: "{dataset_id}") {{ snapshots {{ tag created }} }}'
        " }"
    )


def _openneuro_files_query(dataset_id: str, tag: str) -> str:
    return (
        "query snapshotFiles { "
        f'snapshot(datasetId: "{dataset_id}", tag: "{tag}") '
        "{ files(recursive: true) { filename size directory annexed id urls } }"
        " }"
    )


def _physionet_version_from_landing(html: str, project: str) -> str:
    escaped = re.escape(project)
    return (
        _first_match(rf"/content/{escaped}/([^/\"'?#]+)/", html)
        or _first_match(rf"/files/{escaped}/([^/\"'?#]+)/", html)
        or _first_match(r"Version:\s*([0-9][A-Za-z0-9_.-]*)", html)
        or ""
    )


def _openneuro_download_url(urls: list[Any], filename: str) -> str:
    candidates = [_string(url) for url in urls if _string(url)]
    normalized_name = filename.replace("\\", "/").lstrip("/")
    if normalized_name:
        quoted_name = quote(normalized_name, safe="/")
        for url in candidates:
            parsed_path = unescape(urlparse(url).path).lstrip("/")
            if parsed_path.endswith(normalized_name) or parsed_path.endswith(quoted_name):
                return url
    return candidates[0] if candidates else ""


def _is_archive_name(name: str) -> bool:
    normalized = name.lower()
    return (
        any(normalized.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)
        or bool(SPLIT_ARCHIVE_PATTERN.search(normalized))
        or bool(re.fullmatch(r"rawdata[_-]?part[_-]?\d+", Path(normalized).name))
        or bool(re.fullmatch(r".+[_-]part[_-]?[a-z]", Path(normalized).name))
    )


def _is_directly_loadable_signal_name(name: str) -> bool:
    if not can_load_extension(name):
        return False
    suffix = Path(name.lower()).suffix
    if name.lower().endswith(".fif.gz"):
        suffix = ".fif"
    if suffix in CONFIDENT_SIGNAL_SUFFIXES:
        return True
    if suffix in GENERIC_NUMERIC_SUFFIXES:
        if "starttime" in Path(name.lower()).stem:
            return False
        if "eeg" in Path(name.lower()).stem:
            return True
        return bool(re.search(r"(^|[_./ -])(eeg|raw|signal|signals|recording|subject|sub-[a-z0-9]+|ses-[a-z0-9]+|task-[a-z0-9]+)", name, re.IGNORECASE))
    return False


def _is_generic_signal_supported_by_plan(name: str, plan: AcquisitionPlan) -> bool:
    suffix = Path(name.lower()).suffix
    if plan.provider == "bnci" and suffix == ".mat":
        return True
    hints = set(plan.format_hints)
    if suffix == ".mat" and "mat" in hints:
        return _is_signal_like_generic_path(name)
    if suffix in {".npy", ".npz"} and "numpy" in hints:
        return _is_signal_like_generic_path(name)
    if suffix in {".pt", ".pth"}:
        return _is_torch_signal_file(name, plan)
    if suffix == ".csv" and "csv" in hints:
        return (
            _is_signal_like_generic_path(name)
            or _is_eeg_condition_delimited_file(name, plan)
            or _is_eeg_device_delimited_file(name, plan)
        )
    if suffix in TEXT_SIGNAL_SUFFIXES and "text" in hints:
        return _is_signal_like_generic_path(name)
    if suffix in {".tab", ".tsv"} and hints.intersection({"csv", "text"}):
        return _is_signal_like_generic_path(name)
    return False


def _is_eeg_condition_delimited_file(name: str, plan: AcquisitionPlan) -> bool:
    if not re.search(r"\b(eeg|electroencephalogram)\b", plan.name, flags=re.IGNORECASE):
        return False
    stem = Path(name.lower()).stem
    return stem in {"c", "ec", "eo", "et", "f", "h", "m", "r", "s"}


def _is_eeg_device_delimited_file(name: str, plan: AcquisitionPlan) -> bool:
    if not re.search(r"\b(eeg|electroencephalogram)\b", plan.name, flags=re.IGNORECASE):
        return False
    stem = Path(name.lower()).stem
    return bool(re.search(r"(?:^|[_-])(?:openbci|gtec|g\.tec)(?:[_-]|$)", stem))


def _is_torch_signal_file(name: str, plan: AcquisitionPlan) -> bool:
    text = f"{plan.name} {' '.join(plan.format_hints)}".lower()
    if not re.search(r"\b(eeg|sleep|polysomnography|preclinical)\b", text):
        return False
    return _is_signal_like_generic_path(name)


def _is_signal_like_generic_path(name: str) -> bool:
    normalized = name.replace("\\", "/").lower()
    basename = Path(normalized).name
    stem = Path(basename).stem
    if any(
        token in normalized
        for token in (
            "behavior",
            "behaviour",
            "questionnaire",
            "readme",
            "stimuli",
            "stimulus",
            "intervalmarker",
            "metadata",
            "starttime",
            "score",
            "scores",
            "result",
            "results",
            "plasma",
            "spectra",
            "spectrum",
            "fft",
            "fooof",
            "statistic",
            "statistics",
            "table",
        )
    ):
        return False
    if any(part in normalized for part in ("/eeg/", "/raw/", "/signal/", "/signals/", "filtered_data/", "segmented_data/")):
        return True
    if any(part in normalized for part in ("eeg_csv/", "raw muse data/", "raw_muse_data/")):
        return True
    if normalized.startswith(("raw/", "eeg/", "signals/")):
        return True
    if "eeg" in basename:
        return True
    if "oddball" in basename:
        return True
    if re.search(r"^(data[_-]?s\d+|s\d+[_-]?data|subject[_-]?\d+[_-]?data)", basename):
        return True
    if re.fullmatch(r"s\d+", stem):
        return True
    if re.fullmatch(r"[a-z]_\d+_s\d+", stem):
        return True
    if re.fullmatch(r"s\d+[-_]m?(?:oo|c)", stem):
        return True
    if re.search(r"^data[_-][a-z0-9]+[_-]sub[_-]?\d+", basename):
        return True
    if re.fullmatch(r"[a-z]{0,3}sub\d+", stem):
        return True
    if re.fullmatch(r"s\d+x?_[ab]\d+", stem):
        return True
    if re.fullmatch(r"p\d+[a-z]?_babble_ar", stem):
        return True
    if re.fullmatch(r"(?:dep|hc)ec\d+", stem):
        return True
    if re.fullmatch(r"shhs\d+[-_]\d+", stem):
        return True
    if re.search(r"^data[_-].*(?:combined|pre\d+|post\d+)", stem):
        return True
    if re.search(r"(?:^|[_-])annotated[_-]?sample\d+", stem):
        return True
    if "epochs" in stem:
        return True
    if "stimulationdata" in stem and "blockdesign" in stem:
        return True
    if stem.startswith(("opto_", "tfus_")):
        return True
    if re.search(r"^s\d+_[0-9]+_kmi$", stem):
        return True
    if re.search(r"^user\d+_[0-9]+_[0-9]+$", stem):
        return True
    if re.search(r"^subj(?:ect)?[_-]?\d+[_-]?(?:rest|tms|eeg|raw|task)", basename):
        return True
    if re.fullmatch(r"\d{5,}", stem):
        return True
    if re.search(r"(?:^|[ _/-])(?:class|diff)[_-]?[a-z0-9]+$", stem):
        return True
    if any(
        token in stem
        for token in ("calibration", "singleplayer", "multiplayer", "recording", "session", "trial", "task")
    ):
        return True
    if stem in {"dataica", "ica_data", "icadata"}:
        return True
    if stem in {"ad", "mci", "normal", "control", "healthy", "hc"}:
        return True
    return bool(re.search(r"^(sub-[a-z0-9]+|subject[_-]?\d+|eeg|raw|signal)", basename))


def _loader_materialization_keys(files: tuple[RemoteFileCandidate, ...]) -> set[tuple[str, str]]:
    selected = {_file_key(file) for file in files if file.directly_loadable}
    by_name = {file.name.replace("\\", "/").lower(): file for file in files}
    for file in files:
        if not file.directly_loadable:
            continue
        for companion_name in _loader_companion_names(file.name):
            companion = by_name.get(companion_name)
            if companion:
                selected.add(_file_key(companion))
    return selected


def _actionable_materialization_keys(
    files: tuple[RemoteFileCandidate, ...],
    *,
    include_archives: bool,
) -> set[tuple[str, str]]:
    selected = _loader_materialization_keys(files)
    if include_archives:
        selected.update(_file_key(file) for file in files if file.archive)
    return selected


def _loader_companion_names(name: str) -> tuple[str, ...]:
    normalized = name.replace("\\", "/")
    lower = normalized.lower()
    if lower.endswith(".vhdr"):
        stem = normalized[:-5]
        return (f"{stem}.eeg".lower(), f"{stem}.dat".lower(), f"{stem}.vmrk".lower())
    if lower.endswith(".set"):
        return (f"{normalized[:-4]}.fdt".lower(),)
    if lower.endswith(".lay"):
        return (f"{normalized[:-4]}.dat".lower(),)
    return ()


def _file_key(file: RemoteFileCandidate) -> tuple[str, str]:
    return (file.name, file.url)


def _safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "dataset"


def _safe_filename(name: str, url: str) -> str:
    candidate = Path(urlparse(url).path).name if not name else name
    cleaned = re.sub(r"[^A-Za-z0-9_.() -]+", "_", candidate).strip(" ._")
    return cleaned or "remote-file"


def _safe_relative_path(name: str, url: str) -> Path:
    candidate = name.replace("\\", "/").strip("/") if name else Path(urlparse(url).path).name
    parts = []
    for part in candidate.split("/"):
        if part in {"", ".", ".."}:
            continue
        cleaned = _safe_filename(part, "")
        if cleaned in {"", ".", ".."}:
            continue
        parts.append(cleaned)
    if not parts:
        parts.append("remote-file")
    return Path(*parts)


def _archive_stem(path: Path) -> str:
    name = path.name
    split_match = SPLIT_ARCHIVE_PATTERN.search(name)
    if split_match:
        return name[: split_match.start()]
    for suffix in (".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2", ".zip", ".tar", ".gz", ".7z", ".rar"):
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
