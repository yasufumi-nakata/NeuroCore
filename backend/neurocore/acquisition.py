from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from .datasets import DatasetInventory, DatasetRecord
from .loaders import find_supported_signal_files


DIRECT_API_PROVIDERS = {
    "zenodo",
    "figshare",
    "osf",
    "dataverse",
    "dryad",
    "mendeley",
    "scidb",
    "invenio",
    "bnci",
    "repository_html",
}
TOOLING_PROVIDERS = {"openneuro", "gin", "github", "physionet", "dandi", "huggingface", "nemar", "kaggle"}
ACCOUNT_PROVIDERS = {"pennsieve", "ieee_dataport", "nda"}
FIGSHARE_COMPATIBLE_HOSTS = {
    "data.4tu.nl",
    "data.dtu.dk",
    "bridges.monash.edu",
}
DATAVERSE_COMPATIBLE_HOSTS = {
    "borealisdata.ca",
    "dataverse.harvard.edu",
    "researchdata.ntu.edu.sg",
    "researchdata.lib.cityu.edu.hk",
    "redu.unicamp.br",
}
INVENIO_COMPATIBLE_HOSTS = {
    "openaccessrepository.it",
    "www.openaccessrepository.it",
    "fdr.uni-hamburg.de",
    "www.fdr.uni-hamburg.de",
    "fdat.uni-tuebingen.de",
}
REPOSITORY_HTML_HOSTS = {
    "datashare.ed.ac.uk",
    "deepblue.lib.umich.edu",
}


@dataclass(frozen=True)
class AcquisitionCandidate:
    provider: str
    method: str
    url: str
    role: str
    requires_auth: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "method": self.method,
            "url": self.url,
            "role": self.role,
            "requires_auth": self.requires_auth,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AcquisitionPlan:
    record_id: str
    name: str
    provider: str
    access_status: str
    automation_status: str
    rationale: str
    candidates: tuple[AcquisitionCandidate, ...]
    format_hints: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "name": self.name,
            "provider": self.provider,
            "access_status": self.access_status,
            "automation_status": self.automation_status,
            "rationale": self.rationale,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "format_hints": list(self.format_hints),
        }


def plan_acquisition(record: DatasetRecord) -> AcquisitionPlan:
    provider = detect_provider(record)
    format_hints = tuple(sorted(_format_hints(record)))
    candidates = tuple(_candidate_urls(record, provider))
    if _access_unusable(record.access_status):
        status = "unusable"
        rationale = "inventory marks the dataset as difficult to reuse"
    elif _access_requires_account(record.access_status) or provider in ACCOUNT_PROVIDERS:
        status = "account_required"
        rationale = "provider or inventory access status requires account approval before raw files can be fetched"
    elif provider in DIRECT_API_PROVIDERS:
        status = "direct_api"
        rationale = "provider exposes a public API or stable file listing that can be automated"
    elif provider in TOOLING_PROVIDERS:
        status = "tooling_required"
        rationale = "provider is automatable through a provider-specific CLI, git, or data client"
    elif provider == "doi":
        status = "doi_resolution_required"
        rationale = "DOI must be resolved to a concrete provider before raw files can be fetched"
    else:
        status = "manual_review"
        rationale = "provider is not mapped to a safe automatic fetch path yet"
    return AcquisitionPlan(
        record_id=record.record_id,
        name=record.name,
        provider=provider,
        access_status=record.access_status,
        automation_status=status,
        rationale=rationale,
        candidates=candidates,
        format_hints=format_hints,
    )


def plan_inventory_acquisition(inventory: DatasetInventory) -> tuple[AcquisitionPlan, ...]:
    return tuple(plan_acquisition(record) for record in inventory.records)


def summarize_acquisition_plans(plans: tuple[AcquisitionPlan, ...]) -> dict[str, Any]:
    status_counts = Counter(plan.automation_status for plan in plans)
    provider_counts = Counter(plan.provider for plan in plans)
    with_candidates = sum(1 for plan in plans if plan.candidates)
    return {
        "record_count": len(plans),
        "automation_status_counts": dict(status_counts.most_common()),
        "provider_counts": dict(provider_counts.most_common(30)),
        "records_with_candidates": with_candidates,
        "records_without_candidates": len(plans) - with_candidates,
    }


def local_readiness_for_record(
    record: DatasetRecord,
    cache_root: str | Path,
    *,
    max_files: int | None = 20,
) -> dict[str, Any]:
    root = Path(cache_root).expanduser()
    candidates = []
    for key in _record_cache_keys(record):
        directory = root / key
        candidates.extend(find_supported_signal_files(directory))
    selected = candidates if max_files is None else candidates[:max_files]
    return {
        "record_id": record.record_id,
        "name": record.name,
        "cache_keys": _record_cache_keys(record),
        "local_signal_file_count": len(candidates),
        "local_signal_files": [str(path) for path in selected],
        "local_signal_files_truncated": max_files is not None and len(candidates) > max_files,
    }


def detect_provider(record: DatasetRecord) -> str:
    text = " ".join([record.source_domain, record.url, record.doi, " ".join(record.search_sources)]).lower()
    doi_provider = _doi_provider(record.doi) or _doi_provider(text)
    if doi_provider:
        return doi_provider
    hosts = [_host(record.url), *[_host(source) for source in record.search_sources]]
    for host in hosts:
        provider = _provider_from_host(host)
        if provider != "unknown":
            return provider
    return _provider_from_host(record.source_domain)


def _candidate_urls(record: DatasetRecord, provider: str) -> list[AcquisitionCandidate]:
    landing_urls = [url for url in [record.url, *record.search_sources] if url.startswith(("http://", "https://"))]
    candidates: list[AcquisitionCandidate] = []
    if provider == "zenodo":
        record_id = _first_match(r"zenodo\.(\d+)|records/(\d+)", " ".join(landing_urls + [record.doi]))
        if record_id:
            candidates.append(
                AcquisitionCandidate(
                    provider, "zenodo_api", f"https://zenodo.org/api/records/{record_id}", "file_listing"
                )
            )
    elif provider == "figshare":
        article_id = _first_match(r"articles/(?:dataset/)?[^/]+/(\d+)|articles/(\d+)", " ".join(landing_urls))
        if article_id:
            candidates.append(
                AcquisitionCandidate(
                    provider,
                    "figshare_api",
                    f"https://api.figshare.com/v2/articles/{article_id}",
                    "file_listing",
                )
            )
        elif record.doi:
            candidates.append(
                AcquisitionCandidate(provider, "doi_resolver", f"https://doi.org/{record.doi}", "landing")
            )
    elif provider == "osf":
        node_id = _first_match(r"osf\.io/([a-z0-9]{4,8})", " ".join(landing_urls))
        if node_id:
            candidates.append(
                AcquisitionCandidate(
                    provider, "osf_api", f"https://api.osf.io/v2/nodes/{node_id}/files/", "file_listing"
                )
            )
    elif provider == "openneuro":
        dataset_id = _first_match(r"(ds\d{6,})", " ".join(landing_urls + [record.doi]))
        target = dataset_id or record.url
        candidates.append(
            AcquisitionCandidate(
                provider,
                "openneuro_cli",
                f"openneuro://{target}",
                "tool_download",
                notes=("use openneuro-py or datalad for full dataset materialization",),
            )
        )
    elif provider == "dataverse":
        if record.doi:
            host = _dataverse_host_for_record(record)
            encoded = quote(_dataverse_persistent_id_for_record(record), safe="")
            candidates.append(
                AcquisitionCandidate(
                    provider,
                    "dataverse_api",
                    f"https://{host}/api/datasets/:persistentId/?persistentId={encoded}",
                    "file_listing",
                )
            )
    elif provider == "dryad":
        if record.doi:
            candidates.append(
                AcquisitionCandidate(
                    provider,
                    "dryad_api",
                    f"https://datadryad.org/api/v2/datasets/{quote(record.doi, safe='')}",
                    "file_listing",
                )
            )
    elif provider == "mendeley":
        dataset_id = _first_match(r"data\.mendeley\.com/datasets/([a-z0-9]+)", " ".join(landing_urls))
        if dataset_id:
            candidates.append(
                AcquisitionCandidate(
                    provider,
                    "mendeley_api",
                    f"https://data.mendeley.com/public-api/datasets/{dataset_id}",
                    "file_listing",
                )
            )
    elif provider == "scidb":
        data_set_id = _first_match(r"dataSetId=([a-z0-9]+)", " ".join(landing_urls))
        params = {}
        if data_set_id:
            params["dataSetId"] = data_set_id
        if record.doi:
            params["doi"] = record.doi
        if record.url:
            params["landing"] = record.url
        if params:
            candidates.append(
                AcquisitionCandidate(
                    provider,
                    "scidb_api",
                    "scidb+resolve://?" + "&".join(f"{key}={quote(value, safe='')}" for key, value in params.items()),
                    "file_listing",
                )
            )
    elif provider == "bnci":
        target = record.url or "https://bnci-horizon-2020.eu/database/data-sets"
        candidates.append(AcquisitionCandidate(provider, "bnci_index", target, "file_listing"))
    elif provider == "repository_html":
        if record.url and _host(record.url) != "doi.org":
            candidates.append(AcquisitionCandidate(provider, "http_landing", record.url, "file_listing"))
        elif record.doi:
            candidates.append(
                AcquisitionCandidate(provider, "doi_resolver", f"https://doi.org/{record.doi}", "landing")
            )
    elif provider == "github":
        repo = _first_match(r"github\.com[:/]([^/\s]+/[^/\s?#]+)", " ".join(landing_urls))
        if repo:
            repo = repo.removesuffix(".git")
            candidates.append(
                AcquisitionCandidate(provider, "git_clone", f"https://github.com/{repo}.git", "repository")
            )
    elif provider in TOOLING_PROVIDERS | ACCOUNT_PROVIDERS:
        if record.url:
            candidates.append(
                AcquisitionCandidate(
                    provider,
                    f"{provider}_client",
                    record.url,
                    "tool_download",
                    requires_auth=provider in ACCOUNT_PROVIDERS,
                )
            )
    elif provider == "doi" and record.doi:
        candidates.append(AcquisitionCandidate(provider, "doi_resolver", f"https://doi.org/{record.doi}", "landing"))
    elif record.url:
        candidates.append(AcquisitionCandidate(provider, "http_landing", record.url, "landing"))
    if not candidates:
        if record.url:
            candidates.append(AcquisitionCandidate(provider, "http_landing", record.url, "landing"))
        elif record.doi:
            candidates.append(
                AcquisitionCandidate(provider, "doi_resolver", f"https://doi.org/{record.doi}", "landing")
            )
    return _dedupe_candidates(candidates)


def _provider_from_host(host: str) -> str:
    host = host.lower().removeprefix("www.")
    if not host:
        return "unknown"
    if "zenodo.org" in host:
        return "zenodo"
    if "openneuro.org" in host:
        return "openneuro"
    if host == "osf.io" or host.endswith(".osf.io"):
        return "osf"
    if "figshare.com" in host:
        return "figshare"
    if host in FIGSHARE_COMPATIBLE_HOSTS:
        return "figshare"
    if "dataverse" in host or host in DATAVERSE_COMPATIBLE_HOSTS:
        return "dataverse"
    if host in INVENIO_COMPATIBLE_HOSTS:
        return "invenio"
    if "bnci-horizon-2020.eu" in host:
        return "bnci"
    if host in REPOSITORY_HTML_HOSTS:
        return "repository_html"
    if "mendeley.com" in host:
        return "mendeley"
    if "kaggle.com" in host:
        return "kaggle"
    if "pennsieve.io" in host:
        return "pennsieve"
    if "ieee-dataport.org" in host:
        return "ieee_dataport"
    if "datadryad.org" in host:
        return "dryad"
    if "gin.g-node.org" in host:
        return "gin"
    if "github.com" in host:
        return "github"
    if "physionet.org" in host:
        return "physionet"
    if "dandiarchive.org" in host:
        return "dandi"
    if "huggingface.co" in host:
        return "huggingface"
    if "nemar.org" in host:
        return "nemar"
    if "scidb.cn" in host:
        return "scidb"
    if "nda.nih.gov" in host:
        return "nda"
    if "doi.org" in host:
        return "doi"
    return "unknown"


def _doi_provider(value: str) -> str | None:
    text = value.lower()
    if "10.5281/zenodo" in text or "zenodo." in text:
        return "zenodo"
    if "figshare" in text or "10.6084/m9.figshare" in text:
        return "figshare"
    if any(prefix in text for prefix in ("10.11583/dtu", "10.4121/", "10.4225/03/")):
        return "figshare"
    if "openneuro" in text or "10.18112/openneuro" in text:
        return "openneuro"
    if "10.17605/osf.io" in text:
        return "osf"
    if "10.17632/" in text:
        return "mendeley"
    if "10.7910/dvn/" in text:
        return "dataverse"
    if any(prefix in text for prefix in ("10.5683/sp3/", "10.21979/n9/", "10.25824/redu/", "10.82468/")):
        return "dataverse"
    if "10.5061/dryad" in text:
        return "dryad"
    if "10.57760/sciencedb" in text or "sciencedb." in text:
        return "scidb"
    if "10.7488/ds/" in text:
        return "repository_html"
    return None


def _access_unusable(value: str) -> bool:
    text = value.casefold()
    return any(
        marker in text
        for marker in (
            "利用困難",
            "unusable",
            "unavailable",
            "not available",
            "not reusable",
            "withdrawn",
            "metadata only",
            "no raw",
        )
    )


def _access_requires_account(value: str) -> bool:
    text = value.casefold()
    return any(
        marker in text
        for marker in (
            "要アカウント",
            "利用登録",
            "account",
            "registration",
            "login",
            "approval",
            "request access",
            "restricted",
            "controlled access",
            "credential",
        )
    )


def _format_hints(record: DatasetRecord) -> set[str]:
    text = record.text_for_detection.lower()
    hints: set[str] = set()
    for label, patterns in {
        "bids": (r"\bbids\b",),
        "edf": (r"\.edf\b", r"\bedf\b"),
        "bdf": (r"\.bdf\b", r"\bbdf\b"),
        "eeglab_set": (r"\.set\b", r"\beeglab\b"),
        "brainvision": (r"\.vhdr\b", r"brainvision"),
        "fif": (r"\.fif\b",),
        "gdf": (r"\.gdf\b", r"\bgdf\b"),
        "cnt": (r"\.cnt\b", r"\bcnt\b"),
        "egi": (r"\.egi\b", r"\bmff\b", r"\begi\b"),
        "eximia": (r"\.nxe\b", r"eximia"),
        "nicolet": (r"\.data\b", r"nicolet"),
        "persyst": (r"\.lay\b", r"persyst"),
        "mef": (r"\.mefd\b", r"\bmef3?\b"),
        "nwb": (r"\.nwb\b", r"\bnwb\b", r"neurodata without borders"),
        "xdf": (r"\.xdf\b", r"\bxdf\b"),
        "mat": (r"\.mat\b", r"matlab"),
        "numpy": (r"\.npy\b", r"\.npz\b"),
        "csv": (r"\.csv\b", r"\bcsv\b"),
    }.items():
        if any(re.search(pattern, text) for pattern in patterns):
            hints.add(label)
    return hints


def _record_cache_keys(record: DatasetRecord) -> list[str]:
    keys = [record.record_id]
    if record.doi:
        keys.append(re.sub(r"[^A-Za-z0-9_.-]+", "_", record.doi).strip("_"))
    if record.name:
        keys.append(re.sub(r"[^A-Za-z0-9_.-]+", "_", record.name)[:80].strip("_"))
    return [key for key in keys if key]


def _host(value: str) -> str:
    if not value:
        return ""
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return parsed.netloc.lower()


def _dataverse_host_for_record(record: DatasetRecord) -> str:
    text = record.doi.lower()
    if "10.5683/sp3/" in text:
        return "borealisdata.ca"
    if "10.21979/n9/" in text:
        return "researchdata.ntu.edu.sg"
    if "10.25824/redu/" in text:
        return "redu.unicamp.br"
    if "10.82468/" in text:
        return "researchdata.lib.cityu.edu.hk"
    for value in (record.url, *record.search_sources):
        host = _host(value).removeprefix("www.")
        if host and (host in DATAVERSE_COMPATIBLE_HOSTS or ("dataverse" in host and "." in host)):
            return host
    return "dataverse.harvard.edu"


def _dataverse_persistent_id_for_record(record: DatasetRecord) -> str:
    for value in (record.url, *record.search_sources):
        parsed = urlparse(value if "://" in value else f"https://{value}")
        persistent_values = parse_qs(parsed.query).get("persistentId", [])
        if persistent_values:
            return persistent_values[0]
    return f"doi:{record.doi}"


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    for group in match.groups():
        if group:
            return group
    return match.group(0)


def _dedupe_candidates(candidates: list[AcquisitionCandidate]) -> list[AcquisitionCandidate]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[AcquisitionCandidate] = []
    for candidate in candidates:
        key = (candidate.provider, candidate.method, candidate.url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
