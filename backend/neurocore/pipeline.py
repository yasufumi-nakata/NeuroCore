from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

from .frame import FrameValidationError, NeuroFrame, ValidityIssue
from .kernels import FeatureSet, Kernel


@dataclass(frozen=True)
class PipelineStepReport:
    name: str
    backend: str
    status: str
    duration_ms: float
    output: dict[str, Any]
    warnings: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 3),
            "output": self.output,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class PipelineReport:
    pipeline_id: str
    backend: str
    status: str
    steps: tuple[PipelineStepReport, ...]
    issues: tuple[ValidityIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "backend": self.backend,
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class PipelineResult:
    output: NeuroFrame | FeatureSet
    report: PipelineReport

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.output, NeuroFrame):
            output = {"kind": "frame", "summary": self.output.to_summary()}
        else:
            output = {"kind": "features", "features": self.output.to_dict()}
        return {"output": output, "report": self.report.to_dict()}


class PipelineExecutionError(RuntimeError):
    def __init__(self, message: str, report: PipelineReport | None = None):
        self.report = report
        super().__init__(message)


@dataclass
class Pipeline:
    kernels: Iterable[Kernel]
    name: str = "neurocore-pipeline"
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        self.kernels = tuple(self.kernels)

    def run(self, frame: NeuroFrame, *, backend: str = "cpu") -> PipelineResult:
        if backend != "cpu":
            raise PipelineExecutionError(f"unsupported backend: {backend}. The CPU backend is the reference backend in v0.1.")
        current: NeuroFrame | FeatureSet = frame
        steps: list[PipelineStepReport] = []
        issues: list[ValidityIssue] = []
        for kernel in self.kernels:
            if not isinstance(current, NeuroFrame):
                raise PipelineExecutionError(f"kernel {kernel.name} cannot run after feature extraction")
            started = time.perf_counter()
            try:
                validation_issues = tuple(kernel.validate(current))
                errors = [issue for issue in validation_issues if issue.severity in {"error", "critical"}]
                if errors:
                    raise FrameValidationError(errors)
                current = kernel.run(current)
                duration_ms = (time.perf_counter() - started) * 1000.0
                output = _summarize_output(current)
                warnings = tuple(issue.to_dict() for issue in validation_issues if issue.severity == "warning")
                issues.extend(validation_issues)
                steps.append(PipelineStepReport(kernel.name, backend, "passed", duration_ms, output, warnings))
            except FrameValidationError as exc:
                duration_ms = (time.perf_counter() - started) * 1000.0
                issues.extend(exc.issues)
                steps.append(
                    PipelineStepReport(
                        kernel.name,
                        backend,
                        "failed",
                        duration_ms,
                        {"error_count": len(exc.issues)},
                    )
                )
                report = PipelineReport(self.pipeline_id, backend, "failed", tuple(steps), tuple(issues))
                raise PipelineExecutionError(str(exc), report=report) from exc
        report = PipelineReport(self.pipeline_id, backend, "passed", tuple(steps), tuple(issues))
        return PipelineResult(current, report)


def _summarize_output(output: NeuroFrame | FeatureSet) -> dict[str, Any]:
    if isinstance(output, NeuroFrame):
        return {
            "kind": "frame",
            "shape": [output.samples, output.channel_count],
            "sampling_rate": output.timebase.sampling_rate,
            "duration_seconds": output.duration_seconds,
        }
    return {"kind": "features", "feature_count": len(output.values), "feature_names": sorted(output.values)}
