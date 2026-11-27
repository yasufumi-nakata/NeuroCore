"""Connect the local product to a fixed inference server through an SSH tunnel.

The destination is selected once by the operator at startup. Requests cannot
redirect it, choose another URL, or upload a model. No request is retried.
"""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .service import XavierServiceError


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_remote_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"} or parsed.port is None):
            raise ValueError
        port = parsed.port
        if not 1 <= port <= 65535:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Use an explicit loopback HTTP port for the model server, through an SSH tunnel") from None
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"http://{host}:{port}"


class RemoteXavierService:
    """XavierService-compatible transport, without any GPU dependencies locally."""

    def __init__(self, base_url: str) -> None:
        self.base_url = validate_remote_url(base_url)

    def _request(self, path: str, payload: dict[str, Any] | None = None,
                 *, timeout: float = 15) -> dict[str, Any]:
        try:
            data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            raise XavierServiceError("invalid_input", "Input must be finite JSON") from None
        if data is not None and len(data) > 8 * 1024 * 1024:
            raise XavierServiceError("request_too_large", "Requests are limited to 8 MiB", 413)
        request = urllib.request.Request(self.base_url + "/api/xavier/" + path,
                                         data=data, headers={"Content-Type": "application/json"},
                                         method="GET" if payload is None else "POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                body = response.read(8 * 1024 * 1024 + 1)
                if len(body) > 8 * 1024 * 1024:
                    raise ValueError("Response too large")
                result = json.loads(body)
                if not isinstance(result, dict):
                    raise ValueError("Response is not an object")
                return result
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read(65536)).get("detail", {})
                if not isinstance(detail, dict):
                    detail = {}
            except (ValueError, AttributeError):
                detail = {}
            raise XavierServiceError(str(detail.get("code", "remote_error"))[:100],
                                     str(detail.get("message", "Model server rejected this operation"))[:1000],
                                     exc.code if 400 <= exc.code < 600 else 502) from None
        except (TimeoutError, socket.timeout):
            raise XavierServiceError("inference_timeout", "Model request timed out; check status before retrying", 504) from None
        except urllib.error.URLError:
            raise XavierServiceError("remote_unavailable", "Cannot reach the model server; check the SSH tunnel", 503) from None
        except (ValueError, OSError):
            raise XavierServiceError("remote_invalid_response", "Model server response could not be verified", 502) from None

    def list_models(self) -> dict[str, Any]:
        return self._request("models")

    def inspect(self, model_id: str) -> dict[str, Any]:
        return self._request("models/" + urllib.parse.quote(model_id, safe=""))

    def status(self) -> dict[str, Any]:
        result = self._request("status")
        return {**result, "transport": "loopback_ssh_tunnel"}

    def load(self, model_id: str) -> dict[str, Any]:
        return self._request("load", {"model_id": model_id}, timeout=600)

    def unload(self) -> dict[str, Any]:
        return self._request("unload", {}, timeout=90)

    def forecast(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("forecast", payload, timeout=180)

    def propose(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("propose", payload, timeout=180)
