"""Local application delivery, desktop API boundary, and one-command launch."""
from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
import secrets
from typing import Any, Callable
from urllib.parse import urlsplit


def ui_directory() -> Path:
    configured = os.environ.get("NEUROCORE_UI_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    packaged = Path(__file__).with_name("web")
    if (packaged / "index.html").is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "frontend" / "dist"


class _ProductBoundary:
    """Reject cross-site desktop operations, DNS rebinding, and unbounded bodies."""

    def __init__(self, app, *, token: str) -> None:
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/"):
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        host = headers.get(b"host", b"").decode("latin-1")
        client = (scope.get("client") or ("", 0))[0]
        method = scope.get("method", "GET")
        workspace = scope["path"].startswith("/api/workspace/")

        async def reject(code: str, status: int):
            body = json.dumps({"detail": {"code": code, "message": "Local application request could not be authorized"}}).encode()
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
            await send({"type": "http.response.body", "body": body})

        try:
            hostname = urlsplit("http://" + host).hostname
        except ValueError:
            hostname = None
        # TestClient has no network peer. Network requests must use a loopback host.
        if hostname not in {"localhost", "127.0.0.1", "::1"} and not (client == "testclient" and hostname == "testserver"):
            return await reject("local_host_required", 403)
        origin = headers.get(b"origin", b"").decode("latin-1")
        allowed_origins = {"http://" + host, "https://" + host,
                           "http://localhost:5174", "http://127.0.0.1:5174",
                           "http://localhost:5173", "http://127.0.0.1:5173"}
        if origin and origin not in allowed_origins:
            return await reject("origin_rejected", 403)
        if workspace and client not in {"127.0.0.1", "::1", "testclient"}:
            return await reject("local_client_required", 403)
        if workspace and method not in {"GET", "HEAD", "OPTIONS"}:
            supplied = headers.get(b"x-neurocore-token", b"").decode("latin-1")
            if not hmac.compare_digest(supplied, self.token):
                return await reject("session_token_required", 403)
        if workspace and method == "POST":
            if headers.get(b"content-type", b"").split(b";", 1)[0].strip() != b"application/json":
                return await reject("json_required", 415)
            chunks, total = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body = message.get("body", b"")
                total += len(body)
                if total > 8 * 1024 * 1024:
                    return await reject("request_too_large", 413)
                chunks.append(body)
                if not message.get("more_body", False):
                    break
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
                return await receive()

            return await self.app(scope, replay, send)
        return await self.app(scope, receive, send)


def install_product_routes(app, service_getter: Callable[[], Any]) -> None:
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from .desktop import DesktopError, DesktopSessionManager
    from .xavier.service import XavierServiceError

    token = secrets.token_urlsafe(32)
    manager = DesktopSessionManager(service_getter(), enabled=os.environ.get("NEUROCORE_DESKTOP") == "1")
    app.state.desktop = manager
    app.state.desktop_token = token
    app.add_middleware(_ProductBoundary, token=token)

    def call(method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except (DesktopError, XavierServiceError) as exc:
            raise HTTPException(status_code=exc.http_status, detail=exc.to_dict()) from None
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail={"code": "invalid_input", "message": "Check the session request fields"}) from None

    def checked(payload, required, optional=()):
        if not isinstance(payload, dict) or set(payload) - set(required) - set(optional) or not set(required) <= set(payload):
            raise HTTPException(status_code=422, detail={"code": "invalid_input", "message": "Unexpected session request fields"})
        return payload

    @app.get("/api/workspace/bootstrap")
    def bootstrap():
        return JSONResponse({"token": token, **call(app.state.desktop.status)}, headers={"Cache-Control": "no-store"})

    @app.get("/api/workspace/status")
    def status():
        return call(app.state.desktop.status)

    @app.get("/api/workspace/history")
    def history():
        return call(app.state.desktop.history)

    @app.post("/api/workspace/start")
    def start(payload: dict[str, Any]):
        checked(payload, ("goal",), ("model_id",))
        return call(app.state.desktop.start, payload["goal"], model_id=payload.get("model_id"))

    @app.post("/api/workspace/capture")
    def capture(payload: dict[str, Any]):
        checked(payload, ("session_id",), ("delay_seconds",))
        return call(app.state.desktop.capture, payload["session_id"], delay_seconds=payload.get("delay_seconds", 3))

    @app.post("/api/workspace/propose")
    def propose(payload: dict[str, Any]):
        checked(payload, ("session_id", "eeg"))
        return call(app.state.desktop.propose, payload["session_id"], payload["eeg"])

    @app.post("/api/workspace/apply")
    def apply(payload: dict[str, Any]):
        checked(payload, ("session_id", "proposal_id"), ("delay_seconds",))
        return call(app.state.desktop.apply, payload["session_id"], payload["proposal_id"], delay_seconds=payload.get("delay_seconds", 3))

    @app.post("/api/workspace/stop")
    def stop(payload: dict[str, Any]):
        checked(payload, ("session_id",))
        return call(app.state.desktop.stop, payload["session_id"])

    directory = ui_directory()
    if (directory / "index.html").is_file():
        if (directory / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=directory / "assets"), name="product-assets")

        @app.get("/", include_in_schema=False)
        def home():
            return FileResponse(directory / "index.html", headers={"Cache-Control": "no-cache"})
    else:
        @app.get("/", include_in_schema=False)
        def unavailable():
            return {"service": "NeuroCore", "ui": "not_built", "help": "Run python scripts/build_product.py, then neurocore serve"}


def cmd_serve(args) -> int:
    """Run UI and API on the same loopback port; GPU service remains on SMAC."""
    from .xavier.remote import validate_remote_url
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in 1..65535")
    if args.remote_model:
        remote = validate_remote_url(args.remote_model)
        if urlsplit(remote).port == args.port:
            raise ValueError("The model tunnel and local application must use different ports")
        os.environ["NEUROCORE_REMOTE_MODEL"] = remote
    else:
        os.environ.pop("NEUROCORE_REMOTE_MODEL", None)
    if args.registry:
        os.environ["XAVIER_MODEL_REGISTRY"] = str(args.registry.expanduser().resolve(strict=True))
    os.environ["NEUROCORE_DESKTOP"] = "1" if args.desktop else "0"
    if not args.api_only and not (ui_directory() / "index.html").is_file():
        print("The application UI is not built. Run python scripts/build_product.py or use --api-only.")
        return 1
    import uvicorn
    print(f"NeuroCore: http://127.0.0.1:{args.port}")
    uvicorn.run("neurocore.api:app", host="127.0.0.1", port=args.port, workers=1, access_log=False)
    return 0
