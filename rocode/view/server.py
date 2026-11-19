"""A small local web server for the dashboard. Standard library only.

It only reads the runs directory, plus one write path: operator controls for the episode running in this
process (`rocode run --view`). It binds to 127.0.0.1, and control requests must carry a token that is
embedded in the page it serves, so other web pages cannot drive the robot through it.
"""

from __future__ import annotations

import json
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..core.control import ControlChannel
from .compare import BATCH_ID, compare, load_manifests
from .episodes import IMAGE_NAME, episode_dir, list_episodes, load_episode

STATIC = {"app.js": "text/javascript; charset=utf-8", "app.css": "text/css; charset=utf-8"}
MAX_SAY = 2000


def _asset(name: str) -> bytes:
    return resources.files("rocode.view").joinpath("static", name).read_bytes()


class ViewerServer:
    def __init__(self, runs_root: Path, host: str = "127.0.0.1", port: int = 8765,
                 control: ControlChannel | None = None, live_id: str | None = None):
        self.root = Path(runs_root)
        self.control = control
        self.live_id = live_id
        self.token = secrets.token_urlsafe(24)
        self.httpd = self._bind(host, port)
        self._thread: threading.Thread | None = None

    def _bind(self, host: str, port: int) -> ThreadingHTTPServer:
        handler = _make_handler(self)
        for candidate in [port, *range(port + 1, port + 20), 0] if port else [0]:
            try:
                httpd = ThreadingHTTPServer((host, candidate), handler)
                httpd.daemon_threads = True
                return httpd
            except OSError:
                continue
        raise OSError(f"no free port near {port}")

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}/"

    def start(self) -> "ViewerServer":
        """Serve from a background thread (the episode loop keeps the main thread)."""
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="rocode-viewer", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    # ---- request handling -------------------------------------------------------------------

    def controllable(self, episode_id: str) -> bool:
        return self.control is not None and episode_id == self.live_id

    def handle_control(self, body: dict) -> tuple[int, dict]:
        episode_id, action = body.get("episode"), body.get("action")
        if not self.controllable(episode_id):
            return HTTPStatus.CONFLICT, {"error": "Controls work only for the episode this process is running."}
        if action == "say":
            text = str(body.get("text", "")).strip()
            if not text:
                return HTTPStatus.BAD_REQUEST, {"error": "Nothing to say."}
            self.control.say(text[:MAX_SAY])
        elif action == "pause":
            self.control.pause()
        elif action == "resume":
            self.control.resume()
        elif action == "stop":
            self.control.estop("dashboard")
        else:
            return HTTPStatus.BAD_REQUEST, {"error": f"Unknown action {action!r}."}
        return HTTPStatus.OK, {"ok": True, "paused": self.control.paused}


def _make_handler(viewer: ViewerServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "rocode-viewer"

        def log_message(self, *args):  # keep the terminal for the episode transcript
            pass

        def _send(self, status, body: bytes, ctype: str, cache: str = "no-store"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload, default=str).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            path = unquote(url.path)
            if path == "/api/compare":
                batch = (parse_qs(url.query).get("batch") or [""])[0] or None
                if batch is not None and not BATCH_ID.match(batch):
                    return self._json(HTTPStatus.BAD_REQUEST, {"error": "Invalid batch id."})
                return self._json(HTTPStatus.OK, compare(list_episodes(viewer.root), load_manifests(viewer.root),
                                                         batch))
            if path in ("/", "/index.html"):
                page = _asset("index.html").decode()
                page = page.replace("{{TOKEN}}", viewer.token).replace("{{LIVE}}", viewer.live_id or "")
                return self._send(HTTPStatus.OK, page.encode(), "text/html; charset=utf-8")
            if path.startswith("/static/") and path[8:] in STATIC:
                return self._send(HTTPStatus.OK, _asset(path[8:]), STATIC[path[8:]])
            if path == "/api/episodes":
                return self._json(HTTPStatus.OK, {"episodes": list_episodes(viewer.root), "live": viewer.live_id,
                                                  "root": str(viewer.root)})
            if path.startswith("/api/episodes/"):
                ep = load_episode(viewer.root, path[len("/api/episodes/"):])
                if ep is None:
                    return self._json(HTTPStatus.NOT_FOUND, {"error": "No such episode."})
                ep["controllable"] = viewer.controllable(ep["id"])
                ep["paused"] = bool(viewer.control and ep["controllable"] and viewer.control.paused)
                return self._json(HTTPStatus.OK, ep)
            if path.startswith("/files/"):
                parts = path.split("/")
                if len(parts) == 5 and parts[3] == "observations" and IMAGE_NAME.match(parts[4]):
                    d = episode_dir(viewer.root, parts[2])
                    f = d / "observations" / parts[4] if d else None
                    if f is not None and f.is_file():
                        return self._send(HTTPStatus.OK, f.read_bytes(), "image/png", "max-age=31536000, immutable")
                return self._json(HTTPStatus.NOT_FOUND, {"error": "No such file."})
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/api/control":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            if not secrets.compare_digest(self.headers.get("X-Rocode-Token", ""), viewer.token):
                return self._json(HTTPStatus.FORBIDDEN, {"error": "Missing or wrong token."})
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 16384)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "Body must be JSON."})
            status, payload = viewer.handle_control(body if isinstance(body, dict) else {})
            return self._json(status, payload)

    return Handler
