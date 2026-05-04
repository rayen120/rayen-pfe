import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")
EXCEPTIONS_FILE = os.path.join(DASHBOARD_DIR, "data", "exceptions.json")
PORT_FILE = os.path.join(DASHBOARD_DIR, "data", ".dashboard_port")
# Prefer project asset, then env, then legacy Downloads path.
HERO_CANDIDATES = [
    os.path.join(DASHBOARD_DIR, "assets", "hero.png"),
    os.environ.get("HIFI_DASHBOARD_HERO", "").strip(),
    os.path.join(BASE_DIR, "assets", "hero.png"),
    r"C:\Users\espace info\Downloads\assembly line 5.png",
]
HOST = "127.0.0.1"
PORT_FIRST = 8000
PORT_LAST = 8010


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def log_message(self, format, *args):
        return

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/assets/hero.png":
            body = b""
            for candidate in HERO_CANDIDATES:
                if not candidate or not os.path.isfile(candidate):
                    continue
                try:
                    with open(candidate, "rb") as f:
                        body = f.read()
                    break
                except OSError:
                    body = b""
            if body:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                # Tiny transparent PNG so the page still loads without a broken image.
                body = (
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
                    b"\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
                )
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return
        if parsed.path == "/api/exceptions":
            try:
                with open(EXCEPTIONS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {"events": []}
            except Exception:
                data = {"events": []}
            self._send_json(data, status=200)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        return super().do_GET()


def _write_port_file(port):
    try:
        os.makedirs(os.path.dirname(PORT_FILE), exist_ok=True)
        with open(PORT_FILE, "w", encoding="utf-8") as f:
            f.write(str(int(port)))
    except OSError:
        pass


def main():
    os.makedirs(os.path.dirname(EXCEPTIONS_FILE), exist_ok=True)
    if not os.path.isfile(EXCEPTIONS_FILE):
        with open(EXCEPTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump({"events": []}, f, ensure_ascii=True, indent=2)
    server = None
    port_used = None
    for port in range(PORT_FIRST, PORT_LAST + 1):
        try:
            server = ThreadingHTTPServer((HOST, port), DashboardHandler)
            port_used = port
            break
        except OSError:
            continue
    if server is None:
        print("HIFI dashboard: no free port in range {}-{}.".format(PORT_FIRST, PORT_LAST), file=sys.stderr)
        sys.exit(1)
    _write_port_file(port_used)
    url = "http://{}:{}/".format(HOST, port_used)
    print("HIFI dashboard local server: {}".format(url), flush=True)
    print("Open that URL in your browser (do not open index.html as a file).", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if os.path.isfile(PORT_FILE):
                os.remove(PORT_FILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
