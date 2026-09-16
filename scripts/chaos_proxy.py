"""A TCP proxy that sits between one or more workers and the coordinator so a test can break *the connection*
itself, not the processes: cut it, add latency, or throttle it, per worker, on command.

    python3 scripts/chaos_proxy.py --upstream 127.0.0.1:8000 --via w1=8001 --via w2=8002 --control 8100
    python3 -m dgpt.worker --name w1 --coordinator http://127.0.0.1:8001     # w1 talks through the proxy

Control API (plain HTTP GET, so curl works):
    /set?name=w1&mode=cut          drop every open connection of w1 and refuse new ones (connection reset)
    /set?name=w1&mode=normal       restore
    /set?name=w1&lag_ms=800        delay every chunk in both directions by 800 ms (a slow, high-latency link)
    /set?name=w1&rate_kb_s=200     cap w1's throughput at 200 kB/s (a 43 MB download then takes ~3.5 min)
    /status                        current mode per worker and live connection counts

Signed requests pass through untouched: the signature covers method, path, timestamp, nonce and body, not the
host, so the proxy is transparent to the auth layer. scripts/chaos.py drives this with --cut/--lag/--throttle.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CHUNK = 64 * 1024


class Link:
    """State for one worker's port."""
    def __init__(self, name: str, port: int):
        self.name, self.port = name, port
        self.mode = "normal"
        self.lag_ms = 0.0
        self.rate_kb_s = 0.0
        self.conns: set[asyncio.StreamWriter] = set()
        self.bytes_up = self.bytes_down = 0
        self.cuts = 0

    def status(self) -> dict:
        return {"port": self.port, "mode": self.mode, "lag_ms": self.lag_ms, "rate_kb_s": self.rate_kb_s,
                "open_connections": len(self.conns), "bytes_up": self.bytes_up, "bytes_down": self.bytes_down, "cuts": self.cuts}


class Proxy:
    def __init__(self, upstream: tuple[str, int], links: dict[str, Link]):
        self.upstream, self.links = upstream, links
        self.loop: asyncio.AbstractEventLoop | None = None

    # ---- data path -------------------------------------------------------------------------
    async def pump(self, link: Link, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, direction: str):
        try:
            while True:
                data = await reader.read(CHUNK)
                if not data:
                    break
                if link.mode == "cut":
                    break
                if link.lag_ms:
                    await asyncio.sleep(link.lag_ms / 1000)
                if link.rate_kb_s:
                    await asyncio.sleep(len(data) / (link.rate_kb_s * 1000))
                writer.write(data)
                await writer.drain()
                if direction == "up":
                    link.bytes_up += len(data)
                else:
                    link.bytes_down += len(data)
        except (ConnectionError, asyncio.CancelledError, OSError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def handle(self, link: Link, client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter):
        if link.mode == "cut":
            client_w.close()                     # refuse: the worker sees a reset / connection error
            return
        try:
            up_r, up_w = await asyncio.open_connection(*self.upstream)
        except OSError:
            client_w.close()
            return
        link.conns.add(client_w); link.conns.add(up_w)
        try:
            await asyncio.gather(self.pump(link, client_r, up_w, "up"), self.pump(link, up_r, client_w, "down"))
        finally:
            link.conns.discard(client_w); link.conns.discard(up_w)

    def cut(self, link: Link):
        """Close every live connection of this worker (called from the control thread)."""
        def _do():
            for w in list(link.conns):
                try:
                    w.transport.abort()          # RST, not FIN: the worker's in-flight request fails immediately
                except Exception:
                    pass
            link.conns.clear()
        link.cuts += 1
        if self.loop:
            self.loop.call_soon_threadsafe(_do)

    async def serve(self):
        self.loop = asyncio.get_running_loop()
        servers = []
        for link in self.links.values():
            srv = await asyncio.start_server(lambda r, w, l=link: self.handle(l, r, w), "127.0.0.1", link.port)
            servers.append(srv)
            print(f"[proxy] {link.name}: 127.0.0.1:{link.port} -> {self.upstream[0]}:{self.upstream[1]}", flush=True)
        await asyncio.gather(*(s.serve_forever() for s in servers))


# ---- control API ---------------------------------------------------------------------------

def control_server(proxy: Proxy, port: int) -> ThreadingHTTPServer:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/status":
                return self._json(200, {n: l.status() for n, l in proxy.links.items()})
            if u.path != "/set":
                return self._json(404, {"error": "unknown path"})
            link = proxy.links.get(q.get("name", ""))
            if link is None:
                return self._json(404, {"error": f"unknown worker {q.get('name')!r}", "known": list(proxy.links)})
            if "mode" in q:
                if q["mode"] not in ("normal", "cut"):
                    return self._json(400, {"error": "mode must be normal|cut"})
                link.mode = q["mode"]
                if link.mode == "cut":
                    proxy.cut(link)
            if "lag_ms" in q:
                link.lag_ms = float(q["lag_ms"])
            if "rate_kb_s" in q:
                link.rate_kb_s = float(q["rate_kb_s"])
            print(f"[proxy] {link.name}: {link.status()}", flush=True)
            return self._json(200, link.status())

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True, name="proxy-control").start()
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", default="127.0.0.1:8000", help="the real coordinator host:port")
    ap.add_argument("--via", action="append", default=[], metavar="NAME=PORT", required=True,
                    help="listen on PORT for worker NAME (repeat per worker)")
    ap.add_argument("--control", type=int, default=8100, help="control API port (localhost only)")
    args = ap.parse_args(argv)
    host, port = args.upstream.rsplit(":", 1)
    links = {}
    for spec in args.via:
        name, p = spec.split("=", 1); links[name] = Link(name, int(p))
    proxy = Proxy((host, int(port)), links)
    control_server(proxy, args.control)
    print(f"[proxy] control API on http://127.0.0.1:{args.control}  (/status, /set?name=..&mode=cut|normal&lag_ms=..&rate_kb_s=..)", flush=True)
    try:
        asyncio.run(proxy.serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
