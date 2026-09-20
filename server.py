"""
server.py — punto de entrada único para Render (y para local).

Junta los 4 backends de Teorema en UN solo servicio / UNA sola URL, sin modificar
ninguno de sus archivos:

    /m0/...        -> modulo_0/Back0.py                    (Módulo 0)
    /fac/...       -> factorizadora/Factorizadora_Back.py  (Flask: POST /fac/factorizar)
    /graf/...      -> graficadora/Graficadora_Back.py      (FastAPI)
    todo lo demás  -> modulo_1/backend1.py                 (Módulo 1: /api/options, /api/exam, ...)
    /  y  /health  -> estado de cada módulo

Cómo funciona: Back0, backend1 y la app Flask se levantan en puertos internos de
127.0.0.1 (no expuestos) con sus propias clases/apps, y este servidor les reenvía
las peticiones tal cual. La Graficadora ya es ASGI, así que se monta directo.
Si un módulo falla al importarse, los demás siguen funcionando y ese responde 503.

Local:   python server.py                 -> http://127.0.0.1:8000
Render:  Start Command: python server.py  (lee PORT y escucha en 0.0.0.0)
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import traceback
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent
UPSTREAM_TIMEOUT = 120  # segundos que se espera a un backend interno
FORWARD_HEADERS = ("content-type", "x-api-key", "authorization", "accept")

modules: dict[str, str] = {}    # id -> "ok" | "error: <motivo>"
upstreams: dict[str, str] = {}  # id -> "http://127.0.0.1:<puerto interno>"

logging.getLogger("werkzeug").setLevel(logging.WARNING)


# ── Carga de módulos ────────────────────────────────────────────────────────

def _load(module_name: str, relpath: str):
    """Importa un .py por ruta (sin depender de sys.path ni de que las carpetas sean paquetes)."""
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # necesario para @dataclass con "from __future__ import annotations"
    spec.loader.exec_module(module)
    return module


def _serve_http(key: str, handler_cls) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)  # puerto 0 = uno libre, solo local
    upstreams[key] = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True, name=f"{key}-http").start()


def _serve_wsgi(key: str, wsgi_app) -> None:
    from werkzeug.serving import make_server

    server = make_server("127.0.0.1", 0, wsgi_app, threaded=True)
    upstreams[key] = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, daemon=True, name=f"{key}-wsgi").start()


def _boot(key: str, module_name: str, relpath: str, start) -> None:
    try:
        start(_load(module_name, relpath))
        modules[key] = "ok"
    except Exception as exc:  # un módulo roto no tumba a los demás
        modules[key] = f"error: {type(exc).__name__}: {exc}"
        traceback.print_exc()


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_boot("m1", "backend1", "modulo_1/backend1.py", lambda m: _serve_http("m1", m.Handler))
_boot("m0", "Back0", "modulo_0/Back0.py", lambda m: _serve_http("m0", m.Handler))
_boot("fac", "Factorizadora_Back", "factorizadora/Factorizadora_Back.py", lambda m: _serve_wsgi("fac", m.app))
# El mount de la Graficadora debe registrarse ANTES que las rutas de abajo (el orden importa).
_boot("graf", "Graficadora_Back", "graficadora/Graficadora_Back.py", lambda m: app.mount("/graf", m.app))


# ── Reenvío a los backends internos ─────────────────────────────────────────

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # ignora HTTP(S)_PROXY: es tráfico local


def _error(message: str, status: int) -> Response:
    # Un solo JSON que entienden los 4 frontends (success / ok / status+data / error).
    body = {"success": False, "ok": False, "status": "error", "error": message, "data": message}
    return Response(json.dumps(body, ensure_ascii=False), status, media_type="application/json")


def _forward(method: str, url: str, body: bytes, headers: dict[str, str]):
    req = urllib.request.Request(url, data=body or None, method=method, headers=headers)
    try:
        with _opener.open(req, timeout=UPSTREAM_TIMEOUT) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as exc:  # 4xx/5xx del backend: se devuelven tal cual
        return exc.code, exc.headers, exc.read()


async def _proxy(key: str, path: str, request: Request) -> Response:
    base = upstreams.get(key)
    if base is None:
        return _error(f"El módulo '{key}' no está disponible ({modules.get(key, 'no iniciado')}).", 503)
    url = f"{base}/{quote(path)}" + (f"?{request.url.query}" if request.url.query else "")
    headers = {name: request.headers[name] for name in FORWARD_HEADERS if name in request.headers}
    body = await request.body()
    try:
        status, resp_headers, data = await run_in_threadpool(_forward, request.method, url, body, headers)
    except Exception as exc:
        return _error(f"No se pudo contactar al módulo '{key}': {exc}", 502)
    out = {}
    if resp_headers.get("Content-Disposition"):  # nombre del PDF descargado
        out["Content-Disposition"] = resp_headers["Content-Disposition"]
    return Response(data, status, out, resp_headers.get("Content-Type"))


# ── Rutas (de la más específica a la más general) ───────────────────────────

@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": all(v == "ok" for v in modules.values()), "modules": modules}


@app.api_route("/m0/{path:path}", methods=["GET", "POST"])
async def modulo_0(path: str, request: Request):
    return await _proxy("m0", path, request)


@app.api_route("/fac/{path:path}", methods=["GET", "POST"])
async def factorizadora(path: str, request: Request):
    return await _proxy("fac", path, request)


@app.api_route("/graf/{path:path}", methods=["GET", "POST"])  # solo se alcanza si la Graficadora no se pudo montar
async def graficadora_no_disponible(path: str):
    return _error(f"El módulo 'graf' no está disponible ({modules.get('graf', 'no iniciado')}).", 503)


@app.api_route("/{path:path}", methods=["GET", "POST"])  # Módulo 1 vive en la raíz (/api/...)
async def modulo_1(path: str, request: Request):
    return await _proxy("m1", path, request)


if __name__ == "__main__":
    import uvicorn

    print("Módulos:", modules, flush=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
