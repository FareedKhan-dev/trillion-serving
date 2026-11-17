"""router.py: the fleet router, pins sessions to prefill racks, holds each class to its budget, picks a decode rack."""
import asyncio
import bisect
import hashlib
import random

import aiohttp
from aiohttp import web

import config as cfg

REPLICAS = 24                                                   # router processes, each holds 1/24 of every budget
PREFILL = [f"http://prefill-{i}:30000" for i in range(13)]      # one SGLang prefill server per rack
DECODE = [f"http://decode-{i}:30100" for i in range(8)]
POOLS = {"chat": DECODE[:3], "doc": DECODE[3:4], "agent": DECODE[4:], "long": DECODE[4:], "max": DECODE[4:]}
BUDGET = {name: int(1.03 * 100_000 * c[0] / REPLICAS) for name, c in cfg.CLASSES.items()}   # streams per class
HELD, FILL, DOWN = dict.fromkeys(BUDGET, 0), dict.fromkeys(PREFILL + DECODE, 0.0), set()
RING = sorted((int(hashlib.sha1(f"{url}#{v}".encode()).hexdigest(), 16), url) for url in PREFILL for v in range(64))


def prefill_for(session):
    """The session's prefill rack, or the next distinct one when that rack is down."""
    k = bisect.bisect(RING, (int(hashlib.sha1(session.encode()).hexdigest(), 16), ""))
    walk = [url for _, url in RING[k:] + RING[:k]]
    primary = walk[0]
    secondary = next(url for url in walk if url != primary)     # it holds none of the cache, so it re-prefills
    return secondary if primary in DOWN else primary


async def scrape(http):
    """Once a second, read every rack's fill: the fuller of its KV pool and its KDA state pool."""
    while True:
        for url in FILL:
            try:
                async with http.get(f"{url}/metrics") as r:
                    lines = (await r.text()).splitlines()
                FILL[url] = max(float(s.split()[-1]) for s in lines if s.startswith(("sglang:token_usage", "sglang:mamba_usage")))
                DOWN.discard(url)
            except (aiohttp.ClientError, ValueError):           # no answer, or an answer without pool figures
                DOWN.add(url)
        await asyncio.sleep(1)


async def handle(request):
    """Admit the request, pair its two racks and stream the decode side back."""
    body, name = await request.json(), request.headers.get("x-request-class", "chat")
    live = [u for u in POOLS[name] if u not in DOWN] or POOLS[name]   # a rack that stopped answering takes nothing
    while HELD[name] >= BUDGET[name] or min(FILL[u] for u in live) >= cfg.POOL_FILL:
        await asyncio.sleep(0.05)                              # over budget or every rack full: the request waits here
    HELD[name] += 1
    prefill, decode = prefill_for(request.headers.get("x-session-id", "")), min(live, key=FILL.get)
    body |= {"bootstrap_host": prefill.split("//")[1].split(":")[0], "bootstrap_port": 8998,
             "bootstrap_room": random.getrandbits(63)}
    http, out = request.app["http"], web.StreamResponse()
    try:
        await out.prepare(request)
        first = asyncio.ensure_future(http.post(f"{prefill}/v1/chat/completions", json=body))
        async with http.post(f"{decode}/v1/chat/completions", json=body) as r:
            async for chunk in r.content.iter_any():           # tokens leave as soon as decode writes them
                await out.write(chunk)
        (await first).release()
    finally:
        HELD[name] -= 1
    return out


async def start(app):
    app["http"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=2))
    app["scrape"] = asyncio.create_task(scrape(app["http"]))


def app(argv=None):
    """The aiohttp app: python -m aiohttp.web -H 0.0.0.0 -P 8000 router:app"""
    web_app = web.Application()
    web_app.on_startup.append(start)
    web_app.router.add_post("/v1/chat/completions", handle)
    return web_app
