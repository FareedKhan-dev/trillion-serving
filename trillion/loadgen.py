"""loadgen.py: holds a fixed number of streams open against the router and times every token they get back."""
import asyncio
import json
import random
import sys
import time

import aiohttp

import config as cfg

NAMES = [name for name, c in cfg.CLASSES.items() for _ in range(round(c[0] * 1000))]   # 1,000 slots in class shares
TTFT, GAPS = {name: [] for name in cfg.CLASSES}, {name: [0] * 1000 for name in cfg.CLASSES}   # gaps in 1 ms buckets


def prompt(session, turn, name):
    """A prompt of the class's length in 512-token blocks: the session's cached blocks first, then new ones."""
    _, length, cached, out, _ = cfg.CLASSES[name]
    keys = [f"{session}:{b}" for b in range(cached // 512)] + [f"{session}:{turn}:{b}" for b in range((length - cached) // 512)]
    text = " ".join(" ".join(random.Random(k).choices("the of and to in is was for on that".split(), k=512)) for k in keys)
    return {"model": cfg.MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": out, "stream": True}


async def stream(http, url, name, slot, steady):
    """One stream of one class, with a new session every 256 turns."""
    await asyncio.sleep(random.uniform(0, 600))                # the streams join over a 10 minute ramp
    for turn in range(10**9):
        session, t0, last = f"{name}-{slot}-{turn // 256}", time.perf_counter(), None
        counted = time.time() >= steady                        # only requests sent after the ramp are timed
        head = {"x-request-class": name, "x-session-id": session}
        async with http.post(url, json=prompt(session, turn, name), headers=head) as r:
            async for line in r.content:
                if not line.startswith(b"data: {") or not counted:
                    continue                                   # one server-sent event per token
                now = time.perf_counter()
                if last is None:
                    TTFT[name].append(now - t0)
                else:
                    GAPS[name][min(int((now - last) * 1000), 999)] += 1
                last = now


async def main(url, streams, first, seconds):
    """Hold this process's streams through the ramp and the window, then write the TTFTs and the gap histogram."""
    steady = time.time() + 600
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http:
        tasks = [asyncio.create_task(stream(http, url, NAMES[s % 1000], s, steady)) for s in range(first, first + streams)]
        await asyncio.sleep(600 + seconds)
        for task in tasks:
            task.cancel()
    with open(f"loadgen-{first}.json", "w") as f:
        json.dump({"ttft": TTFT, "gaps": GAPS}, f)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])))
