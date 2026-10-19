"""Minimal Chrome DevTools Protocol driver (Python; no npm deps).

The bundled Node runtime cannot reach 127.0.0.1:9223 on this machine
(ECONNREFUSED while curl and Python connect fine -- a sandbox network policy),
so screenshots and JS evaluation go through this script instead.

    python app/tools/cdp.py 9223 shot  app/screenshots/x.png --wait 2000
    python app/tools/cdp.py 9223 eval  "document.title"
    python app/tools/cdp.py 9223 click "#srSeg button[data-sr='ultrasharp']"
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import urllib.request

import websockets


def _targets(port: int) -> list[dict]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{port}/json/list", timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


async def _run(port: int, action: str, arg: str, wait: int) -> int:
    pages = [t for t in _targets(port) if t.get("type") == "page"]
    if not pages:
        print("no page target", file=sys.stderr)
        return 2

    async with websockets.connect(pages[0]["webSocketDebuggerUrl"],
                                  max_size=64 * 1024 * 1024) as ws:
        seq = 0
        pending: dict[int, asyncio.Future] = {}

        async def reader() -> None:
            async for raw in ws:
                msg = json.loads(raw)
                fut = pending.pop(msg.get("id"), None)
                if fut and not fut.done():
                    fut.set_result(msg)

        task = asyncio.create_task(reader())

        async def send(method: str, params: dict | None = None) -> dict:
            nonlocal seq
            seq += 1
            fut = asyncio.get_running_loop().create_future()
            pending[seq] = fut
            await ws.send(json.dumps({"id": seq, "method": method,
                                      "params": params or {}}))
            return await fut

        await send("Page.enable")
        await send("Runtime.enable")

        if action == "shot":
            await asyncio.sleep(wait / 1000)
            res = await send("Page.captureScreenshot", {"format": "png"})
            data = res.get("result", {}).get("data", "")
            if not data:
                print("screenshot failed: " + json.dumps(res)[:300], file=sys.stderr)
                return 3
            with open(arg, "wb") as fh:
                fh.write(base64.b64decode(data))
            print(f"saved {arg}")
        elif action == "eval":
            res = await send("Runtime.evaluate", {
                "expression": arg, "returnByValue": True, "awaitPromise": True})
            print(json.dumps(res.get("result", {}).get("result", {}), ensure_ascii=False))
        elif action == "click":
            js = (f"(()=>{{const el=document.querySelector({json.dumps(arg)});"
                  f"if(!el)return 'not found';el.click();return 'clicked';}})()")
            res = await send("Runtime.evaluate", {"expression": js, "returnByValue": True})
            print(json.dumps(res.get("result", {}).get("result", {}), ensure_ascii=False))
        else:
            print(f"unknown action {action}", file=sys.stderr)
            return 2

        task.cancel()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("action", choices=["shot", "eval", "click"])
    ap.add_argument("arg", nargs="?", default="")
    ap.add_argument("--wait", type=int, default=1200)
    args = ap.parse_args()
    return asyncio.run(_run(args.port, args.action, args.arg, args.wait))


if __name__ == "__main__":
    raise SystemExit(main())
