"""Log per-node execution times of prompts submitted with client_id kernel-ab (ComfyUI ws events)."""
import asyncio, json, sys, time
import aiohttp

GRAPH = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
async def main():
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect("http://127.0.0.1:8188/ws?clientId=kernel-ab", max_msg_size=0) as ws:
            cur, t0, start = None, None, None
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                m = json.loads(msg.data); typ, d = m.get("type"), m.get("data", {})
                now = time.time()
                if typ == "execution_start":
                    start = now; print(f"--- prompt {d.get('prompt_id','')[:8]} start", flush=True)
                if typ == "executing":
                    if cur is not None:
                        ct = GRAPH.get(cur, {}).get("class_type", "?")
                        print(f"{now - t0:8.1f} s  node {cur:>3} {ct}", flush=True)
                    cur, t0 = d.get("node"), now
                    if cur is None and start:
                        print(f"--- total {now - start:.1f} s", flush=True); start = None
asyncio.run(main())
