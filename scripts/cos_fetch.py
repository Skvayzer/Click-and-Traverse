#!/usr/bin/env python3
"""Download a public Hugging Face dataset repo onto Lustre (run inside a compute job; resumable, skips complete files).

  python scripts/cos_fetch.py --repo ember-lab-berkeley/AMASS_Retargeted_for_G1 --out DIR [--prefix g1/] [--threads 16]
"""
import argparse
import json
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HF = "https://huggingface.co"


def files(repo):
    out, url = [], f"{HF}/api/datasets/{repo}/tree/main?recursive=true"
    while url:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        out += [x for x in r.json() if x["type"] == "file"]
        url = r.links.get("next", {}).get("url")
    return [(x["path"], x.get("size", 0)) for x in out]


def get(repo, path, size, out):
    dst = Path(out) / path
    if dst.exists() and (size == 0 or dst.stat().st_size == size):
        return "skip", path
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF}/datasets/{repo}/resolve/main/{urllib.parse.quote(path)}"
    tmp = dst.with_name(dst.name + ".part")
    err = None
    for attempt in range(5):
        try:
            done = tmp.stat().st_size if tmp.exists() else 0
            hdr = {"Range": f"bytes={done}-"} if done else {}
            with requests.get(url, headers=hdr, stream=True, timeout=120) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                with open(tmp, "ab" if done and r.status_code == 206 else "wb") as f:
                    for b in r.iter_content(1 << 22):
                        f.write(b)
            break
        except Exception as e:  # noqa: BLE001 - retry, then report
            err = e
            time.sleep(5 * (attempt + 1))
    else:
        return "error", f"{path}: {err}"
    tmp.rename(dst)
    return "ok", path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    F = [(p, s) for p, s in files(a.repo) if p.startswith(a.prefix) or p in ("README.md", "LICENSE", "dataset_infos.json")]
    print(f"{a.repo}: {len(F)} files, {sum(s for _, s in F) / 1e9:.2f} GB", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "_files.json").write_text(json.dumps(F))
    t, n = time.time(), {}
    with ThreadPoolExecutor(a.threads) as ex:
        for k, (st, msg) in enumerate(ex.map(lambda x: get(a.repo, x[0], x[1], a.out), F)):
            n[st] = n.get(st, 0) + 1
            if st == "error":
                print("ERROR", msg, flush=True)
            if k % 500 == 0:
                print(k, n, f"{time.time() - t:.0f}s", flush=True)
    print("FETCH DONE", a.repo, n, f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
