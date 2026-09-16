#!/usr/bin/env python3
"""Fetch reproducible CC0 Poly Haven furniture and floor assets for rendering.

This only writes render assets; it does not modify training obstacle fields.
Uses the official API's exact download URLs, sizes and MD5 checksums, and records
SHA256 hashes and provenance in manifest.json. Existing files are reused only
after their size and checksum have been checked against the current API.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import urllib.parse
import urllib.request


API_ROOT = "https://api.polyhaven.com"
USER_AGENT = "CAT realistic scene renderer (research; CC0 asset fetcher)"
LICENSE_URL = "https://polyhaven.com/license"
MODELS = ("wooden_table_02", "SchoolChair_01")
FLOOR_ASSET = "floor_tiles_02"
FLOOR_MAPS = {
    "base_color": "Diffuse",
    "normal_gl": "nor_gl",
    "roughness": "Rough",
    "displacement": "Displacement",
}


def safe_target(root: Path, relative: str) -> Path:
    """Reject traversal, absolute paths and destinations outside the asset root."""
    parts = PurePosixPath(relative)
    if not relative or parts.is_absolute() or ".." in parts.parts or "\\" in relative:
        raise ValueError(f"Unsafe asset path: {relative!r}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes download directory: {relative!r}")
    return target


def request_bytes(url: str, *, timeout: float, max_bytes: int) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "api.polyhaven.com", "dl.polyhaven.org"
    }:
        raise ValueError(f"Unexpected asset URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Response exceeds allowed size: {url}")
    return data


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def verified(data: bytes, spec: dict) -> bool:
    return len(data) == spec["size"] and hashlib.md5(data).hexdigest() == spec["md5"]


def download_file(root: Path, relative: str, spec: dict, timeout: float) -> dict:
    path = safe_target(root, relative)
    if path.is_file() and path.stat().st_size == spec["size"]:
        data = path.read_bytes()
    else:
        data = b""
    reused = verified(data, spec)
    if not reused:
        data = request_bytes(spec["url"], timeout=timeout, max_bytes=spec["size"])
        if not verified(data, spec):
            raise ValueError(f"Download failed size/MD5 verification: {spec['url']}")
        atomic_write(path, data)
    print(f"{'Verified existing' if reused else 'Downloaded'} {relative} ({len(data):,} bytes)", flush=True)
    return {
        "path": relative,
        "url": spec["url"],
        "size_bytes": len(data),
        "md5": spec["md5"],
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def fetch_api(root: Path, asset: str, endpoint: str, timeout: float) -> tuple[dict, dict]:
    url = f"{API_ROOT}/{endpoint}/{asset}"
    data = request_bytes(url, timeout=timeout, max_bytes=5_000_000)
    document = json.loads(data)
    relative = f"{asset}/api_{endpoint}.json"
    atomic_write(safe_target(root, relative), data)
    return document, {"path": relative, "url": url, "sha256": hashlib.sha256(data).hexdigest()}


def fetch_asset(root: Path, asset: str, resolution: str, timeout: float, model: bool) -> dict:
    files, files_source = fetch_api(root, asset, "files", timeout)
    info, info_source = fetch_api(root, asset, "info", timeout)
    record = {
        "asset_id": asset,
        "name": info["name"],
        "kind": "model" if model else "texture",
        "resolution": resolution,
        "source_page": f"https://polyhaven.com/a/{asset}",
        "license": "CC0-1.0",
        "license_url": LICENSE_URL,
        "authors": info.get("authors", {}),
        "api_sources": [files_source, info_source],
        "physical_dimensions_mm": info.get("dimensions"),
        "files": [],
    }
    if model:
        gltf = files["gltf"][resolution]["gltf"]
        basename = PurePosixPath(urllib.parse.urlparse(gltf["url"]).path).name
        record["main_path"] = f"{asset}/{basename}"
        selections = [(record["main_path"], gltf)] + [
            (f"{asset}/{relative}", spec)
            for relative, spec in gltf.get("include", {}).items()
        ]
    else:
        selections = []
        record["texture_paths"] = {}
        for usage, api_key in FLOOR_MAPS.items():
            spec = files[api_key][resolution]["jpg"]
            basename = PurePosixPath(urllib.parse.urlparse(spec["url"]).path).name
            relative = f"{asset}/{basename}"
            record["texture_paths"][usage] = relative
            selections.append((relative, spec))
    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(download_file, root, relative, spec, timeout) for relative, spec in selections]
        record["files"] = [task.result() for task in tasks]
    if model:
        gltf_path = safe_target(root, record["main_path"])
        document = json.loads(gltf_path.read_bytes())
        for entry in document.get("buffers", []) + document.get("images", []):
            uri = entry.get("uri")
            if uri and not uri.startswith("data:"):
                if urllib.parse.urlparse(uri).scheme:
                    raise ValueError(f"Unexpected external glTF dependency: {uri}")
                dependency = safe_target(gltf_path.parent, urllib.parse.unquote(uri))
                if not dependency.is_file():
                    raise ValueError(f"Missing glTF dependency: {dependency}")
    record["total_bytes"] = sum(item["size_bytes"] for item in record["files"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", choices=("1k", "2k", "4k"), default="2k")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--skip-floor", action="store_true")
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    assets = [fetch_asset(root, asset, args.resolution, args.timeout, model=True) for asset in MODELS]
    if not args.skip_floor:
        assets.append(fetch_asset(root, FLOOR_ASSET, args.resolution, args.timeout, model=False))
    manifest = {
        "schema_version": 1,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Visual rendering assets; training geometry is unchanged by this downloader.",
        "license": "CC0-1.0",
        "license_url": LICENSE_URL,
        "assets": {asset["asset_id"]: asset for asset in assets},
        "total_bytes": sum(asset["total_bytes"] for asset in assets),
    }
    manifest_path = root / "manifest.json"
    atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Verified {manifest['total_bytes']:,} bytes of assets.", flush=True)


if __name__ == "__main__":
    main()
