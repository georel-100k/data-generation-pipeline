"""
Download the 3D assets listed in the asset manifest into <data root>/assets/ (default
<repo>/__artifacts, or --data-root), in the layout the generator expects, then report
anything still missing.

  objaverse  GLBs from the public allenai/objaverse Hugging Face dataset
             -> assets/objaverse/glbs/<uid>.glb (object-paths.json is fetched once)
  abo        streams the official ABO 3D-model archive (~166 GB) and extracts only the
             manifest's models -> assets/abo/3dmodels/original/<x>/<item_id>.glb
  gso        per-model zips of Google Scanned Objects from Gazebo Fuel (GoogleResearch)
             -> assets/gscanned/<name>/{meshes/model.obj, materials/textures/texture.png, ...}

Only missing files are fetched and every file is written atomically, so the script can
be re-run to resume or retry. It exits 1 if any asset of the generator's pool
(gemma_location_type indoor/both) is still missing, since a scene's content depends on
the exact set of pool assets present.

Usage (from the repository root):
    python scripts/setup_assets.py                          # all sources, then report
    python scripts/setup_assets.py --data-root ../dataset/georel-cvpr
    python scripts/setup_assets.py --sources objaverse gso  # some sources
    python scripts/setup_assets.py --check                  # only report what is missing
"""

import argparse
import csv
import gzip
import io
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "__artifacts")        # set from --data-root in main()
MANIFEST  = os.path.join(REPO_ROOT, "manifest", "asset_manifest.csv")

OBJAVERSE_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
ABO_URL       = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar"
GSO_URL       = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/{name}.zip"
POOL          = {"indoor", "both"}
USER_AGENT    = {"User-Agent": "georel-setup-assets"}

def repo_path(path: str) -> str:
    """Command-line paths: absolute as given, relative ones from the repository root."""
    return os.path.abspath(os.path.join(REPO_ROOT, os.path.expanduser(path)))


def open_url(url: str, retries: int = 5, timeout: int = 120):
    """urlopen with retries and backoff (longer wait on HTTP 429)."""
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT),
                                          timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or attempt == retries - 1:
                raise
            time.sleep(60 if exc.code == 429 else 2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def write_atomic(dest: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + ".part", "wb") as f:
        f.write(data)
    os.replace(dest + ".part", dest)


def run_parallel(jobs: dict, workers: int, label: str) -> list:
    """Run {key: callable} in a thread pool; returns [(key, error)] for failures."""
    failures, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn): key for key, fn in jobs.items()}
        for fut in as_completed(futures):
            done += 1
            try:
                fut.result()
            except Exception as exc:
                failures.append((futures[fut], repr(exc)))
            if done % 200 == 0 or done == len(jobs):
                print(f"  [{label}] {done:,}/{len(jobs):,} done, {len(failures)} failed, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return failures


# ── Objaverse ────────────────────────────────────────────────────────────────
def objaverse_paths() -> dict:
    """uid -> repo path (glbs/<shard>/<uid>.glb), from DATA_ROOT/object-paths.json."""
    path = os.path.join(DATA_ROOT, "object-paths.json")
    if not os.path.isfile(path):
        print("  [objaverse] fetching object-paths.json")
        with open_url(OBJAVERSE_URL + "object-paths.json.gz") as resp:
            write_atomic(path, gzip.decompress(resp.read()))
    with open(path) as f:
        return json.load(f)


def fetch_objaverse(rows: list, workers: int) -> list:
    if not rows:
        return []
    repo_paths = objaverse_paths()

    def job(uid: str, dest: str):
        with open_url(OBJAVERSE_URL + repo_paths[uid]) as resp:
            data = resp.read()
        if data[:4] != b"glTF":
            raise IOError("not a GLB")
        write_atomic(dest, data)

    jobs = {r["filename"]: (lambda u=r["filename"], d=r["_dest"]: job(u, d)) for r in rows}
    unknown = [u for u in jobs if u not in repo_paths]
    for u in unknown:
        jobs.pop(u)
    return [(u, "no entry in object-paths.json") for u in unknown] + run_parallel(jobs, workers, "objaverse")


# ── ABO ──────────────────────────────────────────────────────────────────────
def fetch_abo(rows: list) -> list:
    """Stream the ABO archive once and extract only the needed members."""
    if not rows:
        return []
    abo_root = os.path.join(DATA_ROOT, "assets", "abo")
    needed = {os.path.relpath(r["_dest"], abo_root): r["_dest"] for r in rows}
    print(f"  [abo] streaming the ABO archive (~166 GB) for {len(needed):,} models")
    t0, found = time.time(), 0
    with open_url(ABO_URL, timeout=600) as resp, tarfile.open(fileobj=resp, mode="r|") as tar:
        for member in tar:
            dest = needed.pop(member.name, None)
            if dest is None or not member.isfile():
                continue
            write_atomic(dest, tar.extractfile(member).read())
            found += 1
            if found % 500 == 0:
                print(f"  [abo] {found:,} extracted, {(time.time() - t0) / 60:.1f} min", flush=True)
            if not needed:
                break
    return [(name, "not in the ABO archive") for name in needed]


# ── GSO ──────────────────────────────────────────────────────────────────────
def fetch_gso(rows: list, workers: int) -> list:
    """Download each missing model's zip from Gazebo Fuel and unpack it into its folder."""
    gso_root = os.path.join(DATA_ROOT, "assets", "gscanned")
    names = sorted({os.path.relpath(r["_dest"], gso_root).split(os.sep)[0] for r in rows})

    def job(name: str):
        with open_url(GSO_URL.format(name=urllib.parse.quote(name))) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        tmp = os.path.join(gso_root, f".{name}.part")
        shutil.rmtree(tmp, ignore_errors=True)
        for info in archive.infolist():
            parts = info.filename.split("/")
            if info.filename.startswith("/") or ".." in parts:
                raise IOError(f"unsafe path in zip: {info.filename}")
            if not info.is_dir():
                target = os.path.join(tmp, *parts)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(archive.read(info))
        final = os.path.join(gso_root, name)
        shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)

    return run_parallel({n: (lambda n=n: job(n)) for n in names}, workers, "gso")


# ── Main ─────────────────────────────────────────────────────────────────────
def load_rows(manifest: str) -> list:
    rows = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("filepath", "")
            if not rel:
                continue
            row["_dest"]   = rel if os.path.isabs(rel) else os.path.join(DATA_ROOT, rel)
            row["_source"] = rel.split("/")[1] if rel.startswith("assets/") else "other"
            tex = row.get("texture_filepath", "")
            row["_texture"] = (tex if os.path.isabs(tex) else os.path.join(DATA_ROOT, tex)) if tex else ""
            rows.append(row)
    return rows


def missing(rows: list) -> list:
    return [r for r in rows
            if not os.path.isfile(r["_dest"]) or (r["_texture"] and not os.path.isfile(r["_texture"]))]


def report(rows: list) -> int:
    """Print per-source coverage; returns the number of missing pool assets."""
    miss = missing(rows)
    print(f"\nAssets under {os.path.join(DATA_ROOT, 'assets')}:")
    pool_missing = 0
    for src in sorted({r["_source"] for r in rows}):
        all_src = [r for r in rows if r["_source"] == src]
        miss_src = [r for r in miss if r["_source"] == src]
        pool_src = [r for r in miss_src if r.get("gemma_location_type", "").strip().lower() in POOL]
        pool_missing += len(pool_src)
        print(f"  {src:10s} {len(all_src) - len(miss_src):>6,}/{len(all_src):<6,} present"
              f"   missing: {len(miss_src):,} (in generator pool: {len(pool_src):,})")
    print("All generator-pool assets present." if not pool_missing else
          f"{pool_missing:,} generator-pool assets missing: generated scenes would differ.")
    return pool_missing


def main() -> None:
    global DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=DATA_ROOT,
                        help="Data root (default: <repo>/__artifacts); assets go to <data root>/assets/")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--sources", nargs="+", default=["objaverse", "abo", "gso"],
                        choices=["objaverse", "abo", "gso"])
    parser.add_argument("--check", action="store_true", help="Only report missing assets")
    parser.add_argument("--workers", type=int, default=16, help="Parallel downloads")
    parser.add_argument("--failures", default=None,
                        help="Failure list (default: <data root>/logs/setup_assets_failures.csv)")
    args = parser.parse_args()
    DATA_ROOT = repo_path(args.data_root)
    args.manifest = repo_path(args.manifest)
    args.failures = repo_path(args.failures) if args.failures else \
        os.path.join(DATA_ROOT, "logs", "setup_assets_failures.csv")

    rows = load_rows(args.manifest)
    print(f"Manifest: {len(rows):,} assets | data root: {DATA_ROOT}")
    if not args.check:
        todo = missing(rows)
        failures = []
        if "objaverse" in args.sources:
            failures += fetch_objaverse([r for r in todo if r["_source"] == "objaverse"], args.workers)
        if "abo" in args.sources:
            failures += fetch_abo([r for r in todo if r["_source"] == "abo"])
        if "gso" in args.sources:
            failures += fetch_gso([r for r in todo if r["_source"] == "gscanned"], args.workers)
        if failures:
            os.makedirs(os.path.dirname(args.failures), exist_ok=True)
            with open(args.failures, "w", newline="") as f:
                csv.writer(f).writerows([("asset", "error"), *failures])
            print(f"{len(failures):,} downloads failed -> {args.failures}")
    sys.exit(1 if report(rows) else 0)


if __name__ == "__main__":
    main()
