"""
Generate the asset manifest, manifest/asset_manifest.csv, in one step:

    python scripts/build_manifest.py

Inputs are in manifest/source/: the recovered source manifest
(gemma_classified_expanded.csv) and the Objaverse licenses (objaverse_licenses.csv).
The steps:

1. Paths become relative to the data root (assets/<source>/...): the original data
   root (--old-prefix) is replaced, and absolute paths from earlier builds are cut at
   their assets/ folder.
2. ShapeNet is removed (not used for generation). In the source manifest every ShapeNet
   row has an empty filepath (notebook cell 28 copied texture_filepath instead of the
   exploded to_expand_filepath); these rows, and any row under assets/shapenet/, are
   dropped.
3. GSO paths: two folder names with non-ASCII characters ("Pokémon_...") lost them in the
   paths ("Pokmon_..."). The correct name is recovered from object_name, which stores
   it with octal escapes ("Pok\\303\\251mon_...").
4. A license column is added, in Objaverse's codes (by, by-sa, by-nc, by-nc-sa, cc0).
   ABO and GSO rows are "by" (both datasets are CC BY 4.0). Objaverse licenses come from
   manifest/source/objaverse_licenses.csv; if that file is missing (or with
   --refresh-licenses) they are fetched from the public allenai/objaverse metadata
   (~550 MB of shards, cached under <data root>/cache/) and the file is written.

Rows keep their order (the generator's sampling depends on it). Needs no assets on
disk, and is idempotent: running it on its own output changes nothing.
"""

import argparse
import csv
import gzip
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_DIR = os.path.join(REPO_ROOT, "manifest")
OLD_PREFIX   = "/raid/scratch/oti/nam_exp/3d_dataset/data/"   # data root in the source manifest
SOURCE_LICENSES = {"abo": "by", "gscanned": "by"}            # CC BY 4.0 datasets
SOURCES = ("objaverse", "abo", "gscanned", "shapenet")
OBJAVERSE_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"


def repo_path(path: str) -> str:
    """Command-line paths: absolute as given, relative ones from the repository root."""
    return os.path.abspath(os.path.join(REPO_ROOT, os.path.expanduser(path)))


# ── Paths ────────────────────────────────────────────────────────────────────
def relative_path(path: str, old_prefix: str) -> str:
    """Path relative to the data root: assets/<source>/..."""
    if not path or path.startswith("assets/"):
        return path
    if path.startswith(old_prefix):
        return "assets/" + path[len(old_prefix):]
    for source in SOURCES:
        marker = f"/assets/{source}/"
        if marker in path:
            return path[path.index(marker) + 1:]
    return path


def decode_octal_escapes(text: str) -> str:
    """'Pok\\303\\251mon' -> 'Pokémon' (escapes are UTF-8 bytes)."""
    if "\\" not in text:
        return text
    raw = re.sub(rb"\\([0-7]{3})", lambda m: bytes([int(m.group(1), 8)]), text.encode("utf-8"))
    return raw.decode("utf-8", errors="replace")


def fix_gso_paths(df: pd.DataFrame) -> int:
    """Restore non-ASCII GSO folder names from object_name; returns rows fixed."""
    prefix, fixed = "assets/gscanned/", 0
    for i in df.index[df.filepath.str.startswith(prefix)]:
        folder = df.at[i, "filepath"][len(prefix):].split("/", 1)[0]
        name = decode_octal_escapes(df.at[i, "object_name"])
        if name == folder or name.encode("ascii", "ignore").decode() != folder:
            continue
        for col in ("filepath", "texture_filepath"):
            if df.at[i, col].startswith(prefix + folder + "/"):
                df.at[i, col] = prefix + name + df.at[i, col][len(prefix + folder):]
        fixed += 1
    return fixed


# ── Objaverse licenses ───────────────────────────────────────────────────────
def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "georel-build-manifest"})
    with urllib.request.urlopen(req, timeout=300) as resp, open(dest + ".part", "wb") as f:
        f.write(resp.read())
    os.replace(dest + ".part", dest)


def fetch_objaverse_licenses(uids: set, data_root: str, out: str) -> None:
    """Write out (uid,license) for uids from the allenai/objaverse metadata shards."""
    paths_file = os.path.join(data_root, "object-paths.json")
    if not os.path.isfile(paths_file):
        gz = paths_file + ".gz"
        _download(OBJAVERSE_URL + "object-paths.json.gz", gz)
        with gzip.open(gz, "rb") as src, open(paths_file + ".part", "wb") as dst:
            dst.write(src.read())
        os.replace(paths_file + ".part", paths_file)
        os.remove(gz)
    with open(paths_file) as f:
        repo_paths = json.load(f)

    cache = os.path.join(data_root, "cache", "objaverse_metadata")
    shards = sorted({repo_paths[u].split("/")[1] for u in uids})

    def shard_file(shard: str) -> str:
        dest = os.path.join(cache, f"{shard}.json.gz")
        if not os.path.isfile(dest):
            _download(OBJAVERSE_URL + f"metadata/{shard}.json.gz", dest)
        return dest

    print(f"fetching Objaverse licenses: {len(uids):,} objects in {len(shards)} metadata shards")
    with ThreadPoolExecutor(max_workers=16) as pool:
        files = list(pool.map(shard_file, shards))
    licenses = {}
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for uid, meta in json.load(f).items():
                if uid in uids:
                    licenses[uid] = meta.get("license") or ""
    missing = uids - licenses.keys()
    if missing:
        raise SystemExit(f"{len(missing)} uids without metadata, e.g. {sorted(missing)[:3]}")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "license"])
        w.writerows(sorted(licenses.items()))
    print(f"wrote {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest",
                        default=os.path.join(MANIFEST_DIR, "source", "gemma_classified_expanded.csv"),
                        help="Source (or already-built) manifest")
    parser.add_argument("--out", default=os.path.join(MANIFEST_DIR, "asset_manifest.csv"))
    parser.add_argument("--objaverse-licenses",
                        default=os.path.join(MANIFEST_DIR, "source", "objaverse_licenses.csv"))
    parser.add_argument("--refresh-licenses", action="store_true",
                        help="Re-fetch the Objaverse licenses even if the file exists")
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "__artifacts"),
                        help="Where object-paths.json and the metadata cache are kept when "
                             "licenses are fetched (default: <repo>/__artifacts)")
    parser.add_argument("--old-prefix", default=OLD_PREFIX,
                        help="Data root used in the source manifest")
    args = parser.parse_args()
    old_prefix = args.old_prefix.rstrip("/") + "/"
    licenses_file = repo_path(args.objaverse_licenses)

    df = pd.read_csv(repo_path(args.manifest), dtype=str, keep_default_na=False)
    df = df.drop(columns=["gemma_location_type_orig", "license"], errors="ignore")
    for col in ("filepath", "texture_filepath"):
        df[col] = df[col].map(lambda p: relative_path(p, old_prefix))

    is_shapenet = (df.filepath == "") | df.filepath.str.startswith("assets/shapenet/")
    out = df[~is_shapenet].copy()
    print(f"ShapeNet rows dropped: {is_shapenet.sum():,}")
    print(f"GSO paths fixed: {fix_gso_paths(out)}")

    source = out.filepath.str.extract(r"^assets/([^/]+)/")[0]
    if args.refresh_licenses or not os.path.isfile(licenses_file):
        fetch_objaverse_licenses(set(out.filename[source == "objaverse"]),
                                 repo_path(args.data_root), licenses_file)
    objaverse = pd.read_csv(licenses_file, dtype=str, keep_default_na=False)
    by_uid = dict(zip(objaverse.uid, objaverse.license))
    out["license"] = [by_uid.get(uid, "") if src == "objaverse" else SOURCE_LICENSES.get(src, "")
                      for uid, src in zip(out.filename, source)]
    unlicensed = (out.license == "").sum()
    if unlicensed:
        raise SystemExit(f"{unlicensed} rows without a license")
    print("licenses:", out.license.value_counts().to_dict())

    out_path = repo_path(args.out)
    out.to_csv(out_path, index=False)
    print(f"rows written: {len(out):,} -> {out_path}")


if __name__ == "__main__":
    main()
