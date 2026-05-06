import bpy
import os
import csv
import json
import random
import math
import mathutils
import colorsys
import urllib.request
import urllib.parse
import ssl
import argparse
import sys

# ----------------------------
# GET ARGUMENTS AFTER '--'
# ----------------------------
#argv = sys.argv

#if "--" in argv:
#    argv = argv[argv.index("--") + 1:]
#else:
#    argv = []

## ----------------------------
## PARSE
## ----------------------------
#parser = argparse.ArgumentParser()

#parser.add_argument("--fstart", type=str, required=True)

#args = parser.parse_args(argv)
# ── Config ────────────────────────────────────────────────────────────────────
# CSV_PATH   = "/Users/namanlal/Desktop/test_data/gemma_classified_expanded2.csv"
# OUTPUT_DIR = f"/Users/namanlal/Desktop/test_data/{args.fstart}"

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = "/workspace/data/gemma_classified_expanded.csv"
OUTPUT_DIR = f"/workspace/data/final_output_scenes7/"
# ── Config ────────────────────────────────────────────────────────────────────
# CSV_PATH   = "/Users/namanlal/Desktop/test_data/gemma_classified_expanded.csv"
# OUTPUT_DIR = "/Users/namanlal/Desktop/test_data/scenes"

N    = 27000   # objects to sample from the filtered pool
M    = 10000    # number of scenes to generate
Z    = 7.3     # mean objects per scene (Poisson, min 1)
SEED = 2026

# Objects are normalised so their largest bbox axis lands in this range
TARGET_SIZE_MIN = 1.8
TARGET_SIZE_MAX = 3.5

# Room dimensions
FLOOR_HALF     = 2.2    # half-extent; floor is (2*FLOOR_HALF) × (2*FLOOR_HALF)
WALL_HEIGHT    = 2.4
WALL_THICKNESS = 0.12

# Placement: distance from scene origin to object XY-centre
PLACE_DIST_MIN   = 0.3   # minimum distance from scene origin
PLACE_DIST_MAX   = 1.6   # spread across floor; clamped to FLOOR_HALF - radius anyway
COLLISION_MARGIN = 0.9   # minimum clear gap between object footprint circles

PLACEMENT_RETRIES    = 20
MIN_GAP              = 0.15  # minimum 3D AABB gap required between objects (Blender units)
MAX_OBJECTS_PER_SCENE = 12   # hard cap on objects loaded per scene (overrides Poisson)

# ── Camera config ─────────────────────────────────────────────────────────────
NUM_ARRANGEMENTS  = 3      # arrangements (re-shuffles) per scene
# Views per scene are sampled from Beta(2,4) scaled to [VIEWS_MIN, VIEWS_MAX].
# E[L] = VIEWS_MEAN  (exactly).  L is split evenly across NUM_ARRANGEMENTS.
VIEWS_MIN  = 6
VIEWS_MEAN = 12
VIEWS_MAX  = 24
CAM_POLAR_MIN_DEG = 30.0   # polar angle from zenith (0 = top-down, 90 = horizon)
CAM_POLAR_MAX_DEG = 60.0
CAM_FOV_DEG       = 70.0   # wider FOV lets the camera sit closer and still see the scene

RENDER_RES_X        = 1024
RENDER_RES_Y        = 1024
RENDER_ENGINE       = "CYCLES"
RENDER_SAMPLES      = 32   # OIDN denoiser makes 16 samples look clean; raise if quality needed
RENDER_SAMPLES_AUX  = 2    # depth / normals / mask passes — pure emission, minimal samples needed
RENDER_DENOISE      = True  # OIDN denoiser on RGB renders
USE_CPU_WITH_GPU    = False  # True → also schedule CPU tiles alongside GPUs
GPU_BACKEND         = "CUDA" # H200: CUDA is most reliable; "" = auto-detect

# ── Cycles light-path limits ──────────────────────────────────────────────────
# Lower bounce counts = fewer rays per sample = significantly faster renders.
# Values below are tuned for indoor synthetic data; raise if you see lost light.
CYCLES_BOUNCES_TOTAL        = 3   # default 12 — biggest single speedup lever
CYCLES_BOUNCES_DIFFUSE      = 2   # default 4
CYCLES_BOUNCES_GLOSSY       = 2   # default 4
CYCLES_BOUNCES_TRANSMISSION = 2   # default 12
CYCLES_BOUNCES_TRANSPARENT  = 4   # default 8
CYCLES_DISABLE_CAUSTICS     = True # caustic computation is expensive, rarely needed

# ── Metric ────────────────────────────────────────────────────────────────────
BLENDER_UNIT_TO_M = 1.0   # 1 Blender unit == 1 metre

# ── BlenderKit config ─────────────────────────────────────────────────────────
# Get a free API key at blenderkit.com → Profile → API Key
# Leave empty to read the key already saved in Blender addon preferences.
BLENDERKIT_API_KEY = ""
BLENDERKIT_CACHE   = os.path.join(OUTPUT_DIR, "_bk_cache")
USE_BLENDERKIT     = True   # False → always use procedural materials

# Keywords searched in rotation; picked randomly per object
_BK_KEYWORDS = [
    "wood floor", "concrete", "fabric soft", "plastic matte",
    "brushed metal", "ceramic tile", "leather worn", "painted wall",
    "stone rough", "rubber matte", "carpet texture", "marble polished",
    "plywood", "cork", "terracotta",
]

# ── Light config ──────────────────────────────────────────────────────────────
NUM_LIGHTS       = 3
LIGHT_DIST_MIN   = 4.0
LIGHT_DIST_MAX   = 8.0
LIGHT_HEIGHT_MIN = 3.0
LIGHT_HEIGHT_MAX = 7.0

# ── Parallelism ───────────────────────────────────────────────────────────────
# Each worker is a separate Blender process pinned to one GPU via
# CUDA_VISIBLE_DEVICES so they never contend for the same device.
N_WORKERS = 8              # one worker per GPU
GPU_IDS   = [0]  # physical CUDA device IDs — worker i gets GPU_IDS[i%len]
                           # set [] to let every worker use all GPUs (single-GPU mode)
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

# ── CLI args (set by coordinator when spawning worker processes) ───────────────
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
_ap   = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--worker",         action="store_true",
                 help="Run as a worker (renders assigned scenes only)")
_ap.add_argument("--scene-indices",  nargs="+", type=int, default=None,
                 help="Scene indices this worker should render")
_ap.add_argument("--worker-seed",    type=int, default=None,
                 help="RNG seed for this worker")
_ap.add_argument("--worker-gpu-id",  type=int, default=None,
                 help="CUDA device index to pin to (optional)")
_args, _ = _ap.parse_known_args(_argv)

IS_WORKER            = _args.worker
WORKER_SCENE_INDICES = _args.scene_indices          # None in coordinator
WORKER_GPU_ID        = _args.worker_gpu_id          # None = no pinning
if _args.worker_seed is not None:
    random.seed(_args.worker_seed)                  # each worker gets its own RNG


# ── Load & filter CSV ─────────────────────────────────────────────────────────
all_rows = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_rows.append(row)

VALID_LOCATIONS = {"indoor", "both"}

available = [
    row for row in all_rows
    if row.get("filepath")
    and os.path.isfile(row["filepath"])
    and row.get("gemma_location_type", "").strip().lower() in VALID_LOCATIONS
]

print(f"\n{'='*60}")
print(f"Total rows in CSV          : {len(all_rows)}")
print(f"After indoor/both filter   : {len(available)}")
print(f"{'='*60}")
print("\nObjects available to load:")
for i, row in enumerate(available):
    name = row.get("object_name") or row.get("filename") or "unknown"
    loc  = row.get("gemma_location_type", "?")
    print(f"  [{i:>4}] [{loc:7s}] {name}  ->  {row['filepath']}")
print(f"{'='*60}\n")

if not available:
    raise RuntimeError("No valid indoor/both objects found.")

n_sample = min(N, len(available))
print(f"Object pool size: {len(available)}  (will resample {n_sample} per scene)\n")


# ── Poisson count (Knuth), min 1 ─────────────────────────────────────────────
def poisson_count(mean: float) -> int:
    if mean <= 0:
        return 1
    L = math.exp(-mean)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= random.random()
    return max(1, k - 1)


def sample_total_views() -> int:
    """
    Sample total view count L for one scene.

    Distribution: Beta(2, 4) scaled to [VIEWS_MIN, VIEWS_MAX].
      E[Beta(2,4)] = 2/6 = 1/3  →  E[L] = VIEWS_MIN + span/3 = VIEWS_MEAN  ✓
      Mode ≈ VIEWS_MIN + span/4  (mild right skew; most scenes near lower end)

    Hard-clamped to [VIEWS_MIN, VIEWS_MAX] after rounding.
    """
    span = VIEWS_MAX - VIEWS_MIN                 # 36
    raw  = random.betavariate(2, 4)              # E = 1/3, in (0,1)
    return max(VIEWS_MIN, min(VIEWS_MAX, round(VIEWS_MIN + span * raw)))


def views_per_arrangement(total: int) -> list:
    """
    Split `total` views across NUM_ARRANGEMENTS as evenly as possible.
    Remainder views go to the earlier arrangements.
    Returns a list of length NUM_ARRANGEMENTS, sum == total.
    """
    base = total // NUM_ARRANGEMENTS
    rem  = total % NUM_ARRANGEMENTS
    return [base + (1 if i < rem else 0) for i in range(NUM_ARRANGEMENTS)]


# ── Material helpers ──────────────────────────────────────────────────────────
def random_color():
    return (random.random(), random.random(), random.random(), 1.0)


def _hsv_color(h: float, s: float = None, v: float = None) -> tuple:
    """
    Return an RGBA tuple from HSV.
    h is wrapped to [0, 1).  s and v are randomised within natural ranges if omitted.
    """
    h = h % 1.0
    s = s if s is not None else random.uniform(0.55, 0.92)
    v = v if v is not None else random.uniform(0.40, 0.92)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, 1.0)


def _scene_hues(n: int) -> list:
    """
    Return n hues that are visually well-separated.
    Uses the golden-ratio increment so hues never cluster regardless of n.
    A random global offset makes each scene's palette different from others.
    The list is shuffled so neighbouring objects don't form a rainbow sequence.
    """
    phi   = 0.6180339887          # 1/golden-ratio
    start = random.random()       # fresh offset per scene
    hues  = [(start + i * phi) % 1.0 for i in range(n)]
    random.shuffle(hues)
    return hues


def _tex_coord_mapping(nodes, links, scale=None):
    """Add TexCoord → Mapping nodes; return mapping node."""
    coord   = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    sc = scale if scale is not None else random.uniform(0.8, 4.0)
    mapping.inputs["Scale"].default_value = (sc, sc, sc)
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    return mapping


def _color_ramp(nodes, links, fac_socket, c1=None, c2=None):
    """Add ValToRGB node driven by fac_socket; return ramp node."""
    ramp = nodes.new("ShaderNodeValToRGB")
    links.new(fac_socket, ramp.inputs["Fac"])
    ramp.color_ramp.elements[0].color = c1 or random_color()
    ramp.color_ramp.elements[1].color = c2 or random_color()
    return ramp


_PROC_TYPES = ["noise", "voronoi", "wave", "marble", "brick", "checker", "wood", "rust"]


def make_procedural_material(name: str, hue: float = None,
                             proc_type: str = None) -> bpy.types.Material:
    """
    Generate a fully procedural Blender material anchored to a specific hue so
    every object in a scene is visually distinct.

    `hue`       — primary hue in [0, 1); randomised if None.
    `proc_type` — one of the 8 texture types; randomised if None.

    8 texture types, all built from shader nodes — no external image files:
      noise    – fractal noise colour ramp
      voronoi  – cellular / tile pattern
      wave     – stripe / ripple pattern
      marble   – wave distorted by noise
      brick    – brick + mortar
      checker  – checkerboard
      wood     – concentric rings distorted by noise
      rust     – noise-driven metallic roughness variation
    """
    if hue is None:
        hue = random.random()

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output   = nodes.new("ShaderNodeOutputMaterial")
    bsdf     = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    proc = proc_type if proc_type in _PROC_TYPES else random.choice(_PROC_TYPES)
    mapping = _tex_coord_mapping(nodes, links)

    # c1 = primary colour (anchored to the assigned hue, vivid)
    # c2 = secondary colour (analogous offset 15-40° away on the colour wheel)
    c1 = _hsv_color(hue,
                    s=random.uniform(0.60, 0.95),
                    v=random.uniform(0.45, 0.90))
    c2 = _hsv_color(hue + random.uniform(0.08, 0.22),
                    s=random.uniform(0.40, 0.85),
                    v=random.uniform(0.35, 0.85))

    if proc == "noise":
        tex = nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value     = random.uniform(2.0, 10.0)
        tex.inputs["Detail"].default_value    = random.uniform(4.0, 16.0)
        tex.inputs["Roughness"].default_value = random.uniform(0.4, 0.8)
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        ramp = _color_ramp(nodes, links, tex.outputs["Fac"], c1, c2)
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.8)

    elif proc == "voronoi":
        tex = nodes.new("ShaderNodeTexVoronoi")
        tex.inputs["Scale"].default_value = random.uniform(2.0, 8.0)
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        ramp = _color_ramp(nodes, links, tex.outputs["Distance"], c1, c2)
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = random.uniform(0.1, 0.6)

    elif proc == "wave":
        tex = nodes.new("ShaderNodeTexWave")
        tex.inputs["Scale"].default_value      = random.uniform(2.0, 8.0)
        tex.inputs["Distortion"].default_value = random.uniform(0.0, 5.0)
        tex.inputs["Detail"].default_value     = random.uniform(1.0, 6.0)
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        ramp = _color_ramp(nodes, links, tex.outputs["Color"], c1, c2)
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = random.uniform(0.2, 0.7)

    elif proc == "marble":
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value     = random.uniform(3.0, 10.0)
        noise.inputs["Detail"].default_value    = random.uniform(6.0, 16.0)
        noise.inputs["Roughness"].default_value = random.uniform(0.5, 0.8)
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        wave = nodes.new("ShaderNodeTexWave")
        wave.inputs["Scale"].default_value      = random.uniform(2.0, 6.0)
        wave.inputs["Distortion"].default_value = random.uniform(3.0, 10.0)
        links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
        # Add noise + wave together via MixRGB
        mix = nodes.new("ShaderNodeMixRGB")
        mix.inputs["Fac"].default_value = 0.5
        links.new(noise.outputs["Color"], mix.inputs["Color1"])
        links.new(wave.outputs["Color"],  mix.inputs["Color2"])
        ramp = _color_ramp(nodes, links, mix.outputs["Color"], c1, c2)
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = random.uniform(0.1, 0.35)

    elif proc == "brick":
        tex = nodes.new("ShaderNodeTexBrick")
        tex.inputs["Scale"].default_value       = random.uniform(2.0, 8.0)
        tex.inputs["Mortar Size"].default_value = random.uniform(0.01, 0.06)
        tex.inputs["Color1"].default_value      = c1
        tex.inputs["Color2"].default_value      = c2
        tex.inputs["Mortar"].default_value      = (0.1, 0.1, 0.1, 1.0)
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = random.uniform(0.5, 0.9)

    elif proc == "checker":
        tex = nodes.new("ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value  = random.uniform(2.0, 8.0)
        tex.inputs["Color1"].default_value = c1
        tex.inputs["Color2"].default_value = c2
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 0.7)

    elif proc == "wood":
        # Noise feeds into the wave vector to distort the rings
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value     = random.uniform(5.0, 15.0)
        noise.inputs["Detail"].default_value    = random.uniform(2.0, 8.0)
        noise.inputs["Roughness"].default_value = random.uniform(0.5, 0.8)
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "RINGS"
        wave.inputs["Scale"].default_value      = random.uniform(3.0, 8.0)
        wave.inputs["Distortion"].default_value = random.uniform(3.0, 10.0)
        links.new(noise.outputs["Color"], wave.inputs["Vector"])
        ramp = nodes.new("ShaderNodeValToRGB")
        links.new(wave.outputs["Color"], ramp.inputs["Fac"])
        # Dark ring grain = darker shade of hue; light ring = lighter shade
        ramp.color_ramp.elements[0].color = _hsv_color(hue, s=0.70, v=0.25)
        ramp.color_ramp.elements[1].color = _hsv_color(hue, s=0.45, v=0.78)
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = random.uniform(0.4, 0.65)

    elif proc == "rust":
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value     = random.uniform(4.0, 12.0)
        noise.inputs["Detail"].default_value    = random.uniform(8.0, 16.0)
        noise.inputs["Roughness"].default_value = random.uniform(0.6, 0.9)
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        # Dark oxidised tone ↔ lighter metallic tone, both anchored to hue
        cramp = _color_ramp(nodes, links, noise.outputs["Fac"],
                            _hsv_color(hue, s=0.80, v=0.30),
                            _hsv_color(hue, s=0.40, v=0.65))
        links.new(cramp.outputs["Color"], bsdf.inputs["Base Color"])
        # Roughness ramp (greyscale, unaffected by hue)
        rramp = _color_ramp(nodes, links, noise.outputs["Fac"],
                            (0.85, 0.85, 0.85, 1.0), (0.20, 0.20, 0.20, 1.0))
        links.new(rramp.outputs["Color"], bsdf.inputs["Roughness"])
        bsdf.inputs["Metallic"].default_value = random.uniform(0.3, 0.75)

    return mat


def make_texture_material(name: str, texture_path: str,
                          hue: float = None) -> bpy.types.Material:
    """
    Load an image texture and pipe it through a HueSaturation node so every
    object gets a distinct colour tint even when textures are similar.
    `hue` is in [0, 1); randomised if None.
    """
    if hue is None:
        hue = random.random()

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf   = nodes.new("ShaderNodeBsdfPrincipled")
    hue_sat = nodes.new("ShaderNodeHueSaturation")
    # Shift the texture's hue toward the assigned hue and add some saturation boost
    hue_sat.inputs["Hue"].default_value        = hue           # [0,1] maps to 0–360°
    hue_sat.inputs["Saturation"].default_value = random.uniform(1.0, 1.6)
    hue_sat.inputs["Value"].default_value      = random.uniform(0.85, 1.15)
    tex = nodes.new("ShaderNodeTexImage")
    try:
        tex.image = bpy.data.images.load(texture_path)
        links.new(tex.outputs["Color"], hue_sat.inputs["Color"])
        links.new(hue_sat.outputs["Color"], bsdf.inputs["Base Color"])
    except Exception as exc:
        print(f"    [WARN] texture load failed ({exc}), using hue color")
        bsdf.inputs["Base Color"].default_value = _hsv_color(hue)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def apply_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    """Replace every material slot on obj with mat (handles multi-slot GLBs)."""
    if not (obj.data and hasattr(obj.data, "materials")):
        return
    if obj.data.materials:
        for i in range(len(obj.data.materials)):
            obj.data.materials[i] = mat
    else:
        obj.data.materials.append(mat)


def _tint_material_with_hue(mat: bpy.types.Material, hue: float) -> None:
    """
    Inject a HueSaturation node into an existing material (e.g. from BlenderKit)
    so it picks up the scene-assigned hue.  Works by finding the first Principled
    BSDF connected to the Material Output and inserting the tint node between
    whatever drives Base Color and the BSDF input.  Safe to call on any material.
    """
    if not mat or not mat.use_nodes:
        return
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    output = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
    if output is None:
        return
    surface_links = [l for l in links
                     if l.to_node == output and l.to_socket.name == "Surface"]
    if not surface_links:
        return
    bsdf = surface_links[0].from_node
    base_links = [l for l in links
                  if l.to_node == bsdf and l.to_socket.name == "Base Color"]

    hue_sat = nodes.new("ShaderNodeHueSaturation")
    hue_sat.inputs["Hue"].default_value        = hue
    hue_sat.inputs["Saturation"].default_value = random.uniform(1.1, 1.8)
    hue_sat.inputs["Value"].default_value      = random.uniform(0.85, 1.15)

    if base_links:
        color_from = base_links[0].from_socket
        links.remove(base_links[0])
        links.new(color_from, hue_sat.inputs["Color"])
    else:
        hue_sat.inputs["Color"].default_value = _hsv_color(hue)
    links.new(hue_sat.outputs["Color"], bsdf.inputs["Base Color"])


# ── BlenderKit realistic materials ───────────────────────────────────────────
def _bk_api_key() -> str:
    """Return API key from script config or from Blender addon preferences."""
    if BLENDERKIT_API_KEY:
        return BLENDERKIT_API_KEY
    for addon_name in ("blenderkit",
                       "bl_ext.blenderkit_client.blenderkit",
                       "bl_ext.system_blenderkit.blenderkit"):
        if addon_name in bpy.context.preferences.addons:
            prefs = bpy.context.preferences.addons[addon_name].preferences
            key = getattr(prefs, "api_key", "") or getattr(prefs, "apiKey", "")
            if key:
                return key
    return ""


def _bk_ssl_ctx():
    """Return an SSL context; fall back to unverified if CA bundle is missing."""
    try:
        return ssl.create_default_context()
    except Exception:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        return ctx


def _bk_request(url: str) -> dict | None:
    """GET url with Bearer auth; return parsed JSON or None."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "BlenderKit-script/1.0"})
        key = _bk_api_key()
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=20, context=_bk_ssl_ctx()) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        print(f"    [BK] request error ({url[:60]}…): {exc}")
        return None


def _bk_search(keyword: str, n: int = 20) -> list:
    """Return up to n free material asset dicts matching keyword."""
    params = urllib.parse.urlencode({
        "query":      keyword,
        "asset_type": "material",
        "is_free":    "true",
        "page_size":  n,
        "order":      "-quality_count",
    })
    data = _bk_request(f"https://www.blenderkit.com/api/v1/search/?{params}")
    return (data or {}).get("results", [])


def _bk_signed_url(asset_base_id: str) -> str | None:
    """Fetch the signed S3 download URL for a material .blend file."""
    params = urllib.parse.urlencode({
        "asset_base_id": asset_base_id,
        "resolution":    "blend",
    })
    data = _bk_request(
        f"https://www.blenderkit.com/api/v1/assets/download/?{params}")
    return (data or {}).get("filePath")


def _bk_download(url: str, dest: str) -> bool:
    """Download url → dest (binary); return True on success."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "BlenderKit-script/1.0"})
        with urllib.request.urlopen(req, timeout=120, context=_bk_ssl_ctx()) as r, \
             open(dest, "wb") as f:
            f.write(r.read())
        return True
    except Exception as exc:
        print(f"    [BK] download failed: {exc}")
        return False


def _bk_load_first_material(blend_path: str, rename: str) -> "bpy.types.Material | None":
    """
    Append the first material from blend_path into the current scene.
    Tracks before/after bpy.data.materials so Blender's .001 auto-rename
    doesn't cause us to return a stale copy.
    Returns the material (renamed to `rename`) or None if the file has none.
    """
    names  = []
    before = set(bpy.data.materials)
    try:
        with bpy.data.libraries.load(blend_path, link=False) as (src, dst):
            names = list(src.materials)
            if names:
                dst.materials = [names[0]]
    except Exception as exc:
        print(f"    [BK] library load error: {exc}")
        return None
    if not names:
        return None
    # Prefer the freshly-added material; fall back to name lookup
    new_mats = [m for m in bpy.data.materials if m not in before]
    mat = new_mats[0] if new_mats else bpy.data.materials.get(names[0])
    if mat:
        mat.name = rename
    return mat


# Pool of cached .blend paths, built lazily from BLENDERKIT_CACHE on disk.
_bk_pool: list     = []
_bk_warned: bool   = False   # suppress repeated "no API key" messages


def make_blenderkit_material(name: str) -> "bpy.types.Material | None":
    """
    Return a realistic BlenderKit material:
      1. If cached .blend files exist on disk, pick one at random.
      2. Periodically (or when the pool is empty) search + download a new one.
      3. Append the first material from the chosen .blend and return it.
    Returns None on any failure so callers can fall back to procedural.
    """
    global _bk_pool, _bk_warned

    if not USE_BLENDERKIT:
        return None

    if not _bk_api_key():
        if not _bk_warned:
            print("    [BK] no API key – set BLENDERKIT_API_KEY or log in via the addon; "
                  "falling back to procedural materials")
            _bk_warned = True
        return None

    os.makedirs(BLENDERKIT_CACHE, exist_ok=True)

    # Populate pool from disk cache on first call
    if not _bk_pool:
        _bk_pool = [
            os.path.join(BLENDERKIT_CACHE, f)
            for f in os.listdir(BLENDERKIT_CACHE)
            if f.endswith(".blend")
        ]

    # Download a new material when the pool is thin or randomly (30% chance)
    if len(_bk_pool) < 6 or random.random() < 0.30:
        keyword = random.choice(_BK_KEYWORDS)
        assets  = _bk_search(keyword)
        if assets:
            asset    = random.choice(assets[:8])
            base_id  = asset.get("assetBaseId") or asset.get("id", "")
            cached   = os.path.join(BLENDERKIT_CACHE, f"{base_id}.blend")
            if not os.path.isfile(cached):
                signed = _bk_signed_url(base_id)
                if signed and _bk_download(signed, cached):
                    _bk_pool.append(cached)
                    print(f"    [BK] cached '{keyword}' → {base_id}.blend")
            elif cached not in _bk_pool:
                _bk_pool.append(cached)

    if not _bk_pool:
        return None

    blend_path = random.choice(_bk_pool)
    mat = _bk_load_first_material(blend_path, name)
    if mat:
        print(f"    [BK] applied '{mat.name}'")
    return mat


# ── Hierarchy helpers ─────────────────────────────────────────────────────────
def get_roots(objects: list) -> list:
    """Top-level objects whose parent is None or not inside this import group."""
    obj_set = set(objects)
    return [o for o in objects if o.parent is None or o.parent not in obj_set]


def get_meshes(objects: list) -> list:
    """All MESH-type objects in the list."""
    return [o for o in objects if o.type == "MESH"]


# ── Bounding-box utilities ────────────────────────────────────────────────────
def _depsgraph_bbox(obj: bpy.types.Object) -> list:
    """
    Get evaluated (post-modifier, post-scale) world-space bbox corners.
    Using the depsgraph ensures scale is baked and modifiers applied.
    """
    dg   = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(dg)
    return [eval_obj.matrix_world @ mathutils.Vector(c) for c in eval_obj.bound_box]


def group_world_bbox(meshes: list):
    """World-space (min_xyz, max_xyz) of a list of mesh objects via depsgraph."""
    mn = [math.inf]*3
    mx = [-math.inf]*3
    for obj in meshes:
        for w in _depsgraph_bbox(obj):
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    return mn, mx


def group_footprint_radius(meshes: list) -> float:
    mn, mx = group_world_bbox(meshes)
    dx, dy = mx[0] - mn[0], mx[1] - mn[1]
    return math.sqrt(dx*dx + dy*dy) * 0.5


def group_center_xy(meshes: list):
    mn, mx = group_world_bbox(meshes)
    return (mn[0]+mx[0])*0.5, (mn[1]+mx[1])*0.5


_ROOM_NAMES = {"Floor", "Wall_PosY", "Wall_NegY", "Wall_PosX", "Wall_NegX"}


def _objects_bbox_corners() -> list:
    """
    Tight bbox of only placed objects (no floor/walls) for camera framing.
    Padded by 0.6 units on XY and always includes Z=0 so the floor shows.
    This keeps ortho_scale small so objects fill the frame.
    """
    obj_meshes = [o for o in bpy.data.objects
                  if o.type == "MESH" and o.name not in _ROOM_NAMES]
    if not obj_meshes:
        h = FLOOR_HALF
        return [(-h,-h,0),(h,-h,0),(h,h,0),(-h,h,0),
                (-h,-h,1),(h,-h,1),(h,h,1),(-h,h,1)]
    mn = [math.inf]*3
    mx = [-math.inf]*3
    for obj in obj_meshes:
        for w in _depsgraph_bbox(obj):
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    pad = 0.6
    mn[0] = max(mn[0] - pad, -FLOOR_HALF)
    mn[1] = max(mn[1] - pad, -FLOOR_HALF)
    mx[0] = min(mx[0] + pad,  FLOOR_HALF)
    mx[1] = min(mx[1] + pad,  FLOOR_HALF)
    mn[2] = 0.0   # always show floor
    return [
        (mn[0],mn[1],mn[2]),(mx[0],mn[1],mn[2]),(mx[0],mx[1],mn[2]),(mn[0],mx[1],mn[2]),
        (mn[0],mn[1],mx[2]),(mx[0],mn[1],mx[2]),(mx[0],mx[1],mx[2]),(mn[0],mx[1],mx[2]),
    ]


def permute_materials(mesh_groups: list) -> None:
    """
    Randomly shuffle materials across all placed objects.
    mesh_groups: list of lists — each inner list is the meshes from one import.
    Materials are reshuffled so each object gets a different object's material.
    """
    # Collect every (obj, slot_idx, material) for non-room objects
    slots = []
    for meshes in mesh_groups:
        for obj in meshes:
            if obj.name in _ROOM_NAMES:
                continue
            for i, slot in enumerate(obj.material_slots):
                if slot.material:
                    slots.append([obj, i, slot.material])

    if len(slots) < 2:
        return

    mats = [s[2] for s in slots]
    random.shuffle(mats)
    for (obj, idx, _), new_mat in zip(slots, mats):
        obj.material_slots[idx].material = new_mat

    print(f"    [PERMUTE] shuffled {len(slots)} material slot(s)")


# ── Object transform pipeline ─────────────────────────────────────────────────
def _select_roots(roots: list) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in roots:
        obj.select_set(True)
    if roots:
        bpy.context.view_layer.objects.active = roots[0]


def normalise_scale(roots: list, meshes: list) -> None:
    """
    Scale root objects uniformly so the combined mesh bbox largest axis
    falls in [TARGET_SIZE_MIN, TARGET_SIZE_MAX], then bake scale.
    Baking (transform_apply) is essential so bound_box reflects true size.
    """
    if not roots or not meshes:
        return
    bpy.context.view_layer.update()
    mn, mx  = group_world_bbox(meshes)
    max_dim = max(mx[i] - mn[i] for i in range(3))
    if max_dim < 1e-6:
        return
    factor = random.uniform(TARGET_SIZE_MIN, TARGET_SIZE_MAX) / max_dim
    for obj in roots:
        obj.scale = obj.scale * factor
    # Bake scale into mesh data so bound_box is reliable going forward
    _select_roots(roots)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.view_layer.update()


def place_on_floor(roots: list, meshes: list, floor_z: float = 0.0) -> None:
    """
    Shift root objects along Z so the lowest mesh vertex sits at floor_z.
    Reads bbox from evaluated depsgraph (post-scale), moves only roots.
    """
    if not roots or not meshes:
        return
    bpy.context.view_layer.update()
    mn, _ = group_world_bbox(meshes)
    delta_z = floor_z - mn[2]
    for obj in roots:
        obj.location.z += delta_z
    bpy.context.view_layer.update()


def move_group_xy(roots: list, meshes: list, tx: float, ty: float) -> None:
    """Move root objects so the mesh footprint centre lands at (tx, ty)."""
    bpy.context.view_layer.update()
    cx, cy = group_center_xy(meshes)
    dx, dy = tx - cx, ty - cy
    for obj in roots:
        obj.location.x += dx
        obj.location.y += dy
    bpy.context.view_layer.update()


def random_rotate(roots: list, meshes: list) -> None:
    """
    Apply a random Z rotation (full 360°) to each root so objects face
    different directions.  A small random tilt (±8°) on X/Y adds life.
    After rotation the group is re-snapped to the floor by the caller.
    Rotation pivot = world XY centre of the mesh group so position stays fixed.
    """
    bpy.context.view_layer.update()
    cx, cy = group_center_xy(meshes)

    yaw   = random.uniform(0.0, 2 * math.pi)
    tilt  = math.radians(random.uniform(-8.0, 8.0))

    for obj in roots:
        obj.rotation_euler.z += yaw
        obj.rotation_euler.x += tilt * random.choice([-1, 1])

    # Keep the XY centre at its original spot after rotation
    bpy.context.view_layer.update()
    new_cx, new_cy = group_center_xy(meshes)
    for obj in roots:
        obj.location.x += cx - new_cx
        obj.location.y += cy - new_cy
    bpy.context.view_layer.update()


def _scale_down_group(roots: list, meshes: list, factor: float = 0.85) -> None:
    """Scale a group down by factor, bake scale, then re-snap to floor."""
    if not roots:
        return
    bpy.context.view_layer.update()
    for obj in roots:
        obj.scale = obj.scale * factor
    _select_roots(roots)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.view_layer.update()
    place_on_floor(roots, meshes, floor_z=0.0)


def resolve_overlaps(roots_list: list, mesh_groups: list,
                     placed_circles: list, max_iter: int = 8) -> list:
    """
    Post-placement pass: check every pair of objects for 3D AABB overlap.
    For each overlapping pair push the smaller-footprint object outward along the
    inter-centre vector.  If pushing would leave the floor bounds, scale down the
    larger object by 0.85 and recheck.
    Returns an updated placed_circles list (positions and radii synced to meshes).
    """
    n = len(roots_list)
    if n < 2:
        return placed_circles

    for _iteration in range(max_iter):
        found_overlap = False

        for i in range(n):
            for j in range(i + 1, n):
                mn_i, mx_i = group_world_bbox(mesh_groups[i])
                mn_j, mx_j = group_world_bbox(mesh_groups[j])

                # 3D AABB overlap test with MIN_GAP clearance on every axis
                overlap = (
                    mn_i[0] < mx_j[0] + MIN_GAP and mn_j[0] < mx_i[0] + MIN_GAP and
                    mn_i[1] < mx_j[1] + MIN_GAP and mn_j[1] < mx_i[1] + MIN_GAP and
                    mn_i[2] < mx_j[2] + MIN_GAP and mn_j[2] < mx_i[2] + MIN_GAP
                )
                if not overlap:
                    continue

                found_overlap = True
                print(f"    [OVERLAP] resolving pair ({i+1},{j+1})")

                ci_x = (mn_i[0] + mx_i[0]) * 0.5
                ci_y = (mn_i[1] + mx_i[1]) * 0.5
                cj_x = (mn_j[0] + mx_j[0]) * 0.5
                cj_y = (mn_j[1] + mx_j[1]) * 0.5
                dx   = cj_x - ci_x
                dy   = cj_y - ci_y
                dist = math.sqrt(dx * dx + dy * dy)

                ri = placed_circles[i][2]
                rj = placed_circles[j][2]
                push_needed = ri + rj + MIN_GAP

                # Normalise push direction; use +X if objects share same XY centre
                if dist < 1e-4:
                    dx, dy, dist = 1.0, 0.0, 1.0

                # Try to push object j outward to reach required separation
                scale_factor = push_needed / dist
                new_cx = ci_x + dx * scale_factor
                new_cy = ci_y + dy * scale_factor
                max_reach = FLOOR_HALF - rj - 0.05

                if abs(new_cx) <= max_reach and abs(new_cy) <= max_reach:
                    move_group_xy(roots_list[j], mesh_groups[j], new_cx, new_cy)
                    place_on_floor(roots_list[j], mesh_groups[j], floor_z=0.0)
                    actual_cx, actual_cy = group_center_xy(mesh_groups[j])
                    placed_circles[j] = (actual_cx, actual_cy, placed_circles[j][2])
                else:
                    # Can't push j within floor — scale down the larger object
                    scale_idx = i if ri >= rj else j
                    print(f"    [OVERLAP] scaling down object {scale_idx+1}")
                    _scale_down_group(roots_list[scale_idx], mesh_groups[scale_idx], 0.85)
                    new_r    = group_footprint_radius(mesh_groups[scale_idx])
                    cx_s, cy_s = group_center_xy(mesh_groups[scale_idx])
                    placed_circles[scale_idx] = (cx_s, cy_s, new_r)

        if not found_overlap:
            break

    # Final sync: update all circle entries to true post-op mesh positions/radii
    for idx in range(n):
        cx, cy = group_center_xy(mesh_groups[idx])
        r      = group_footprint_radius(mesh_groups[idx])
        placed_circles[idx] = (cx, cy, r)

    return placed_circles


# ── Collision-aware XY placement ─────────────────────────────────────────────
def circles_overlap(cx1, cy1, r1, cx2, cy2, r2) -> bool:
    dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    return dist < (r1 + r2 + COLLISION_MARGIN)


def find_free_position(radius: float, placed: list) -> tuple | None:
    """
    Sample a random (x,y) on the floor plane at distance [PLACE_DIST_MIN,
    PLACE_DIST_MAX] from origin, guaranteed collision-free.
    The search radius expands every 10 failed attempts.
    """
    max_reach   = min(PLACE_DIST_MAX, FLOOR_HALF - radius - 0.05)
    effective   = max_reach
    for attempt in range(PLACEMENT_RETRIES):
        angle   = random.uniform(0, 2 * math.pi)
        dist    = random.uniform(PLACE_DIST_MIN, effective)
        cx, cy  = dist * math.cos(angle), dist * math.sin(angle)
        if not any(circles_overlap(cx, cy, radius, px, py, pr) for px, py, pr in placed):
            return cx, cy
        if attempt % 10 == 9:
            effective = min(effective + 0.4, FLOOR_HALF - radius - 0.05)
    return None


# ── Import ────────────────────────────────────────────────────────────────────
def import_file(filepath: str) -> list:
    """Import file and return ALL new objects (empties, meshes, armatures, etc.)."""
    before = set(bpy.data.objects)
    ext    = os.path.splitext(filepath)[1].lower()
    try:
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=filepath)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=filepath)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=filepath)
        else:
            print(f"    [SKIP] Unsupported format: {filepath}")
            return []
    except Exception as exc:
        print(f"    [ERR]  {filepath}: {exc}")
        return []
    all_new    = [o for o in bpy.data.objects if o not in before]
    mesh_count = sum(1 for o in all_new if o.type == "MESH")
    print(f"    [OK]   {os.path.basename(filepath)}  "
          f"({mesh_count} mesh(es), {len(all_new)} total objects)")
    return all_new


# ── Shared material factory ───────────────────────────────────────────────────
def _make_plain_material(name, color, roughness=0.85):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out  = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value  = roughness
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


# ── Floor + Walls ─────────────────────────────────────────────────────────────
def create_room() -> None:
    h, wh, wt = FLOOR_HALF, WALL_HEIGHT, WALL_THICKNESS

    # Floor
    bpy.ops.mesh.primitive_plane_add(size=h * 2, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(
        _make_plain_material("Floor_Mat", (0.75, 0.75, 0.75, 1.0), roughness=0.9)
    )

    wall_mat = _make_plain_material("Wall_Mat", (0.85, 0.83, 0.80, 1.0), roughness=0.8)

    # Each wall: cube of size=2, scaled to a slab, placed flush with floor edge.
    # Scale tuple is (half_width, half_depth, half_height) → cube ×2 = full size.
    wall_defs = [
        ("Wall_PosY", ( 0,          h+wt*0.5,  wh*0.5), (h,      wt*0.5, wh*0.5)),
        ("Wall_NegY", ( 0,        -(h+wt*0.5), wh*0.5), (h,      wt*0.5, wh*0.5)),
        ("Wall_PosX", ( h+wt*0.5,  0,          wh*0.5), (wt*0.5, h,      wh*0.5)),
        ("Wall_NegX", (-(h+wt*0.5), 0,         wh*0.5), (wt*0.5, h,      wh*0.5)),
    ]
    for name, loc, scale in wall_defs:
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=loc)
        wall       = bpy.context.active_object
        wall.name  = name
        wall.scale = mathutils.Vector(scale)
        bpy.ops.object.select_all(action="DESELECT")
        wall.select_set(True)
        bpy.context.view_layer.objects.active = wall
        bpy.ops.object.transform_apply(scale=True)
        wall.data.materials.append(wall_mat)


# ── Lights ────────────────────────────────────────────────────────────────────
def create_lights(cx=0.0, cy=0.0) -> None:
    for i in range(NUM_LIGHTS):
        az   = random.uniform(0, 2*math.pi)
        dist = random.uniform(LIGHT_DIST_MIN, LIGHT_DIST_MAX)
        lz   = random.uniform(LIGHT_HEIGHT_MIN, LIGHT_HEIGHT_MAX)
        lx   = cx + dist * math.cos(az)
        ly   = cy + dist * math.sin(az)
        direction = mathutils.Vector((cx-lx, cy-ly, -lz))

        if i == 0:
            bpy.ops.object.light_add(type="SUN", location=(lx, ly, lz))
            light = bpy.context.active_object
            light.name = "KeyLight"
            light.data.energy = random.uniform(3.0, 6.0)
            light.data.color  = (random.uniform(0.9,1.0),
                                  random.uniform(0.75,0.9),
                                  random.uniform(0.5,0.75))
        else:
            bpy.ops.object.light_add(type="AREA", location=(lx, ly, lz))
            light = bpy.context.active_object
            light.name = f"FillLight_{i}"
            light.data.energy = random.uniform(200.0, 600.0)
            light.data.size   = random.uniform(1.5, 4.0)
            light.data.color  = (random.uniform(0.6,0.85),
                                  random.uniform(0.75,0.95),
                                  random.uniform(0.9,1.0))
        light.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


# ── Camera helpers ────────────────────────────────────────────────────────────
def _sample_sphere_band(pmin, pmax):
    cos_min   = math.cos(math.radians(pmax))
    cos_max   = math.cos(math.radians(pmin))
    cos_theta = random.uniform(cos_min, cos_max)
    polar     = math.degrees(math.acos(cos_theta))
    return random.uniform(0.0, 360.0), 90.0 - polar   # (azimuth, elevation)


def _cam_location(az_deg, el_deg, dist, target):
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return mathutils.Vector((
        target.x + dist * math.cos(el) * math.cos(az),
        target.y + dist * math.cos(el) * math.sin(az),
        target.z + dist * math.sin(el),
    ))


def _point_at(cam_obj, target):
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def _ortho_scale_for_cam(cam_loc, target, corners, aspect=1.0, pad=1.15):
    fwd = (target - cam_loc).normalized()
    up  = mathutils.Vector((0, 0, 1))
    if abs(fwd.dot(up)) > 0.999:
        up = mathutils.Vector((0, 1, 0))
    right  = fwd.cross(up).normalized()
    cam_up = right.cross(fwd).normalized()
    hw = hh = 0.0
    for c in corners:
        v  = mathutils.Vector(c) - target
        hw = max(hw, abs(v.dot(right)))
        hh = max(hh, abs(v.dot(cam_up)))
    return max(hw*2.0, hh*2.0/aspect) * pad


def create_cameras(target: mathutils.Vector, n_cams: int) -> list:
    aspect  = RENDER_RES_X / RENDER_RES_Y
    corners = _objects_bbox_corners()
    fov_rad = math.radians(CAM_FOV_DEG)
    cameras = []
    for i in range(n_cams):
        az, el = _sample_sphere_band(CAM_POLAR_MIN_DEG, CAM_POLAR_MAX_DEG)
        # Estimate scene half-span using ortho helper, then convert to perspective distance
        probe_dist = FLOOR_HALF * 3 + WALL_HEIGHT
        probe_loc  = _cam_location(az, el, probe_dist, target)
        ortho_sc   = _ortho_scale_for_cam(probe_loc, target, corners, aspect=aspect, pad=1.05)
        half_span  = ortho_sc * 0.5
        dist       = max(half_span / math.tan(fov_rad * 0.5), FLOOR_HALF * 1.1)
        loc = _cam_location(az, el, dist, target)

        # Enforce camera above wall top so no view looks through a wall.
        # If the candidate Z is too low, recompute elevation from the same
        # horizontal ring (preserving azimuth and distance) so cam_z ≥ wall_top.
        wall_top = WALL_HEIGHT + 0.05
        if loc.z < wall_top:
            dxy = math.hypot(loc.x - target.x, loc.y - target.y)
            dz_needed = wall_top - target.z
            el = math.degrees(math.atan2(dz_needed, dxy)) if dxy > 1e-6 else 90.0
            loc = _cam_location(az, el, dist, target)
            loc.z = max(loc.z, wall_top)   # safety hard-clamp

        bpy.ops.object.camera_add(location=loc)
        cam       = bpy.context.active_object
        label     = f"cam{i:02d}_az{az:.1f}_el{el:.1f}"
        cam.name  = label
        _point_at(cam, target)
        cam.data.type       = "PERSP"
        cam.data.angle      = fov_rad   # horizontal FOV
        cam.data.clip_start = 0.1
        cam.data.clip_end   = dist * 3.0
        cameras.append((cam, label))
        print(f"    [CAM] {label}  polar={90-el:.1f}°  dist={dist:.2f}")
    return cameras


# ── Cycles GPU setup ─────────────────────────────────────────────────────────
def configure_cycles_gpu() -> None:
    """
    Detect and activate every available GPU for Cycles.

    Backend priority: OptiX (NVIDIA RT cores) → CUDA → HIP (AMD) →
                      Metal (Apple) → OneAPI (Intel Arc) → CPU fallback.

    All GPU devices found under the winning backend are enabled.
    CPU tiles are added when USE_CPU_WITH_GPU is True.
    """
    if RENDER_ENGINE != "CYCLES":
        return

    scene = bpy.context.scene
    scene.cycles.device = "GPU"

    cycles_prefs = bpy.context.preferences.addons.get("cycles")
    if cycles_prefs is None:
        print("[GPU] Cycles addon not found – rendering on CPU")
        scene.cycles.device = "CPU"
        return

    prefs = cycles_prefs.preferences

    order = [GPU_BACKEND] if GPU_BACKEND else ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]
    for backend in order:
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if not gpus:
                continue
            for d in prefs.devices:
                d.use = (d.type != "CPU") or USE_CPU_WITH_GPU
            active = [d.name for d in prefs.devices if d.use]
            print(f"[GPU] backend={backend}  devices={active}")
            return
        except Exception as exc:
            print(f"[GPU] {backend} failed: {exc}")
            continue

    # No GPU found
    scene.cycles.device = "CPU"
    print("[GPU] No GPU found – falling back to CPU rendering")


# ── Render helpers ────────────────────────────────────────────────────────────
def _eevee_engine() -> str:
    """Return the correct EEVEE engine identifier for this Blender version."""
    major, minor = bpy.app.version[:2]
    return "BLENDER_EEVEE_NEXT" if (major, minor) >= (4, 2) else "BLENDER_EEVEE"


# Cache the EGL check so we only probe the filesystem once per process.
_egl_available: bool | None = None

def _aux_engine() -> str:
    """
    Return the fastest engine suitable for aux passes (normals, euclidean depth).

    On workstations with full OpenGL drivers this is EEVEE (~instant renders).
    On headless servers libEGL.so is often missing, causing Blender to crash when
    EEVEE initialises.  In that case we transparently fall back to Cycles with
    minimal samples — still very fast thanks to use_persistent_data.

    Detection order:
      1. DISPLAY env var set        → assume desktop, use EEVEE
      2. libEGL found in ld cache   → EGL present, use EEVEE
      3. libEGL found in /usr/lib*  → EGL present, use EEVEE
      4. Nothing found              → headless server, use Cycles
    """
    global _egl_available
    if _egl_available is None:
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            _egl_available = True
        else:
            import ctypes.util, glob
            found_via_ldconfig = ctypes.util.find_library("EGL") is not None
            found_on_disk = bool(
                glob.glob("/usr/lib/x86_64-linux-gnu/libEGL.so*") +
                glob.glob("/usr/lib64/libEGL.so*") +
                glob.glob("/usr/lib/aarch64-linux-gnu/libEGL.so*") +
                glob.glob("/usr/lib/libEGL.so*")
            )
            _egl_available = found_via_ldconfig or found_on_disk
        print(f"[AUX] EGL available={_egl_available}  "
              f"→ aux engine={'EEVEE' if _egl_available else 'CYCLES (headless fallback)'}")
    return _eevee_engine() if _egl_available else RENDER_ENGINE


def _setup_render(engine=RENDER_ENGINE, fmt="PNG", color_mode="RGBA", color_depth="8",
                  samples=None):
    scene = bpy.context.scene
    scene.render.engine                     = engine
    scene.render.resolution_x               = RENDER_RES_X
    scene.render.resolution_y               = RENDER_RES_Y
    scene.render.resolution_percentage      = 100
    scene.render.image_settings.file_format = fmt
    scene.render.image_settings.color_mode  = color_mode
    scene.render.image_settings.color_depth = color_depth

    # Keep BVH + textures in GPU memory across multiple renders of the same scene.
    scene.render.use_persistent_data = True

    if engine == "CYCLES":
        n      = samples if samples is not None else RENDER_SAMPLES
        is_aux = samples is not None and samples < RENDER_SAMPLES
        scene.cycles.samples               = n
        # Adaptive sampling must be OFF for aux passes: emission shaders have zero
        # variance and adaptive sampling can terminate at 0 valid samples → black.
        scene.cycles.use_adaptive_sampling = not is_aux
        if not is_aux:
            scene.cycles.adaptive_threshold   = 0.05
            scene.cycles.adaptive_min_samples = 4

        # ── Light-path limits (biggest single Cycles speedup) ─────────────────
        scene.cycles.max_bounces             = CYCLES_BOUNCES_TOTAL
        scene.cycles.diffuse_bounces         = CYCLES_BOUNCES_DIFFUSE
        scene.cycles.glossy_bounces          = CYCLES_BOUNCES_GLOSSY
        scene.cycles.transmission_bounces    = CYCLES_BOUNCES_TRANSMISSION
        scene.cycles.transparent_max_bounces = CYCLES_BOUNCES_TRANSPARENT
        scene.cycles.caustics_reflective     = not CYCLES_DISABLE_CAUSTICS
        scene.cycles.caustics_refractive     = not CYCLES_DISABLE_CAUSTICS

        vl = scene.view_layers[0]
        if RENDER_DENOISE and not is_aux:
            vl.cycles.use_denoising             = True
            scene.cycles.denoiser               = "OPENIMAGEDENOISE"
            scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        else:
            vl.cycles.use_denoising = False
    else:
        # EEVEE – 1 sample is enough for pure-geometry / material-override passes
        eevee = getattr(scene, "eevee", None)
        if eevee is not None:
            eevee.taa_render_samples = max(1, samples if samples is not None else 1)

    scene.use_nodes = False
    scene.view_layers[0].material_override = None


def render_all(cameras: list, rgb_dir: str) -> None:
    os.makedirs(rgb_dir, exist_ok=True)
    _setup_render(fmt="PNG", color_mode="RGBA")
    scene = bpy.context.scene
    for cam, label in cameras:
        scene.camera          = cam
        scene.render.filepath = os.path.join(rgb_dir, f"{label}.png")
        bpy.ops.render.render(write_still=True)
        print(f"    [RENDER] rgb/{label}.png")


# ── Object-ID assignment (for instance masks) ─────────────────────────────────
def assign_object_ids(mesh_groups: list) -> None:
    """Set obj.pass_index = 1-based group index on every mesh in each group."""
    for g_idx, meshes in enumerate(mesh_groups):
        for obj in meshes:
            obj.pass_index = g_idx + 1


# ── COLMAP coordinate conversion ──────────────────────────────────────────────
def _blender_cam_to_colmap(cam_obj):
    """
    Return (R_colmap 3×3 Matrix, t_colmap Vector) in COLMAP convention.
    Blender camera local: X right, Y up, −Z look.
    COLMAP camera local:  X right, Y down, +Z look.
    Flip = diag(1, −1, −1) applied to the world-to-cam rotation and translation.
    """
    M_inv  = cam_obj.matrix_world.inverted()
    R_w2c  = M_inv.to_3x3()
    t_w2c  = M_inv.translation.copy()
    flip   = mathutils.Matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return flip @ R_w2c, flip @ t_w2c


# ── Depth / normal material overrides ────────────────────────────────────────
def _make_euclidean_material():
    mat = bpy.data.materials.new(name="_Euclidean_Override")
    mat.use_nodes = True
    nd, lk = mat.node_tree.nodes, mat.node_tree.links
    nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial")
    emit = nd.new("ShaderNodeEmission")
    cam  = nd.new("ShaderNodeCameraData")
    lk.new(cam.outputs["View Distance"], emit.inputs["Strength"])
    emit.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    lk.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _make_world_normal_material():
    mat = bpy.data.materials.new(name="_WorldNorm_Override")
    mat.use_nodes = True
    nd, lk = mat.node_tree.nodes, mat.node_tree.links
    nd.clear()
    out  = nd.new("ShaderNodeOutputMaterial")
    emit = nd.new("ShaderNodeEmission")
    geo  = nd.new("ShaderNodeNewGeometry")
    mul  = nd.new("ShaderNodeVectorMath"); mul.operation = "MULTIPLY"
    add  = nd.new("ShaderNodeVectorMath"); add.operation = "ADD"
    mul.inputs[1].default_value = (0.5, 0.5, 0.5)
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    lk.new(geo.outputs["Normal"],    mul.inputs[0])
    lk.new(mul.outputs["Vector"],    add.inputs[0])
    lk.new(add.outputs["Vector"],    emit.inputs["Color"])
    emit.inputs["Strength"].default_value = 1.0
    lk.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _render_override(scene, cam, out_path, override_mat):
    """Render one camera with a material override; output is 32-bit EXR."""
    scene.view_layers[0].material_override = override_mat
    scene.use_nodes = False
    scene.camera          = cam
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)


def render_depth(cameras: list, depth_dir: str) -> None:
    os.makedirs(depth_dir, exist_ok=True)
    _setup_render(fmt="OPEN_EXR", color_mode="RGB", color_depth="32",
                  samples=RENDER_SAMPLES_AUX)
    scene = bpy.context.scene
    vl    = scene.view_layers[0]

    # ── Z depth via compositor Depth pass (correct in Cycles; emission override
    #    does NOT reliably produce Z depth in path-tracing mode) ────────────────
    vl.use_pass_z = True
    scene.use_nodes = True
    tree = scene.node_tree
    for cam, label in cameras:
        tree.nodes.clear()
        rl   = tree.nodes.new("CompositorNodeRLayers")
        comp = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(rl.outputs["Depth"], comp.inputs["Image"])
        scene.camera          = cam
        scene.render.filepath = os.path.join(depth_dir, f"{label}_zdepth")
        bpy.ops.render.render(write_still=True)
    vl.use_pass_z   = False
    scene.use_nodes = False

    # ── Euclidean ray distance via ShaderNodeCameraData "View Distance" ────────
    euclidean = _make_euclidean_material()
    for cam, label in cameras:
        _render_override(scene, cam, os.path.join(depth_dir, f"{label}_euclidean"),
                         euclidean)
        print(f"    [DEPTH] {label}")
    vl.material_override = None
    bpy.data.materials.remove(euclidean)


def render_normals(cameras: list, normals_dir: str) -> None:
    os.makedirs(normals_dir, exist_ok=True)
    _setup_render(fmt="OPEN_EXR", color_mode="RGB", color_depth="32",
                  samples=RENDER_SAMPLES_AUX)
    scene    = bpy.context.scene
    vl       = scene.view_layers[0]
    wnorm_mat = _make_world_normal_material()

    for cam, label in cameras:
        # World-space normals via material override (remapped to [0,1])
        _render_override(scene, cam, os.path.join(normals_dir, f"{label}_worldspace"), wnorm_mat)

        # Camera-space normals via compositor Normal pass (raw float, ~[-1,1])
        vl.material_override = None
        vl.use_pass_normal   = True
        scene.use_nodes      = True
        tree = scene.node_tree
        tree.nodes.clear()
        rl   = tree.nodes.new("CompositorNodeRLayers")
        comp = tree.nodes.new("CompositorNodeComposite")
        tree.links.new(rl.outputs["Normal"], comp.inputs["Image"])
        scene.camera          = cam
        scene.render.filepath = os.path.join(normals_dir, f"{label}_camspace")
        bpy.ops.render.render(write_still=True)
        print(f"    [NORM] {label}")

    vl.use_pass_normal   = False
    vl.material_override = None
    scene.use_nodes      = False
    bpy.data.materials.remove(wnorm_mat)


def _split_object_masks(combined_exr: str, label: str,
                         masks_dir: str, n_obj: int) -> None:
    """
    Load the combined IndexOB EXR (R channel = float object-id) and write
    one binary EXR per object: 1.0 where that object, 0.0 everywhere else.
    Output filenames: {label}_obj{id:03d}.exr
    """
    tmp = f"_combined_{label}"
    if tmp in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[tmp])
    img = bpy.data.images.load(combined_exr, check_existing=False)
    img.name = tmp
    img.colorspace_settings.name = "Non-Color"   # raw float, no gamma

    W, H   = img.size
    pixels = list(img.pixels)                     # flat RGBA, W*H*4

    for obj_id in range(1, n_obj + 1):
        binary = []
        for i in range(W * H):
            bit = 1.0 if abs(pixels[i * 4] - obj_id) < 0.5 else 0.0
            binary.extend([bit, bit, bit, 1.0])

        name = f"_mask_{label}_obj{obj_id:03d}"
        if name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[name])
        out = bpy.data.images.new(name, W, H, float_buffer=True, alpha=False)
        out.pixels = binary
        out.filepath_raw = os.path.join(masks_dir, f"{label}_obj{obj_id:03d}.exr")
        out.file_format  = "OPEN_EXR"
        out.save()
        bpy.data.images.remove(out)

    bpy.data.images.remove(img)


def render_masks(cameras: list, mesh_groups: list, masks_dir: str) -> None:
    os.makedirs(masks_dir, exist_ok=True)
    _setup_render(fmt="OPEN_EXR", color_mode="RGB", color_depth="32",
                  samples=RENDER_SAMPLES_AUX)
    scene  = bpy.context.scene
    vl     = scene.view_layers[0]
    n_obj  = len(mesh_groups)
    vl.use_pass_object_index = True
    scene.use_nodes = True
    tree   = scene.node_tree

    for cam, label in cameras:
        # Render combined IndexOB (one GPU pass per camera angle)
        tree.nodes.clear()
        rl        = tree.nodes.new("CompositorNodeRLayers")
        comp      = tree.nodes.new("CompositorNodeComposite")
        mn        = tree.nodes.new("CompositorNodeMath")
        mn.operation = "ADD"; mn.inputs[1].default_value = 0.0
        tree.links.new(rl.outputs["IndexOB"], mn.inputs[0])
        tree.links.new(mn.outputs["Value"],   comp.inputs["Image"])
        scene.camera          = cam
        combined_stem         = os.path.join(masks_dir, f"{label}_combined")
        scene.render.filepath = combined_stem
        bpy.ops.render.render(write_still=True)

        # Split CPU-side into N binary masks (no extra GPU render needed)
        _split_object_masks(combined_stem + ".exr", label, masks_dir, n_obj)
        print(f"    [MASK] {label}  → {n_obj} per-object masks")

    vl.use_pass_object_index = False
    scene.use_nodes = False


# ── Combined-pass renderer (replaces the 4 separate render_* calls) ───────────
def render_all_passes(cameras: list, arr_dir: str, mesh_groups: list) -> None:
    """
    Render all 6 output types per camera in only 3 render calls instead of 6:

      1. ONE Cycles render  → RGB (denoised) + Z-depth + camera normals + IndexOB
                              All saved in one shot via compositor OutputFile nodes.
      2. ONE EEVEE render   → world-space normals  (material override, ~instant)
      3. ONE EEVEE render   → Euclidean ray distance (material override, ~instant)

    Replaces render_all + render_depth + render_normals + render_masks.
    """
    rgb_dir     = os.path.join(arr_dir, "rgb")
    depth_dir   = os.path.join(arr_dir, "depth")
    normals_dir = os.path.join(arr_dir, "normals")
    masks_dir   = os.path.join(arr_dir, "masks")
    for d in (rgb_dir, depth_dir, normals_dir, masks_dir):
        os.makedirs(d, exist_ok=True)

    scene  = bpy.context.scene
    vl     = scene.view_layers[0]
    n_obj  = len(mesh_groups)
    eevee  = _eevee_engine()

    # ── Pass 1: Main Cycles render — all compositor-based outputs ─────────────
    _setup_render(engine=RENDER_ENGINE, fmt="PNG", color_mode="RGBA",
                  samples=RENDER_SAMPLES)
    vl.use_pass_z             = True
    vl.use_pass_normal        = True
    vl.use_pass_object_index  = True
    scene.use_nodes           = True
    tree = scene.node_tree

    for cam, label in cameras:
        scene.camera = cam
        tree.nodes.clear()
        rl = tree.nodes.new("CompositorNodeRLayers")

        # OutputFile node — one node, four output slots, one render call
        out = tree.nodes.new("CompositorNodeOutputFile")
        out.base_path = arr_dir   # all paths below are relative to arr_dir

        # Slot 0 (exists by default): RGB → PNG
        out.file_slots[0].path             = f"rgb/{label}_"
        out.file_slots[0].use_node_format  = False
        out.file_slots[0].format.file_format = "PNG"
        out.file_slots[0].format.color_mode  = "RGBA"
        out.file_slots[0].format.color_depth = "8"
        tree.links.new(rl.outputs["Image"], out.inputs[0])

        # Slot 1: Z-depth → EXR
        out.file_slots.new(f"depth/{label}_zdepth_")
        out.file_slots[1].use_node_format  = False
        out.file_slots[1].format.file_format = "OPEN_EXR"
        out.file_slots[1].format.color_mode  = "RGB"
        out.file_slots[1].format.color_depth = "32"
        tree.links.new(rl.outputs["Depth"], out.inputs[1])

        # Slot 2: Camera-space normals → EXR
        out.file_slots.new(f"normals/{label}_camspace_")
        out.file_slots[2].use_node_format  = False
        out.file_slots[2].format.file_format = "OPEN_EXR"
        out.file_slots[2].format.color_mode  = "RGB"
        out.file_slots[2].format.color_depth = "32"
        tree.links.new(rl.outputs["Normal"], out.inputs[2])

        # Slot 3: IndexOB → EXR  (Value → Color implicit conversion via Math node)
        mn = tree.nodes.new("CompositorNodeMath")
        mn.operation = "ADD"; mn.inputs[1].default_value = 0.0
        tree.links.new(rl.outputs["IndexOB"], mn.inputs[0])
        out.file_slots.new(f"masks/{label}_combined_")
        out.file_slots[3].use_node_format  = False
        out.file_slots[3].format.file_format = "OPEN_EXR"
        out.file_slots[3].format.color_mode  = "RGB"
        out.file_slots[3].format.color_depth = "32"
        tree.links.new(mn.outputs["Value"], out.inputs[3])

        # Render once — OutputFile node writes all 4 files automatically.
        # write_still=False: we don't need scene.render.filepath output.
        bpy.ops.render.render(write_still=False)

        # Blender appends the frame number (0001) to every OutputFile path.
        # Rename to the canonical names expected by downstream writers.
        _rename_output(arr_dir, f"rgb/{label}_0001.png",
                                f"rgb/{label}.png")
        _rename_output(arr_dir, f"depth/{label}_zdepth_0001.exr",
                                f"depth/{label}_zdepth.exr")
        _rename_output(arr_dir, f"normals/{label}_camspace_0001.exr",
                                f"normals/{label}_camspace.exr")
        _rename_output(arr_dir, f"masks/{label}_combined_0001.exr",
                                f"masks/{label}_combined.exr")

        # Split combined IndexOB mask CPU-side into per-object binary EXRs
        combined_exr = os.path.join(masks_dir, f"{label}_combined.exr")
        _split_object_masks(combined_exr, label, masks_dir, n_obj)
        print(f"    [PASS1] {label}  RGB + zdepth + cam-normals + masks")

    # Clean up main-pass settings
    vl.use_pass_z            = False
    vl.use_pass_normal       = False
    vl.use_pass_object_index = False
    scene.use_nodes          = False

    aux_eng  = _aux_engine()   # EEVEE on workstation, Cycles on headless server
    aux_tag  = "EEVEE" if "EEVEE" in aux_eng else "Cycles"
    aux_samp = 1 if "EEVEE" in aux_eng else RENDER_SAMPLES_AUX

    # ── Pass 2: World-space normals (material override) ───────────────────────
    _setup_render(engine=aux_eng, fmt="OPEN_EXR", color_mode="RGB", color_depth="32",
                  samples=aux_samp)
    wnorm_mat = _make_world_normal_material()
    for cam, label in cameras:
        _render_override(scene, cam,
                         os.path.join(normals_dir, f"{label}_worldspace"),
                         wnorm_mat)
        print(f"    [PASS2] {label}  world-normals ({aux_tag})")
    vl.material_override = None
    bpy.data.materials.remove(wnorm_mat)

    # ── Pass 3: Euclidean ray distance (material override) ────────────────────
    eucl_mat = _make_euclidean_material()
    for cam, label in cameras:
        _render_override(scene, cam,
                         os.path.join(depth_dir, f"{label}_euclidean"),
                         eucl_mat)
        print(f"    [PASS3] {label}  euclidean-depth ({aux_tag})")
    vl.material_override = None
    bpy.data.materials.remove(eucl_mat)

    # Restore Cycles for subsequent renders
    scene.render.engine = RENDER_ENGINE
    print(f"    [PASSES] {len(cameras)} cameras × 3 renders "
          f"(1 Cycles + 2 {aux_tag}) complete")


def _rename_output(base: str, src_rel: str, dst_rel: str) -> None:
    """Rename base/src_rel → base/dst_rel if src exists (no-op otherwise)."""
    src = os.path.join(base, src_rel)
    dst = os.path.join(base, dst_rel)
    if os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.rename(src, dst)


# ── COLMAP writer ─────────────────────────────────────────────────────────────
def write_colmap(cameras: list, colmap_dir: str) -> None:
    os.makedirs(colmap_dir, exist_ok=True)
    aspect = RENDER_RES_X / RENDER_RES_Y

    with open(os.path.join(colmap_dir, "cameras.txt"), "w") as f:
        f.write("# camera_id model width height fx fy cx cy\n")
        for i, (cam, _label) in enumerate(cameras):
            fov_h = cam.data.angle
            fov_v = 2 * math.atan(math.tan(fov_h * 0.5) / aspect)
            fx    = (RENDER_RES_X * 0.5) / math.tan(fov_h * 0.5)
            fy    = (RENDER_RES_Y * 0.5) / math.tan(fov_v * 0.5)
            cx    = RENDER_RES_X * 0.5
            cy    = RENDER_RES_Y * 0.5
            f.write(f"{i+1} PINHOLE {RENDER_RES_X} {RENDER_RES_Y} "
                    f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    with open(os.path.join(colmap_dir, "images.txt"), "w") as f:
        f.write("# image_id QW QX QY QZ TX TY TZ camera_id name\n")
        for i, (cam, label) in enumerate(cameras):
            R, t = _blender_cam_to_colmap(cam)
            q    = R.to_quaternion()
            f.write(f"{i+1} {q.w:.8f} {q.x:.8f} {q.y:.8f} {q.z:.8f} "
                    f"{t.x:.8f} {t.y:.8f} {t.z:.8f} {i+1} {label}.png\n\n")

    with open(os.path.join(colmap_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list — empty (poses only, no sparse reconstruction)\n")

    print(f"    [COLMAP] cameras.txt  images.txt  points3D.txt")


# ── Metadata writer ───────────────────────────────────────────────────────────
def write_metadata(cameras: list, mesh_groups: list,
                   placed_circles: list, scene_dir: str) -> None:
    aspect = RENDER_RES_X / RENDER_RES_Y
    U      = BLENDER_UNIT_TO_M

    cam_entries = []
    for i, (cam, label) in enumerate(cameras):
        fov_h = cam.data.angle
        fov_v = 2 * math.atan(math.tan(fov_h * 0.5) / aspect)
        fx    = (RENDER_RES_X * 0.5) / math.tan(fov_h * 0.5)
        fy    = (RENDER_RES_Y * 0.5) / math.tan(fov_v * 0.5)
        R, t  = _blender_cam_to_colmap(cam)
        pos   = cam.matrix_world.translation
        cam_entries.append({
            "id": i, "label": label,
            "intrinsics": {
                "fx_px": fx, "fy_px": fy,
                "cx_px": RENDER_RES_X * 0.5, "cy_px": RENDER_RES_Y * 0.5,
                "fov_h_deg": math.degrees(fov_h), "fov_v_deg": math.degrees(fov_v),
            },
            "extrinsics": {
                "position_world_m": [pos.x*U, pos.y*U, pos.z*U],
                "R_w2c_colmap": [[R[r][c] for c in range(3)] for r in range(3)],
                "t_w2c_colmap_m": [t.x*U, t.y*U, t.z*U],
            },
        })

    obj_entries = []
    centers_3d  = []
    for g_idx, (meshes, circle) in enumerate(zip(mesh_groups, placed_circles)):
        cx2, cy2, radius = circle
        mn, mx = group_world_bbox(meshes)
        c3d    = [(mn[k]+mx[k])*0.5 for k in range(3)]
        dims   = [(mx[k]-mn[k])*U   for k in range(3)]
        centers_3d.append(c3d)
        obj_entries.append({
            "id": g_idx + 1,
            "center_3d_m":       [v*U for v in c3d],
            "bbox_min_m":        [v*U for v in mn],
            "bbox_max_m":        [v*U for v in mx],
            "dimensions_m":      dims,
            "footprint_radius_m": radius*U,
        })

    dist_entries = []
    for i in range(len(centers_3d)):
        for j in range(i+1, len(centers_3d)):
            a, b   = centers_3d[i], centers_3d[j]
            dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
            d3d    = math.sqrt(dx**2 + dy**2 + dz**2) * U
            d2d    = math.sqrt(dx**2 + dy**2) * U
            r1, r2 = placed_circles[i][2]*U, placed_circles[j][2]*U
            dist_entries.append({
                "object_ids":                [i+1, j+1],
                "center_to_center_3d_m":     d3d,
                "center_to_center_2d_m":     d2d,
                "surface_to_surface_approx_m": max(0.0, d2d - r1 - r2),
                "delta_z_m":                 abs(dz) * U,
            })

    meta = {
        "render":   {"resolution_x": RENDER_RES_X, "resolution_y": RENDER_RES_Y,
                     "num_cameras": len(cameras)},
        "cameras":  cam_entries,
        "objects":  obj_entries,
        "inter_object_distances": dist_entries,
    }
    out = os.path.join(scene_dir, "metadata.json")
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"    [META] metadata.json  ({len(obj_entries)} obj, "
          f"{len(dist_entries)} dist pairs)")


# ── Scene graph ──────────────────────────────────────────────────────────────
def _spatial_relations(ci, cj, ri, rj, dims_i, dims_j) -> dict:
    """
    Return spatial relation labels describing how object-j sits relative to
    object-i.  All inputs in metres.

    Keys returned
    ─────────────
    proximity   : "adjacent" | "near" | "far"
    vertical    : "above" | "below" | "same_level"
    horizontal  : "east_of" | "north_of" | "west_of" | "south_of"
                  (world-space: +X=east, +Y=north)
    """
    dx, dy, dz = cj[0]-ci[0], cj[1]-ci[1], cj[2]-ci[2]
    d2d = math.sqrt(dx**2 + dy**2)
    s2s = max(0.0, d2d - ri - rj)

    # proximity
    if s2s < 0.30:
        prox = "adjacent"
    elif s2s < 1.50:
        prox = "near"
    else:
        prox = "far"

    # vertical  (compare centre-Z shift vs object half-heights)
    hh_i = dims_i[2] * 0.5
    hh_j = dims_j[2] * 0.5
    if dz > hh_i + hh_j * 0.5:
        vert = "above"       # j is above i
    elif dz < -(hh_i * 0.5 + hh_j):
        vert = "below"       # j is below i
    else:
        vert = "same_level"

    # horizontal  (four-quadrant compass from i → j)
    if d2d > 1e-4:
        ang = math.degrees(math.atan2(dy, dx)) % 360
        if ang < 45 or ang >= 315:
            horiz = "east_of"
        elif ang < 135:
            horiz = "north_of"
        elif ang < 225:
            horiz = "west_of"
        else:
            horiz = "south_of"
    else:
        horiz = "coincident"

    return {"proximity": prox, "vertical": vert, "horizontal": horiz}


def write_scene_graph(mesh_groups: list, placed_circles: list,
                      obj_rows: list, scene_id: str, scene_dir: str) -> None:
    """
    Write scene_graph.json to scene_dir.

    Graph schema
    ────────────
    nodes[]       – one entry per placed object
      id            1-based int (matches pass_index / mask value)
      object_name   from CSV
      filepath      source asset path
      center_3d_m   [x, y, z] world-space centroid
      bbox_min/max  world-space corners
      dimensions_m  [width, depth, height]
      footprint_radius_m
      material_name Blender material name on first mesh slot

    edges[]       – one entry per ORDERED pair (i→j), so N*(N-1) total
      source / target  node ids
      distances_m
        center_3d      Euclidean 3D
        center_2d      XY-plane only
        surface_to_surface  max(0, d2d − r_i − r_j)
        delta_z        |z_i − z_j|
      relations        {proximity, vertical, horizontal}  (j relative to i)

    adjacency     – dict[str, dict[str, float]]  keyed by node id (as string)
                    value = surface-to-surface distance for quick lookup
    """
    U = BLENDER_UNIT_TO_M

    # ── nodes ──────────────────────────────────────────────────────────────────
    nodes      = []
    centers    = []
    dims_list  = []

    for g_idx, (meshes, circle, row) in enumerate(
            zip(mesh_groups, placed_circles, obj_rows)):
        _, _, radius = circle
        mn, mx = group_world_bbox(meshes)
        c3d  = [(mn[k]+mx[k])*0.5*U for k in range(3)]
        dims = [(mx[k]-mn[k])*U      for k in range(3)]
        centers.append(c3d)
        dims_list.append(dims)

        # material name from first mesh's first slot
        mat_name = ""
        for obj in meshes:
            if obj.material_slots:
                m = obj.material_slots[0].material
                if m:
                    mat_name = m.name
                    break

        nodes.append({
            "id":                 g_idx + 1,
            "object_name":        row.get("object_name") or row.get("filename", ""),
            "filepath":           row.get("filepath", ""),
            "class":              row.get("class", ""),
            "center_3d_m":        c3d,
            "bbox_min_m":         [v*U for v in mn],
            "bbox_max_m":         [v*U for v in mx],
            "dimensions_m":       dims,
            "footprint_radius_m": radius * U,
            "material_name":      mat_name,
        })

    n = len(nodes)

    # ── edges (all ordered pairs i → j) ───────────────────────────────────────
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ci, cj = centers[i], centers[j]
            ri = placed_circles[i][2] * U
            rj = placed_circles[j][2] * U
            dx  = cj[0]-ci[0]; dy = cj[1]-ci[1]; dz = cj[2]-ci[2]
            d3d = math.sqrt(dx**2 + dy**2 + dz**2)
            d2d = math.sqrt(dx**2 + dy**2)
            s2s = max(0.0, d2d - ri - rj)
            edges.append({
                "source":    nodes[i]["id"],
                "target":    nodes[j]["id"],
                "distances_m": {
                    "center_3d":          round(d3d, 4),
                    "center_2d":          round(d2d, 4),
                    "surface_to_surface": round(s2s, 4),
                    "delta_z":            round(abs(dz), 4),
                },
                "relations": _spatial_relations(
                    ci, cj, ri, rj, dims_list[i], dims_list[j]),
            })

    # ── adjacency matrix  (node_id → {node_id → s2s distance}) ───────────────
    adjacency = {str(nd["id"]): {} for nd in nodes}
    for e in edges:
        adjacency[str(e["source"])][str(e["target"])] = \
            e["distances_m"]["surface_to_surface"]

    graph = {
        "scene_id":    scene_id,
        "num_objects": n,
        "nodes":       nodes,
        "edges":       edges,
        "adjacency":   adjacency,
    }

    out = os.path.join(scene_dir, "scene_graph.json")
    with open(out, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"    [GRAPH] scene_graph.json  "
          f"({n} nodes, {len(edges)} directed edges)")


# ── Arrangement helpers ───────────────────────────────────────────────────────
def rearrange_objects(roots_list: list, mesh_groups: list,
                      old_circles: list) -> list:
    """
    Move every object group to a new random, collision-free position on the
    floor.  Processes groups in shuffled order so placement bias is avoided.
    Returns a new placed_circles list in the original group order.
    """
    new_map = {}                             # idx → (cx, cy, radius)
    order   = list(range(len(roots_list)))
    random.shuffle(order)

    for idx in order:
        roots  = roots_list[idx]
        meshes = mesh_groups[idx]

        # Rotate BEFORE computing footprint so radius reflects post-rotation shape
        random_rotate(roots, meshes)
        place_on_floor(roots, meshes, floor_z=0.0)
        radius = group_footprint_radius(meshes)   # post-rotation radius

        placed_so_far = list(new_map.values())
        spot = find_free_position(radius, placed_so_far)
        if spot is None:
            spot = (random.uniform(-FLOOR_HALF * 0.35, FLOOR_HALF * 0.35),
                    random.uniform(-FLOOR_HALF * 0.35, FLOOR_HALF * 0.35))

        move_group_xy(roots, meshes, *spot)
        place_on_floor(roots, meshes, floor_z=0.0)

        cx, cy        = group_center_xy(meshes)
        actual_radius = group_footprint_radius(meshes)
        new_map[idx]  = (cx, cy, actual_radius)

    return [new_map[i] for i in range(len(roots_list))]


def _clear_cameras_and_lights() -> None:
    """Remove all camera and light objects (called between arrangements)."""
    for obj in list(bpy.data.objects):
        if obj.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(obj, do_unlink=True)
    for col in (bpy.data.cameras, bpy.data.lights):
        for block in list(col):
            if block.users == 0:
                col.remove(block)


# ── Scene cleanup ─────────────────────────────────────────────────────────────
def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for col in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                bpy.data.cameras, bpy.data.lights):
        for block in list(col):
            if block.users == 0:
                col.remove(block)


# ── Main ──────────────────────────────────────────────────────────────────────

def _find_script_path() -> str:
    """Return the absolute path of this script as passed via --python."""
    for i, arg in enumerate(sys.argv):
        if arg in ("--python", "-P") and i + 1 < len(sys.argv):
            return os.path.abspath(sys.argv[i + 1])
    return ""


def _configure_gpu_for_worker(worker_gpu_id) -> None:
    """
    Enable Cycles GPU rendering for this worker process.

    The coordinator already set CUDA_VISIBLE_DEVICES=<id> in the child's
    environment, so Blender only sees the one assigned physical GPU.
    We simply tell Cycles to use all visible GPU devices (just the one).
    """
    if RENDER_ENGINE != "CYCLES":
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(f"[GPU] worker {worker_gpu_id}  CUDA_VISIBLE_DEVICES={visible}")

    scene = bpy.context.scene
    scene.cycles.device = "GPU"

    cycles_prefs = bpy.context.preferences.addons.get("cycles")
    if cycles_prefs is None:
        scene.cycles.device = "CPU"
        print("[GPU] Cycles addon not found – CPU fallback")
        return

    prefs = cycles_prefs.preferences
    order = [GPU_BACKEND] if GPU_BACKEND else ["OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"]

    for backend in order:
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if not gpus:
                continue
            # Enable all devices visible to this process (env var already filtered them)
            for d in prefs.devices:
                d.use = (d.type != "CPU") or USE_CPU_WITH_GPU
            active = [d.name for d in prefs.devices if d.use]
            print(f"[GPU] backend={backend}  active={active}")
            return
        except Exception as exc:
            print(f"[GPU] {backend} failed: {exc}")

    scene.cycles.device = "CPU"
    print("[GPU] No GPU found – CPU fallback")


def _render_one_scene(scene_idx: int) -> int:
    """
    Build, render, and write outputs for a single scene.
    Returns the number of objects actually loaded, or 0 on fatal error.
    Arrangement-level errors are caught individually so surviving arrangements
    are still written even when one fails.
    """
    import traceback
    scene_label = f"scene_{scene_idx+1:03d}"
    scene_dir   = os.path.join(OUTPUT_DIR, scene_label)

    try:
        os.makedirs(scene_dir, exist_ok=True)

        sampled   = random.sample(available, n_sample)
        k         = min(poisson_count(Z), MAX_OBJECTS_PER_SCENE)
        picks     = random.choices(sampled, k=k)
        total_L   = sample_total_views()
        arr_views = views_per_arrangement(total_L)
        hue_list       = _scene_hues(k)
        proc_type_pool = _PROC_TYPES[:]
        random.shuffle(proc_type_pool)

        print(f"\n{'='*60}")
        print(f"[Scene {scene_idx+1}]  {scene_label}  "
              f"objects={k}  total_views={total_L}  per_arr={arr_views}")

        # ── Step 1: load scene geometry once ────────────────────────────────
        clear_scene()
        create_room()

        roots_list     = []
        mesh_groups    = []
        obj_rows       = []
        placed_circles = []
        loaded         = 0

        for pick_idx, row in enumerate(picks):
            try:
                filepath     = row["filepath"]
                texture_path = row.get("texture_filepath", "").strip()

                all_new = import_file(filepath)
                if not all_new:
                    continue
                roots  = get_roots(all_new)
                meshes = get_meshes(all_new)
                if not meshes:
                    continue

                normalise_scale(roots, meshes)
                random_rotate(roots, meshes)
                place_on_floor(roots, meshes, floor_z=0.0)
                radius = group_footprint_radius(meshes)

                spot = find_free_position(radius, placed_circles)
                if spot is None:
                    spot = (random.uniform(-FLOOR_HALF * 0.4, FLOOR_HALF * 0.4),
                            random.uniform(-FLOOR_HALF * 0.4, FLOOR_HALF * 0.4))

                move_group_xy(roots, meshes, *spot)
                place_on_floor(roots, meshes, floor_z=0.0)

                cx, cy        = group_center_xy(meshes)
                actual_radius = group_footprint_radius(meshes)
                placed_circles.append((cx, cy, actual_radius))
                roots_list.append(roots)
                mesh_groups.append(meshes)
                obj_rows.append(row)

                has_tex   = bool(texture_path) and os.path.isfile(texture_path)
                mat_name  = f"{scene_label}_mat_{pick_idx:03d}"
                hue       = hue_list[pick_idx % len(hue_list)]
                proc_type = proc_type_pool[pick_idx % len(proc_type_pool)]

                if has_tex:
                    mat = make_texture_material(mat_name, texture_path, hue=hue)
                else:
                    bk_mat = make_blenderkit_material(mat_name)
                    if bk_mat:
                        _tint_material_with_hue(bk_mat, hue)
                        mat = bk_mat
                    else:
                        mat = make_procedural_material(mat_name, hue=hue,
                                                       proc_type=proc_type)
                for obj in meshes:
                    apply_material(obj, mat)

                loaded += 1
                print(f"    [LOAD] {pick_idx+1}/{k}  cx={cx:.2f} cy={cy:.2f} r={radius:.2f}")

            except Exception as exc:
                print(f"    [WARN] object {pick_idx+1} skipped: {exc}")
                traceback.print_exc()
                continue   # skip this object, keep going

        assign_object_ids(mesh_groups)
        placed_circles = resolve_overlaps(roots_list, mesh_groups, placed_circles)

        # ── Step 2: arrangements ─────────────────────────────────────────────
        arr_ok = 0
        for arr_idx in range(NUM_ARRANGEMENTS):
            arr_label = f"arr_{arr_idx+1:03d}"
            arr_dir   = os.path.join(scene_dir, arr_label)
            try:
                os.makedirs(arr_dir, exist_ok=True)
                print(f"\n  [{scene_label}] arrangement {arr_idx+1}/{NUM_ARRANGEMENTS}")

                if arr_idx > 0:
                    placed_circles = rearrange_objects(roots_list, mesh_groups, placed_circles)
                    placed_circles = resolve_overlaps(roots_list, mesh_groups, placed_circles)

                avg_cx    = sum(c[0] for c in placed_circles) / max(len(placed_circles), 1)
                avg_cy    = sum(c[1] for c in placed_circles) / max(len(placed_circles), 1)
                target_pt = mathutils.Vector((avg_cx, avg_cy, WALL_HEIGHT * 0.4))

                n_cams = arr_views[arr_idx]
                _clear_cameras_and_lights()
                create_lights(cx=avg_cx, cy=avg_cy)
                cameras = create_cameras(target_pt, n_cams)

                render_all_passes(cameras, arr_dir, mesh_groups)
                write_colmap(cameras,  os.path.join(arr_dir, "colmap"))
                write_metadata(cameras, mesh_groups, placed_circles, arr_dir)
                write_scene_graph(mesh_groups, placed_circles, obj_rows,
                                  f"{scene_label}_{arr_label}", arr_dir)

                blend_path = os.path.join(arr_dir, "scene.blend")
                bpy.ops.wm.save_as_mainfile(filepath=blend_path)
                print(f"    [BLEND] saved {blend_path}")
                print(f"    => arrangement {arr_idx+1} done  "
                      f"({n_cams} views × 6 outputs + scene.blend)")
                arr_ok += 1

            except Exception as exc:
                print(f"\n  [ERROR] {scene_label} arrangement {arr_idx+1} failed: {exc}")
                traceback.print_exc()
                # Write an error marker so we know which arrangements failed
                try:
                    with open(os.path.join(arr_dir, "ERROR.txt"), "w") as ef:
                        ef.write(f"{exc}\n\n")
                        traceback.print_exc(file=ef)
                except Exception:
                    pass
                continue   # skip to next arrangement

        print(f"\n  [{scene_label}] complete — "
              f"{loaded} objects, {arr_ok}/{NUM_ARRANGEMENTS} arrangements OK")
        return loaded

    except Exception as exc:
        print(f"\n[ERROR] {scene_label} aborted: {exc}")
        traceback.print_exc()
        try:
            with open(os.path.join(scene_dir, "ERROR.txt"), "w") as ef:
                ef.write(f"{exc}\n\n")
                traceback.print_exc(file=ef)
        except Exception:
            pass
        return 0


def _run_coordinator() -> None:
    """
    Spawn one dedicated Blender process per scene.
    Concurrency is limited to len(GPU_IDS) simultaneous processes via a GPU
    slot queue — each slot holds one physical GPU ID.  When a scene finishes
    (success or failure) its slot is released and the next waiting scene starts.
    Errors in individual scenes are logged but never stop the pipeline.
    """
    import subprocess, threading
    from queue import Queue
    from concurrent.futures import ThreadPoolExecutor, as_completed

    blender = bpy.app.binary_path
    script  = _find_script_path()
    if not script:
        raise RuntimeError(
            "Cannot determine script path. "
            "Run with:  blender --background --python /abs/path/to/blender_script.py"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_slots = len(GPU_IDS) if GPU_IDS else N_WORKERS

    # Each entry in the queue is one available GPU ID (or None for no pinning).
    # A scene must acquire a slot before launching; it releases on completion.
    gpu_queue: Queue = Queue()
    for g in (GPU_IDS if GPU_IDS else [None] * N_WORKERS):
        gpu_queue.put(g)

    results: dict = {}   # scene_idx → exit_code
    lock = threading.Lock()

    print(f"\n{'='*60}")
    print(f"[COORD] {M} scenes  |  {n_slots} concurrent slots  "
          f"|  GPUs={GPU_IDS or 'all'}  |  MAX_OBJ={MAX_OBJECTS_PER_SCENE}")
    print(f"{'='*60}")

    def _run_scene(scene_idx: int) -> int:
        """Acquire a GPU slot, run one Blender process, release slot."""
        gpu_id = gpu_queue.get()   # blocks until a GPU is free
        tag    = f"S{scene_idx+1:03d}|GPU{gpu_id if gpu_id is not None else 'all'}"
        try:
            seed_i = SEED + scene_idx * 997
            cmd = [
                blender, "--background", "--python", script,
                "--",
                "--worker",
                "--scene-indices", str(scene_idx),
                "--worker-seed",   str(seed_i),
                "--worker-gpu-id", str(scene_idx),
            ]
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["HIP_VISIBLE_DEVICES"]  = str(gpu_id)

            with lock:
                print(f"  [{tag}] starting  seed={seed_i}")

            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(f"  [{tag}] {line}")
                sys.stdout.flush()
            code = proc.wait()

            with lock:
                status = "OK" if code == 0 else f"FAILED (exit {code})"
                print(f"  [{tag}] {status}")
            return code

        except Exception as exc:
            with lock:
                print(f"  [{tag}] exception: {exc}")
            return -1

        finally:
            gpu_queue.put(gpu_id)   # always return the slot, even on error

    # Submit all M scenes; the GPU queue caps actual concurrency at n_slots
    with ThreadPoolExecutor(max_workers=M) as pool:
        future_map = {pool.submit(_run_scene, i): i for i in range(M)}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = -1
                print(f"  [S{idx+1:03d}] unhandled exception: {exc}")

    ok     = sum(1 for c in results.values() if c == 0)
    failed = sorted(i + 1 for i, c in results.items() if c != 0)
    print(f"\n{'='*60}")
    print(f"[COORD] finished  —  {ok}/{M} OK  —  {len(failed)} failed")
    if failed:
        print(f"  Failed scene numbers: {failed}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


def _run_worker(scene_indices: list) -> None:
    """Render all scenes in `scene_indices` sequentially inside this process."""
    _configure_gpu_for_worker(WORKER_GPU_ID)

    object_counts = []
    for scene_idx in scene_indices:
        object_counts.append(_render_one_scene(scene_idx))

    actual_mean = sum(object_counts) / len(object_counts) if object_counts else 0
    print(f"\n{'='*60}")
    print(f"[Worker] scenes={scene_indices}  "
          f"avg_objects={actual_mean:.2f}  max={MAX_OBJECTS_PER_SCENE}")
    print(f"Arrangements/scene: {NUM_ARRANGEMENTS}")
    print(f"Views/scene dist  : min={VIEWS_MIN} mean={VIEWS_MEAN} max={VIEWS_MAX}")
    print(f"Output            : {OUTPUT_DIR}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if IS_WORKER:
    # Called by the coordinator — render only the assigned scenes
    _run_worker(WORKER_SCENE_INDICES)
else:
    # Top-level invocation — act as coordinator and fan out to N_WORKERS
    if N_WORKERS <= 1 or M <= 1:
        # Degenerate case: just run inline without spawning sub-processes
        _run_worker(list(range(M)))
    else:
        _run_coordinator()
