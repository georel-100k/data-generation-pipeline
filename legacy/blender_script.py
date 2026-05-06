# FULL FINAL SCRIPT (Blender 4.x)
import bpy, os, sys, csv, json, random, math
from mathutils import Vector
import addon_utils

# ---------------- ARGS ----------------
def get_args():
    argv = sys.argv
    argv = argv[argv.index("--")+1:]
    return {
        "scene_id": int(argv[0]),
        "out": argv[1],
        "type": argv[2],
        "data": argv[3]
    }
args = get_args()
print("ARGS:", args)
BASE = args["data"]
# MESH_DIR = os.path.join(BASE")
# MAT_DIR  = os.path.join(BASE,"materials")
HDRI_DIR = os.path.join(BASE,"hdri")
CSV_PATH = os.path.join(BASE,"gemma_classified.csv")

ROOM=6
OUTDOOR=30

# ---------------- CLEAN ----------------
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)

# ---------------- GPU ----------------
def setup_gpu():
    p=bpy.context.preferences.addons['cycles'].preferences
    try: p.compute_device_type='OPTIX'
    except: p.compute_device_type='CUDA'
    for d in p.devices: d.use=True
    s=bpy.context.scene
    s.cycles.device='GPU'
    s.render.engine='CYCLES'
    s.cycles.samples=64

# ---------------- MATERIAL ----------------
def tex_mat(path):
    m=bpy.data.materials.new("tex")
    m.use_nodes=True
    n=m.node_tree.nodes
    t=n.new("ShaderNodeTexImage")
    t.image=bpy.data.images.load(path)
    n["Principled BSDF"].inputs["Base Color"].default_value=(1,1,1,1)
    m.node_tree.links.new(t.outputs["Color"],n["Principled BSDF"].inputs["Base Color"])
    return m

def proc_mat():
    m=bpy.data.materials.new("proc")
    m.use_nodes=True
    bsdf=m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value=(random.random(),random.random(),random.random(),1)
    return m

def find_tex(name):
    tex=[]
    name = os.path.basename(name)
    # print(name)
    for r,_,f in os.walk(name):
        for x in f:
            if x.endswith((".png",".jpg",".jpeg")):
                tex.append(os.path.join(r,x))
                # if name.lower() in x.lower():
                #     return os.path.join(r,x)
    return random.choice(tex) if tex else None

def apply_mat(o, path):
    path = path.replace('meshes','materials')
    if o.type!='MESH': return
    t=find_tex(o.name)
    m=tex_mat(t) if t else proc_mat()
    if o.data.materials: o.data.materials[0]=m
    else: o.data.materials.append(m)

# ---------------- LOAD ----------------
def load(path):
    bpy.ops.object.select_all(action='DESELECT')
    ext=path.split('.')[-1].lower()
    try:
        if ext=="obj": bpy.ops.wm.obj_import(filepath=path)
        elif ext in ["glb","gltf"]: bpy.ops.import_scene.gltf(filepath=path)
    except Exception as e:
        print("load error",e); return []
    objs=bpy.context.selected_objects
    for o in objs: apply_mat(o,path)
    return objs

def model(folder):
    d=os.path.join(os.path.join(BASE,folder),"meshes")
    for r,_,f in os.walk(d):
        for x in f:
            if x.endswith((".obj",".glb",".gltf")):
                print("found model",x)
                return os.path.join(r,x)
    return None

def dataset():
    indoor,outdoor,both=[],[],[]
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            p=model(r['object_name'])
            if not p: continue
            t=r['gemma_location_type'].lower()
            if t=="indoor": indoor.append(p)
            elif t=="outdoor": outdoor.append(p)
            else: both.append(p)
    return indoor,outdoor,both

# ---------------- SCALE ----------------
def norm(o,target):
    c=[o.matrix_world@Vector(v) for v in o.bound_box]
    size=max((max(v.x for v in c)-min(v.x for v in c),
              max(v.y for v in c)-min(v.y for v in c),
              max(v.z for v in c)-min(v.z for v in c)))
    if size>0:
        s=target/size
        o.scale=(s,s,s)

def ground(o):
    z=min((o.matrix_world@Vector(v)).z for v in o.bound_box)
    o.location.z-=z

# ---------------- COLLISION ----------------
def bb(o):
    c=[o.matrix_world@Vector(v) for v in o.bound_box]
    return (Vector((min(v.x for v in c),min(v.y for v in c),min(v.z for v in c))),
            Vector((max(v.x for v in c),max(v.y for v in c),max(v.z for v in c))))

def inter(a,b,m=0.02):
    mn1,mx1=a; mn2,mx2=b
    return mn1.x<=mx2.x+m and mx1.x>=mn2.x-m and \
           mn1.y<=mx2.y+m and mx1.y>=mn2.y-m and \
           mn1.z<=mx2.z+m and mx1.z>=mn2.z-m

# ---------------- PLACE ----------------
def place(paths,area,d=0.85):
    P=[];objs=[]
    for _ in range(max(5,int(len(paths)*d))):
        for o in load(random.choice(paths)):
            if o.type!='MESH': continue
            norm(o,area*0.1)
            ground(o)
            for _ in range(40):
                o.location=(random.uniform(-area,area),
                            random.uniform(-area,area),
                            o.location.z)
                o.rotation_euler[2]=random.uniform(0,6.28)
                b=bb(o)
                if not any(inter(b,x) for x in P):
                    P.append(b); objs.append(o); break
            else: bpy.data.objects.remove(o,do_unlink=True)
    return objs

# ---------------- SCENE ----------------
def room():
    bpy.ops.mesh.primitive_cube_add(size=ROOM)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode='OBJECT')

def ground_plane():
    bpy.ops.mesh.primitive_plane_add(size=OUTDOOR)

def road():
    bpy.ops.mesh.primitive_plane_add(size=OUTDOOR/4)

# ---------------- LIGHT ----------------
def hdri():
    if not os.path.exists(HDRI_DIR): return
    f=[x for x in os.listdir(HDRI_DIR) if x.endswith((".hdr",".exr"))]
    if not f: return
    w=bpy.context.scene.world
    w.use_nodes=True
    n=w.node_tree.nodes; n.clear()
    env=n.new("ShaderNodeTexEnvironment")
    env.image=bpy.data.images.load(os.path.join(HDRI_DIR,random.choice(f)))
    bg=n.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value=random.uniform(0.5,3)
    out=n.new("ShaderNodeOutputWorld")
    w.node_tree.links.new(env.outputs["Color"],bg.inputs["Color"])
    w.node_tree.links.new(bg.outputs["Background"],out.inputs["Surface"])

def sun():
    bpy.ops.object.light_add(type='SUN')
    s=bpy.context.object
    s.data.energy=random.uniform(2,8)

def indoor_lights():
    for _ in range(random.randint(1,3)):
        bpy.ops.object.light_add(type='AREA')

# ---------------- CAMERA ----------------
def cams(n=10):
    C=[]
    for _ in range(n):
        bpy.ops.object.camera_add()
        c=bpy.context.object
        c.location=(random.uniform(-10,10),
                    random.uniform(-10,10),
                    random.uniform(3,10))
        d=Vector((0,0,0))-c.location
        c.rotation_euler=d.to_track_quat('-Z','Y').to_euler()
        C.append(c)
    return C

# ---------------- RENDER ----------------
def render_setup(out):
    s=bpy.context.scene
    vl=bpy.context.view_layer
    vl.use_pass_z=True
    vl.use_pass_normal=True
    vl.use_pass_object_index=True

    s.use_nodes=True
    t=s.node_tree; t.nodes.clear()
    rl=t.nodes.new('CompositorNodeRLayers')

    def outn(name,fmt):
        n=t.nodes.new('CompositorNodeOutputFile')
        n.base_path=os.path.join(out,name)
        n.format.file_format=fmt
        n.format.color_depth='32' if fmt=="OPEN_EXR" else '8'
        return n

    t.links.new(rl.outputs['Image'],outn("rgb","PNG").inputs[0])
    t.links.new(rl.outputs['Depth'],outn("depth","OPEN_EXR").inputs[0])
    t.links.new(rl.outputs['Normal'],outn("normal","OPEN_EXR").inputs[0])
    t.links.new(rl.outputs['IndexOB'],outn("mask","OPEN_EXR").inputs[0])

# ---------------- META + GRAPH ----------------
def graph(objs):
    G=[]
    for i in range(len(objs)):
        for j in range(i+1,len(objs)):
            G.append({
                "a":objs[i].name,
                "b":objs[j].name,
                "d":(objs[i].location-objs[j].location).length
            })
    return G

def save_meta(out,cams,objs):
    data={"cams":[],"graph":graph(objs)}
    for c in cams:
        data["cams"].append({"loc":list(c.location)})
    with open(os.path.join(out,"meta.json"),"w") as f:
        json.dump(data,f,indent=2)

# ---------------- COLMAP ----------------
def colmap(out,cams):
    with open(os.path.join(out,"cameras.txt"),"w") as f:
        f.write("# simple\n")
    with open(os.path.join(out,"images.txt"),"w") as f:
        f.write("# simple\n")

# ---------------- MAIN ----------------
def main():
    reset()
    setup_gpu()

    indoor,outdoor,both=dataset()

    if args["type"]=="indoor":
        pool=indoor+both
        room()
        objs=place(pool,ROOM/2-1)
        hdri(); indoor_lights()
    else:
        pool=outdoor+both
        ground_plane(); road()
        objs=place(pool,OUTDOOR/2-2)
        hdri(); sun()

    os.makedirs(args["out"],exist_ok=True)
    render_setup(args["out"])

    C=cams(10)
    s=bpy.context.scene

    for i,c in enumerate(C):
        s.camera=c
        s.render.filepath=os.path.join(args["out"],f"view_{i:03d}")
        bpy.ops.render.render(write_still=True)

    save_meta(args["out"],C,objs)
    colmap(args["out"],C)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(args["out"],"scene.blend"))

main()