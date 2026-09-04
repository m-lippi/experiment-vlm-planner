import math
from pathlib import Path
import numpy as np
import trimesh

OUT = Path(__file__).parent / 'meshes'
OUT.mkdir(exist_ok=True)

def loft_elliptical_x(rings, n=40):
    # rings: [(x, width_y, thickness_z), ...]
    verts=[]
    for x,w,t in rings:
        for i in range(n):
            a=2*math.pi*i/n
            y=(w/2)*math.cos(a)
            z=(t/2)*math.sin(a)
            verts.append([x,y,z])
    faces=[]
    nr=len(rings)
    for r in range(nr-1):
        for i in range(n):
            j=(i+1)%n
            a=r*n+i; b=r*n+j; c=(r+1)*n+j; d=(r+1)*n+i
            faces += [[a,b,c],[a,c,d]]
    # end caps
    c0=len(verts); verts.append([rings[0][0],0,0])
    c1=len(verts); verts.append([rings[-1][0],0,0])
    for i in range(n):
        j=(i+1)%n
        faces += [[c0,j,i]]
        a=(nr-1)*n+i; b=(nr-1)*n+j
        faces += [[c1,a,b]]
    return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)

def box(name, extents, center=(0,0,0)):
    m=trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    m.export(OUT/name)

def cylinder(name, radius, height, center=(0,0,0), axis='z', sections=48):
    m=trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    # cylinder default axis z
    if axis=='x': m.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2,[0,1,0]))
    if axis=='y': m.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2,[1,0,0]))
    m.apply_translation(center)
    m.export(OUT/name)

# Wooden handle: hand-sized, tapered, slightly flattened.
rings=[
    (-0.185,0.026,0.015),
    (-0.165,0.031,0.017),
    (-0.135,0.034,0.018),
    (-0.105,0.030,0.017),
    (-0.080,0.036,0.018),
    (-0.055,0.043,0.018),
    (-0.040,0.046,0.017),
]
handle=loft_elliptical_x(rings)
handle.export(OUT/'wood_handle.stl')

# Ferrule and subtle front/back plates.
box('ferrule_body.stl', (0.070,0.052,0.017), (-0.005,0,0))
box('ferrule_face_top.stl', (0.068,0.050,0.0012), (-0.005,0,0.0091))
box('ferrule_face_bottom.stl', (0.068,0.050,0.0012), (-0.005,0,-0.0091))
# Crimp bands across ferrule.
for idx,x in enumerate([-0.024,-0.010,0.006]):
    box(f'ferrule_crimp_{idx+1}.stl', (0.0022,0.053,0.019), (x,0,0))

# Bristle pack: multiple tapered slabs for visible variation.
def tapered_prism_x(x0,x1,w0,w1,t0,t1, yoff=0,zoff=0):
    # rectangular prism with changing width/thickness along x
    verts=[]
    for x,w,t in [(x0,w0,t0),(x1,w1,t1)]:
        verts += [[x,-w/2+yoff,-t/2+zoff],[x,w/2+yoff,-t/2+zoff],[x,w/2+yoff,t/2+zoff],[x,-w/2+yoff,t/2+zoff]]
    faces=[[0,1,2],[0,2,3],[4,6,5],[4,7,6],[0,4,5],[0,5,1],[1,5,6],[1,6,2],[2,6,7],[2,7,3],[3,7,4],[3,4,0]]
    return trimesh.Trimesh(vertices=np.array(verts),faces=np.array(faces),process=True)

# central bristle mass
m=tapered_prism_x(0.030,0.110,0.050,0.047,0.014,0.010)
m.export(OUT/'bristles_center.stl')
# side wisps, slightly longer/shorter and offset
specs=[
    ('bristles_side_l.stl',0.031,0.108,0.008,0.007,0.012,0.008,-0.021,0.0005),
    ('bristles_side_r.stl',0.031,0.113,0.008,0.006,0.012,0.007, 0.021,-0.0005),
    ('bristles_top.stl',0.032,0.106,0.034,0.032,0.0032,0.0022,0,0.0062),
    ('bristles_bottom.stl',0.032,0.109,0.034,0.031,0.0032,0.0020,0,-0.0062),
]
for name,x0,x1,w0,w1,t0,t1,y,z in specs:
    tapered_prism_x(x0,x1,w0,w1,t0,t1,y,z).export(OUT/name)

# Hanging hole visual inserts on both faces (dark recess effect)
cylinder('hole_top.stl',0.0048,0.0015,(-0.158,0,0.0090),'z')
cylinder('hole_bottom.stl',0.0048,0.0015,(-0.158,0,-0.0090),'z')

# Small metal ferrule pins/rivets on both sides.
for side,z in [('top',0.0099),('bottom',-0.0099)]:
    for i,x in enumerate([-0.018,0.014]):
        cylinder(f'rivet_{side}_{i+1}.stl',0.0023,0.0012,(x,0,z),'z',32)

# End cap accent, tiny flattened ellipse-ish cylinder at handle end.
cylinder('handle_end_cap.stl',0.008,0.0015,(-0.186,0,0),'x',40)

print('Generated', len(list(OUT.glob('*.stl'))), 'STL files')
