"""Render actual fixed Dex3 meshes from four views; audit enclosing centers."""
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")


def audit(side):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial import ConvexHull
    from cat_ppo.furniture.grippers import hand_mesh_vertices, hand_sphere
    meshes = hand_mesh_vertices(side)
    vertices = np.concatenate(list(meshes.values()))
    palm = meshes[f"{side}_hand_palm_link"]
    palm_center = (palm.min(0) + palm.max(0)) / 2
    current = hand_sphere(side)
    hull = vertices[ConvexHull(vertices).vertices] * 1000
    start = np.r_[np.asarray(current["center"]) * 1000, (current["radius"]-.005)*1000]
    result = minimize(lambda q: q[3], start, jac=lambda q: np.array([0., 0., 0., 1.]),
        method="SLSQP", constraints=[{"type": "ineq",
            "fun": lambda q: q[3]-np.linalg.norm(hull-q[:3], axis=1),
            "jac": lambda q: np.column_stack([(hull-q[:3])/np.linalg.norm(hull-q[:3], axis=1)[:, None],
                                              np.ones(len(hull))])}],
        options={"ftol": 1e-10, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(result.message)
    center = result.x[:3]/1000
    radius = float(np.linalg.norm(vertices-center, axis=1).max()+.005)
    return {"current": current, "optimal": {"center": center.tolist(), "radius": radius},
            "palm": {"center": palm_center.tolist(),
                     "required_radius": float(np.linalg.norm(vertices-palm_center, axis=1).max()+.005)},
            "hand_dimensions_m": (vertices.max(0)-vertices.min(0)).tolist(),
            "current_to_optimal_center_m": float(np.linalg.norm(center-np.asarray(current["center"]))),
            "margin_m": .005, "geometry_changed": False}


def main():
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from cat_ppo.furniture.grippers import ASSET_ROOT, fixed_wrist, hand_mesh_vertices
    reports = {side: audit(side) for side in ("left", "right")}
    report = reports["left"]
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    for name in hand_mesh_vertices("left"):
        ET.SubElement(asset, "mesh", name=name, file=str(ASSET_ROOT/"assets"/(name+".STL")))
    world = ET.SubElement(root, "worldbody")
    wrist = fixed_wrist("left")
    wrist.set("pos", "0 0 0")
    for body in wrist.iter("body"):
        for item in list(body):
            if item.tag == "inertial" or (item.tag == "geom" and "_hand_" not in item.get("mesh", "")):
                body.remove(item)
            elif item.tag == "geom":
                item.attrib.pop("class", None)
                item.set("type", "mesh")
                item.set("rgba", ".45 .49 .54 1")
    world.append(wrist)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="700", offheight="590", fovy="35")
    ET.SubElement(visual, "headlight", ambient=".5 .5 .5", diffuse=".6 .6 .6")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Independent compiled-model containment; catches a wrong source transform.
    validation = {}
    for geom in range(model.ngeom):
        mesh = int(model.geom_dataid[geom]); start=int(model.mesh_vertadr[mesh]); count=int(model.mesh_vertnum[mesh])
        vertices=np.einsum('ij,nj->ni',data.geom_xmat[geom].reshape(3,3),model.mesh_vert[start:start+count])+data.geom_xpos[geom]
        for name in ("current", "optimal"):
            margin=float((report[name]["radius"]-np.linalg.norm(vertices-report[name]["center"],axis=1)).min())
            validation[name]=min(validation.get(name, float("inf")), margin)
    if min(validation.values()) < .005-2e-6:
        raise ValueError(f"Sphere misses compiled mesh: {validation}")
    report["compiled_mesh_minimum_margin_m"] = validation
    canvas = Image.new("RGB", (1400, 1460), "white")
    font_path = next((path for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(path).is_file()), None)
    make_font = lambda size: ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size=size)
    font=make_font(26); small=make_font(22)
    draw=ImageDraw.Draw(canvas)
    draw.text((35,20), "Dex3 hand: current sphere and better center", font=make_font(36),fill="#19374b")
    legend=[("#2173bc", "Current: radius 10.35 cm"), ("#25945a", "Optimized: radius 9.73 cm"), ("#cc612c", "Palm center")]
    for x,(color,text) in zip([35,480,990],legend):
        draw.line((x,89,x+30,89),fill=color,width=4); draw.text((x+40,74),text,font=small,fill=color)
    focal=590/(2*np.tan(np.deg2rad(35)/2))
    views=[("Palm-facing view",90,0),("Back view",270,0),("Side view",0,0),("Oblique view",35,-30)]
    with mujoco.Renderer(model,height=590,width=700) as renderer:
        for index,(title,azimuth,elevation) in enumerate(views):
            camera=mujoco.MjvCamera(); camera.lookat[:]=report["optimal"]["center"]
            camera.distance=.57; camera.azimuth=azimuth; camera.elevation=elevation
            renderer.disable_segmentation_rendering(); renderer.update_scene(data,camera=camera)
            rgb=renderer.render().copy(); cams=renderer.scene.camera
            cam=(np.asarray(cams[0].pos)+np.asarray(cams[1].pos))/2
            forward=np.asarray(cams[0].forward); up=np.asarray(cams[0].up); right=np.cross(forward,up)
            renderer.enable_segmentation_rendering(); seg=renderer.render(); rgb[seg[:,:,0]<0]=255
            ox=(index%2)*700; oy=135+(index//2)*615
            canvas.paste(Image.fromarray(rgb),(ox,oy))
            draw.text((ox+25,oy+8),title,font=font,fill="#19374b")
            def project(points):
                delta=np.asarray(points)-cam; depth=np.einsum('ni,i->n',delta,forward)
                return np.column_stack([ox+350+focal*np.einsum('ni,i->n',delta,right)/depth,
                                        oy+295-focal*np.einsum('ni,i->n',delta,up)/depth])
            for name,color in [("current","#2173bc"),("optimal","#25945a")]:
                center=np.asarray(report[name]["center"]); radius=report[name]["radius"]
                # Perspective silhouette of a true sphere: tangent circle to camera.
                to_cam=cam-center; distance=np.linalg.norm(to_cam); normal=to_cam/distance
                axis=np.cross(normal,up); axis/=np.linalg.norm(axis); other=np.cross(normal,axis)
                circle_center=center+(radius*radius/distance)*normal
                circle_radius=radius*np.sqrt(1-(radius/distance)**2)
                t=np.linspace(0,2*np.pi,361)
                points=circle_center+circle_radius*(np.cos(t)[:,None]*axis+np.sin(t)[:,None]*other)
                draw.line([tuple(p) for p in project(points)], fill=color,width=3)
                x,y=project([center])[0]; draw.ellipse((x-6,y-6,x+6,y+6),fill=color,outline="white",width=1)
            x,y=project([report["palm"]["center"]])[0]
            draw.rectangle((x-6,y-6,x+6,y+6),fill="#cc612c",outline="white",width=1)
            # Actual 5 cm reference segment in the camera plane through the hand center.
            a=np.asarray(report["optimal"]["center"])-.085*up-.025*right
            p=project([a,a+.05*right]); draw.line([tuple(v) for v in p],fill="#19374b",width=3)
            draw.text((float(p[0,0]),float(p[0,1])+8),"5 cm",font=small,fill="#19374b")
    draw.text((35,1382),"Palm-centered enclosing sphere would need radius 14.10 cm. Both outlines include every finger + 5 mm.",font=small,fill="#19374b")
    draw.text((35,1420),"Actual fixed-pose meshes. Right hand is mirrored. This inspection does not change the training geometry.",font=small,fill="#5d6d77")
    output=ROOT/"docs/assets/hand-sphere-inspection"; output.mkdir(parents=True,exist_ok=True)
    path=output/"hand-sphere-views.png"; canvas.save(path)
    (output/"geometry-audit.json").write_text(json.dumps(reports,indent=2)+"\n")
    target=Path.home()/"Downloads/CAT-hand-sphere-views.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path,target)
    print(target)


if __name__ == "__main__":
    main()
