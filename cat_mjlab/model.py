"""Original CAT + Dex3 model assembly, with no JAX dependency.

Assembly is intentionally identical to env_cat_wholebody. A numerical contract
check compares compiled model arrays between both implementations.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
from cat_mjlab import constants as consts
from cat_ppo.furniture.grippers import install_fixed_hands, hand_envelope, hand_sphere

def _numbers(values):
    return " ".join(format(float(value), ".10g") for value in values)

def assemble_training_xml(asset_root=None):
    """Original physics plus one field-query sphere/hand and one site/elbow.

    No obstacle geoms are introduced. The approved hand sphere (grippers.hand_sphere,
    the envelope that encloses the fixed fingers) is given explicit contact pairs with
    both thigh and both shin capsules, so fingers can no longer pass through the legs
    in simulation and hand/leg contact is reportable (task.hand_self_contact). Before
    this, the only hand/thigh pairs referenced the original palm capsule and the finger
    meshes penetrated the thighs in 8 of 17 recorded walks. Sphere geoms add no mass;
    their centers reuse CAT's palm sites.
    """
    asset_root = Path(asset_root or consts.ROOT_PATH).resolve()
    root = ET.parse(asset_root / "g1_mjx_feetonly_torque.xml").getroot()
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str(asset_root / mesh.get("file")))
    # Merge the scene wrapper's floor, sensor and rendering settings.  Keeping
    # its original floor attributes also preserves the explicit foot pairs.
    scene = ET.parse(asset_root / "scene_mjx_feetonly_flat_terrain.xml").getroot()
    for child in scene:
        if child.tag == "include":
            continue
        existing = root.find(child.tag)
        if existing is not None and child.tag in ("asset", "worldbody", "sensor"):
            existing.extend(deepcopy(list(child)))
        else:
            root.append(deepcopy(child))
    install_fixed_hands(root)
    bodies = {body.get("name"): body for body in root.findall(".//body")}
    for side in ("left", "right"):
        envelope = hand_envelope(side)
        sphere = hand_sphere(side)
        ET.SubElement(
            bodies[f"{side}_wrist_yaw_link"], "geom",
            name=f"furniture_{side}_hand_envelope", type="box", **{
                "class": "collision", "pos": _numbers(envelope["center"]),
                "size": _numbers(envelope["half_size"]), "density": "0",
                "contype": "0", "conaffinity": "0", "group": "3",
                "rgba": "0.25 0.6 0.85 0.3",
            },
        )
        wrist = bodies[f"{side}_wrist_yaw_link"]
        wrist.find(f"site[@name='{side}_palm']").set("pos", _numbers(sphere["center"]))
        ET.SubElement(wrist, "geom", name=f"furniture_{side}_hand_sphere", type="sphere",
                      pos=_numbers(sphere["center"]), size=str(sphere["radius"]),
                      density="0", contype="0", conaffinity="0", group="4", rgba="0.25 0.6 0.85 0.2")
        ET.SubElement(bodies[f"{side}_elbow_link"], "site", name=f"{side}_elbow_probe",
                      pos="0 0 0", size="0.005", group="5")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for side in ("left", "right"):
        for leg in ("left_thigh", "right_thigh", "left_shin", "right_shin"):
            ET.SubElement(contact, "pair", name=f"{side}_hand_sphere_{leg}",
                          geom1=f"furniture_{side}_hand_sphere", geom2=leg, condim="1")
    return ET.tostring(root, encoding="unicode")
