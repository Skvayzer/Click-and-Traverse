"""Native geometry, inertial and contact regressions for fixed Dex3 hands."""
import xml.etree.ElementTree as ET
import json

import mujoco
import numpy as np
import pytest

from cat_ppo.envs.g1 import constants
from cat_ppo.envs.g1.env_furniture import assemble_scene_xml
from cat_ppo.furniture.control import JOINT_NAMES, PROBE_SPECS, posture_actions
from cat_ppo.furniture.grippers import (
    ASSET_ROOT, ENVELOPE_MARGIN_M, geometry_contract, hand_envelope,
    hand_mesh_vertices, hand_sphere, validate_hand_envelopes, validate_hand_spheres,
)
from cat_ppo.furniture.scenes import generate_scene


@pytest.fixture(scope="module")
def model():
    scene = generate_scene(seed=3, family="table", difficulty="open_floor")
    return mujoco.MjModel.from_xml_string(assemble_scene_xml(scene))


@pytest.fixture(scope="module")
def sphere_model():
    scene = generate_scene(seed=3, family="table", difficulty="open_floor")
    root = ET.fromstring(assemble_scene_xml(scene))
    for side in ("left", "right"):
        sphere = hand_sphere(side)
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        # Query/visualization sphere does not modify physical contact geometry.
        ET.SubElement(wrist, "geom", name=f"furniture_{side}_hand_sphere",
                      type="sphere", pos=" ".join(map(str, sphere["center"])),
                      size=str(sphere["radius"]), density="0", contype="0", conaffinity="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def test_persisted_geometry_metadata_matches_current_assets_and_pose():
    assert json.loads((ASSET_ROOT / "geometry-contract.json").read_text()) == geometry_contract()


@pytest.mark.parametrize("posture", ["nominal", "raised", "tucked"])
@pytest.mark.parametrize("wrists", [(0., 0., 0.), (.7, -.5, .6), (-.9, .6, -.7)])
def test_fixed_spheres_contain_every_compiled_mesh_vertex_at_arbitrary_wrist_pose(sphere_model, posture, wrists):
    data = mujoco.MjData(sphere_model)
    data.qpos[:] = constants.DEFAULT_QPOS
    data.qpos[7:] += .8 * posture_actions(np.zeros(29), JOINT_NAMES, posture)
    for side, sign in (("left", 1), ("right", -1)):
        for axis, angle in zip(("roll", "pitch", "yaw"), wrists):
            joint = sphere_model.joint(f"{side}_wrist_{axis}_joint")
            data.qpos[int(joint.qposadr[0])] = sign * angle
    mujoco.mj_forward(sphere_model, data)
    names = {side: f"furniture_{side}_hand_sphere" for side in ("left", "right")}
    report = validate_hand_spheres(sphere_model, data)
    for side in ("left", "right"):
        sphere = hand_sphere(side)
        wrist = sphere_model.body(f"{side}_wrist_yaw_link").id
        geom = sphere_model.geom(names[side]).id
        # The center follows the articulated wrist frame, while radius is fixed.
        np.testing.assert_allclose(
            data.geom_xpos[geom],
            data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ sphere["center"],
            atol=1e-12,
        )
        assert report[side]["radius_m"] == sphere["radius"]
        assert report[side]["minimum_margin_m"] == pytest.approx(ENVELOPE_MARGIN_M, abs=2e-6)
        assert report[side]["checked_vertices"] > 80_000
        assert len(report[side]["checked_meshes"]) == 8
        assert sum("thumb" in mesh for mesh in report[side]["checked_meshes"]) == 3


def test_virtual_spheres_preserve_all_robot_inertias_and_contacts(model, sphere_model):
    np.testing.assert_array_equal(sphere_model.body_mass, model.body_mass)
    np.testing.assert_array_equal(sphere_model.body_inertia, model.body_inertia)
    np.testing.assert_array_equal(sphere_model.body_ipos, model.body_ipos)
    np.testing.assert_array_equal(sphere_model.body_iquat, model.body_iquat)
    for side in ("left", "right"):
        sphere = sphere_model.geom(f"furniture_{side}_hand_sphere").id
        assert sphere_model.geom_contype[sphere] == sphere_model.geom_conaffinity[sphere] == 0
        for field in ("geom_type", "geom_size", "geom_contype", "geom_conaffinity"):
            name = f"furniture_{side}_hand_envelope"
            np.testing.assert_array_equal(
                getattr(sphere_model, field)[sphere_model.geom(name).id],
                getattr(model, field)[model.geom(name).id],
            )


def test_sphere_validator_rejects_an_envelope_that_misses_hand_mesh(sphere_model):
    data = mujoco.MjData(sphere_model)
    data.qpos[:] = constants.DEFAULT_QPOS
    mujoco.mj_forward(sphere_model, data)
    names = {side: f"furniture_{side}_hand_sphere" for side in ("left", "right")}
    geom = sphere_model.geom(names["left"]).id
    original_radius = sphere_model.geom_size[geom, 0]
    try:
        sphere_model.geom_size[geom, 0] = original_radius - .01
        with pytest.raises(ValueError, match="left Dex3 hand not contained in sphere"):
            validate_hand_spheres(sphere_model, data, geom_names=names)
    finally:
        sphere_model.geom_size[geom, 0] = original_radius


@pytest.mark.parametrize("posture", ["nominal", "raised", "tucked"])
@pytest.mark.parametrize("wrists", [(0., 0., 0.), (.7, -.5, .6), (-.9, .6, -.7)])
def test_every_native_mesh_vertex_and_thumb_stays_inside_physical_box(model, posture, wrists):
    data = mujoco.MjData(model)
    data.qpos[:] = constants.DEFAULT_QPOS
    data.qpos[7:] += .8 * posture_actions(np.zeros(29), JOINT_NAMES, posture)
    for side, sign in (("left", 1), ("right", -1)):
        for axis, angle in zip(("roll", "pitch", "yaw"), wrists):
            joint = model.joint(f"{side}_wrist_{axis}_joint")
            data.qpos[int(joint.qposadr[0])] = sign * angle
    mujoco.mj_forward(model, data)
    report = validate_hand_envelopes(model, data)
    for side in ("left", "right"):
        assert report[side]["minimum_margin_m"] == pytest.approx(ENVELOPE_MARGIN_M, abs=2e-6)
        assert report[side]["checked_vertices"] > 80_000
        assert len(report[side]["checked_meshes"]) == 8
        assert sum("thumb" in mesh for mesh in report[side]["checked_meshes"]) == 3
        envelope = model.geom(f"furniture_{side}_hand_envelope").id
        # Independently require corner probes to coincide with native box corners.
        for name, _, _, _ in PROBE_SPECS:
            if name.startswith(side) and "hand_corner" in name:
                point = data.site_xpos[model.site("furniture_probe_" + name).id]
                local = data.geom_xmat[envelope].reshape(3, 3).T @ (point - data.geom_xpos[envelope])
                np.testing.assert_allclose(np.abs(local), model.geom_size[envelope], atol=1e-8)


def test_cat_body_coordinate_contract_and_source_distal_inertias(model):
    assert (model.nq, model.nv, model.nu, model.njnt) == (36, 35, 29, 30)
    assert [model.actuator(i).name for i in range(model.nu)] == JOINT_NAMES
    assert [model.joint(i).name for i in range(1, model.njnt)] == JOINT_NAMES
    assert all("rubber_hand" not in model.mesh(i).name for i in range(model.nmesh))
    root = ET.parse(constants.ROOT_PATH / "g1_mjx_feetonly_torque.xml").getroot()
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str(constants.ROOT_PATH / mesh.get("file")))
    ET.SubElement(root.find("worldbody"), "geom", name="floor", type="plane", size="0 0 .01")
    original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for side in ("left", "right"):
        wrist_id = model.body(f"{side}_wrist_yaw_link").id
        finger_ids = [i for i in range(model.nbody) if model.body(i).name.startswith(f"{side}_hand_")]
        assert len(finger_ids) == 7
        assert model.body_mass[wrist_id] == pytest.approx(.457415)
        assert model.body_mass[wrist_id] + model.body_mass[finger_ids].sum() == pytest.approx(.7811226)
        assert np.all(model.body_mass[finger_ids] > 0)
        assert np.all(model.body_inertia[finger_ids] > 0)
    # No unrelated body mass, joint range, armature or actuator force changed.
    for body_id in range(1, original.nbody):
        name = original.body(body_id).name
        if "wrist_yaw" not in name:
            assert model.body_mass[model.body(name).id] == original.body_mass[body_id]
    np.testing.assert_array_equal(model.jnt_range, original.jnt_range)
    np.testing.assert_array_equal(model.dof_armature, original.dof_armature)
    np.testing.assert_array_equal(model.actuator_forcerange, original.actuator_forcerange)


@pytest.mark.parametrize("side", ["left", "right"])
def test_real_contact_just_outside_thumb_is_detected_where_old_box_missed(side):
    scene = generate_scene(seed=3, family="table", difficulty="open_floor")
    root = ET.fromstring(assemble_scene_xml(scene))
    obstacle_body = ET.SubElement(root.find("worldbody"), "body", name="thumb_test_body", mocap="true")
    ET.SubElement(obstacle_body, "geom", name="thumb_test_obstacle", type="sphere",
                  size=".001", pos="0 0 3", contype="2", conaffinity="1", condim="3")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    data.qpos[:] = constants.DEFAULT_QPOS
    mujoco.mj_forward(model, data)
    wrist = model.body(f"{side}_wrist_yaw_link").id
    rotation = data.xmat[wrist].reshape(3, 3)
    direction = -1. if side == "left" else 1.
    thumb = np.concatenate([vertices for name, vertices in hand_mesh_vertices(side).items() if "thumb" in name])
    extreme = thumb[np.argmax(direction * thumb[:, 1])].copy()
    # A 1 mm sphere 3 mm beyond the outer thumb sits inside the 5 mm guard.
    point = extreme.copy()
    point[1] += direction * .003
    assert abs(point[1]) > .04 + .001  # The previous generic envelope misses it.
    obstacle = model.geom("thumb_test_obstacle").id
    envelope = model.geom(f"furniture_{side}_hand_envelope").id

    def contacts_at(local_point):
        data.mocap_pos[0] = data.xpos[wrist] + rotation @ local_point - model.geom_pos[obstacle]
        mujoco.mj_forward(model, data)
        return [contact for contact in data.contact
                if set(contact.geom) == {obstacle, envelope} and contact.dist < 0]

    assert contacts_at(point)
    bounds = hand_envelope(side)
    point[1] = bounds["center"][1] + direction * (bounds["half_size"][1] + .003)
    assert not contacts_at(point)
