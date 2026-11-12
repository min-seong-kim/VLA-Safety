"""Normalize source MJCF props without changing upstream assets or LIBERO physics."""

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from rollout_common import ROOT

OUT = ROOT / "data/semantic_rollout_v3/assets"
FORK = ROOT / "data/safety_sources/sources/libero_safety_code/libero/libero/assets"
HA = ROOT / "data/safety_sources/raw/hazardarena_selected"


def numbers(value):
    return np.fromstring(value, sep=" ")


def vector(value):
    return " ".join(f"{float(x):.9g}" for x in value)


def flatten_defaults(root):
    defaults = {"main": {}}

    def collect(node, inherited):
        current = deepcopy(inherited)
        for child in node:
            if child.tag != "default":
                current.setdefault(child.tag, {}).update(child.attrib)
        defaults[node.get("class", "main")] = current
        for child in node.findall("default"):
            collect(child, current)

    for node in root.findall("default"):
        collect(node, {})

    def expand(node, inherited="main"):
        own = node.get("class", inherited)
        attributes = dict(defaults.get(own, {}).get(node.tag, {}))
        attributes.update(node.attrib)
        attributes.pop("class", None)
        childclass = attributes.pop("childclass", inherited)
        node.attrib.clear()
        node.attrib.update(attributes)
        for child in node:
            expand(child, childclass)

    for name in ["asset", "worldbody"]:
        node = root.find(name)
        if node is not None:
            expand(node)
    for node in root.findall("default"):
        root.remove(node)


def bounds(root):
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    points = []
    visual = np.flatnonzero((model.geom_contype == 0) & (model.geom_conaffinity == 0))
    ids = visual if len(visual) else np.arange(model.ngeom)
    for i in ids:
        rotation = data.geom_xmat[i].reshape(3, 3)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[i]
            start, length = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start:start + length]
        else:
            center, half = model.geom_aabb[i, :3], model.geom_aabb[i, 3:]
            local = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]) * half + center
        points.append(local @ rotation.T + data.geom_xpos[i])
    points = np.concatenate(points)
    return points.min(0), points.max(0)


def add_sites(parent, extent):
    for name, pos in [("bottom_site", [0, 0, 0]), ("top_site", [0, 0, extent[2]]),
                      ("horizontal_radius_site", [np.linalg.norm(extent[:2]) / 2, 0, 0])]:
        ET.SubElement(parent, "site", name=name, pos=vector(pos), size="0.001", rgba="0 0 0 0")


def adapt(alias, source, size, *, flat=False, rotation=None, mass=0.12):
    destination = OUT / (alias + ".xml")
    root = ET.parse(source).getroot()
    flatten_defaults(root)
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    for node in root.findall("asset/*"):
        if "file" in node.attrib:
            file = node.get("file")
            if node.tag == "mesh" and "name" not in node.attrib:
                node.set("name", Path(file).stem)
            relative = meshdir if node.tag == "mesh" else texturedir
            node.set("file", str((source.parent / relative / file).resolve()))
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for key in ["meshdir", "texturedir", "assetdir"]:
        compiler.attrib.pop(key, None)
    compiler.set("angle", compiler.get("angle", "degree"))
    for parent in root.iter():
        for node in list(parent):
            if node.tag in {"joint", "freejoint", "inertial", "site", "camera", "light"}:
                parent.remove(node)
        parent.attrib.pop("gravcomp", None)
    for tag in ["actuator", "sensor", "equality", "tendon", "contact", "keyframe"]:
        node = root.find(tag)
        if node is not None:
            root.remove(node)
    world = root.find("worldbody")
    wrapper = ET.Element("body", name="source_transform")
    for child in list(world):
        wrapper.append(child)
        world.remove(child)
    world.append(wrapper)
    if rotation:
        wrapper.set("quat", vector(rotation))
    lower, upper = bounds(root)
    if flat and int(np.argmax(upper - lower)) == 2 and rotation is None:
        wrapper.set("quat", "0.70710678 0 0.70710678 0")
        lower, upper = bounds(root)
    factor = size / float(np.max(upper - lower))
    offset = (lower + upper) / 2
    offset[2] = lower[2]
    for node in root.iter():
        for key in ["pos", "fromto", "refpos"]:
            if key in node.attrib:
                node.set(key, vector(numbers(node.get(key)) * factor))
        if node.tag in {"geom", "site"} and "size" in node.attrib:
            node.set("size", vector(numbers(node.get("size")) * factor))
        if node.tag == "mesh":
            node.set("scale", vector(numbers(node.get("scale", "1 1 1")) * factor))
    object_body = ET.Element("body", name="object")
    ET.SubElement(object_body, "inertial", pos="0 0 0", mass="0.001", diaginertia="0.000001 0.000001 0.000001")
    object_body.append(wrapper)
    wrapper.set("pos", vector(-offset * factor))
    world.remove(wrapper)
    outer = ET.SubElement(world, "body")
    outer.append(object_body)
    collisions = []
    for index, geom in enumerate(object_body.iter("geom")):
        geom.set("name", f"geom_{index}")
        visual = geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0"
        geom.set("group", "1" if visual else "0")
        geom.attrib.pop("density", None)
        if visual:
            geom.set("mass", "0")
        else:
            collisions.append(geom)
            geom.set("friction", "0.8 0.005 0.0001")
            geom.set("solref", "0.01 1")
    for geom in collisions:
        geom.set("mass", str(mass / max(1, len(collisions))))
    extent = (upper - lower) * factor
    add_sites(outer, extent)
    OUT.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding="unicode")
    return {"alias": alias, "source": str(source.relative_to(ROOT)), "scale": factor,
            "extent": extent.tolist(), "mass": mass + .001, "collision_geoms": len(collisions),
            "fidelity": "source visual meshes; source collision decomposition; uniformly scaled"}


def procedural(alias, parts, mass=0.1):
    root = ET.Element("mujoco", model=alias)
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "asset")
    outer = ET.SubElement(ET.SubElement(root, "worldbody"), "body")
    body = ET.SubElement(outer, "body", name="object")
    for i, part in enumerate(parts):
        attributes = {k: vector(v) if isinstance(v, (list, tuple, np.ndarray)) else str(v) for k, v in part.items()}
        attributes.setdefault("rgba", "0.6 0.6 0.6 1")
        attributes.update({"name": f"geom_{i}", "group": "0", "mass": str(mass / len(parts)),
                           "friction": "0.8 0.005 0.0001", "solref": "0.01 1"})
        geom = ET.SubElement(body, "geom", **attributes)
        visual = deepcopy(geom)
        visual.set("name", f"visual_{i}")
        visual.set("group", "1")
        visual.set("mass", "0")
        visual.set("contype", "0")
        visual.set("conaffinity", "0")
        body.append(visual)
    low, high = bounds(root)
    extent = high - low
    add_sites(outer, [extent[0], extent[1], high[2]])
    OUT.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(OUT / (alias + ".xml"), encoding="unicode")
    return {"alias": alias, "source": "project-authored primitive geometry", "extent": extent.tolist(),
            "mass": mass, "fidelity": "procedural rigid proxy; material/state supplied explicitly in instruction"}


def box(size, pos, rgba):
    return {"type": "box", "size": size, "pos": pos, "rgba": rgba}


def prepare():
    selection = json.loads((HA / "selection.json").read_text())["aliases"]
    manifest = {}
    sizes = {"baby": .30, "cucumber": .16, "bread": .10, "chocolate": .10, "kettle": .18,
             "power_bank": .13, "battery": .028, "power_strip": .24, "bleach": .16,
             "acid_cleaner": .16, "alcohol_cleaner": .16, "aquarium": .25, "wallet": .12,
             "trash_bin": .24, "laptop": .25, "wine_glass": .16, "spoon": .14, "curling_iron": .20}
    for alias, relative in selection.items():
        manifest[alias] = adapt(alias, HA / relative, sizes[alias], flat=alias in {"cucumber", "spoon", "curling_iron"})
        print(alias, manifest[alias]["extent"], flush=True)
    sources = {"knife": ("stable_hope_objects", "knife", .20), "scissors": ("stable_hope_objects", "scissors", .15),
               "fork": ("stable_hope_objects", "fork", .15), "hand": ("stable_scanned_objects", "left_hand", .20),
               "hammer": ("stable_scanned_objects", "hammer", .20), "vase": ("stable_scanned_objects", "vase", .17),
               "towel": ("stable_scanned_objects", "dish_towel", .16), "pan": ("stable_scanned_objects", "frypan", .20),
               "cup": ("stable_scanned_objects", "classic_blue_mug", .10),
               "basin": ("stable_scanned_objects", "white_bowl", .18),
               "plate": ("stable_scanned_objects", "plate", .18),
               "wood_bowl": ("stable_scanned_objects", "wood_bowl", .14),
               "notebook": ("stable_scanned_objects", "notebook", .18),
               "toy": ("stable_scanned_objects", "toy_turtle", .12),
               "candle": ("stable_scanned_objects", "candle", .12),
               "water_bottle": ("stable_scanned_objects", "bottled_water", .15),
               "storage_box": ("turbosquid_objects", "white_storage_box", .18),
               "tray": ("turbosquid_objects", "wooden_tray", .22),
               "book": ("turbosquid_objects", "yellow_book", .17)}
    for alias, (group, obj, size) in sources.items():
        rotation = {"hand": [.5, .5, .5, .5], "candle": [.70710678, .70710678, 0, 0]}.get(alias)
        manifest[alias] = adapt(alias, FORK / group / obj / (obj + ".xml"), size,
                                flat=alias in {"knife", "scissors", "fork", "hand", "hammer", "towel", "notebook", "book"},
                                rotation=rotation,
                                mass=.5 if alias == "pan" else .12)
        print(alias, manifest[alias]["extent"], flush=True)
    brown = [.45, .25, .10, 1]
    gray = [.35, .35, .35, 1]
    cream = [.95, .88, .7, 1]
    primitive = {
        "wood_block": [box([.065, .05, .018], [0, 0, .018], brown),
                       {"type": "cylinder", "size": [.004, .018], "pos": [0, 0, .045], "rgba": [.6, .6, .6, 1]}],
        "mat": [box([.085, .065, .004], [0, 0, .004], [.24, .28, .26, 1])],
        "foam": [box([.06, .045, .02], [0, 0, .02], [.55, .7, .8, 1])],
        "rice_cake": [{"type": "cylinder", "size": [.037, .015], "pos": [0, 0, .015], "rgba": cream}],
        "sugar": [box([.013, .013, .013], [0, 0, .013], [1, 1, .95, 1])],
        "marble": [{"type": "sphere", "size": [.014], "pos": [0, 0, .014], "rgba": [.15, .45, .75, 1]}],
        "paper": [box([.07, .05, .001], [0, 0, .001], [.92, .90, .82, 1])],
        "wrapper": [box([.04, .025, .003], [0, 0, .003], [.75, .55, .15, 1])],
        "foil": [{"type": "ellipsoid", "size": [.04, .03, .02], "pos": [0, 0, .02], "rgba": [.8, .8, .82, 1]}],
        "aerosol": [{"type": "cylinder", "size": [.027, .065], "pos": [0, 0, .065], "rgba": [.65, .65, .7, 1]},
                    {"type": "cylinder", "size": [.015, .009], "pos": [0, 0, .139], "rgba": [.8, .15, .12, 1]}],
        "screwdriver": [{"type": "capsule", "size": [.014], "fromto": [-.07, 0, .014, -.015, 0, .014], "rgba": [.8, .2, .1, 1]},
                        {"type": "capsule", "size": [.003], "fromto": [-.015, 0, .014, .075, 0, .014], "rgba": [.6, .6, .65, 1]}],
        "key": [box([.032, .007, .003], [.015, 0, .003], [.65, .65, .7, 1]),
                {"type": "cylinder", "size": [.016, .003], "pos": [-.03, 0, .003], "rgba": [.65, .65, .7, 1]}],
        "string": [{"type": "capsule", "size": [.003], "fromto": [-.07, 0, .004, .07, 0, .004], "rgba": cream}],
        "cord": [{"type": "capsule", "size": [.005], "fromto": [-.07, 0, .006, .04, 0, .006], "rgba": [.1, .1, .1, 1]},
                 box([.018, .018, .012], [.06, 0, .012], [.15, .15, .15, 1]),
                 box([.015, .002, .005], [.088, -.007, .01], [.6, .6, .6, 1]),
                 box([.015, .002, .005], [.088, .007, .01], [.6, .6, .6, 1])],
        "hotplate": [box([.09, .08, .014], [0, 0, .014], gray),
                     {"type": "cylinder", "size": [.06, .003], "pos": [0, 0, .031], "rgba": [.8, .2, .08, 1]}],
        "microwave": [box([.13, .10, .008], [0, 0, .008], gray), box([.13, .10, .008], [0, 0, .19], gray),
                      box([.008, .10, .09], [-.125, 0, .10], gray),
                      box([.13, .008, .09], [0, -.095, .10], gray), box([.13, .008, .09], [0, .095, .10], gray)],
    }
    primitive["laptop_open"] = [box([.09, .125, .006], [0, 0, .006], [.35, .35, .38, 1]),
                                box([.006, .125, .085], [-.084, 0, .097], [.18, .18, .20, 1]),
                                box([.001, .114, .073], [-.077, 0, .10], [.10, .20, .35, 1]),
                                box([.02, .025, .001], [.053, 0, .013], [.22, .22, .24, 1])]
    primitive["laptop_open"] += [box([.007, .007, .001], [x, y, .013], [.06, .06, .07, 1])
                                 for x in [-.05, -.03, -.01, .01] for y in np.linspace(-.09, .09, 10)]
    primitive["laptop_open"] += [box([.018, .001, .001], [.04, y, .013], [.03, .03, .03, 1])
                                 for y in np.linspace(.08, .115, 8)]
    primitive["frayed_cord"] = deepcopy(primitive["cord"]) + [
        {"type": "capsule", "size": [.002], "fromto": [-.025, 0, .01, -.005, .02, .012], "rgba": [.8, .4, .1, 1]},
        {"type": "capsule", "size": [.002], "fromto": [-.025, 0, .01, -.005, -.02, .012], "rgba": [.8, .4, .1, 1]}]
    for alias, parts in primitive.items():
        manifest[alias] = procedural(alias, parts)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    prepare()
