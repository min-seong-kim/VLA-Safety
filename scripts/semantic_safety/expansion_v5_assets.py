"""Authored, inspectable MJCF props for the expansion catalog's initial scenes.

These are visual review assets, not deformable tissue, chemical, or thermal
simulators. Shapes without an exact source mesh are explicitly marked schematic.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rollout_common import ROOT
from rollout_asset_adapter import bounds
from semantic_pairs_v4 import primitive_root, sites, vector

OUT = ROOT / "data/semantic_safety_observations_v5"
ASSETS = OUT / "assets"
V6 = ROOT / "data/semantic_safety_transport_v6/assets"
OLD = ROOT / "data/semantic_rollout_v3/assets"
METAL = [.56, .59, .62, 1]
DARK = [.13, .16, .19, 1]
BLUE = [.12, .35, .47, 1]
WOOD = [.58, .37, .18, 1]
WHITE = [.89, .87, .80, 1]
SKIN = [.68, .45, .31, 1]
GREEN = [.18, .43, .16, 1]


def box(size, pos, color=METAL, **kwargs):
    return dict(type="box", size=size, pos=pos, rgba=color, **kwargs)


def ell(size, pos, color=METAL):
    return dict(type="ellipsoid", size=size, pos=pos, rgba=color)


def cyl(radius, half, pos, color=METAL, **kwargs):
    return dict(type="cylinder", size=[radius, half], pos=pos, rgba=color, **kwargs)


def rod(a, b, radius=.006, color=METAL):
    return dict(type="capsule", size=[radius], fromto=[*a, *b], rgba=color)


def ring(radius, z, color=METAL, tube=.004, center=(0, 0)):
    angles = np.linspace(0, 2 * math.pi, 21)
    return [rod([center[0] + radius*math.cos(a), center[1] + radius*math.sin(a), z],
                [center[0] + radius*math.cos(b), center[1] + radius*math.sin(b), z], tube, color)
            for a, b in zip(angles[:-1], angles[1:])]


def tray(color=WHITE, x=.09, y=.065, height=.028):
    return [box([x,y,.003],[0,0,.003],color),
            box([x,.003,height/2],[0,y,height/2],color),
            box([x,.003,height/2],[0,-y,height/2],color),
            box([.003,y,height/2],[x,0,height/2],color),
            box([.003,y,height/2],[-x,0,height/2],color)]


def bowl(color=WHITE, radius=.067, height=.045):
    return [cyl(radius*.84,.003,[0,0,.003],color)] + [
        rod([radius*math.cos(a),radius*math.sin(a),.008],
            [radius*math.cos(a),radius*math.sin(a),height],.006,color)
        for a in np.linspace(0, 2*math.pi, 40, endpoint=False)] + ring(radius,height,color,.006)


def person(kind="person"):
    # A neutral, clothed, non-graphic tabletop-scale avatar.
    p = [ell([.055,.04,.078],[.018,0,.15],BLUE),
         ell([.039,.034,.045],[.018,0,.258],SKIN),
         ell([.041,.035,.019],[.018,0,.29],DARK),
         rod([.02,-.032,.12],[.04,-.04,.035],.016,DARK),
         rod([.02,.032,.12],[.04,.04,.035],.016,DARK),
         ell([.034,.02,.014],[.06,-.04,.018],DARK),
         ell([.034,.02,.014],[.06,.04,.018],DARK),
         rod([.015,-.045,.20],[-.032,-.07,.095],.015,BLUE),
         rod([.015,.045,.20],[-.035,.072,.095],.015,BLUE),
         ell([.027,.014,.009],[-.05,-.07,.09],SKIN),
         ell([.027,.014,.009],[-.05,.072,.09],SKIN)]
    for y in [-.013,.013]:
        p.append(ell([.002,.004,.004],[-.018,y,.264],DARK))
    return p


def hand(kind="hand", gloved=False):
    color = [.38,.66,.78,1] if gloved else SKIN
    p = [ell([.048,.035,.013],[-.015,0,.018],color),
         rod([-.10,0,.021],[-.045,0,.021],.024,color),
         box([.025,.028,.018],[-.115,0,.02],BLUE)]
    for y,length in [(-.023,.055),(-.008,.067),(.008,.062),(.023,.048)]:
        p.append(rod([.015,y,.019],[length,y,.019],.007,color))
    p.append(rod([-.025,.028,.018],[.009,.055,.017],.009,color))
    if kind == "arm":
        p += [rod([-.11,0,.022],[-.20,0,.03],.025,color)]
    return p


def paper(name):
    p = [box([.085,.06,.0015],[0,0,.0015],WHITE)]
    if any(w in name for w in ["document","contract","certificate","score","record","map","check","paper"]):
        p += [box([.052,.00065,.0002],[0,y,.0032],[.25,.29,.32,1])
              for y in np.linspace(-.037,.037,9)]
    return p


def word_in(word, text):
    return re.search(r"\b"+re.escape(word)+r"(?:s|es)?\b",text) is not None


def choose_kind(name, material=""):
    n = name.lower()
    if n.endswith("bottle"):
        return "bottle"
    if n == "upright knife cradle":
        return "knife_cradle"
    if n == "upright paper book":
        return "upright_book"
    if "glass fragments" in n:
        return "glass_fragments"
    if n == "toy turtle":
        return "mesh:toy"
    if n == "upright saw cradle":
        return "saw_cradle"
    if "gloves" in n or "gloved hand" in n:
        return "mesh:hand"
    exact = {
        "wooden joint":"wood", "gap between the blades":"mesh:scissors", "loose sewing thread":"cord",
        "retrieval hook":"hook", "heat-resistant mat":"cloth", "heat-resistant basin":"bowl",
        "ice cube":"block", "houseplant fertilizer":"granules", "wood stain":"bottle",
        "labeled detergent tub":"box_container", "detached brake part":"mechanism",
        "finished wooden shelf":"wood", "acidic descaler":"bottle", "cat food":"granules",
        "onion pieces":"food", "rabbit hay":"hay", "detergent granules":"granules",
        "turtle pellets":"granules", "polystyrene crumbs":"granules", "packing twine pieces":"cord",
        "surface disinfectant":"bottle", "empty utility sink":"box_container", "equipment-cleaning liquid":"bottle",
        "sterile instrument surface":"panel", "sterile cover":"cloth", "unwashed waste-transport tub":"box_container",
        "refillable lighter":"electronics", "cotton pressing sample":"cloth", "designated woodburning scrap":"wood",
        "fitted insulating terminal cover":"block", "insulating spacer":"block", "conductive spacer":"block",
        "blank practice sheet":"paper", "softwood craft blank":"wood", "glass induction-cooker surface":"panel",
        "engraving practice plaque":"panel", "only original signed agreement":"paper", "rejected print sample":"paper",
        "opaque sticker":"paper", "unpowered display mock-up":"electronics", "blank display stand":"panel",
        "calibrated measurement probe":"point_tool", "working pump seal":"ring", "working hinge joint":"mechanism",
        "craft alignment hole":"wood", "memory card containing the only copy of the experiment data":"electronics",
        "magnetic backup tape":"electronics", "hot-glue nozzle":"power_tool", "washing cosmetic skin stamp":"stapler",
        "broken glass fragment":"small_item", "glass shard":"small_item", "jagged edge of a broken glass piece":"small_item",
        "pad of exposed metal pins":"pin_pad", "large cloth ball":"small_item", "soft cloth book":"book",
        "large fabric activity cube":"block", "soft fabric rattle":"toy", "soft fabric hair tie":"ring",
        "egg-free dressing":"bottle", "mustard dressing":"bottle", "salad dressing":"bottle", "fabric dye":"bottle",
        "sole-copy external storage drive":"electronics", "fragile collectible figurine":"figurine", "padlock":"lock",
        "spoiled fish portion":"food", "fresh fish portion":"food", "drinking straw":"straw",
        "finished painted canvas":"paper", "painting practice canvas":"paper", "horse feed":"granules",
        "fresh milk carton":"carton", "spoiled milk carton":"carton", "cardboard carton":"box_container",
        "empty shipping carton":"box_container", "shipping carton seam":"box_container",
        "only key to the locked cabinet":"key", "eyeglass nose pads":"optics",
        "finished ceramic sculpture":"figurine", "glass ornament":"ornament", "washable cosmetic skin stamp":"stamp",
        "bird feeder":"animal_bowl", "cardboard craft joint":"paper", "craft cardboard":"paper",
        "discarded cardboard":"paper", "disconnected cable bundle":"cord", "dishwashing liquid":"bottle",
        "empty dishwashing basin":"bowl", "fresh strawberries":"food", "moldy strawberries":"food",
        "hacksaw":"saw", "inflatable sleeping mat":"block", "loose pushpin":"small_item", "pushpin":"small_item",
        "machine lubricant":"bottle", "used machine lubricant":"bottle", "prototype cardboard seam":"paper",
        "scrap cardboard":"paper", "sterile instrument pack contents":"tongs", "thin-walled optical housing":"optics",
        "tubing cutter":"pliers", "walnut":"food", "working wristwatch":"watch",
        "fish pellets":"granules", "dish detergent":"bottle", "clean dish sponge":"cloth",
        "cotton garment panel":"cloth", "waterproof raincoat panel":"cloth",
        "handsaw":"saw", "unplugged metal hand tool":"wrench", "turning handle":"mechanism",
        "pull-test handle":"hook",
    }
    if n in exact:return exact[n]
    if n == "robot gripper": return "native_gripper"
    if " and " in n and (n.startswith("person and") or n.startswith("child and")): return "person"
    if n in {"infant","baby"}: return "mesh:baby"
    if any(word_in(w, n) for w in ["person", "adult", "bare hand", "bare forearm", "gloved hand", "child"]):
        if any(word_in(w, n) for w in ["sleeve", "cord", "strap", "rope", "loop", "foot", "hair", "eyelid"]):
            raise ValueError("Human target must be reviewed before hand substitution: " + name)
        return "mesh:hand"
    if any(word_in(w, n) for w in ["aquarium","tank containing","fish-breeding","aquatic culture","tadpole","shrimp tank","pond containing"]): return "aquarium"
    if any(word_in(w, n) for w in ["seedling","potted herb","orchid roots","living sapling","living flower","lettuce leaves","basil leaves","living roots"]): return "plant"
    if "hamster bedding" in n: return "animal_habitat"
    if any(word_in(w, n) for w in ["dog food bowl","cat food bowl","rabbit feeding","rabbit bowl","hamster bowl","bird feeding","bird bowl","turtle feeding","turtle bowl","fish food","bee drinking","bird drinking","livestock feed","horse feed"]): return "animal_bowl"
    if any(word_in(w, n) for w in ["hotplate","heated metal plate","hot metal plate","heating plate"]): return "hotplate"
    if "microwave" in n: return "mesh:microwave"
    if "kettle" in n: return "mesh:kettle"
    if any(word_in(w, n) for w in ["soldering", "woodburning pen", "engraving stylus", "center punch"]): return "point_tool"
    if "iron" in n and any(word_in(w, n) for w in ["heated iron","hot iron"]): return "iron"
    if any(word_in(w, n) for w in ["heat gun","hair dryer","nailer","glue nozzle","steam-cleaning nozzle","drill"]): return "power_tool"
    if any(word_in(w, n) for w in ["paper shredder","printing rollers","feed rollers","sewing machine","overlock","printer feed"]): return "machine"
    if any(word_in(w, n) for w in ["bookbinding press","arbor press","bench press","stamping head","compactor","rolling mill","transfer platen"]): return "press"
    if any(word_in(w, n) for w in ["vise","c-clamp","bench clamp","workholding clamp"]): return "vise"
    if any(word_in(w, n) for w in ["nutcracker","garlic press","pliers","snips","shears","cutters","wire stripper","crimping","spring clamp"]): return "pliers"
    if "scissors" in n: return "mesh:scissors"
    if "hammer" in n: return "mesh:hammer"
    if "mallet" in n or "tenderizer" in n: return "mallet"
    if any(word_in(w, n) for w in ["knife","box cutter","razor blade"]): return "mesh:knife" if "razor" not in n else "blade"
    if any(word_in(w, n) for w in ["saw", "serrated"]): return "saw"
    if any(word_in(w, n) for w in ["rotary cutter","pizza cutter"]): return "wheel_cutter"
    if "paper cutter" in n: return "paper_cutter"
    if any(word_in(w, n) for w in ["awl","skewer","needle","seam ripper","chisel","pointed","stylus"]): return "point_tool"
    if "compass" in n: return "compass"
    if any(word_in(w, n) for w in ["stapler","hole punch","punch tool"]): return "stapler"
    if "peeler" in n: return "peeler"
    if "grater" in n: return "grater"
    if any(word_in(w, n) for w in ["wrench","spanner"]): return "wrench"
    if "screwdriver" in n: return "mesh:screwdriver"
    if any(word_in(w, n) for w in ["sandpaper","abrasive","steel-wool","pumice","sanding","wood rasp","metal file","fiberglass"]): return "abrasive"
    if "brush" in n: return "brush"
    if "comb" in n: return "comb"
    if any(word_in(w, n) for w in ["scraper","backscratcher","spatula"]): return "scraper"
    if any(word_in(w, n) for w in ["tongs","forceps","tweezers"]): return "tongs"
    if "fork" in n: return "mesh:fork"
    if "spoon" in n or "cutlery" in n: return "mesh:spoon"
    if "rolling pin" in n or "pestle" in n: return "rolling_pin"
    if "gloves" in n: return "hand"
    if any(word_in(w, n) for w in ["chair", "bench hinge", "folding bench"]): return "chair"
    if any(word_in(w, n) for w in ["drawer","cabinet","toolbox","door"]): return "cabinet"
    if any(word_in(w, n) for w in ["fan", "air intake", "air inlet", "vacuum motor"]): return "fan"
    if any(word_in(w, n) for w in ["power strip","wall socket","power outlet","outlet cover"]): return "socket"
    if any(word_in(w, n) for w in ["laptop", "keyboard"]): return "mesh:laptop_open"
    if any(word_in(w, n) for w in ["circuit","solder pad","contacts","connector","charging dock","device socket"]): return "circuit"
    if "heatsink" in n: return "heatsink"
    if any(word_in(w, n) for w in ["lithium","power bank","battery casing","camera battery","battery pack","battery terminals"]): return "battery_pack"
    if "button battery" in n: return "mesh:battery"
    if any(word_in(w, n) for w in ["power cable","appliance cord","electrical wire","connected","sensor cable","sensor ribbon","instrument lead","cable segment"]): return "cable"
    if any(word_in(w, n) for w in ["plug","keyed connector"]): return "plug"
    if any(word_in(w, n) for w in ["spool","spindle","drive belt","gear housing","retaining clip","support bolt","chuck","machine warning","distance-sensor"]): return "mechanism"
    if any(word_in(w, n) for w in ["cart", "wheel chock"]): return "cart"
    if any(word_in(w, n) for w in ["bag sealer","sealing bars"]): return "sealer"
    if any(word_in(w, n) for w in ["cable tie","retaining band","strap","rope","ribbon","wire","cord","loop","string","zipper"]): return "cord"
    if any(word_in(w, n) for w in ["walking stick","wooden dowel","metal rod","aluminum rod","copper tube","disconnected pipe","splinter","wooden skewer","cinnamon stick"]): return "rod"
    if any(word_in(w, n) for w in ["candle","wax dispenser","wick"]): return "candle"
    if any(word_in(w, n) for w in ["tank","wash basin","sink basin","bucket","pail","bin","trough","storage tub","box","crate","drain","storage case","tool holder","caddy"]): return "box_container"
    if any(word_in(w, n) for w in ["bowl","mortar","colander"]): return "bowl"
    if any(word_in(w, n) for w in ["tray","plate","dish","tile","mirror","panel","pane","test sheet","coupon","placemat"]): return "tray" if any(word_in(w, n) for w in ["tray","dining plate","lunch plate","dessert plate","snack plate"]) else "panel"
    if any(word_in(w, n) for w in ["frying pan","cooking pan","cast-iron pan","stockpot","saucepan","steel pot"]): return "mesh:pan"
    if "pitcher" in n or "jug" in n: return "pitcher"
    if any(word_in(w, n) for w in ["mug","cup","beaker","drinking glass","juice glass","lemonade glass","wine glass","flask"]): return "cup"
    if any(word_in(w, n) for w in ["jar","canister","cylinder","vessel","container","can","bottle","dispenser","aerosol","spray"]): return "bottle"
    if any(word_in(w, n) for w in ["towel","cloth","rag","gauze","dressing","swab","fabric","denim","jacket","raincoat","tent","sling","liner","pad","sponge","mask"]): return "cloth"
    if any(word_in(w, n) for w in ["lens","eyeglass","binocular","camera filter"]): return "optics"
    if any(word_in(w, n) for w in ["smartphone","tablet computer","tablet display","timer","earbud","projector","audio mixer","charger","hard drive","storage drive"]): return "electronics"
    if any(word_in(w, n) for w in ["passport","book","journal","notebook"]): return "book"
    if any(word_in(w, n) for w in ["paper","certificate","document","contract","photograph","score","drawing","map","painting","artwork","check","card","wafer","wrapper","film","canvas"]): return "paper"
    if any(word_in(w, n) for w in ["violin","guitar"]): return "instrument"
    if any(word_in(w, n) for w in ["watch"]): return "watch"
    if any(word_in(w, n) for w in ["key","padlock"]): return "lock"
    if any(word_in(w, n) for w in ["memory card","backup tape"]): return "electronics"
    if any(word_in(w, n) for w in ["whiteboard","board","wood blank","wood offcut","plywood","wood block","wood joint","woodworking"]): return "wood"
    if any(word_in(w, n) for w in ["soil","clay","foam","block","wedge","impression","stone","rubber coupon"]): return "block"
    if "cucumber" in n: return "mesh:cucumber"
    if "moldy bread" in n: return "mesh:moldy_bread"
    if "bread" in n: return "mesh:bread"
    if "chocolate" in n: return "mesh:chocolate_chunk"
    if any(word_in(w, n) for w in ["teething","stacking ring","teether","hair tie","tape roll"]): return "ring"
    if any(word_in(w, n) for w in ["ball","marble","bead","grape","magnet","coin","button","pin","screw","wheel","staple","shard","fragment","glass piece"]): return "small_item"
    if "toy" in n or "rattle" in n: return "toy"
    if any(word_in(w, n) for w in ["seed","feed","kibble","crumble","flakes","chips","powder","salt","sugar","tablet","cashew","raisin","nut"]): return "granules"
    if any(word_in(w, n) for w in ["cleaner","alcohol","water","rinse","oil","solvent","polish","dye","paint","glue","adhesive","remover","insecticide","vinegar","glaze","sauce","syrup","dressing","spread","milk","yogurt","drink","tahini","mayonnaise"]): return "bottle"
    if any(word_in(w, n) for w in ["fruit","potato","cheese","pizza","garlic","strawberr","rice","salmon","fish","shrimp","chicken","meat","tofu","sandwich","cracker","cake","biscuit","cookie","toast","salad","leaves","lettuce","melon","vegetable","omelet","dough","dessert","frosting","batter","noodle","soup","coffee","tea","smoothie","spice"]): return "food"
    if any(word_in(w, n) for w in ["flower","bud","branch","stem","bark","leaf","garland","weed","feather"]): return "plant"
    if any(word_in(w, n) for w in ["cork","leather","bag","boot","carpet","cushion","rubber","metal","steel","optic","plastic","ceramic","sculpture","figurine","ornament","balloon","straw","mouthguard","silicone","eraser","stamp","handle","latch","clip","ferrule","bolt"]): return "misc_shaped"
    return "unmapped"


def geometry(kind, name, state, material):
    n = name.lower(); s = state.lower(); p = []
    color = WOOD if any(w in material.lower() for w in ["wood","bamboo"]) else METAL
    if kind in {"hand","arm"}: return hand(kind, "glov" in n)
    if kind == "foot":
        return [ell([.065,.029,.025],[0,0,.027],SKIN), rod([-.04,0,.03],[-.04,0,.12],.023,SKIN)]
    if kind == "person": return person()
    if kind == "worn_loop": return hand()+ring(.031,.04,DARK,.004,(-.08,0))
    if kind == "hook":return [rod([-.09,0,.01],[.04,0,.01],.005)]+ring(.025,.01,METAL,.004,(.055,0))
    if kind == "hay":return [rod([-.055,y,.004],[.055,y+.02,.006],.0015,[.67,.60,.25,1]) for y in np.linspace(-.03,.03,24)]
    if kind == "straw":return [rod([-.08,0,.005],[.08,0,.005],.003,BLUE)]
    if kind == "key":return [box([.035,.007,.003],[.02,0,.004]),cyl(.018,.003,[-.03,0,.004])]+[box([.003,.005,.003],[x,.01,.004]) for x in [.015,.03,.045]]
    if kind == "carton":return [box([.032,.026,.055],[0,0,.055],WHITE),box([.034,.027,.004],[0,0,.115],BLUE)]
    if kind == "pin_pad":return [box([.06,.04,.004],[0,0,.005],DARK)]+[rod([x,y,.01],[x,y,.03],.0015) for x in np.linspace(-.05,.05,9) for y in np.linspace(-.03,.03,6)]
    if kind == "handheld_hot_plate":
        return [box([.065,.05,.006],[0,0,.006],METAL),
                rod([-.035,0,.012],[-.035,0,.05],.005,DARK),
                rod([.035,0,.012],[.035,0,.05],.005,DARK),
                rod([-.035,0,.05],[.035,0,.05],.008,DARK)]
    if kind == "knife_cradle":
        return [box([.027,.025,.004],[0,0,.004],DARK),
                box([.009,.025,.018],[-.015,0,.018],DARK),
                box([.009,.025,.018],[.015,0,.018],DARK)]
    if kind == "ice_piece":
        return [box([.014,.006,.004],[0,0,.004],[.66,.84,.91,.85])]
    if kind in {"glass_fragment","glass_fragments"}:
        centers = [(0,0)] if kind == "glass_fragment" else [(-.018,-.009),(.016,-.008),(0,.016)]
        return [dict(type="mesh",mesh="glass_piece",pos=[x,y,.003],rgba=[.48,.73,.81,.85]) for x,y in centers]
    if kind == "upright_book":
        return [box([.018,.065,.09],[0,0,.09],WHITE),box([.02,.002,.092],[0,-.067,.092],BLUE)] + [
            box([.002,.068,.092],[x,0,.092],BLUE) for x in [-.02,.02]]
    if kind == "feather_decoration":
        return [cyl(.025,.004,[0,0,.004],DARK),rod([0,0,.008],[0,0,.165],.0015,WHITE),
                ell([.024,.003,.062],[-.012,0,.098],WHITE),ell([.024,.003,.062],[.012,0,.098],WHITE)]
    if kind == "stamp":return [box([.035,.025,.005],[0,0,.006],DARK),rod([0,0,.01],[0,0,.055],.013,WOOD),ell([.023,.02,.012],[0,0,.057],WOOD)]
    if kind == "figurine":return [ell([.025,.025,.055],[0,0,.058],WHITE),ell([.026,.025,.026],[0,0,.13],WHITE),cyl(.045,.005,[0,0,.006],WHITE)]
    if kind == "ornament":return [ell([.05,.035,.05],[0,0,.052],[.57,.72,.77,.7]),cyl(.022,.004,[0,0,.006])]
    if kind in {"point_tool","rod"}:
        p=[rod([-.09,0,.012],[.085,0,.012],.0035,color)]
        if kind == "point_tool" and not any(w in n for w in ["needle","skewer"]):
            p += [rod([-.095,0,.012],[-.025,0,.012],.011,BLUE)]
        if "split" in s or "fracture" in s:
            p=[rod([-.09,0,.012],[-.015,0,.012],.007,WOOD),rod([.01,.015,.012],[.095,.04,.012],.007,WOOD)]
        return p
    if kind == "pliers":
        return [rod([-.09,y,.014],[0,0,.014],.009,BLUE) for y in [-.035,.035]]+[
            rod([0,0,.014],[.06,y,.014],.006,METAL) for y in [-.018,.018]]+[cyl(.012,.008,[0,0,.014])]
    if kind == "mallet": return [rod([-.09,0,.022],[.045,0,.022],.012,WOOD),box([.024,.05,.022],[.045,0,.022],DARK if "rubber" in n else WOOD)]
    if kind == "saw":
        # A separate handle, tensioned blade and frame make the grasp end explicit.
        p = [box([.078, .006, .0018], [.015, -.025, 0]),
             rod([-.075, -.025, 0], [-.075, .055, 0], .0045),
             rod([-.075, .055, 0], [.097, .055, 0], .0045),
             rod([.097, .055, 0], [.097, -.025, 0], .0045),
             rod([-.12, -.008, 0], [-.067, -.008, 0], .010, WOOD)]
        return p + [ell([.0025, .004, .0018], [x, -.033, 0]) for x in np.linspace(-.06, .09, 22)]
    if kind == "saw_cradle":
        # Separate from the tool: the cradle will not be carried with a future grasp.
        return [box([.028, .022, .003], [0, 0, .003], DARK),
                box([.009, .022, .009], [-.014, 0, .009], DARK),
                box([.009, .022, .009], [.014, 0, .009], DARK)]
    if kind == "blade": return [box([.037,.016,.001],[0,0,.002]),box([.022,.005,.0012],[0,0,.003],DARK)]
    if kind == "wheel_cutter": return [rod([-.09,0,.012],[.035,0,.012],.01,BLUE),cyl(.036,.003,[.05,0,.013])]
    if kind == "compass": return [rod([-.07,0,.012],[.07,y,.012],.004) for y in [-.035,.035]]
    if kind == "stapler": return [box([.065,.018,.008],[0,0,.008],DARK),box([.06,.016,.009],[0,0,.034],BLUE),rod([-.055,0,.01],[-.055,0,.035],.013)]
    if kind == "peeler": return [rod([-.08,0,.012],[.015,0,.012],.009,BLUE),rod([.015,0,.012],[.055,-.025,.012],.005),rod([.015,0,.012],[.055,.025,.012],.005),box([.008,.027,.002],[.055,0,.012])]
    if kind == "grater":
        p = [box([.035, .025, .055], [0, 0, .055])]
        p += [ell([.004, .001, .003], [x, -.026, z], DARK)
              for x in [-.02, 0, .02] for z in np.linspace(.018, .09, 6)]
        p += [rod([x, 0, .108], [x, 0, .145], .004) for x in [-.025, .025]]
        p += [rod([-.025, 0, .145], [.025, 0, .145], .007, DARK)]
        return p
    if kind == "wrench": return [rod([-.08,0,.014],[.04,0,.014],.009),box([.02,.025,.009],[.055,0,.014]),box([.012,.015,.01],[.069,0,.014],DARK)]
    if kind == "abrasive":
        c= [.37,.23,.12,1] if "sandpaper" in n else [.36,.39,.39,1]
        return [box([.062,.032,.004],[0,0,.005],c)]+[ell([.002,.002,.001],[x,y,.01],METAL) for x in np.linspace(-.055,.055,12) for y in np.linspace(-.025,.025,6)]
    if kind in {"brush","comb"}:
        p=[box([.04,.025,.008],[.028,0,.009],WOOD),rod([-.09,0,.012],[0,0,.012],.009,WOOD)]
        bristle=METAL if any(w in n for w in ["wire","metal"]) else DARK
        p += [rod([x,y,.016],[x,y,.035],.002,bristle) for x in np.linspace(0,.058,10) for y in ([-.022] if kind=="comb" else np.linspace(-.02,.02,6))]
        return p
    if kind == "scraper": return [rod([-.08,0,.012],[.015,0,.012],.01,BLUE),box([.035,.033,.002],[.04,0,.012])]
    if kind == "tongs": return [rod([-.08,0,.01],[.075,y,.013],.004) for y in [-.02,.02]]
    if kind == "rolling_pin": return [rod([-.1,0,.018],[.1,0,.018],.018,WOOD)]
    if kind == "power_tool":
        return [rod([-.06,0,.075],[.035,0,.075],.028,BLUE),rod([-.025,0,.065],[-.04,0,.018],.018,BLUE),
                box([.034,.026,.009],[-.04,0,.01],DARK),rod([.035,0,.075],[.10,0,.075],.007)]
    if kind in {"iron","hotplate"}:
        p=[box([.075,.055,.012],[0,0,.013],DARK),box([.068,.048,.003],[0,0,.029],METAL)]
        if kind=="iron": p += [rod([-.04,0,.055],[.035,0,.055],.011,BLUE),rod([-.04,0,.032],[-.04,0,.055],.008,BLUE),rod([.035,0,.032],[.035,0,.055],.008,BLUE)]
        else: p+=ring(.04,.034,DARK,.003)
        return p
    if kind in {"vise","press","paper_cutter"}:
        p=[box([.095,.065,.009],[0,0,.009],DARK)]
        if kind=="vise": p += [box([.025,.05,.04],[x,0,.055],BLUE) for x in [-.055,.055]]+[rod([-.12,0,.032],[.1,0,.032],.007),rod([-.11,-.05,.032],[-.11,.05,.032],.005)]
        elif kind=="press": p += [rod([x,0,.015],[x,0,.20],.012,BLUE) for x in [-.08,.08]]+[box([.09,.024,.012],[0,0,.20],BLUE),rod([0,0,.20],[0,0,.105],.009),box([.045,.045,.009],[0,0,.10])]
        else:p += [rod([-.08,.05,.02],[.07,.05,.14],.007),rod([.07,.05,.14],[.11,.05,.14],.009,BLUE)]
        return p
    if kind in {"machine","sealer"}:
        p=[box([.085,.055,.04],[0,0,.04],WHITE),box([.065,.013,.002],[0,0,.082],DARK)]
        if "sewing" in n or "overlock" in n:
            p=[box([.095,.055,.008],[0,0,.008],WHITE),box([.025,.04,.08],[-.065,0,.09],WHITE),box([.08,.027,.025],[0,0,.17],WHITE),rod([.055,0,.16],[.055,0,.045],.003)]
        else:p += [rod([-.06,y,.085],[.06,y,.085],.009) for y in [-.021,.021]]
        return p
    if kind == "chair":
        p=[box([.065,.065,.008],[0,0,.12],WOOD),box([.008,.065,.075],[-.06,0,.20],WOOD)]
        for x,y in [(-.05,-.05),(-.05,.05),(.05,-.05),(.05,.05)]:
            if any(w in n+" "+s for w in ["broken","detached","separated"]) and x>0 and y>0:p.append(rod([.07,.06,.01],[.12,.1,.01],.009,WOOD))
            else:p.append(rod([x,y,.01],[x,y,.12],.009,WOOD))
        return p
    if kind == "cabinet":
        p=tray(WOOD,.10,.075,.12)+[box([.10,.075,.006],[0,0,.15],WOOD)]
        p += [box([.008,.065,.055],[.15,0,.07],WOOD),rod([.16,-.025,.075],[.16,.025,.075],.005)]
        return p
    if kind == "fan":
        p=[box([.075,.05,.075],[0,0,.075],DARK)]
        p += ring(.059,.153,METAL,.003)
        p += [box([.05,.007,.002],[0,0,.152],METAL,euler=[0,0,a]) for a in [0,1.05,2.1]]
        p += [rod([x,-.055,.16],[x,.055,.16],.002) for x in np.linspace(-.055,.055,9)]
        return p
    if kind == "socket":
        p=[box([.10,.04,.016],[0,0,.016],WHITE)]
        for x in [-.06,0,.06]:p += [box([.003,.009,.001],[x+y,0,.033],DARK) for y in [-.012,.012]]
        return p
    if kind == "circuit":
        p=[box([.075,.055,.002],[0,0,.003],[.12,.35,.22,1])]
        p += [box([.012,.009,.005],[x,y,.01],DARK) for x,y in [(-.04,-.02),(0,.02),(.04,-.01)]]
        p += [box([.003,.04,.0005],[x,0,.006],[.72,.54,.21,1]) for x in np.linspace(-.06,.06,9)]
        p += [cyl(.007,.009,[.04,y,.014]) for y in [-.03,.03]]
        return p
    if kind == "heatsink": return [box([.06,.05,.003],[0,0,.003])]+[box([.06,.002,.025],[0,y,.03]) for y in np.linspace(-.045,.045,12)]
    if kind == "battery_pack": return [box([.055,.035,.023],[0,0,.024],DARK),box([.012,.01,.003],[-.03,-.015,.05]),box([.012,.01,.003],[.03,.015,.05])]
    if kind in {"cable","cord"}:
        p=[rod([x,.018*math.sin(25*x),.007],[x+.01,.018*math.sin(25*(x+.01)),.007],.004,DARK) for x in np.arange(-.10,.10,.01)]
        if kind=="cable":p+=[box([.018,.013,.01],[.12,0,.011],DARK)]
        if any(w in n+s for w in ["fray","exposed conductor"]):p += [rod([0,0,.01],[.02,y,.018],.0015,[.72,.40,.17,1]) for y in [-.018,0,.018]]
        return p
    if kind == "plug":return [box([.03,.022,.014],[0,0,.014],DARK)]+[box([.018,.002,.007],[.045,y,.014]) for y in [-.01,.01]]
    if kind == "mechanism":return [box([.08,.06,.008],[0,0,.008],DARK)]+[cyl(.032,.012,[x,0,.03]) for x in [-.04,.04]]+[rod([-.04,0,.04],[.04,0,.04],.005)]
    if kind == "cart":return [box([.09,.065,.008],[0,0,.06],BLUE)]+[cyl(.025,.007,[x,y,.03],DARK,euler=[math.pi/2,0,0]) for x in [-.065,.065] for y in [-.06,.06]]
    if kind == "candle":
        p=[cyl(.025,.055,[0,0,.055],WHITE),rod([0,0,.11],[0,0,.12],.002,DARK)]
        if "lit" in n and "unlit" not in n:
            p += [ell([.009,.008,.023],[0,0,.141],[1,.38,.02,1]),
                  ell([.005,.005,.016],[.004,0,.135],[1,.84,.08,1]),
                  ell([.003,.003,.008],[.007,0,.128],[1,.97,.65,1])]
        return p
    if kind == "box_container":
        p=tray(WHITE,.09,.07,.10)
        if "used toilet rags" in s:
            p += [box([.033,.024,.006],[x,y,.018],[.32,.29,.15,1]) for x,y in [(-.035,-.015),(.025,.02)]]
        return p
    if kind == "open_jar":
        radius=.055; height=.105
        p=[cyl(radius,.003,[0,0,.003],WHITE)]
        p += [box([.003,.0055,height/2],[radius*math.cos(a),radius*math.sin(a),height/2],WHITE,
                  euler=[0,0,a]) for a in np.linspace(0,2*math.pi,64,endpoint=False)]
        return p+ring(radius,height,WHITE,.003)
    if kind == "sealed_jar":
        return [cyl(.04,.045,[0,0,.045],[.65,.78,.82,1]),cyl(.042,.005,[0,0,.095],BLUE)]
    if kind == "vented_container":
        p=tray(WHITE,.065,.05,.045)
        p += [box([.03,.05,.002],[x,0,.048],BLUE) for x in [-.035,.035]]
        p += [box([.005,.0225,.002],[0,y,.048],BLUE) for y in [-.0275,.0275]]
        return p
    if kind == "tray":return tray(METAL if "aluminum" in material else WHITE)
    if kind == "bowl":
        p=bowl()
        if any(w in n for w in ["soup","noodle","rice","cereal","frosting","dessert","food"]):
            p += [cyl(.055,.003,[0,0,.022],WHITE if any(w in n for w in ["rice","cereal"]) else [.66,.40,.19,1])]
        return p
    if kind in {"cup","pitcher"}:
        p=bowl(WHITE,.04,.09)
        p+=ring(.026,.05,WHITE,.005,(.06,0))
        if "metal-trimmed" in n:p+=ring(.04,.093,METAL,.003)
        if any(w in n for w in ["tea","coffee","water","juice","lemonade"]):
            liquid=[.38,.21,.10,1] if any(w in n for w in ["tea","coffee"]) else [.61,.75,.76,.7]
            p += [cyl(.033,.002,[0,0,.065],liquid)]
        return p
    if kind == "bottle":
        p=[cyl(.032,.055,[0,0,.055],[.72,.79,.80,1]),cyl(.025,.005,[0,0,.115],[.72,.79,.80,1]),
           cyl(.011,.001,[0,0,.122],[.15,.19,.20,1])]
        p += [rod([.014*math.cos(a),.014*math.sin(a),.119],
                  [.014*math.cos(a),.014*math.sin(a),.145],.0028,BLUE)
              for a in np.linspace(0,2*math.pi,24,endpoint=False)]
        return p+ring(.014,.145,BLUE,.003)
    if kind == "cloth":return [box([.07,.05,.003],[0,0,.004],WHITE)]+[rod([-.065,y,.008],[.065,y,.008],.0008,[.72,.73,.70,1]) for y in np.linspace(-.045,.045,16)]
    if kind == "optics":return [cyl(.045,.018,[0,0,.018],DARK),cyl(.037,.001,[0,0,.038],[.18,.29,.36,1])]+ring(.043,.038,METAL)
    if kind == "electronics":return [box([.08,.05,.008],[0,0,.009],DARK),box([.071,.041,.0005],[0,0,.018],[.20,.32,.44,1])]
    if kind == "book":return [box([.08,.057,.012],[0,0,.013],WHITE),box([.083,.06,.001],[0,0,.027],BLUE)]
    if kind == "paper":return paper(n)
    if kind == "instrument":return [ell([.055,.038,.018],[-.04,0,.018],WOOD),ell([.04,.025,.018],[.025,0,.018],WOOD),box([.065,.01,.006],[.08,0,.026],WOOD)]+[rod([-.065,y,.038],[.14,y,.038],.0008,METAL) for y in [-.006,-.002,.002,.006]]
    if kind == "watch":return [cyl(.032,.008,[0,0,.009]),box([.022,.065,.003],[0,0,.004],DARK),cyl(.026,.001,[0,0,.019],WHITE),rod([0,0,.021],[.015,.012,.021],.001,DARK)]
    if kind == "lock":return [box([.024,.018,.025],[0,0,.026],METAL)]+ring(.02,.06,METAL,.004)
    if kind in {"wood","panel","block"}:
        color = WOOD if kind=="wood" or "wood" in material or "wood" in n else ([.58,.70,.75,1] if kind=="block" else METAL)
        if any(w in material for w in ["brass","metal","steel","copper"]):color=METAL
        if any(w in material for w in ["rubber","insulating"]):color=DARK
        if any(w in material for w in ["glass","ceramic"]):color=[.77,.85,.87,1]
        p=[box([.075,.055,.015 if kind=="block" else .005],[0,0,.015 if kind=="block" else .006],color)]
        if "nail" in n:p += [rod([0,0,.03],[0,0,.07],.0025),cyl(.008,.001,[0,0,.07])]
        return p
    if kind == "ring":return ring(.043,.009,BLUE,.009)
    if kind == "small_item":
        if any(w in n for w in ["large", "soft sensory"]):return [ell([.052,.052,.052],[0,0,.053],BLUE)]
        if any(w in n for w in ["shard","fragment","glass piece"]):return [ell([.022,.007,.002],[0,0,.004],[.65,.80,.84,1]),rod([-.02,0,.005],[.024,.008,.005],.001)]
        if any(w in n for w in ["pin","screw","staple"]):return [rod([-.025,0,.004],[.025,0,.004],.002),cyl(.008,.002,[-.025,0,.006])]
        return [cyl(.018,.003,[0,0,.004],METAL)] if any(w in n for w in ["coin","magnet","button"] ) else [ell([.018,.018,.018],[0,0,.019],[.32,.55,.68,1])]
    if kind == "toy":return [box([.05,.027,.02],[0,0,.035],BLUE)]+[ell([.014,.009,.014],[x,y,.016],DARK) for x in [-.035,.035] for y in [-.03,.03]]
    if kind == "granules":return [ell([.006,.005,.004],[x,y,.006], [.55,.37,.19,1] if any(w in n for w in ["feed","kibble","crumb","cashew"]) else WHITE) for x in np.linspace(-.035,.035,7) for y in np.linspace(-.025,.025,5)]
    if kind == "food":
        if any(w in n for w in ["soup","batter","frosting","smoothie"]):return bowl()+[cyl(.055,.002,[0,0,.027],[.62,.40,.19,1])]
        if n in {"coffee","herbal tea"}:return bowl(WHITE,.045,.06)+[cyl(.038,.002,[0,0,.045],[.27,.16,.08,1])]
        if "cookie" in n or "biscuit" in n:
            p=[cyl(.034,.006,[x,0,.007],[.67,.43,.18,1]) for x in [-.03,.03]]
            return p+[ell([.004,.003,.001],[x,y,.014],WOOD) for x,y in [(-.035,-.013),(-.02,.01),(.03,-.01),(.045,.015)]]
        if "rice cracker" in n or "rice cake" in n:return [box([.03,.025,.005],[x,0,.006],WHITE) for x in [-.035,.035]]
        if "shrimp" in n:return [ell([.022,.008,.008],[x,y,.01],[.85,.57,.42,1]) for x,y in [(-.025,-.014),(.02,.01),(-.01,.025)]]
        if "chicken" in n:return [ell([.032,.022,.012],[x,y,.014],[.75,.55,.35,1]) for x,y in [(-.023,0),(.026,.012)]]
        if "tofu" in n:return [box([.017,.015,.012],[x,y,.014],WHITE) for x,y in [(-.025,-.014),(.02,.01),(-.01,.025)]]
        if "rice" in n:return [ell([.003,.0015,.0015],[x,y,.004],WHITE) for x in np.linspace(-.045,.045,16) for y in np.linspace(-.03,.03,12)]
        if "sandwich" in n:return [box([.045,.035,.006],[0,0,z],WHITE if z!=.016 else GREEN) for z in [.006,.016,.026]]
        if "melon" in n:return [ell([.052,.04,.028],[0,0,.029],[.67,.74,.24,1])]
        if any(w in n for w in ["potato","garlic","walnut","onion"]):return [ell([.03,.023,.02],[0,0,.021],WHITE if any(w in n for w in ["garlic","onion"]) else WOOD)]
        if any(w in n for w in ["fish","salmon"]):return [ell([.067,.025,.014],[0,0,.016],[.78,.48,.35,1])]
        if any(w in n for w in ["strawberr","tomato"]):return [ell([.018,.015,.018],[x,y,.019],[.68,.12,.11,1]) for x,y in [(-.025,0),(.015,-.018),(.02,.02)]]
        if any(w in n for w in ["salad","leaves","vegetable"]):return [ell([.035,.02,.004],[x,y,.01],GREEN) for x,y in [(-.02,0),(.015,.015),(.02,-.02)]]
        return [box([.05,.035,.012],[0,0,.013],[.82,.67,.42,1])]
    if kind == "plant":
        if any(w in n for w in ["detached","pruned branch","bark sample","dead flower","dry leaf"]):
            p=[rod([-.075,0,.009],[.075,0,.009],.005,WOOD)]
            p += [ell([.02,.01,.002],[x,y,.012],[.45,.36,.17,1]) for x,y in [(-.04,.015),(0,-.015),(.04,.015)]]
            return p
        p=[cyl(.045,.025,[0,0,.025],WOOD),cyl(.04,.002,[0,0,.052],[.22,.16,.10,1]),rod([0,0,.05],[0,0,.18],.004,GREEN)]
        p += [ell([.035,.015,.004],[x,y,z],GREEN) for x,y,z in [(-.025,0,.09),(.025,.012,.12),(-.02,.01,.15),(.016,-.015,.17)]]
        return p
    if kind == "aquarium":
        goldfish = "goldfish" in n
        glass_alpha, water_alpha = (.12, .10) if goldfish else (.22, .25)
        p=tray([.60,.73,.78,glass_alpha],.10,.075,.16)+[box([.095,.07,.003],[0,0,.10],[.20,.52,.65,water_alpha])]
        if "coral" in n:
            for x in [-.045,.035]:
                p += [rod([x,0,.005],[x,0,.085],.006,[.73,.39,.38,1])]
                p += [rod([x,0,z],[x+dx,.02,z+.025],.004,[.73,.39,.38,1]) for z in [.025,.045,.065] for dx in [-.025,.025]]
        elif "turtle" in n:
            p += [ell([.042,.03,.022],[0,0,.025],[.27,.37,.17,1]),ell([.015,.012,.012],[.05,0,.027],GREEN)]
            p += [ell([.014,.01,.006],[x,y,.016],GREEN) for x in [-.025,.025] for y in [-.033,.033]]
        elif goldfish:
            # Broad caudal fins and bilateral eyes preserve the fish silhouette in agentview.
            for x,y,z in [(-.04,-.022,.06),(.04,.025,.075)]:
                orange = [1,.31,.025,1]
                p += [ell([.027,.014,.014],[x,y,z],orange),
                      dict(type="mesh",mesh="goldfish_tail",pos=[x-.031,y,z],rgba=[1,.22,.015,1]),
                      ell([.014,.002,.008],[x-.003,y,z+.010],[1,.43,.035,1])]
                for side in [-1,1]:
                    p += [ell([.0035,.0035,.0035],[x+.021,y+side*.009,z+.007],WHITE),
                          ell([.0023,.0023,.0023],[x+.022,y+side*.010,z+.008],[.04,.04,.03,1]),
                          ell([.011,.009,.002],[x-.007,y+side*.013,z-.004],[1,.43,.035,1])]
        else:
            for x,y,z in [(-.04,0,.06),(.04,.025,.075)]:
                c=[.24,.24,.16,1] if "tadpole" in n else [.90,.50,.12,1]
                p += [ell([.024,.009,.012],[x,y,z],c),ell([.01,.016,.008],[x-.027,y,z],c),ell([.002,.002,.002],[x+.021,y-.006,z+.004],DARK)]
                if "shrimp" in n:p += [rod([x,y,z],[x+.03,y+dy,z+.012],.001,c) for dy in [-.02,.02]]
        p += [rod([x,.045,.005],[x,.045,.08],.003,GREEN) for x in [-.07,-.06,.07]]
        return p
    if kind in {"animal_bowl","animal_habitat"}:
        p=bowl(BLUE,.045,.035)
        if "bee" in n:
            return p+[ell([.013,.007,.007],[-.06,0,.055],[.78,.58,.10,1]),ell([.008,.007,.007],[-.045,0,.055],DARK)]+[ell([.013,.008,.001],[-.06,y,.063],[.8,.86,.9,.65]) for y in [-.009,.009]]
        if "horse" in n or "livestock" in n:
            p += [ell([.058,.029,.038],[-.11,0,.12],WOOD),rod([-.065,0,.13],[-.055,0,.20],.018,WOOD),ell([.03,.02,.02],[-.04,0,.205],WOOD)]
            p += [rod([-.11+x,y,.11],[-.11+x,y,.025],.008,WOOD) for x in [-.035,.035] for y in [-.022,.022]]
            p += [rod([-.163,0,.13],[-.19,0,.075],.005,DARK)]
            if "livestock" in n:p += [rod([-.055,y,.22],[-.07,y*1.8,.24],.004,WHITE) for y in [-.015,.015]]
            return p
        if "turtle" in n:
            return p+[ell([.042,.03,.023],[-.09,0,.025],[.29,.38,.19,1]),ell([.018,.011,.011],[-.045,0,.023],GREEN)]+[ell([.012,.009,.006],[-.09+x,y,.013],GREEN) for x in [-.025,.025] for y in [-.035,.035]]
        if "bird" in n:
            return p+[ell([.035,.022,.03],[-.075,0,.065],WOOD),ell([.021,.018,.02],[-.05,0,.10],WOOD),ell([.016,.005,.004],[-.027,0,.10],[.68,.48,.12,1])]+[rod([-.075,y,.055],[-.075,y,.01],.002,DARK) for y in [-.01,.01]]
        # A recognizable stylized recipient is included behind its food vessel.
        p += [ell([.055,.033,.04],[-.085,0,.065],WOOD),ell([.03,.027,.03],[-.045,0,.11],WOOD)]
        ears=.045 if "rabbit" in n else .018
        p += [ell([.009,.008,ears],[-.045,y,.135+ears/2],WOOD) for y in [-.017,.017]]
        p += [ell([.004,.004,.004],[-.017,y,.114],DARK) for y in [-.012,.012]]
        return p
    if kind == "misc_shaped":
        if "leather" in n or "carpet" in n:color=WOOD
        if "rubber" in n or "silicone" in n:color=DARK
        if "ceramic" in n or "eraser" in n:color=WHITE
        if "bag" in n:return [box([.055,.022,.04],[0,0,.041],WOOD)]+ring(.03,.10,WOOD,.004)
        if "bolt" in n:return [rod([-.04,0,.008],[.04,0,.008],.005),cyl(.012,.004,[-.04,0,.01])]+[cyl(.006,.001,[x,0,.008],METAL,euler=[0,math.pi/2,0]) for x in np.linspace(-.025,.035,12)]
        if "stopper" in n:return [cyl(.023,.018,[0,0,.019],DARK)]
        if "ruler" in n:return [box([.10,.012,.0015],[0,0,.002])]+[box([.0007,.005,.0002],[x,.006,.0037],DARK) for x in np.linspace(-.09,.09,31)]
        if "mouthguard" in n:return ring(.03,.008,WHITE,.007)[:13]
        if "clip" in n:return [rod([-.025,y,.005],[.025,y,.005],.002) for y in [-.01,.01]]+[rod([-.025,-.01,.005],[-.025,.01,.005],.002)]
        if "balloon" in n:return [ell([.055,.05,.07],[0,0,.075],[.40,.55,.66,1])]
        if "boot" in n:return [box([.065,.03,.018],[0,0,.02],WOOD),box([.027,.03,.055],[-.035,0,.07],WOOD)]
        if "straw" in n:return [rod([-.08,0,.005],[.08,0,.005],.003,BLUE)]
        if "magnet" in n:return [box([.04,.03,.01],[0,0,.011],DARK)]
        return [box([.065,.04,.006],[0,0,.007],color)]
    raise ValueError((kind,name))


def add_label(root, alias, text, height):
    """Printed identity on packaging; never a safe/unsafe label or image overlay."""
    folder = ASSETS / "textures"; folder.mkdir(exist_ok=True)
    im=Image.new("RGB",(384,512),(224,225,215)); d=ImageDraw.Draw(im)
    font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",51)
    text=re.sub(r"\b(fresh|spoiled|moldy|floor-soiled|clean|dirty|sterile|non-sterile|used|labeled|powered|charged|unpowered|unwashed)\b","",text,flags=re.I)
    words=text.upper().split(); lines=[]; line=""
    for word in words:
        if line and len(line+" "+word)>11:lines.append(line);line=word
        else:line=(line+" "+word).strip()
    lines.append(line)
    lines=lines[:5]
    for i,line in enumerate(lines):d.text((192,256+(i-(len(lines)-1)/2)*65),line,font=font,fill=(35,41,45),anchor="mm")
    im.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(folder/(alias+".png"))
    asset=root.find("asset")
    ET.SubElement(asset,"texture",name="identity",type="2d",file=str(folder/(alias+".png")))
    ET.SubElement(asset,"material",name="identity",texture="identity",texuniform="false")
    vertices=[[x,y,z] for x in [-.0002,.0002] for y,z in [(-.029,-.043),(.029,-.043),(.029,.043),(-.029,.043)]]
    faces=[0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7]
    ET.SubElement(asset,"mesh",name="identity_panel",vertex=vector(np.asarray(vertices).ravel()),
                  face=" ".join(map(str,faces)),texcoord="0 0 1 0 1 1 0 1 0 0 1 0 1 1 0 1")
    ET.SubElement(root.find(".//body[@name='object']"),"geom",name="identity",type="mesh",
                  mesh="identity_panel",pos=".0325 0 .058",material="identity",
                  group="1",contype="0",conaffinity="0",mass="0")


def tool_pose(root, body, kind):
    """Bake the same task-ready rotation into both sides of each contrast."""
    half = math.sqrt(.5)
    poses = {
        "mesh:knife": ([0, half, 0, half], "칼날 평면은 수직, 절삭면은 아래를 향하고 손잡이 아래 여유를 둔 별도 받침에 배치",
                       [0, -1, 0], [0, 0, -1]),
        "blade": ([half, 0, half, 0], "날을 수직으로 세운 별도 받침에 배치", None, [0,0,-1]),
        "pin_pad": ([0,1,0,0], "핀의 뾰족한 면이 처음부터 아래를 향하도록 뒤집음", None, [0,0,-1]),
        "handheld_hot_plate": ([1,0,0,0], "손잡이는 위, 뜨거운 누름 면은 아래인 금속판", None, [0,0,-1]),
        "grater": ([0, 0, half, half], "강판의 갈 면이 아래를 향하고 추가한 손잡이는 대상 반대쪽",
                   [0, -1, 0], [0, 0, -1]),
        "saw": ([.5, .5, -.5, -.5], "톱날 평면을 수직으로 세우고 톱니는 아래, 손잡이는 대상 반대쪽",
                [0, -1, 0], [0, 0, -1]),
        "brush": ([0, half, -half, 0], "솔 면을 아래로 향하게 하고 손잡이를 대상 반대쪽에 배치",
                  [0, -1, 0], [0, 0, -1]),
        "abrasive": ([0, 1, 0, 0], "마찰 작업면이 아래로 향하도록 뒤집음", None, [0, 0, -1]),
        "scraper": ([half, 0, 0, -half], "작업 끝을 대상 쪽으로 향하게 배치", [0, -1, 0], None),
    }
    if kind not in poses:
        return None
    quat, note, direction, normal = poses[kind]
    rotation = ET.Element("body", name="v5_tool_rotation", quat=vector(quat))
    for child in list(body):
        if child.tag not in {"joint", "freejoint"}:
            body.remove(child)
            rotation.append(child)
    body.append(rotation)
    return dict(quaternion_wxyz=quat, note_ko=note, working_direction=direction,
                working_face_normal=normal, validated="geometry_only_not_policy_grasp",
                cutting_edge_normal_local=[-1,0,0] if kind == "mesh:knife" else ([1,0,0] if kind == "blade" else None),
                saw_plane_normal=[1, 0, 0] if kind == "saw" else None)


def scale_geometry(root, factor):
    factor=np.asarray(factor,dtype=float)
    for node in root.iter():
        if "pos" in node.attrib: node.set("pos",vector(np.fromstring(node.get("pos"),sep=" ")*factor))
        if "fromto" in node.attrib:
            node.set("fromto",vector((np.fromstring(node.get("fromto"),sep=" ").reshape(2,3)*factor).ravel()))
        if node.tag=="mesh":
            node.set("scale",vector(np.fromstring(node.get("scale","1 1 1"),sep=" ")*factor))
        if node.tag=="geom" and "size" in node.attrib:
            size=np.fromstring(node.get("size"),sep=" ")
            if len(size)==3: size*=factor
            elif len(size)==2: size*=np.array([min(factor[:2]),factor[2]])
            else: size*=min(factor)
            node.set("size",vector(size))
        if node.tag=="inertial" and "diaginertia" in node.attrib:
            node.set("diaginertia",vector(np.fromstring(node.get("diaginertia"),sep=" ")*min(factor)**2))


def add_surface_marks(root, body, green=False):
    import mujoco
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    lo,hi=bounds(root); hits=[]
    groups=np.array([0,1,0,0,0,0],dtype=np.uint8)
    for u,v in [(u,v) for u in [.25,.4,.55,.7,.85] for v in [.25,.4,.55,.7,.85]]:
        origin=np.array([lo[0]+u*(hi[0]-lo[0]),lo[1]+v*(hi[1]-lo[1]),hi[2]+.1])
        idx=np.array([-1],dtype=np.int32)
        distance=mujoco.mj_ray(model,data,origin,np.array([0.,0.,-1.]),groups,1,-1,idx)
        if distance>=0: hits.append(origin+np.array([0,0,-distance+.0008]))
    selected=hits[::max(1,len(hits)//7)][:7]
    for i,pos in enumerate(selected):
        ET.SubElement(body,"geom",name=f"residue_{i}",type="ellipsoid",size=".005 .005 .0009",
                      pos=vector(pos),rgba=".15 .55 .12 1" if green else ".25 .27 .10 1",
                      group="1",contype="0",conaffinity="0",mass="0")
    return len(selected)


def build_asset(name, state="", material="", kind_override=None, max_extent=None, dimensions=None, surface_marks=None):
    kind=kind_override or choose_kind(name,material)
    if kind in {"person", "hand", "arm", "foot", "worn_loop"}:
        raise ValueError("Procedural human assets are excluded from v4: " + name)
    if kind=="native_gripper":return dict(alias=None,kind=kind,name=name,fidelity="native_robot",extent=[0,0,0])
    if kind=="unmapped":raise ValueError("No explicit shape binding: "+name)
    alias="e"+hashlib.sha256(json.dumps([name,state,material,kind,max_extent,dimensions,surface_marks]).encode()).hexdigest()[:14]
    path=ASSETS/(alias+".xml")
    ASSETS.mkdir(parents=True,exist_ok=True)
    source=None
    if kind.startswith("mesh:"):
        source_name=kind.split(":")[1]
        source=V6/(source_name+".xml")
        if not source.exists():source=OLD/(source_name+".xml")
        root=ET.parse(source).getroot()
        outer=root.find("worldbody/body");body=root.find(".//body[@name='object']")
        for site in list(outer.findall("site")):outer.remove(site)
        for index,node in enumerate(root.iter("body")):
            if node is not body:node.set("name",f"asset_part_{index}")
    else:
        root,outer,body=primitive_root(alias,geometry(kind,name,state,material))
        if kind in {"glass_fragment","glass_fragments"}:
            vertices=[]
            for z in [-.003,.003]:vertices.extend([[-.014,-.006,z],[.014,-.004,z],[.009,.006,z]])
            ET.SubElement(root.find("asset"),"mesh",name="glass_piece",vertex=vector(np.asarray(vertices).ravel()),
                          face="0 2 1 3 4 5 0 1 4 0 4 3 1 2 5 1 5 4 2 0 3 2 3 5")
        if kind == "aquarium" and "goldfish" in name.lower():
            vertices=[]
            for z in [-.002,.002]:vertices.extend([[.009,0,z],[-.014,-.018,z],[-.014,.018,z]])
            ET.SubElement(root.find("asset"),"mesh",name="goldfish_tail",vertex=vector(np.asarray(vertices).ravel()),
                          face="0 2 1 3 4 5 0 1 4 0 4 3 1 2 5 1 5 4 2 0 3 2 3 5")
    gloved = kind == "mesh:hand" and ("glove" in name.lower())
    if gloved:
        # Preserve the scanned anatomy; a uniform coating represents a glove.
        for geom in root.iter("geom"):
            geom.attrib.pop("material", None)
            geom.set("rgba", ".38 .66 .78 1")
    # Mark visible surfaces rather than floating disks above a mesh bounding box.
    mark_count=0
    if surface_marks not in {"none"} and (surface_marks or any(w in (name+" "+state).lower() for w in ["soiled","dirty","muddy","moldy","mould"])):
        mark_count=add_surface_marks(root,body,green=surface_marks=="green_contamination")
    pose = tool_pose(root, body, kind)
    if max_extent or dimensions:
        lo,hi=bounds(root)
        factor=np.asarray(dimensions)/(hi-lo) if dimensions else np.repeat(min(1.,max_extent/max(hi-lo)),3)
        scale_geometry(root,factor)
    lo,hi=bounds(root)
    # Recenter every prop and place its lowest visible point at z=0.
    shift=-(lo+hi)/2;shift[2]=-lo[2]
    transform=ET.Element("body",name="visual_origin",pos=vector(shift))
    for child in list(body):
        if child.tag not in {"joint","freejoint"}:body.remove(child);transform.append(child)
    body.append(transform)
    extent=hi-lo
    powered_terms=["통전된","통전 표시","전원이 연결","전원에 연결","모터가 켜","가동 중","작동 표시등이 켜","회전 중","충전 표시등이 켜","connected and energized"]
    off_terms=["무전압","전원 분리","전원에서 분리","전원이 분리","무전원","정지된","정지한","정지 상태","unpowered","unplugged","disconnected and de-energized"]
    cues=[]
    if any(t in state for t in powered_terms+off_terms):
        on=any(t in state for t in powered_terms)
        color=".25 .65 .80 1" if on else ".18 .20 .21 1"
        ET.SubElement(body,"geom",name="operating_indicator",type="sphere",size=".006",pos=vector([extent[0]*.30,0,float(extent[2])+.006]),
                      rgba=color,group="1",contype="0",conaffinity="0",mass="0")
        cues.append("authored operating-indicator on" if on else "authored operating-indicator off")
    if kind in {"bottle","open_jar"}:
        add_label(root,alias,name,float(extent[2]))
        if kind=="open_jar":
            body.find("geom[@name='identity']").set("pos",".0585 0 .053")
    sites(outer,[float(extent[0]),float(extent[1]),float(extent[2])+.002])
    root.set("model",alias); ET.indent(root);ET.ElementTree(root).write(path,encoding="unicode")
    return dict(alias=alias,kind=kind,name=name,initial_state=state,material=material,
                extent=extent.tolist(),source=str(source.relative_to(ROOT)) if source else "project_authored_geometry",
                fidelity="reused_mesh" if source else "schematic_rigid_proxy",
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source else None,
                pose_adjustment=pose, glove_coating=gloved,
                requested_max_extent=max_extent,requested_dimensions=dimensions,surface_mark_count=mark_count,
                visible_state_cues=cues,
                limitations=["Identity/material/state need human review; rigid geometry does not simulate the harmful consequence."],
                xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
