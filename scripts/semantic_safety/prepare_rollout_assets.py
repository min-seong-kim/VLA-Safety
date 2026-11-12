"""Acquire a pinned, small subset of HazardArena meshes for LIBERO scene adapters."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data/safety_sources/raw/hazardarena_selected"
BASE = "HazardArena/assets/obj/meshes/"
SELECTED = {
    "baby": "characters/baby/model_stable.xml",
    "cucumber": "ingredients/cucumber/cucumber_0/model.xml",
    "bread": "breads/bagel/bagel_0/model.xml",
    "chocolate": "snacks/chocolate/chocolate_0/model.xml",
    "kettle": "kitchen_appliances/electric_kettle/electric_kettle/electric_kettle.xml",
    "power_bank": "electronics/powerbank/powerbank/powerbank.xml",
    "battery": "electronics/button_battery/button_battery/button_battery.xml",
    "power_strip": "electronics/power_strip/powerstrip/powerstrip.xml",
    "bleach": "tools/cleaning_agents/84_disinfectant/84disinfection/84disinfection.xml",
    "acid_cleaner": "tools/cleaning_agents/toilet_cleaner_1/toilet_cleaner_1/toilet_cleaner_1.xml",
    "alcohol_cleaner": "tools/cleaning_agents/alcohol_cleaner/alcohol_cleaner/alcohol_cleaner.xml",
    "aquarium": "containers/fish_tank/fish_tank/fish_tank/fish_tank.xml",
    "wallet": "trash/wallet/wallet/wallet.xml",
    "trash_bin": "trash/trashcan_new/trashcan/trashcan.xml",
    "laptop": "tools/laptop/laptop_1/laptop/laptop.xml",
    "wine_glass": "tablewares/glasses/wine_glass/wine_glass/wine_glass.xml",
    "spoon": "tablewares/spoons/spoon_0/model.xml",
    "curling_iron": "tools/curling_iron/curling_iron_1/curling_iron.xml",
}


def main():
    metadata = json.loads((ROOT / "data/safety_sources/catalog/hazardarena_assets.json").read_text())
    prefixes = {str(Path(BASE + value).parent) + "/" for value in SELECTED.values()}
    rows = [row for row in metadata["siblings"] if any(row["rfilename"].startswith(p) for p in prefixes)]
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "selection.json").write_text(json.dumps({"repository": metadata["id"], "revision": metadata["sha"],
                                                     "aliases": {k: BASE + v for k, v in SELECTED.items()},
                                                     "file_count": len(rows), "bytes": sum(r["size"] for r in rows)}, indent=2))

    def download(row):
        name = row["rfilename"]
        path = DEST / name
        expected = row.get("lfs", {}).get("sha256")
        if path.exists() and path.stat().st_size == row["size"]:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not expected or digest == expected:
                return {"path": name, "sha256": digest, "bytes": row["size"]}
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{metadata['id']}/resolve/{metadata['sha']}/{quote(name)}?download=true"
        for retry in range(5):
            try:
                response = requests.get(url, stream=True, timeout=(30, 120))
                response.raise_for_status()
                temporary = path.with_suffix(path.suffix + ".part")
                digest = hashlib.sha256()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
                        digest.update(chunk)
                if temporary.stat().st_size != row["size"] or (expected and digest.hexdigest() != expected):
                    raise ValueError("Asset integrity check failed: " + name)
                temporary.replace(path)
                return {"path": name, "sha256": digest.hexdigest(), "bytes": row["size"]}
            except Exception:
                if retry == 4:
                    raise
                time.sleep(retry + 1)

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(download, row) for row in rows]):
            result = future.result()
            results.append(result)
            print(json.dumps({"done": len(results), "total": len(rows), **result}), flush=True)
    (DEST / "integrity.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
