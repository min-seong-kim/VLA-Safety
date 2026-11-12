"""Apply the repository's runtime integration to a newly created Conda environment."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import sysconfig


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--role", choices=["train", "render"], required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--bind-legacy-path", action="store_true",
                        help="Create the old interpreter path only if it is absent; never replace an existing environment")
    args = parser.parse_args()
    prefix = Path(sys.prefix)
    if not args.check_only and not (prefix / "conda-meta").is_dir():
        raise RuntimeError("Activate the newly created Conda environment before applying this setup")
    expected_python = (3, 11) if args.role == "train" else (3, 12)
    assert sys.version_info[:2] == expected_python, (sys.version, expected_python)
    site = Path(sysconfig.get_paths()["purelib"])
    checked = []
    if args.role == "train":
        assert importlib.metadata.version("transformers") == "4.53.2"
        source = ROOT / "openpi/src/openpi/models_pytorch/transformers_replace"
        target = Path(importlib.metadata.distribution("transformers").locate_file("transformers"))
        assert target.is_relative_to(prefix), "Refuse to patch packages outside this environment"
        for path in sorted(source.rglob("*.py")):
            destination = target / path.relative_to(source)
            if not args.check_only:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            assert destination.exists() and path.read_bytes() == destination.read_bytes(), str(destination)
            checked.append(str(path.relative_to(source)))
        assert len(checked) == 5, "Unexpected OpenPI Transformers patch set"
    else:
        paths = [ROOT / "third_party/libero", ROOT / "third_party/robosuite_vanilla"]
        assert all(path.is_dir() for path in paths), "Copy the preserved simulator sources with this repository"
        content = "".join(str(path) + "\n" for path in paths)
        path = site / "vla_safety_simulator_sources.pth"
        if not args.check_only:
            path.write_text(content)
        assert path.exists() and path.read_text() == content, str(path)
        checked = [str(p.relative_to(ROOT)) for p in paths]
        if not args.check_only:
            (ROOT / "envs").mkdir(exist_ok=True)

    bound = None
    if args.bind_legacy_path:
        legacy = ROOT / ("openpi/.venv" if args.role == "train" else "envs/libero-render")
        if legacy.is_symlink() and legacy.resolve() == prefix.resolve():
            bound = str(legacy)
        elif legacy.exists() or legacy.is_symlink():
            raise FileExistsError(f"Existing environment was preserved: {legacy}; use direct conda run commands instead")
        elif args.check_only:
            raise FileNotFoundError(legacy)
        else:
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.symlink_to(prefix, target_is_directory=True)
            bound = str(legacy)
    print(json.dumps({"role": args.role, "prefix": str(prefix), "checked": checked,
                      "legacy_path": bound, "check_only": args.check_only}, ensure_ascii=False))


if __name__ == "__main__":
    main()
