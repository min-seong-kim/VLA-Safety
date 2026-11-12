"""Package the local comparison viewer and recorded media without copying training data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

from report_v6_lora_final import MODELS, load_results
from rollout_common import write_json
from v6_lora_data import RUN


def main():
    global RUN
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN)
    args = parser.parse_args()
    RUN = args.run_root.resolve()
    import report_v6_lora_final
    report_v6_lora_final.RUN = RUN
    results = load_results()
    document = (RUN / "review.html").read_text()
    members = {}
    for (bench, model), result in results.items():
        assert model in MODELS
        for key in result["rows"]:
            folder = result["folder"] / "episodes" / key
            for name in ["views.mp4", "initial.png"]:
                path = folder / name
                assert path.exists()
                archive = f"media/{bench}/{model}/{key}/{name}"
                document = document.replace(os.path.relpath(path, RUN), archive)
                members[archive] = path
    target = RUN / "review_bundle.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("review.html", document)
        archive.writestr("README.txt", "압축을 푼 뒤 review.html을 브라우저에서 여세요.\n"
                         "운반 성공은 최종 위험 행동의 완료나 위해 발생을 뜻하지 않습니다.\n"
                         "원본 trajectory.h5 및 학습 데이터는 저장 공간 때문에 이 영상 묶음에 포함하지 않았습니다.\n")
        for name in ["learning_curves.png", "learning_curves.svg", "comparison_summary.csv", "comparison_summary.json",
                     "paired_model_comparisons.csv", "representation_candidate_pairs.json", "experiment_protocol.json",
                     "test_subgroups.json", "training_comparison_audit.json", "final_cross_model_input_audit.json",
                     "test_condition_summary.csv", "test_condition_summary.json",
                     "browser_audit.json", "review_desktop.png", "review_mobile.png"]:
            archive.write(RUN / name, name)
        for name, source in sorted(members.items()):
            archive.write(source, name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
    digest = hashlib.sha256()
    with target.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    result = dict(passed=True, media_files=len(members), bytes=target.stat().st_size, sha256=digest.hexdigest(),
                  filename=target.name, included_hdf5=False)
    write_json(RUN / "review_bundle_audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
