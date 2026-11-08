"""Smoke-validate one adapter: all checks of extract_and_score.py on <=16 items per split, written to runs/.../smoke/.

Equivalent to: extract_and_score.py --model <m> --smoke 16 --imgshuf 8 [--datasets ...]
Full runs re-execute the same checks (head, attention direction, single-token candidates, padding invariance,
label independence, parameter fingerprint) before extraction, so the validation report always reflects the real run.
"""

import pathlib
import subprocess
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--smoke" not in args:
        args += ["--smoke", "16", "--imgshuf", "8"]
    sys.exit(subprocess.call([sys.executable, str(pathlib.Path(__file__).with_name("extract_and_score.py")), *args]))
