
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(project_root / "main.py"),
        "--scenario",
        "demo",
        "--episodes",
        "10",
        "--max-steps",
        "200",
        "--plot",
        "--output-dir",
        str(project_root / "results"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
