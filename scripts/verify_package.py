"""Verify archived files, the run provenance, and the manuscript estimates."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
STUDIES = (
    ("single_commit", "verify_selective_wait.py"),
    ("rolling_wait", "verify_rolling_wait.py"),
    ("matched_wait", "verify_matched_wait.py"),
    ("distance9", "verify_robustness.py"),
)


def verify_hashes():
    manifest = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in manifest:
        digest, path = line.split("  ", 1)
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"SHA-256 mismatch for {path}")
    print(f"Verified SHA-256 for {len(manifest)} files.", flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def verify_lineage():
    rolling = EXPERIMENTS / "rolling_wait" / "results"
    matched = EXPERIMENTS / "matched_wait"
    frozen = read_json(matched / "priority1_design.json")
    source = read_json(rolling / "run.json")
    if any(source.get(key) != value for key, value in frozen.items()):
        raise ValueError("The matched-wait design differs from its rolling-wait source.")
    source_counts = [
        row for row in read_csv(rolling / "commit_waits.csv") if row["policy"] == "budgeted"
    ]
    if source_counts != read_csv(matched / "priority1_commit_waits.csv"):
        raise ValueError("The frozen commit counts differ from their rolling-wait source.")

    distance9 = EXPERIMENTS / "distance9"
    original = read_json(distance9 / "results" / "run.json")
    replica = read_json(distance9 / "replication" / "run.json")

    def run_seeds(run):
        return {
            run["settings"]["seed"],
            run["trigger_seed"],
            *run["calibration_seeds"],
            *(seed for job in run["test_jobs"] for seed in job[4:]),
        }

    if run_seeds(original) & run_seeds(replica):
        raise ValueError("The original and replicated distance-9 runs share a seed.")
    print("Verified the frozen rolling-wait inputs and disjoint distance-9 seeds.", flush=True)


def run_python(directory, *args):
    subprocess.run([sys.executable, "-B", *args], cwd=directory, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-only", action="store_true", help="skip the dependency-based checks")
    args = parser.parse_args()
    verify_hashes()
    if args.hash_only:
        return
    verify_lineage()
    for name, verifier in STUDIES:
        directory = EXPERIMENTS / name
        print(f"\n{name}", flush=True)
        run_python(directory, "-m", "unittest", "discover", "-v")
        run_python(directory, verifier, "results")
    print("\ndistance9 independent replication", flush=True)
    run_python(EXPERIMENTS / "distance9", "verify_robustness.py", "replication")


if __name__ == "__main__":
    main()
