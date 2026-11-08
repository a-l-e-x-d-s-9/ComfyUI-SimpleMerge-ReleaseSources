#!/usr/bin/env python3
"""Check, explicitly apply, or revert the reviewed ComfyUI cached-output patch."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="Modify the three reviewed core files; restart ComfyUI afterward")
    action.add_argument("--revert", action="store_true", help="Restore the exact original core versions; restart afterward")
    args = parser.parse_args()
    root = args.comfy_root.resolve()
    integration = Path(__file__).resolve().parents[1] / "integration"
    manifest = json.loads((integration / "base_sha256.json").read_text())
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in manifest}
    before = all(hashes[name] == value["before"] for name, value in manifest.items())
    after = all(hashes[name] == value["after"] for name, value in manifest.items())
    if not before and not after:
        parser.error("Core files differ from the reviewed before/after versions. No changes made; review this ComfyUI version first.")
    if (args.apply and after) or (args.revert and before):
        print("Already in the requested state; no changes made.")
        return
    if not args.apply and not args.revert:
        print("Integration already applied." if after else "Exact reviewed base matched. Patch is ready; no files changed.")
        return
    command = ["git", "-C", str(root), "apply"]
    if args.revert:
        command.append("--reverse")
    patch = str(integration / "cached-output-release.patch")
    subprocess.run([*command, "--check", patch], check=True)
    subprocess.run([*command, patch], check=True)
    expected = "before" if args.revert else "after"
    for name, value in manifest.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != value[expected]:
            raise RuntimeError(f"Unexpected contents after patching {name}; inspect before restarting.")
    print("Integration reverted. Restart ComfyUI." if args.revert else "Integration applied. Restart ComfyUI.")


if __name__ == "__main__":
    main()
