import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parts = [json.loads(path.read_text()) for path in args.inputs]
    config = parts[0]["config"]
    config["seeds"] = [part["config"]["seed"] for part in parts]
    arms = [
        {**arm, "seed": part["config"]["seed"]}
        for part in parts
        for arm in part["arms"]
    ]
    args.output.write_text(
        json.dumps({"config": config, "arms": arms}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
