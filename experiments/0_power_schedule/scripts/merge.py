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
    config["seeds"] = [seed for part in parts for seed in part["config"]["seeds"]]
    config["output"] = str(args.output)
    records = [row for part in parts for row in part["records"]]
    records.sort(key=lambda row: (row["seed"], row["step"]))
    args.output.write_text(
        json.dumps({"config": config, "records": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
