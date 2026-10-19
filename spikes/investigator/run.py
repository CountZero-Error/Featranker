"""
Run the spike against a real LLM through an OpenAI-compatible endpoint.

The API key is read from an environment variable (default DEEPSEEK_API_KEY);
it is never printed or written. Results go to results/<model>-<timestamp>.json
and are rewritten after every seed, so a failed run keeps what finished.

    python spikes/investigator/run.py --model <model-id>
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

from openai import OpenAI

import spike

HERE = Path(__file__).resolve().parent


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="model id at the endpoint")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--key-env", default="DEEPSEEK_API_KEY", help="env var holding the key")
    parser.add_argument("--seeds", default="5,6,7,8,9")  # D8: v2 seeds; v1 used 0-4
    parser.add_argument("--arms", default=",".join(spike.ARMS))
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ[args.key_env], base_url=args.base_url)

    def llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return response.choices[0].message.content or ""

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = tuple(args.arms.split(","))
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = HERE / "results" / f"{args.model.replace('/', '_')}-{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    results = {
        "model": args.model, "base_url": args.base_url, "seeds_requested": seeds, "arms": list(arms),
        "commit": git("rev-parse", "HEAD"), "spike_dirty": bool(git("status", "--porcelain", ".")),
        # The freeze check: must equal the hash recorded in FINDINGS.md.
        "spike_sha256": hashlib.sha256((HERE / "spike.py").read_bytes()).hexdigest(),
        "started": stamp, "seeds": [],
    }
    for seed in seeds:
        print(f"[*] seed {seed}", flush=True)
        results["seeds"].append(spike.run_seed(seed, llm, arms))
        out.write_text(json.dumps(results, indent=2))
        for arm in arms:
            run = results["seeds"][-1]["arms"][arm]
            print(f"    {arm:<12} recall {run['recall']:.1f}  false {run['false_findings']}  calls {run['calls']}")

    if arms == spike.ARMS:
        results["decision"] = spike.decide([s["arms"] for s in results["seeds"]])
        out.write_text(json.dumps(results, indent=2))
        decision = results["decision"]
        print(f"\n[*] outcome: {decision['outcome']}  borderline: {decision['borderline']}")
        print(f"    mean recall: {decision['mean_recall']}")
    print(f"[*] results: {out}")


if __name__ == "__main__":
    main()
