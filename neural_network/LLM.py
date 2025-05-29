#!/usr/bin/env python3
"""
collect_and_rank.py

Iterates through every sub-directory of the top-level folder **experiment/**,
finds all *.txt files, and sends the *contents* of each file to the OpenAI
Chat Completions API with the question:

    “Below is an experiment log.  Suggest the best hyper-parameters.”

The model’s response is written to a new file that sits next to the original
log, named <original>_chatgpt.txt.

Prerequisites
-------------
1.  `pip install openai`
2.  Set your OpenAI API key in the environment  
    `export OPENAI_API_KEY="sk-..."`

Usage
-----
Simply run:

    python collect_and_rank.py

All `.txt` results will be produced alongside the input logs.
"""

import os
from pathlib import Path
import openai
import time
from typing import List

# --------------------------------------------------------------------------- #
# Configurable knobs
# --------------------------------------------------------------------------- #
EXPERIMENT_ROOT = Path("experiment")  # top-level directory
MODEL          = "gpt-4o-preview"     # or "gpt-4o-mini" / "gpt-3.5-turbo-0125"
RATE_LIMIT_QPS = 3                    # OpenAI free-tier ≈3 req/s; adjust if needed
SYSTEM_ROLE    = (
    "You are an expert ML researcher. "
    "Based on the experiment log you will receive, "
    "recommend the *single best* hyper-parameter configuration "
    "and briefly justify why."
)
USER_PROMPT_TEMPLATE = (
    "Below is an experiment log from a Max-Cut model.  "
    "Suggest the best set of hyper-parameters (embedding_dim, hidden_dim, "
    "optimizer, learning rate, batch size, epochs, etc.) that you would try "
    "next, and explain your reasoning in 2-3 sentences.\n\n"
    "{log_contents}"
)
# --------------------------------------------------------------------------- #


def find_txt_logs(root: Path) -> List[Path]:
    """Return a list of paths to all *.txt files under *root* (recursive)."""
    return [p for p in root.rglob("*.txt") if p.is_file()]


def call_chatgpt(log_text: str) -> str:
    """Send *log_text* to ChatGPT and return the assistant’s reply (string)."""
    response = openai.chat.completions.create(
        model=MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_ROLE},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(log_contents=log_text),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    txt_files = find_txt_logs(EXPERIMENT_ROOT)
    if not txt_files:
        print(f"No .txt logs found under {EXPERIMENT_ROOT.resolve()}")
        return

    print(f"Found {len(txt_files)} log file(s).  Querying ChatGPT ...")

    for idx, txt_path in enumerate(sorted(txt_files), 1):
        output_path = txt_path.with_suffix(".chatgpt.txt")
        if output_path.exists():
            print(f"[{idx}/{len(txt_files)}] Skipping {txt_path.name} "
                  f"(already processed).")
            continue

        print(f"[{idx}/{len(txt_files)}] Processing {txt_path.relative_to(EXPERIMENT_ROOT)}")

        try:
            log_contents = txt_path.read_text(encoding="utf-8")
            reply        = call_chatgpt(log_contents)
            output_path.write_text(reply, encoding="utf-8")

        except Exception as exc:
            print(f"  ⚠️  Error on {txt_path.name}: {exc}")
            continue

        # Respect rate limits
        time.sleep(1.0 / RATE_LIMIT_QPS)

    print("✓ Done.")


if __name__ == "__main__":
    main()
