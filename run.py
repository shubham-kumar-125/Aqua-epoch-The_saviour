#!/usr/bin/env python3
"""Convenience entry point for the Aqua-Epoch project."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aqua-Epoch project commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train, evaluate, and generate reports")
    train_parser.add_argument("--data", type=Path, default=None, help="path to the source CSV")
    train_parser.add_argument("--seed", type=int, default=15, help="hold-out split seed (15 matches main2.ipynb)")

    serve_parser = subparsers.add_parser("serve", help="start the REST API and dashboard")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    subparsers.add_parser("report", help="rebuild Round 1 documents from saved artifacts")
    args = parser.parse_args()

    if args.command == "train":
        from aqua_epoch.train import train

        train(data_path=args.data, seed=args.seed)
    elif args.command == "serve":
        from aqua_epoch.api import serve

        serve(args.host, args.port)
    else:
        from aqua_epoch.report import build_report

        build_report()


if __name__ == "__main__":
    main()
