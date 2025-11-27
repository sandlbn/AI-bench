import argparse

import torch

from ai_bench.harness import runner


def main(args):
    if args.xpu:
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    kb_runner = runner.KernelBenchRunner(device=device)
    kb_runner.run_kernels()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xpu",
        action=argparse.BooleanOptionalAction,
        help="Run on Intel GPU",
    )
    args = parser.parse_args()
    main(args)
