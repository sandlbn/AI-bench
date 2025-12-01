import argparse

import torch

from ai_bench.harness import core
from ai_bench.harness import runner


def main(args):
    if args.xpu:
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    spec_type = core.SpecKey.V_CI
    if args.bench:
        if args.xpu:
            spec_type = core.SpecKey.V_BENCH_GPU
        else:
            spec_type = core.SpecKey.V_BENCH_CPU

    kb_runner = runner.KernelBenchRunner(spec_type=spec_type, device=device)
    kb_runner.run_kernels()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run on Intel GPU",
    )
    parser.add_argument(
        "--bench",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Benchmark execution",
    )
    args = parser.parse_args()
    main(args)
