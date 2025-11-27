from ai_bench.harness import runner


def main():
    kb_runner = runner.KernelBenchRunner()
    kb_runner.run_kernels()


if __name__ == "__main__":
    main()
