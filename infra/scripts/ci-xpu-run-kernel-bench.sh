#!/usr/bin/env bash
#
# Script for CI - XPU job.
#
# Run KernelBench on Intel GPU.

SCRIPTS_DIR=$(realpath $(dirname $0))

# Setup environment
source /swtools/intel/2025.2.0/setvars.sh --force
source /swtools/intel-gpu/latest/intel_gpu_vars.sh

# Configure project
git submodule update --init
pip install --upgrade --user uv
AI_BENCH_UV=$HOME/.local/bin/uv
${AI_BENCH_UV} sync --extra xpu

# Run KernelBench
${AI_BENCH_UV} run python ${SCRIPTS_DIR}/run_kernel_bench.py --xpu
