#!/bin/bash
source ~/PycharmProjects/rcpsp/.venv/bin/activate

python mrcpsp_benchmark.py j30.mm --time_limit 600 --num_concurrent_processes 2

