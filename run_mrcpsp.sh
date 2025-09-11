#!/bin/bash
source ~/PycharmProjects/rcpsp/.venv/bin/activate

python mrcpsp_benchmark.py MMLIB50 --time_limit 600 --num_concurrent_processes 4 --save_interval_seconds 300

python mrcpsp_benchmark.py MMLIB100 --time_limit 600 --num_concurrent_processes 4 --save_interval_seconds 300

