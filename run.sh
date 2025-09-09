#!/bin/bash
source ~/PycharmProjects/rcpsp/.venv/bin/activate

python main.py j120.sm --time_limit 600 --num_concurrent_processes 2 --continue_from ./result/j120.sm_STAIRCASE_2025-09-04-15-06-56.csv

