#!/bin/bash
source ~/PycharmProjects/rcpsp/.venv/bin/activate
python main.py j120.sm sat --time_limit 600 --num_concurrent_processes 2 --continue_from ./result/j120.sm_SAT_2025-07-18-23-17-22.csv
