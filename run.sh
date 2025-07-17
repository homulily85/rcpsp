#!/bin/bash
source ~/PycharmProjects/rcpsp/.venv/bin/activate
python main.py j90.sm sat --time_limit 600 --num_concurrent_processes 2 --continue_from ./result/j90.sm_SAT_2025-07-16-21-36-44.csv

python main.py pack_d sat --time_limit 600 --num_concurrent_processes 2 --continue_from ./result/pack_d_SAT_2025-07-17-00-30-00.csv
