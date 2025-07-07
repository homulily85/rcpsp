#!/bin/bash
source ~/PycharmProjects/rcpsp/.venv/bin/activate
python main.py j120.sm sat --time_limit 600 --continue_from ./result/j120.sm_SAT_2025-07-06-14-54-12.csv
