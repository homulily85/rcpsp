#!/bin/bash
source ~/PycharmProjects/rcpsp/.venv/bin/activate

python main.py pack_d --time_limit 600 --num_concurrent_processes 2
