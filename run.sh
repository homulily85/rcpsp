#!/bin/bash
source ~/PycharmProjects/rcpsp/.venv/bin/activate
python main.py pack sat --time_limit 600
python main.py pack_d sat --time_limit 600
