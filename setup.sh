#!/bin/bash
sudo apt-get update
sudo apt-get install -y build-essential gdb
sudo apt install -y python3.12-dev python3.12-venv
python3 -m venv ./.venv
source ./.venv/bin/activate
pip install -r requirements.txt
chmod +x ./bin/tt-open-wbo-inc-Glucose4_1_static
chmod +x ./bin/mrcpsp2smt
chmod +x ./bin/yices-2.6.0/install-yices
cd ./bin/yices-2.6.0 || exit
sudo ./install-yices
