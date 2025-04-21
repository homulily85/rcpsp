sudo apt-get update
sudo apt-get install -y build-essential gdb
sudo apt install -y python3.12-dev python3.12-venv
python3 -m venv ./
source ./bin/activate
pip install -r requirements.txt
chmod +x ./tt-open-wbo-inc-Glucose4_1_static