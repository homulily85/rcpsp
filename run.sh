#!/bin/bash

# Check if encoder_type argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <encoder_type>"
  exit 1
fi

# Check if timeout argument is provided
if [ -z "$2" ]; then
  echo "Usage: $0 <encoder_type> <timeout>"
  exit 1
fi

encoder_type=$1
timeout=$2

# List of datasets
datasets=("pack" "pack_d" "j30.sm" "j60.sm" "j90.sm" "j120.sm")

# Loop through datasets and run the benchmark
for data_set in "${datasets[@]}"; do
  echo "Running benchmark for dataset: $data_set with encoder: $encoder_type"
  python benchmark.py "$data_set" "$encoder_type" "$timeout" --verbose --verify --show_solution
done
