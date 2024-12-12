#!/usr/bin/env bash

# Activate python venv if on Uni HPC
./manually_source_venv.sh

python llama2_eval_own.py -d head -m zero_shot -o llama2_zero_shot_head.csv
python llama2_eval_own.py -d tail -m zero_shot -o llama2_zero_shot_tail.csv
python llama2_eval_own.py -d head -m few_shot -o llama2_few_shot_head.csv
python llama2_eval_own.py -d tail -m few_shot -o llama2_few_shot_tail.csv
python llama2_eval_own.py -d head -m CoT -o llama2_CoT_head.csv
python llama2_eval_own.py -d tail -m CoT -o llama2_CoT_tail.csv
