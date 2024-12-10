#!/usr/bin/env bash

# Redirect all stdout and stderr to tee in append mode
exec > >(tee -a output.log) 2>&1

# Run your Python commands sequentially
python llama2_eval_own.py -d head -m zero_shot -o llama2_zero_shot_head.csv
python llama2_eval_own.py -d tail -m zero_shot -o llama2_zero_shot_tail.csv
python llama2_eval_own.py -d head -m few_shot -o llama2_few_shot_head.csv
python llama2_eval_own.py -d tail -m few_shot -o llama2_few_shot_tail.csv
python llama2_eval_own.py -d head -m CoT -o llama2_CoT_head.csv
python llama2_eval_own.py -d tail -m CoT -o llama2_CoT_tail.csv