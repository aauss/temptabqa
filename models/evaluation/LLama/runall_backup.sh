#!/usr/bin/env bash

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J Eval_TempTabQA_LLama2-70b
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A MENG-SL3-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=10:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
#SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime. 

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

# Activate python venv if on Uni HPC
source /home/aa2613/rds/hpc-work/temptabqa/models/evaluation/LLama/manually_source_venv.sh
echo $PATH
which python
# python /home/aa2613/rds/hpc-work/temptabqa/models/evaluation/LLama/llama2_eval_own.py -d head -m zero_shot -o llama2_zero_shot_head.csv
# python /home/aa2613/rds/hpc-work/temptabqa/models/evaluation/LLama/llama2_eval_own.py -d tail -m zero_shot -o llama2_zero_shot_tail.csv
python /home/aa2613/rds/hpc-work/temptabqa/models/evaluation/LLama/llama2_eval_own.py -d head -m few_shot -o llama2_few_shot_head.csv
python /home/aa2613/rds/hpc-work/temptabqa/models/evaluation/LLama/llama2_eval_own.py -d tail -m few_shot -o llama2_few_shot_tail.csv
# python llama2_eval_own.py -d head -m CoT -o llama2_CoT_head.csv
# python llama2_eval_own.py -d tail -m CoT -o llama2_CoT_tail.csv
