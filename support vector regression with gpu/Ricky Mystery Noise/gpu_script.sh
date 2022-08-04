#!/bin/bash
#SBATCH --time=05:00:00
	# walltime format is h:mm:ss.
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=svr_test
#SBATCH --account=PAS1066

source /fs/project/PAS1066/zhang_anaconda/anaconda3/bin/activate
conda activate rapids-22.06
module load cuda/11.2.2

set -x

cd $SLURM_SUBMIT_DIR

date +"%T"

python -u svr_with_cudf_ricky.py

#sstat -j $SLURM_JOB_ID --format=JobID,AveCPU,AveRSS,MaxRSS,AveVMSize,MaxVMSize,NTasks

date +"%T"

seff $SLURM_JOB_ID

echo 'DONE'
