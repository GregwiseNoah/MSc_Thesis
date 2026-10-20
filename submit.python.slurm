#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH -t 1:00:00
#SBATCH --job-name=create_slices
#SBATCH --mail-user=ageorge@aip.de
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

module load autotools
module load prun
module load gnu9
module load gsl
module load hdf5
module load fftw
module load hypre
module load python

wdir=$SLURM_SUBMIT_DIR
job_id=$SLURM_JOBID

mkdir ./logs
python -u create_slices.py 1> $wdir/logs/job.$job_id.out 2> $wdir/logs/job.$job_id.err
