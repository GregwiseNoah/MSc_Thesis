#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=72
#SBATCH --partition=test24ex
#SBATCH -t 24:00:00
#SBATCH --job-name=radcool_t400
#SBATCH --mail-user=ageorge@aip.de
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

module load autotools
module load prun
module load gnu9
module load openmpi4
module load gsl
module load hdf5
module load fftw
module load hypre

wdir=$SLURM_SUBMIT_DIR
job_id=$SLURM_JOBID

mkdir ./logs
prun ./Arepo param.txt 1> $wdir/logs/job.$job_id.out 2> $wdir/logs/job.$job_id.err
#prun ./Arepo param.txt > out.txt 2> error.txt
