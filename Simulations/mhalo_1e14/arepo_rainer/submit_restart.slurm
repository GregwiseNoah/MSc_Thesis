#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --partition=huge
#SBATCH -t 12:00:00
#SBATCH --job-name=061_M13j46t25
#SBATCH --mail-user=rweinberger@aip.de
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

#mkdir ./logs
prun ./Arepo param.txt 1 1> $wdir/logs/job.$job_id.out 2> $wdir/logs/job.$job_id.err
#prun ./Arepo param.txt > out.txt 2> error.txt
