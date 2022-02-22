#!/bin/bash

# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

#SBATCH --account=your-account
#SBATCH --job-name=memphis
#SBATCH --partition=short,batch
 
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --time=4:00:0

module load python3
export PYTHONPATH=/users/your-user-name/romans:/users/your-user-name/romans/romans

profile=default
ipython profile create --parallel

echo "Launching Controller ..."
ipcontroller --ip='*' &
sleep 1m

echo "Launching Engines ..."
srun ipengine &
sleep 1m

echo "Launching Job ..."

python -m reduce --ensemble phase-field/inc-auto-PCA/test/workdir.%d \
    --input-files out.cahn_hilliard_inc_auto_PCA_100.rd.npy \
    --output-dir phase-field/time-aligned-PCA/test \
    --output-file out.cahn_hilliard_time_aligned_PCA.rd.npy \
    --input-model phase-field/time-aligned-PCA/train/time-aligned-PCA.pkl \
    --over-write \
    --field-var phase_field \
    --csv-out time-aligned-PCA-test.csv \
    --csv-header "Time-Aligned PCA" \
    --parallel