#!/bin/bash

# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

#SBATCH --account=your-account
#SBATCH --job-name=memphis
#SBATCH --partition=batch
 
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=8
#SBATCH --time=36:00:0

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

python -m reduce --ensemble phase-field/training_data/workdir.%d \
    --input-files out.cahn_hilliard_%d.vtk \
    --output-dir phase-field/inc-PCA/train \
    --output-file out.cahn_hilliard_inc_whiten_PCA_1500.rd.npy \
    --algorithm incremental-PCA \
    --num-dim 1500 \
    --whiten \
    --over-write \
    --field-var phase_field \
    --csv-out inc-whiten-PCA-train.csv \
    --csv-header "Incremental Whitened PCA" \
    --file-batch-size 2020 \
    --output-model inc-whiten-PCA-1500.pkl \
    --parallel