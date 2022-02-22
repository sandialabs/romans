#!/bin/bash

# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

#SBATCH --account=your-account
#SBATCH --job-name=memphis
#SBATCH --partition=short,batch
 
#SBATCH --nodes=24
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

python -m convert --ensemble phase-field/test_data/workdir.%d \
    --input-files out.cahn_hilliard_50000000.vtk \
    --output-dir phase-field/test_media \
    --output-format jpg \
    --over-write \
    --field-var phase_field \
    --color-scale 0 1 \
    --csv-out end-state.csv \
    --csv-header "End State" \
    --parallel

python -m convert --ensemble phase-field/test_data/workdir.%d \
    --input-files out.cahn_hilliard_%d.vtk \
    --output-dir phase-field/test_media \
    --output-format mp4 \
    --over-write \
    --field-var phase_field \
    --color-scale 0 1 \
    --csv-out movies.csv \
    --csv-header "Movie" \
    --parallel

