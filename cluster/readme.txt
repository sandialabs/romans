This directory contains sbatch scripts to launch jobs on the parallel
cluster for dimension reduction and model training tasks.  These sripts
are made to work with Slurm.

To launch a job on the cluster use (for example):

$ sbatch run-inc-PCA-train.sh

To check your status in the queue, use:

$ squeue -u user-name

To cancel a job use:

$ scancel job-id

The jod-id is shown using squeue.

NOTES:

1) The media for Slycat is generated using the scripts in the media
directory.

2) The time-aligned-PCA and Isomap scripts are designed to be run after the
run-inc-PCA-100-train/test scripts in the inc-auto-PCA directory.
They use the 100 component auto-correlated reduction as a starting
point.

3) To generate parameter space/videoswarm models, use the ps-csv.py and
vs-dir.py scripts to generate inputs for Slycat.  The vs-dir.py script
generates the "local" version of the files used in the videoswarm wizard.
On the cluster these scripts can be run using run-ps-csv.sh and run-vs-dir.sh.
