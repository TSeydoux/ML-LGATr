#!/bin/bash

### Script to run 'data_prep.py' and preprocess data

source /afs/cern.ch/work/t/thseydou/public/Miniconda/etc/profile.d/conda.sh    # Activates conda
conda activate /afs/cern.ch/work/t/thseydou/public/Miniconda/envs/conda_ML     # Activates the specific conda environment
python /afs/cern.ch/work/t/thseydou/public/LGATr/data_prep.py                  # Runs 'data_prep.py'
