#!/bin/bash

### Script to create plots from 'plots.py'

source /afs/cern.ch/work/t/thseydou/public/Miniconda/etc/profile.d/conda.sh    # Activates conda
conda activate /afs/cern.ch/work/t/thseydou/public/Miniconda/envs/conda_ML     # Activates the specific conda environment
python /afs/cern.ch/work/t/thseydou/public/LGATr/ plots.py                     # Runs 'plots.py'
