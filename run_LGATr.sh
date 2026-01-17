#!/bin/bash

### Script to run the training of L-GATr and generate plots

source /afs/cern.ch/work/t/thseydou/public/Miniconda/etc/profile.d/conda.sh    # Activates conda
conda activate /afs/cern.ch/work/t/thseydou/public/Miniconda/envs/conda_ML     # Activates the specific conda environment
python /afs/cern.ch/work/t/thseydou/public/LGATr/run_LGATr.py                  # Run 'run_LGATr.py'
