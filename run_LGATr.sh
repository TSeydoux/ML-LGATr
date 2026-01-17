#!/bin/bash
source /afs/cern.ch/work/t/thseydou/public/Miniconda/etc/profile.d/conda.sh    # Activates conda
conda activate /afs/cern.ch/work/t/thseydou/public/Miniconda/envs/conda_ML     # Activates the specific conda environment
cd /afs/cern.ch/work/t/thseydou/public/LGATr
python run_LGATr.py                                                            # Rusn the 'run_LGATr.py' script
