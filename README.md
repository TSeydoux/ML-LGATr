# L-GATr Tagger

This repository provides an implementation of the **L-GATr architecture** (https://github.com/heidelberg-hepml/lgatr). This code was developed to create an **ML-based tagger for the $$B_s^0 \rightarrow \tau^+ \tau^-$$ decay, where $$\tau^\pm \rightarrow 3\pi^\pm$$**, in the context of FCC-ee studies.

It includes tools to:

- Preprocess data from ROOT files
- Train the LGATr model
- Evaluate its performance
- Automatically generate all relevant plots

---

## Installation

### 1. Install Miniconda

If you do not already have Conda installed, download **Miniconda** and install it (choose the installation path when prompted):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Initialize Conda:
```
source ~/.bashrc
source <path_to_miniconda>/etc/profile.d/conda.sh
```


### 2. Create and activate the environment

Create a dedicated Conda environment:
```
conda create -p <path>/conda_LGATr python=3.10
conda activate <path>/conda_LGATr
```


### 3. Install dependencies

Install the required Python packages:
```
conda install -c conda-forge \
    pandas numpy matplotlib scikit-learn uproot tqdm
pip install torch torchvision torchaudio
pip install psutil requests
pip install torch-geometric torchinfo lgatr
```


---

## Usage

### 1. Data preprocessing

Data preparation is handled by the `data_prep.py` script via its `preprocessing()` function.

To run the preprocessing step, execute the associated shell script:
```
./data_prep.sh
```
You might need the permission first:
```
chmod +x data_prep.sh
./data_prep.sh
```
This step reads ROOT files, preprocesses the data, and stores it in a format suitable for training. Note that input variables can be rescaled during this step if required. You may want to change paths in `data_prep.py` and `data_prep.sh` to use `preprocessing()`.


### 2. Training the model

Training parameters (e.g. learning rate, batch size, number of epochs) can be configured in `run_LGATr.py`. To start training, run:
```
./run_LGATr.sh
```
You may want to change paths and training parameters in `run_LGATr.py` and `run_LGATr.sh`.

This script will:

- Train the LGATr model using the selected parameters
- Save model results
- Produce performance plots automatically


### 3. Plot customization

The appearance and layout of the plots can be modified in `plots.py`. After making changes, regenerate the plots by running:
```
./run_plots.sh
```
You may want to change path in `plots.py` and `run_plots.sh` when running them standalone.

---
