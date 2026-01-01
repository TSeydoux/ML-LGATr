### Config file for LGATr training and evaluation



## imports
import os
import pandas as pd
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.nn.aggr import MeanAggregation
from lgatr.interface import extract_scalar
from lgatr import LGATr





# Training variables
train_vars = ["TauCand1_x", "TauCand1_y", "TauCand1_z", "TauCand1_chi2", "TauCand1_thrustangle",
              "TauCand2_x", "TauCand2_y", "TauCand2_z", "TauCand2_chi2", "TauCand2_thrustangle",
              "TauCand1_px", "TauCand1_py", "TauCand1_pz", "TauCand1_p", "TauCand1_m", "TauCand1_q",
              "TauCand2_px", "TauCand2_py", "TauCand2_pz", "TauCand2_p", "TauCand2_m", "TauCand2_q",
              "TauCand1_pion1px", "TauCand1_pion1py", "TauCand1_pion1pz", "TauCand1_pion1q", "TauCand1_pion1d0", "TauCand1_pion1z0",
              "TauCand1_pion2px", "TauCand1_pion2py", "TauCand1_pion2pz", "TauCand1_pion2q", "TauCand1_pion2d0", "TauCand1_pion2z0",
              "TauCand1_pion3px", "TauCand1_pion3py", "TauCand1_pion3pz", "TauCand1_pion3q", "TauCand1_pion3d0", "TauCand1_pion3z0",
              "TauCand2_pion1px", "TauCand2_pion1py", "TauCand2_pion1pz", "TauCand2_pion1q", "TauCand2_pion1d0", "TauCand2_pion1z0",
              "TauCand2_pion2px", "TauCand2_pion2py", "TauCand2_pion2pz", "TauCand2_pion2q", "TauCand2_pion2d0", "TauCand2_pion2z0",
              "TauCand2_pion3px", "TauCand2_pion3py", "TauCand2_pion3pz", "TauCand2_pion3q", "TauCand2_pion3d0", "TauCand2_pion3z0"]   # The full list of variables that are extracted from the root files in preprocessing() function

fourv_vars = ["TauCand1_pion1px", "TauCand1_pion1py", "TauCand1_pion1pz",
              "TauCand1_pion2px", "TauCand1_pion2py", "TauCand1_pion2pz",
              "TauCand1_pion3px", "TauCand1_pion3py", "TauCand1_pion3pz",
              "TauCand2_pion1px", "TauCand2_pion1py", "TauCand2_pion1pz",
              "TauCand2_pion2px", "TauCand2_pion2py", "TauCand2_pion2pz",
              "TauCand2_pion3px", "TauCand2_pion3py", "TauCand2_pion3pz"]   # The momenta that can be used to reconstruct the four-vectors (E, p) for each pion

scalar_vars = ["TauCand1_x", "TauCand1_y", "TauCand1_z", "TauCand1_chi2", "TauCand1_thrustangle",
               "TauCand2_x", "TauCand2_y", "TauCand2_z", "TauCand2_chi2", "TauCand2_thrustangle",
               "TauCand1_px", "TauCand1_py", "TauCand1_pz",
               "TauCand2_px", "TauCand2_py", "TauCand2_pz"]

full_scalar_vars = ["TauCand1_x", "TauCand1_y", "TauCand1_z", "TauCand1_chi2", "TauCand1_thrustangle",
                    "TauCand2_x", "TauCand2_y", "TauCand2_z", "TauCand2_chi2", "TauCand2_thrustangle",
                    "TauCand1_px", "TauCand1_py", "TauCand1_pz", "TauCand1_p", "TauCand1_m", "TauCand1_q",
                    "TauCand2_px", "TauCand2_py", "TauCand2_pz", "TauCand2_p", "TauCand2_m", "TauCand2_q",
                    "TauCand1_pion1q", "TauCand1_pion1d0", "TauCand1_pion1z0",
                    "TauCand1_pion2q", "TauCand1_pion2d0", "TauCand1_pion2z0",
                    "TauCand1_pion3q", "TauCand1_pion3d0", "TauCand1_pion3z0",
                    "TauCand2_pion1q", "TauCand2_pion1d0", "TauCand2_pion1z0",
                    "TauCand2_pion2q", "TauCand2_pion2d0", "TauCand2_pion2z0",
                    "TauCand2_pion3q", "TauCand2_pion3d0", "TauCand2_pion3z0"]   # All the scalar variables, including charges and transverse/longitudinal impact parameter (d0, z0)

scalars_wo_tau = ["TauCand1_x", "TauCand1_y", "TauCand1_z", "TauCand1_chi2", "TauCand1_thrustangle",
                  "TauCand2_x", "TauCand2_y", "TauCand2_z", "TauCand2_chi2", "TauCand2_thrustangle",
                  "TauCand1_q", "TauCand2_q",
                  "TauCand1_pion1q", "TauCand1_pion1d0", "TauCand1_pion1z0",
                  "TauCand1_pion2q", "TauCand1_pion2d0", "TauCand1_pion2z0",
                  "TauCand1_pion3q", "TauCand1_pion3d0", "TauCand1_pion3z0",
                  "TauCand2_pion1q", "TauCand2_pion1d0", "TauCand2_pion1z0",
                  "TauCand2_pion2q", "TauCand2_pion2d0", "TauCand2_pion2z0",
                  "TauCand2_pion3q", "TauCand2_pion3d0", "TauCand2_pion3z0"]   # All the scalar variables, without the variables related to the tau momentum and mass; so the model will not rely on these variables to discriminate signal from bkg

alt_vars = ["Tau23PiCandidates_x", "Tau23PiCandidates_y", "Tau23PiCandidates_z",
            "Tau23PiCandidates_xErr", "Tau23PiCandidates_yErr", "Tau23PiCandidates_zErr",
            "Tau23PiCandidates_chi2", "Tau23PiCandidates_hemEmin", "Tau23PiCandidates_pion1px",
            "Tau23PiCandidates_pion1py", "Tau23PiCandidates_pion1pz"]   # alt_vars stands for alternative variables; these are other possible tau -> 3 pions decays

alt_fourv_vars = ["Tau23PiCandidates_pion1px", "Tau23PiCandidates_pion1py", "Tau23PiCandidates_pion1pz",
                  "Tau23PiCandidates_pion2px", "Tau23PiCandidates_pion2py", "Tau23PiCandidates_pion2pz",
                  "Tau23PiCandidates_pion3px", "Tau23PiCandidates_pion3py", "Tau23PiCandidates_pion3pz"]

alt_scalar_vars = ["Tau23PiCandidates_x", "Tau23PiCandidates_y", "Tau23PiCandidates_z",
                   "Tau23PiCandidates_xErr", "Tau23PiCandidates_yErr", "Tau23PiCandidates_zErr",
                   "Tau23PiCandidates_chi2", "Tau23PiCandidates_hemEmin"]

# Normalization parameters for each variable
var_norm = {
    "TauCand1_x": {"mean": 0.0, "std": 2.0, "log": False},              "TauCand2_x": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_y": {"mean": 0.0, "std": 2.0, "log": False},              "TauCand2_y": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_z": {"mean": 0.0, "std": 2.0, "log": False},              "TauCand2_z": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0 and 1000.0
    "TauCand1_chi2": {"mean": 0.0, "std": 350.0, "log": False},         "TauCand2_chi2": {"mean": 0.0, "std": 2500.0, "log": False}, # 350.0 instead of 250.0
    "TauCand1_thrustangle": {"mean": 0.0, "std": 1.0, "log": False},    "TauCand2_thrustangle": {"mean": 0.0, "std": 1.0, "log": False}, # mean of 0.0 isntead of -3.0
    
    "TauCand1_px": {"mean": 0.0, "std": 5.0, "log": False},             "TauCand2_px": {"mean": 0.0, "std": 5.0, "log": False},
    "TauCand1_py": {"mean": 0.0, "std": 5.0, "log": False},             "TauCand2_py": {"mean": 0.0, "std": 5.0, "log": False},
    "TauCand1_pz": {"mean": 0.0, "std": 5.0, "log": False},             "TauCand2_pz": {"mean": 0.0, "std": 5.0, "log": False},
    "TauCand1_p": {"mean": 15.0, "std": 3.0, "log": False},             "TauCand2_p": {"mean": 15.0, "std": 3.0, "log": False}, # never used when tau=True
    "TauCand1_m": {"mean": 0.0, "std": 1.0, "log": False},              "TauCand2_m": {"mean": 0.0, "std": 1.0, "log": False}, # mean of 0.0 instead of 1.0, std of 1.0 instead of 0.5
    "TauCand1_q": {"mean": 0.0, "std": 1.0, "log": False},              "TauCand2_q": {"mean": 0.0, "std": 1.0, "log": False},

    "TauCand1_pion1px": {"mean": 0.0, "std": 2.0, "log": False},        "TauCand2_pion1px": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_pion1py": {"mean": 0.0, "std": 2.0, "log": False},        "TauCand2_pion1py": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_pion1pz": {"mean": 0.0, "std": 2.0, "log": False},        "TauCand2_pion1pz": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_pion1q": {"mean": 0.0, "std": 1.0, "log": False},         "TauCand2_pion1q": {"mean": 0.0, "std": 1.0, "log": False},
    "TauCand1_pion1d0": {"mean": 0.0, "std": 1.0, "log": False},        "TauCand2_pion1d0": {"mean": 0.0, "std": 1.0, "log": False},
    "TauCand1_pion1z0": {"mean": 0.0, "std": 1.0, "log": False},        "TauCand2_pion1z0": {"mean": 0.0, "std": 1.0, "log": False},

    "TauCand1_pion2px": {"mean": 0.0, "std": 2.0, "log": False},        "TauCand2_pion2px": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_pion2py": {"mean": 0.0, "std": 2.0, "log": False},        "TauCand2_pion2py": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_pion2pz": {"mean": 0.0, "std": 2.0, "log": False},        "TauCand2_pion2pz": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_pion2q": {"mean": 0.0, "std": 1.0, "log": False},         "TauCand2_pion2q": {"mean": 0.0, "std": 1.0, "log": False},
    "TauCand1_pion2d0": {"mean": 0.0, "std": 1.0, "log": False},        "TauCand2_pion2d0": {"mean": 0.0, "std": 1.0, "log": False},
    "TauCand1_pion2z0": {"mean": 0.0, "std": 1.0, "log": False},        "TauCand2_pion2z0": {"mean": 0.0, "std": 1.0, "log": False},

    "TauCand1_pion3px": {"mean": 0.0, "std": 2.0, "log": False},        "TauCand2_pion3px": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_pion3py": {"mean": 0.0, "std": 2.0, "log": False},        "TauCand2_pion3py": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_pion3pz": {"mean": 0.0, "std": 2.0, "log": False},        "TauCand2_pion3pz": {"mean": 0.0, "std": 2.0, "log": False}, # 2.0 instead of 3.0
    "TauCand1_pion3q": {"mean": 0.0, "std": 1.0, "log": False},         "TauCand2_pion3q": {"mean": 0.0, "std": 1.0, "log": False},
    "TauCand1_pion3d0": {"mean": 0.0, "std": 1.0, "log": False},        "TauCand2_pion3d0": {"mean": 0.0, "std": 1.0, "log": False},
    "TauCand1_pion3z0": {"mean": 0.0, "std": 1.0, "log": False},        "TauCand2_pion3z0": {"mean": 0.0, "std": 1.0, "log": False},
}



# Dataset class
class LGATrDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        dat = {"mv":self.data["mv"][index], "sc":self.data["sc"][index]}
        return dat, self.labels[index]



# Wrapper class
class LGATrWrapper(nn.Module):
    def __init__(self, net, mean_aggregation=False, mv_only=False):
        super().__init__()
        self.net = net
        self.aggregation = MeanAggregation() if mean_aggregation else None
        self.mv_only = mv_only
        

    def forward(self, embedding):
        if embedding is None:
            return
        else:
            multivector = embedding["mv"].float()
            scalars = embedding["sc"].float()
     
        if self.mv_only:
            multivector_outputs, scalar_outputs = self.net(multivector)
        else:
            multivector_outputs, scalar_outputs = self.net(multivector, scalars=scalars)

        logits = self.extract_from_ga(multivector_outputs, scalar_outputs)
        return torch.sigmoid(logits)


    def extract_from_ga(self, multivector, batch):
        outputs = extract_scalar(multivector)[:, :, :, 0]
        if self.aggregation is not None:
            logits = self.aggregation(outputs, index=batch)
        return logits



# Models
standard_model = LGATr(in_mv_channels=1,
                       out_mv_channels=1,
                       hidden_mv_channels=16,
                       in_s_channels=16,
                       out_s_channels=0,
                       hidden_s_channels=32,
                       attention=dict(num_heads=8),
                       mlp=dict(),
                       num_blocks=12   # For classification, adjust as needed
                      )

small_model = LGATr(in_mv_channels=1,
                    out_mv_channels=1,
                    hidden_mv_channels=8,
                    in_s_channels=16,
                    out_s_channels=0,
                    hidden_s_channels=16,
                    attention=dict(num_heads=6),
                    mlp=dict(),
                    num_blocks=6
                   )

full_var_model = LGATr(in_mv_channels=1,
                       out_mv_channels=1,
                       hidden_mv_channels=16,
                       in_s_channels=40,
                       out_s_channels=0,
                       hidden_s_channels=64,
                       attention=dict(num_heads=8),  
                       mlp=dict(),
                       num_blocks=6
                      )

pion_only_model = LGATr(in_mv_channels=1,
                        out_mv_channels=1,
                        hidden_mv_channels=16,
                        in_s_channels=0,
                        out_s_channels=0,
                        hidden_s_channels=32,
                        attention=dict(num_heads=8),  
                        mlp=dict(),
                        num_blocks=6
                       )



# Physics constants
Pion_Mass = 0.13957061   # GeV (from the PDG)