### Initital data preparation and loading data for training and validation. The preprocessing() function prepares data, extracting training variables defined in 'LGATr_config.py', rescaling them, shuffling, splitting and then saving them in pickle files.
### This function has to be run before training, since it is not called by the train_LGATr() function. The 'preprocessingPlots()' function create plots of the training variables, allowing to check the quality of the applied rescaling. The data_prepper()
### function reconstructs multivector and scalar variables and is called in the train_LGATr() function.
### Paths may need to be modified in the main when ran directly



## imports
import uproot
import os
import torch
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lgatr.interface import embed_vector
from LGATr_config import train_vars, fourv_vars, scalar_vars, full_scalar_vars, scalars_wo_tau, alt_fourv_vars, alt_scalar_vars, var_norm, Pion_Mass





def read_data(path, vars_list, label, frac=1):
    '''Preparation of raw data.
    
       Inputs:
        path: str, path of the data to read
        vars_list: list of str, list of variables to read
        label: list of int, list of labels to read
        frac: float, fraction of data to read
    '''

    with uproot.open(path) as file:
        event = file["events"]
        df = event.arrays(expressions=vars_list, library="pd")   # Convert to pandas dataframe
        df = df.sample(frac=frac)
        df = df[vars_list]

        if label is not None:
            df["label"] = label
    return df



def preprocessing(train_path_sig, train_path_bkg, preprocessed_dir):
    '''Reads the ROOT file containing all data, splits them into training, testing, and validation samples.
       Then, saves them into pickle files.
       
       Inputs:
        train_path_sig: str, path of the signal data to train
        train_path_bkg: str, path of the signal + background data to train
        preprocessed_dir: str, path of the folder where outputs will be stored
    ''' 
    
    print('Reading data')

    df_sig = read_data(path=train_path_sig, vars_list=train_vars, label=1)
    df_train_sig, df_test_sig = train_test_split(df_sig, train_size=0.7, test_size=0.3, random_state=12)   # Split signal data into training and testing samples

    df_bkg = pd.DataFrame()
    for i in range(0,100):
        if i == 51:   # This chunk is somehow corrupted in the current data
            continue
        chunk_path = f'{train_path_bkg}/chunk_{i}.root'
        if os.path.exists(chunk_path):
            temp_df = read_data(path=chunk_path, vars_list=train_vars, label=0)
            df_bkg = pd.concat([df_bkg, temp_df], ignore_index=True)

    df_train_bkg, df_test_bkg = train_test_split(df_bkg, train_size=0.7, test_size=0.3, random_state=12)   # Split background data into training and testing samples

    df_training_data = pd.concat([df_train_sig, df_train_bkg], ignore_index=True).sample(frac=1).reset_index(drop=True)
    df_testing_data = pd.concat([df_test_sig, df_test_bkg], ignore_index=True).sample(frac=1).reset_index(drop=True)
    df_testing_data, df_val_data = train_test_split(df_testing_data, train_size=0.5, test_size=0.5, random_state=12)   # Split testing data into validation and testing samples    

    for var in var_norm:
        mean = var_norm[var]["mean"]
        std = var_norm[var]["std"]
        log = var_norm[var]["log"]
        
        if log:
            df_training_data[var] = (np.log(df_training_data[var]) - mean) / std
            df_val_data[var] = (np.log(df_val_data[var]) - mean) / std
            df_testing_data[var] = (np.log(df_testing_data[var]) - mean) / std
        else:
            df_training_data[var] = (df_training_data[var] - mean) / std
            df_val_data[var] = (df_val_data[var] - mean) / std
            df_testing_data[var] = (df_testing_data[var] - mean) / std

    with open(f'{preprocessed_dir}/Training.pkl', "wb") as f:   # Repackaging the data
        pkl.dump(df_training_data, f)   # Saving data in files
    with open(f'{preprocessed_dir}/Validation.pkl',"wb") as f:
        pkl.dump(df_val_data, f)
    with open(f'{preprocessed_dir}/Testing.pkl',"wb") as f:
        pkl.dump(df_testing_data, f)

    print('Data saved')
    return



def preprocessingPlots(preprocessed_dir):
    '''Plots the distributions of input variables from the training data.

       Inputs:
        preprocessed_dir: str, path of the folder where outputs will be stored
    '''

    with open(os.path.join(preprocessed_dir, 'Training.pkl'), 'rb') as f:
        df_train = pkl.load(f)
    
    output_dir = os.path.join(preprocessed_dir, "plots_input")
    os.makedirs(output_dir, exist_ok=True)

    # Plotting loop
    for var in var_norm:
        ## If not already normalized by the preprocessing function, normalize here:
        mean = var_norm[var]["mean"]
        std = var_norm[var]["std"]
        log = var_norm[var]["log"]
        
        if var not in df_train.columns:
            print(f"Warning: {var} not found in DataFrame.")
            continue

        # if log:
        #     df_train[var] = (np.log(df_train[var]) - mean) / std
        # else:
        #     df_train[var] = (df_train[var] - mean) / std

        plt.figure(figsize=(8, 6))
        plt.hist(df_train[var], bins=200, histtype='step', color='blue', alpha=0.8)

        title = var
        if log:
            title = "log(" + title + ")"
        if mean != 0:
            title += " - " + str(mean)
            if std != 1:
                title = "(" + title + ") / " + str(std)
        elif mean == 0 and std != 1:
            title += " / " + str(std)
        
        plt.title(f'Distribution of: {title}', fontsize=16)
        plt.xlabel("Value", fontsize=14)
        plt.ylabel('Counts', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{var}_dist.png"))
        plt.close()




def data_prepper(preprocessed_dir, debug=False, tau=False):
    '''Prepares training, validation, and testing data.
    
       Inputs:
        preprocessed_dir: str, path of the folder where outputs will be stored
        debug: bool, enable/disable debugging mode. If 'debug = True', only 1% of the data is treated.
        tau: bool, enable/disable variables associated to tau in 'LGATr_config.py'
    '''

    print('Start prepping the data')

    with open(f'{preprocessed_dir}/Training.pkl', 'rb') as file:
        df_train_data = pkl.load(file)
    with open(f"{preprocessed_dir}/Validation.pkl", 'rb') as file:
        df_val_data = pkl.load(file)
    with open(f"{preprocessed_dir}/Testing.pkl", 'rb') as file:
        df_test_data = pkl.load(file)

    if debug:   # Debugging mode activated -> only 1% of data treated
        df_train_data = df_train_data.sample(frac=0.01).reset_index(drop=True)
        df_val_data = df_val_data.sample(frac=0.01).reset_index(drop=True)
        df_test_data = df_test_data.sample(frac=0.01).reset_index(drop=True)
    else:
        df_train_data = df_train_data.sample(frac=1).reset_index(drop=True)
        df_val_data = df_val_data.sample(frac=1).reset_index(drop=True)
        df_test_data = df_test_data.sample(frac=1).reset_index(drop=True)

    print(f"Training data size   : {len(df_train_data)}")
    print(f"Validation data size : {len(df_val_data)}")
    print(f"Testing data size    : {len(df_test_data)}")

    train_mv = construct_mv(df_train_data, tau=tau)
    train_sc = construct_sc(df_train_data, tau=tau)
    train_data = embedding(multivectors=train_mv, scalars=train_sc)
    train_labels = torch.tensor(df_train_data["label"].to_numpy(), dtype=torch.float32)

    val_mv = construct_mv(df_val_data, tau=tau)
    val_sc = construct_sc(df_val_data, tau=tau)
    val_data = embedding(multivectors=val_mv, scalars=val_sc)
    val_labels = torch.tensor(df_val_data["label"].to_numpy(), dtype=torch.float32)

    test_mv = construct_mv(df_test_data, tau=tau)
    test_sc = construct_sc(df_test_data, tau=tau)
    test_data = embedding(multivectors=test_mv, scalars=test_sc)
    test_labels = torch.tensor(df_test_data["label"].to_numpy(), dtype=torch.float32)

    print("Prepping done")
    return train_data, train_labels, val_data, val_labels, test_data, test_labels



def construct_mv(input_df, tau=False):
    """
    Takes momentum of pions and their mass in the PDG to reconstruct the four vector and returns multivectors.
    
    Inputs:
        input_df: pandas dataframe, containing all training variables
        tau: bool, enable/disable scalar variables associated to tau in 'LGATr_config.py'
    """

    local_fourv_vars = list(fourv_vars)

    if tau:
        local_fourv_vars = local_fourv_vars + ['TauCand1_px', 'TauCand1_py', 'TauCand1_pz', 'TauCand2_px', 'TauCand2_py', 'TauCand2_pz', 'TauCand1_m', 'TauCand2_m'] 
    df = input_df[local_fourv_vars]   # Keep only variables relevant to four vector construction
    data = []

    for _, row in df.iterrows():   # Loop over all row in the dataframe
        # Form 2-line tensors for each pion, where each line contains px, py, pz
        p1 = torch.tensor([[row[var] for var in local_fourv_vars[:3]],[row[var] for var in local_fourv_vars[9:12]]])
        p2 = torch.tensor([[row[var] for var in local_fourv_vars[3:6]],[row[var] for var in local_fourv_vars[12:15]]])
        p3 = torch.tensor([[row[var] for var in local_fourv_vars[6:9]],[row[var] for var in local_fourv_vars[15:18]]])

        # Compute energies
        E1 = (Pion_Mass**2 + (p1**2).sum(dim=-1, keepdim=True))**0.5
        E2 = (Pion_Mass**2 + (p2**2).sum(dim=-1, keepdim=True))**0.5
        E3 = (Pion_Mass**2 + (p3**2).sum(dim=-1, keepdim=True))**0.5

        # Form 4-vectors
        P1 = torch.cat((E1, p1), dim=-1)
        P2 = torch.cat((E2, p2), dim=-1)
        P3 = torch.cat((E3, p3), dim=-1)

        # Embed to multivectors
        multivector1 = embed_vector(P1)
        multivector2 = embed_vector(P2)
        multivector3 = embed_vector(P3)
        multivector = torch.cat([multivector1, multivector2, multivector3], 0)   # Places the three multivectors on the same line of the new tensor. Size: [48]

        if tau:
            Tau_Cand1_mass = torch.tensor(row['TauCand1_m'])
            Tau_Cand2_mass = torch.tensor(row['TauCand2_m'])
            p_tau1 = torch.tensor([row['TauCand1_px'], row['TauCand1_py'], row['TauCand1_pz']]).unsqueeze(0)   # Momentum vector
            p_tau2 = torch.tensor([row['TauCand2_px'], row['TauCand2_py'], row['TauCand2_pz']]).unsqueeze(0)
            E_tau1 = (Tau_Cand1_mass**2 + (p_tau1**2).sum(dim=-1, keepdim=True))**0.5   # E = sqrt(p^2 + m^2)
            E_tau2 = (Tau_Cand2_mass**2 + (p_tau2**2).sum(dim=-1, keepdim=True))**0.5
            P_tau1 = torch.cat((E_tau1, p_tau1), dim=-1)   # Quadri-vector energy-momentum (E, p)
            P_tau2 = torch.cat((E_tau2, p_tau2), dim=-1)
            multivector_tau1 = embed_vector(P_tau1)   # Embed a four-vector in a 16-dim multivector for geometric algebra
            multivector_tau2 = embed_vector(P_tau2)
            multivector = torch.cat([multivector, multivector_tau1, multivector_tau2], 0)   # If 'tau=True', also concatenate the tau multivectors. Size: [80]

        data.append(multivector)

    multivectors = torch.stack(data).unsqueeze(2)   # Places the N multivectors along one dim. Size: [N, 48] (or [N, 80]) and then adds a new dim at 'dim=2'. Size: [N, 48, 1]
    return multivectors



def construct_sc(input_df, tau=False):
    """
    Takes scalar variables from the dataframe and prepares them for training.

    Inputs:
        input_df: pandas dataframe, containing all training variables
        tau: bool, enable/disable variables associated to tau in 'LGATr_config.py'
    """

    local_scalar_vars = full_scalar_vars  # Specify variables to be used as scalars
    if tau:
        local_scalar_vars = scalars_wo_tau
    df = input_df[local_scalar_vars]
    df = df.to_numpy()   # Converts panda dataframe to NumPy array
    scalars = torch.tensor(df, dtype=torch.float32)   # Converts the NumPy array into a PyTorch tensor of size [nb events, nb of scalars]

    scalars = scalars.unsqueeze(1)   # [nb events, 1, nb of scalars]
    scalars = torch.cat([scalars, scalars, scalars, scalars, scalars, scalars], 1)   # [nb events, 6, nb of scalars]
    return scalars



def embedding(multivectors, scalars):
    """
    Create a dictionnary that contains multivectors and scalars.
    
    Inputs:
        multivectors: torch.tensor, tensor that contains all multivectors
        scalars: torch.tensor, tensor that contains all scalars
    """

    return {"mv": multivectors, "sc": scalars}



def df_expander(vars_list, df):
    '''Expands columns in the dataframe that contain lists into separate columns.
    
       Inputs:
        input_df: pandas dataframe, containing all training variables
    '''

    for vars in vars_list:
        expanded = pd.DataFrame(df[vars].tolist(), columns=[f'{vars}_a', f'{vars}_b'], index=df.index)
        df = df.drop(columns=[vars]).join(expanded)
    return df



def alt_construct_mv(input_df):
    """
    Takes momentum of pions and the mass in the PDG to reconstruct their four-vectors.
    
    Inputs:
        input_df: pandas dataframe, containing all the training variables
    """
    
    local_fourv_vars = list(alt_fourv_vars)
    df = input_df[local_fourv_vars]   # Keep only the variables relevant to four vector construction
    multivectors = torch.empty(0, dtype=torch.float32)

    data = []
    for _, row in df.iterrows():   # Loop over all row in the dataframe
        # Form 2-line tensors for each pion, where each line contains px, py, pz
        p1 = torch.tensor([row[var] for var in local_fourv_vars[:3]]).unsqueeze(0)
        p2 = torch.tensor([row[var] for var in local_fourv_vars[3:6]]).unsqueeze(0)
        p3 = torch.tensor([row[var] for var in local_fourv_vars[6:9]]).unsqueeze(0)

        # Compute energies
        E1 = (Pion_Mass**2 + (p1**2).sum(dim=-1,keepdim=True))**0.5
        E2 = (Pion_Mass**2 + (p2**2).sum(dim=-1, keepdim=True))**0.5
        E3 = (Pion_Mass**2 + (p3**2).sum(dim=-1, keepdim=True))**0.5

        # Form 4-vectors
        P1 = torch.cat((E1, p1), dim=-1)
        P2 = torch.cat((E2, p2), dim=-1)
        P3 = torch.cat((E3, p3), dim=-1)

        # Embed to multivectors
        multivector1 = embed_vector(P1)
        multivector2 = embed_vector(P2)
        multivector3 = embed_vector(P3)
        multivector = torch.cat([multivector1, multivector2, multivector3], 0)

        data.append(multivector)

    multivectors = torch.stack(data).unsqueeze(2)   # Places the N multivectors along one dim. Size: [N, 48] (or [N, 80]) and then adds a new dim at 'dim=2'. Size: [N, 48, 1]
    return multivectors



def alt_construct_sc(input_df):
    """
    Takes scalar variables from the dataframe and prepares them for training.

    Inputs:
        input_df: pandas dataframe, containing all the training variables
    """
    
    local_scalar_vars = list(alt_scalar_vars)
    df = input_df[local_scalar_vars]
    df = df_expander(vars_list=alt_scalar_vars, df=df)
    df = df.to_numpy()
    scalars = torch.tensor(df, dtype=torch.float32)

    scalars = scalars[:,:32].unsqueeze(1)
    scalars = torch.cat([scalars, scalars, scalars, scalars, scalars, scalars], 1)
    scalars = torch.cat([scalars, scalars], 2)
    return scalars





if __name__ == '__main__':
    preprocessed_dir = '/afs/cern.ch/work/t/thseydou/public/LGATr/preprocessed_data'
    train_path_sig = '/eos/experiment/fcc/ee/analyses_storage/flavor/Bs2TauTau/flatNtuples/winter2023/analysis_stage1_withSimpleCut/p8_ee_Zbb_ecm91_EvtGen_Bs2TauTauTAUHADNU/chunk_0.root'   # We only use chunk_0 because we want to have ~50/50% of signal/background in the training sample, so chunk_0 is already enough
    train_path_bkg = '/eos/experiment/fcc/ee/analyses_storage/flavor/Bs2TauTau/flatNtuples/winter2023/analysis_stage1_withSimpleCut/p8_ee_Zbb_ecm91'
    
    preprocessing(train_path_sig=train_path_sig, train_path_bkg=train_path_bkg, preprocessed_dir=preprocessed_dir)
    preprocessingPlots(preprocessed_dir=preprocessed_dir)
