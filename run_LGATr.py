### Trains an LGATr model and performs evaluation automatically; saves everything in one directory



## imports
import os
from LGATr_config import standard_model, small_model, full_var_model, pion_only_model
from data_prep import data_prepper
from train_LGATr import train_LGATr
from plots import *





def main():
    """
    Training a new LGATr iteration and completing the whole evaluation process. LGATr iteration and evaluation plots are saved in a
    new directory. 
    """
    
    ####################################### All the options to configure before training ##########################################
    
    model_used = small_model   # Model you want to use, check 'LGATr_config.py' for details
    model_name = "small_model"   # Name of the model (free but better to keep the name of the variable)
    num_epochs = 100   # Nnumber of epochs to train
    patience = 100   # Number of epochs with no progress before early stop
    
    directory = f"/afs/cern.ch/work/t/thseydou/public/LGATr/trained_models/LGATr_{model_name}_{num_epochs}"   # Directory of the model
    preprocessed_dir = "/afs/cern.ch/work/t/thseydou/public/LGATr/preprocessed_data"   # Directory where to find the preprocessed data
    
    ###############################################################################################################################

    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' is ready.")
    except PermissionError:
        print(f"Permission denied: unable to create '{directory}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Get data
    train_data, train_labels, val_data, val_labels, test_data, test_labels = data_prepper(preprocessed_dir=preprocessed_dir, debug=False, tau=True)

    # Train the model and save scores and labels for analysis
    train_LGATr(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        test_data=test_data,
        test_labels=test_labels,
        model_name=model_name,
        directory=directory,
        model=model_used,
        lr=0.0001,
        batch_size=32,
        num_epochs=num_epochs,
        patience=patience
    )
    
    # Create plots
    trainingPlots(directory=directory)
    evaluationPlots(directory=directory)
    ROC(directory=directory)

 

    

if __name__ == '__main__':
    main()