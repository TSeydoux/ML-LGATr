### Training and initial validation of the LGATr model. It saves results of the testing sample directly in a pickle file to improve efficiency. 



## imports
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import numpy as np
from torch.utils.data import DataLoader
from lgatr import LGATr
from LGATr_config import LGATrWrapper, LGATrDataset





def move_batch_to_device(batch:dict, device:torch.device) -> dict:
    '''Utilitary function to move batch to device.'''

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch



def train_LGATr(train_data, train_labels, val_data, val_labels, test_data, test_labels, model_name, directory, model=None, lr=0.0001, batch_size=32, num_epochs=10, patience=10):
    """
    Function used to train the model and to save the output scores and labels.
    
    Inputs:
        train_data: dataset, training data
        train_labels: dataset, training labels
        val_data: dataset, validation data
        val_labels: dataset, validation labels
        test_data: dataset, testing data
        test_labels: dataset, testing labels
        lr: float, learning rate
        batch_size: int, size of the batch
        num_epochs: int, number of epochs
        patience: int, number of epochs without improvement in model performances before the training stops
        model: LGATr, model to use, see 'LGATr_config.py' for more informations
        directory: str, directory of the model
    """

    print("Start training")
    if directory is not None:
        logfile = f"{directory}/LGATr_log.txt"   # Creates log file
        csvfile = f"{directory}/LGATr_log.csv"   # Creates .csv file
    
    if model is None:
        model = LGATr(in_mv_channels=1,
                      out_mv_channels=1,
                      hidden_mv_channels=16,
                      in_s_channels=16,
                      out_s_channels=0,
                      hidden_s_channels=32,
                      attention=dict(num_heads=8),
                      mlp=dict(),
                      num_blocks=12
                     )

    wrapper = LGATrWrapper(net=model, mean_aggregation=True, mv_only=True)   # Set mv_only to True if you only use multivectors
    best_accuracy = -1
    best_model_state = None
    best_epoch = -1
        
    with open(logfile, "a") as file:
        file.write(f'LGATr model created at {datetime.datetime.now()}\n')
        file.write(f'Using model: {model_name}\n')
        file.write(f'Using batch size: {batch_size}\n')
        file.write(f'Using learning rate: {lr}\n')
        file.write(f'Using number of epochs: {num_epochs}\n')
        file.write(f'Using patience: {patience}\n')

    # Prepare training data
    print(f'Training data size: {len(train_data["mv"])} mv, {len(train_data["sc"])} sc')
    print(f'Training labels size: {len(train_labels)}')
    with open(logfile, "a") as file:
        file.write(f'Training data size: {len(train_data["mv"])} mv, {len(train_data["sc"])} sc\n')
        file.write(f'Training labels size: {len(train_labels)}\n')
    train_dataset = LGATrDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare validation data
    print(f'Validation data size: {len(val_data["mv"])} mv, {len(val_data["sc"])} sc')
    print(f'Validation labels size: {len(val_labels)}')
    with open(logfile, "a") as file:
        file.write(f'Validation data size: {len(val_data["mv"])} mv, {len(val_data["sc"])} sc\n')
        file.write(f'Validation labels size: {len(val_labels)}\n')
    if len(val_data["mv"]) != len(val_labels):
        with open(logfile, "a") as file:
            file.write("Error: validation data and labels must have the same length.")
        raise ValueError("Validation data and labels must have the same length.")
    val_dataset = LGATrDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Count number of signal and background events in validation set
    n_signal = (val_labels == 1.0).sum().item()
    n_bkg = (val_labels == 0.0).sum().item()
    print(f"Number of validation signal samples: {n_signal}")
    print(f"Number of validation background samples: {n_bkg}")
    with open(logfile, "a") as file:
        file.write(f"Number of validation signal samples: {n_signal}\n")
        file.write(f"Number of validation background samples: {n_bkg}\n")    

    # Set loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    wrapper.to(device)
    model.train()
    for param in model.parameters():
        assert param.requires_grad

    # Header for the .csv file
    with open(csvfile, "w") as f:
        f.write("epoch,training_loss,validation_loss,training_accuracy,validation_accuracy\n")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        cum_loss = 0
        train_cor = 0
        train_tot = 0

        for batch_data, batch_labels in train_loader:
            batch_data = move_batch_to_device(batch=batch_data, device=device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()   # Reinitialize gradients of the model to 0
           
            outputs = wrapper(batch_data)   # Forward pass of the data to get the outputs
            outputs = outputs.squeeze()
            outputs = outputs.to(device)

            if outputs.numel() != batch_size:   # Checks that the outputs have the same number of elements as the batch
                continue
            if batch_labels.numel() != batch_size:
                continue
            
            loss = criterion(outputs, batch_labels)   # Calculates loss by comparing predictions and labels
            loss.backward()   # Backward pass, calculate the new gradients
            optimizer.step()   # Update the parameters of the model using the new gradients
            cum_loss += loss.item()

            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).long()
            train_tot += batch_labels.size(0)
            train_cor += (predicted == batch_labels.long()).sum().item()

        # Validation loop
        val_loss = 0
        val_cor = 0
        val_tot = 0
        model.eval()
        for val_batch, val_label in val_loader:
            with torch.no_grad():
                val_batch = move_batch_to_device(batch=val_batch, device=device)
                val_label = val_label.to(device)
                outputs = wrapper(val_batch)
                outputs = outputs.squeeze()
                outputs.to(device)

                probs = torch.sigmoid(outputs)
                if probs.numel() != batch_size:
                    continue
                if val_label.numel() != batch_size:
                    continue
                
                loss = criterion(outputs, val_label)
                val_loss += loss.item()

                val_tot += val_label.size(0)
                predicted = (probs > 0.5).long()
                val_cor += (predicted == val_label.long()).sum().item()

        avg_train_loss = cum_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_cor / train_tot
        val_accuracy = 100 * val_cor / val_tot
    
        print(f'Epoch {epoch} - Training loss   : {avg_train_loss:.6f}')
        print(f'Epoch {epoch} - Validation loss : {avg_val_loss:.6f}')
        print(f'Epoch {epoch} - Training accuracy   : {train_accuracy:.4f}%')
        print(f'Epoch {epoch} - Validation accuracy : {val_accuracy:.4f}%')
        with open(logfile, "a") as file:
            file.write(f'Epoch {epoch} - Training loss   : {avg_train_loss:.6f}\n')
            file.write(f'Epoch {epoch} - Validation loss : {avg_val_loss:.6f}\n')
            file.write(f'Epoch {epoch} - Training accuracy   : {train_accuracy:.4f}%\n')
            file.write(f'Epoch {epoch} - Validation accuracy : {val_accuracy:.4f}%\n')

        with open(csvfile, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.12f},{avg_val_loss:.12f},{train_accuracy:.12f},{val_accuracy:.12f}\n")
                
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = model.state_dict()
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                if logfile:
                    with open(logfile, "a") as f:
                        f.write("Early stopping triggered.\n")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Saving the best model iteration
    model.eval()
    model_path = f"{directory}/LGATr.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, model_path)

    print(f'Best model (epoch {best_epoch}) saved to {model_path}')
    with open(logfile, "a") as file:
        file.write(f'Best model (epoch {best_epoch}) saved to {model_path}\n')

    # Save the output scores and labels in a pickle file (efficient, no need to load the whole model since its already done)
    print(f'Testing data size: {len(test_data["mv"])} mv, {len(test_data["sc"])} sc')
    print(f'Testing labels size: {len(test_labels)}')
    with open(logfile, "a") as file:
        file.write(f'Testing data size: {len(test_data["mv"])} mv, {len(test_data["sc"])} sc\n')
        file.write(f'Testing labels size: {len(test_labels)}\n')
    if len(test_data["mv"]) != len(test_labels):
        with open(logfile, "a") as file:
            file.write("Error: testing data and labels must have the same length.")
        raise ValueError("Testing data and labels must have the same length.")
    test_data = move_batch_to_device(batch=test_data, device=device)
    test_labels = test_labels.to(device)

    with torch.no_grad():
        predictions = wrapper(test_data).squeeze()
        predictions = torch.atleast_1d(predictions)
        predictions = predictions.detach().cpu().numpy()

    labels = test_labels.detach().cpu().numpy()

    # Count signal and background samples
    n_signal = (labels == 1.0).sum()
    n_bkg = (labels == 0.0).sum()
    print(f"Number of testing signal samples: {n_signal}")
    print(f"Number of testing background samples: {n_bkg}")
    with open(logfile, "a") as file:
        file.write(f"Number of testing signal samples: {n_signal}\n")
        file.write(f"Number of testing background samples: {n_bkg}\n")    

    # Check if predictions for 'label==1' are higher than for 'label==0'
    mean_1 = predictions[labels == 1].mean() if np.any(labels == 1) else 0
    mean_0 = predictions[labels == 0].mean() if np.any(labels == 0) else 0

    if mean_1 < mean_0:
        predictions = 1 - predictions   # Reverse the predictions if necessary to always have avg signal scores > avg bkg scores
        print("Predictions required an inversion to keep average signal scores > average background scores")
    with open(logfile, "a") as file:
        file.write("Predictions required an inversion to keep average signal scores > average background scores\n")

    # Save to pickle
    with open(f"{directory}/scores.pkl", 'wb') as f:
        pkl.dump({'scores': predictions, 'labels': labels}, f)

    print(f"Saved predictions and labels to {directory}/scores.pkl")
    with open(logfile, "a") as file:
        file.write(f"Saved predictions and labels to {directory}/scores.pkl\n")
    print("Training done")
