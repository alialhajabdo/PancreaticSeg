from monai.utils import first
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm

def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value

# This is the train function that will be used to train the model. you can define the amount of epochs you allow your model to keep training without improving the mean dice in pateince .

def train(model, data_in, loss, optim, max_epochs, model_dir, val_interval=1, device=torch.device("cuda")):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_val = []
    save_metric_train = []
    save_metric_val = []
    train_loader, val_loader = data_in

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:
            
            train_step += 1

            volume = batch_data["vol"]
            label = batch_data["seg"]
            label = label != 0
            volume, label = (volume.to(device), label.to(device))

            optim.zero_grad()
            outputs = model(volume)
            
            train_loss = loss(outputs, label)
            
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        print('-'*20)
        
        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        
        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        if (epoch + 1) % val_interval == 0:


            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0
                val_metric = 0
                epoch_metric_val = 0
                val_step = 0

                for val_data in val_loader:

                    val_step += 1

                    val_volume = val_data["vol"]
                    val_label = val_data["seg"]
                    val_label = val_label != 0
                    val_volume, val_label = (val_volume.to(device), val_label.to(device),)
                    
                    val_outputs = model(val_volume)
                    
                    val_loss = loss(val_outputs, val_label)
                    val_epoch_loss += val_loss.item()
                    val_metric = dice_metric(val_outputs, val_label)
                    epoch_metric_val += val_metric
  
                
                val_epoch_loss /= val_step
                print(f'val_loss_epoch: {val_epoch_loss:.4f}')
                save_loss_val.append(val_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_val.npy'), save_loss_val)

                epoch_metric_val /= val_step
                print(f'val_dice_epoch: {epoch_metric_val:.4f}')
                save_metric_val.append(epoch_metric_val)
                np.save(os.path.join(model_dir, 'metric_val.npy'), save_metric_val)
                checkpoint_filename = os.path.join(model_dir, f"checkpoint_epoch_{epoch + 1}_{epoch_metric_val}.pth")
                torch.save(model.state_dict(), checkpoint_filename)
                print(f"Checkpoint saved at epoch {epoch + 1}")
                if epoch_metric_val > best_metric:
                    best_metric = epoch_metric_val
                    best_metric_epoch = epoch + 1
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))
               
                else:
                    epochs_without_improvement += 1
                    #define patience i.e. the number of epochs without improvement 
                    if epochs_without_improvement >= 10:
                        print(f"Early stopping at epoch {epoch}.")
                        break
                print(
                    f"current epoch: {epoch + 1} current mean dice: {val_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )


    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")


def show_patient(data, SLICE_NUMBER=1, train=True, val=False):
    """
    This function is to show one patient from your datasets, so that you can si if the it is okay or you need 
    to change/delete something.

    `data`: this parameter should take the patients from the data loader, which means you need to can the function
    prepare first and apply the transforms that you want after that pass it to this function so that you visualize 
    the patient with the transforms that you want.
    `SLICE_NUMBER`: this parameter will take the slice number that you want to display/show
    `train`: this parameter is to say that you want to display a patient from the training data (by default it is true)
    `val`: this parameter is to say that you want to display a patient from the validation of patients.
    """

    check_patient_train, check_patient_val = data

    view_train_patient = first(check_patient_train)
    view_val_patient = first(check_patient_val)

    
    if train:
        plt.figure("Visualization Train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_train_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_train_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()
    
    if val:
        plt.figure("Visualization val", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"vol {SLICE_NUMBER}")
        plt.imshow(view_val_patient["vol"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"seg {SLICE_NUMBER}")
        plt.imshow(view_val_patient["seg"][0, 0, :, :, SLICE_NUMBER])
        plt.show()


