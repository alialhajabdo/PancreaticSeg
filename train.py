from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
import torch
from preprocess import prepare
from utilities import train
from utilities import dice_metric



verbose = 1
#device = torch.device("cuda:0")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
if verbose:
    print('device is :', device)

# define where the dataset is placed according to the format defined in the readme file or in the preprocess.py script
data_dir = '/home/alhajabdo/PancreaticSegmentation/SubTask1/'

#define where the model will save the metrics : mean dice, loss etc..

model_dir = '/home/alhajabdo/PancreaticSegmentation/test/'
# define the ROI size in spatial_size and the batch size of through batch_size_def .
 
data_in = prepare(data_dir, cache=True,spatial_size=[128,128,64] ,batch_size_def = 2)  


# define the model : you can use the MONAI library which contains a lot of networks to use : https://docs.monai.io/en/stable/networks.html

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


# define the loss function 
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)

#define the optimizer function, learning rate (lr) , and weight decay 
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay=1e-3, amsgrad=True)

# define the maximum number of epochs in max_epochs. It is important to note that within the train function patience can be change. c.f. utilities.py
if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, max_epochs = 300, model_dir = model_dir, device=device)
