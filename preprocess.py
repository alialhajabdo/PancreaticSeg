import os
from glob import glob
from monai.transforms import(
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    NormalizeIntensityd, 
    ScaleIntensityRanged,
    CropForegroundd,
    RandFlipd,
    RandRotate90d,
    RandCropByPosNegLabeld,
    RandHistogramShiftd,
    RandGaussianNoised,
    RandSpatialCropSamplesd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism

"""
This file is for preporcessing only, it contains all the functions that you need
to make your data ready for training.

You need to install the required libraries if you do not already have them.

pip install os, ...
"""


def prepare(in_dir, pixdim=(1, 1, 1.0), a_min=-150, a_max=170, spatial_size=[128,128,64], cache=True, batch_size_def = 2):

    """
    This function is for preprocessing, it contains only the basic transforms, but you can add more operations that you 
    find in the Monai documentation.
    https://monai.io/docs.html
    """
    

    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainSegmentation", "*.nii.gz")))

    path_val_volumes = sorted(glob(os.path.join(in_dir, "ValVolumes", "*.nii.gz")))
    path_val_segmentation = sorted(glob(os.path.join(in_dir, "ValSegmentation", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    val_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_val_volumes, path_val_segmentation)]

    train_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            NormalizeIntensityd(keys = ["vol"]),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            #RandGaussianNoised(keys = ["vol", "seg"], prob = 0.1),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),
            #RandSpatialCropSamplesd(keys = ["vol", "seg"], num_samples = 4, roi_size = spatial_size, random_size = False),   
            RandFlipd(keys=["vol", "seg"], prob = 0.5, spatial_axis=0),
            RandRotate90d(keys=["vol", "seg"], prob=0.5, spatial_axes=(0,1)),
            #RandCropByPosNegLabeld(keys=["vol", "seg"], label_key=['seg'], spatial_size = spatial_size, pos = 0.8, neg = 0.2),
            #RandHistogramShiftd(keys=["vol", "seg"]),
            ToTensord(keys=["vol", "seg"]),
    
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            NormalizeIntensityd(keys = ["vol"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),   
            ToTensord(keys=["vol", "seg"]),

            
        ]
    )

    

    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms,cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size= batch_size_def)

        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
        val_loader = DataLoader(val_ds, batch_size= batch_size_def)

        return train_loader, val_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1)

    return train_loader, val_loader
    for data in train_loader:
            print("Train data - After LoadImaged:", data["vol"].shape, data["seg"].shape)
            print("Train data - After AddChanneld:", data["vol"].shape, data["seg"].shape)
            print("Train data - After Spacingd:", data["vol"].shape, data["seg"].shape)
            print("Train data - After Orientationd:", data["vol"].shape, data["seg"].shape)
            print("Train data - After ScaleIntensityRanged:", data["vol"].shape, data["seg"].shape)
            print("Train data - After CropForegroundd:", data["vol"].shape, data["seg"].shape)
            print("Train data - After Resized:", data["vol"].shape, data["seg"].shape)
            print("Train data - After ToTensord:", data["vol"].shape, data["seg"].shape)
