#PancreaticSeg
PancreaticSeg is the internship project of Ali ALHAJ ABDO, a third year medical student in the MD-PHD program of the University of Strasbourg, France.

Contact : ali.al-haj-abdo@etu.unistra.fr

Supervised by : professor Cedric WEMMERT - SDC Project Manager, ICube Laboraty, Illkirch-Graffenstaden, France 

Project : Segmentation of the pancreas from CT scans.

Dates : 03/07/2023 - 25/08/2023

The code was inspired by and adapted from amine0110's Liver segmentation project :  https://github.com/amine0110/Liver-Segmentation-Using-Monai-and-PyTorch

The report will be submitted to professor WEMMERT. You can also feel free to contact me for a copy.

User guide:  

The ensemble of folders is located on the network through PC1stagiaire2023
a copy will be put on my local account on Cabria.

The folder is named PancreaticSegmentation. It contains the following python scripts of the model :

- preprocess.py → script contains the prepare function necessary to pre-processs the images. The transformation functions all come from the MONAI library.

- utilities.py → script contains the training function and the loss function.

- train.py → script that allows you to define the directories, train the function, change networks, and hyperparameters
                 → the data is usually separated in the following format : 
	      define the directory you want to place your folders 
	      6 folders are created  : TestSegmentation, TestVolumes, TrainSegmentation, TrainVolumes, ValSegmentation, ValVolumes
	      where volumes refers to the CT scans in nii.gz format and the Segmentation folders contain the binary masks in nii.gz format.


- val.ipynb → The jupyter notebook format was chosen for this script as it felt easier to follow how the model was training by producing graphs and images. 
	      →It also allows saving the images in the nii.gz format to be further viewed using 3DSlicer, ITKsnap or any NIfTI reader.

- test.ipynb → A jupyter notebook script that allows to test the model on never seen before data.

- Data_preparation.ipynb → A jupyer notebook script that contains all the functions used for preparing the data 

The foler contains additional folders : 

- Model → contains the best trained model tested. Hyperparameters will be the default ones in the train.py script. 

- output_nii → contains examples of the NIfTI files created from the model. 

- Subtask1 → example of how the folders should be organized, also contains the dataset tested 

- AP-HP Scans → contains annotated CT scans of the hospital
		PA[x] (x = 0,1,2) → contains the non prepared scans
		AP-HP_segmented → prepared AP-HP scan segmentations 
		AP-HP_volume → prepared AP-HP scan volumes




