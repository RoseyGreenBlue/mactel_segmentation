# mactel 10 layer segmentation

## this is the code

#### evaluation_CPS.ipynb generates and saves a .csv file containing per-layer IOUs for the semi-supervised model and evaluation_UNet.ipynb does the same for the UNet (both standard and double) and also imports that csv file and plots the data and calculates confidence intervals for the data using bootstrapping

#### data_collection_creation.ipynb creates all the data: the input data, the ground truths, as well as the images used to evaluate the baseline Heidelberg auto model

#### call_cps.py is a  file I made that generates a prediction for a list of images and masks using the top model (CPS+CutMix trained on 100% of unlabeled data)

#### train_standard_unet.py is the training loop used to train the standard unet
#### train_double_unet.py is the training loop used to train the double unet
#### train_semi_seg/train.py is the training loop used to train the SemiSupervised model, see:
https://github.com/charlesCXK/TorchSemiSeg
(I modified dataloader.py and config.py)

###### -----------------------------------------------------------------
###### git pull https://github.com/RoseyGreenBlue/mactel_segmentation.git 

## abstract link: https://iovs.arvojournals.org/article.aspx?articleid=2790110
## paper link: https://tvst.arvojournals.org/article.aspx?articleid=2802212
