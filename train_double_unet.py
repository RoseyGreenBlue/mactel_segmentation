
import torch.nn as nn
import torch.nn.functional as F
import torch as torch
import torch.utils.data as TData
from torch import optim
from torch.nn.parameter import Parameter
from torchsummary import summary
from torch.autograd import Variable
import torchvision.transforms as transforms
import random

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

from tqdm import tqdm
from einops import rearrange
import copy

import HeidelbergTraining.UNet.UNet_functions as unet_functions
import HeidelbergTraining.UNet.UNet_model as unet_models

N_CLASSES = 13
img_size = (1, 512, 512)  # (channels, width, height)
LAYER_NAMES_13 = [
    "ILM", 
    "RNFL",
    "GCL", 
    "IPL", 
    "INL", 
    "OPL", 
    "ELM", 
    "PR1", 
    "PR2", 
    "RPE", 
    "Interlayer holes", 
    "BG top", 
    "BG bottom"
]
LAYER_NAMES_14 = [
    "ILM", 
    "RNFL",
    "GCL", 
    "IPL", 
    "INL", 
    "OPL", 
    "ELM", 
    "PR1", 
    "PR2", 
    "RPE", 
    "Collapsed Layers",
    "Cycsts",
    "Vitreous", 
    "Choroid/Sclera"
]
LAYER_NAMES_15 = [
    "ILM",   
    "RNFL", 
    "GCL", 
    "IPL", 
    "INL", 
    "OPL", 
    "ONL", 
    "ELM", 
    "PR1", 
    "PR2", 
    "RPE", 
    "BM", 
    "Interlayer holes", 
    "BG top", 
    "BG bottom"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class weights calculated using average percentage of image that each layer takes up
# class_weights_13 = Variable(torch.tensor([12, 12, 4, 5, 5, 6, 2, 12, 12, 12, 12, 1, 1]).float()).to(device) 
class_weights_13 = Variable(torch.tensor([9, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 1, 1]).float()).to(device) 
# class_weights_15 = Variable(torch.tensor([9, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 1, 1, 9, 9]).float()).to(device)
total_path = "./"


def make_list(path):
    ret = []
    for root,dirs,files in os.walk(path):
        ret = sorted([path + "/" + f for f in files if f[-4:] == ".png"])
        if len(ret) > 0:
            break
    return ret

class CustomDataset(TData.Dataset):
    def __init__(
        self, 
        image_paths, 
        diseased_paths, 
        augs=None, 
        image_specs=None, 
        n_classes=13,
        prev_model=None
    ):
        super(CustomDataset, self).__init__()
        
        sample = []
        
        for i,d in zip(image_paths, diseased_paths):
            sample.append((i,d))
        
        self.samples = sample
        self.channels = image_specs[0]
        self.rows = image_specs[1]
        self.col = image_specs[2]
        self.classes = n_classes
        self.augments = augs
        self.model = prev_model
        self.total_std = 63.13706830280864  # calculated on training set
        self.total_mean = 87.1074833179442 # calculated on training set
    
    def __getitem__(self, index):
        image = Image.open(self.samples[index][0]).convert("L")
        mask = np.zeros((self.rows, self.col), dtype=np.float32)
        
        ssl = True if self.samples[index][1] is None else False
        
        if not ssl:
            mask = Image.open(self.samples[index][1]).convert("L")
            
        if self.augments is not None:
            seed = random.random() * 100
            random.seed(seed)
            image = self.augments(image)
            if not ssl:
                random.seed(seed)
                mask = self.augments(mask)
        
        image = np.array(image)
        mean = np.mean(image)
        std = np.std(image)
        image = ((image - mean) / std)[np.newaxis, :]
        
        if self.model is not None:
            self.model = self.model.train()
            image2 = image[np.newaxis, :]
            image2 = torch.from_numpy(image2)
            image2 = Variable(image2.float()).to(device)
            
            image2 = self.model(image2)
            image2 = torch.argmax(rearrange(image2[0].detach(), 'c h w -> h w c'), dim=-1).cpu().numpy()
            image2 = image2[np.newaxis, :]
            
            image = np.concatenate((image, image2), axis=0)
        
        if not ssl:
            mask = np.array(mask, dtype=np.float32)

        return image, mask
    
    def __len__(self):
        return len(self.samples)

image_input_labeled_path = os.path.join(total_path, "HeidelbergTraining/images_input_labeled")
image_mask_path = os.path.join(total_path, "HeidelbergTraining/images_masks")

# go to images_input_labeled_paths.txt, images_masks_paths.txt for more info on these numbers
# 80:10:10 split at patient level
num_diseased = 292
diseased_train_cutoff = 230
normal_train_cutoff = 1381
diseased_valid_cutoff = 262
normal_valid_cutoff = 1521

images_all_labeled = make_list(image_input_labeled_path)
masks_all = make_list(image_mask_path)

image_list_train = images_all_labeled[num_diseased:normal_train_cutoff]
mask_list_train_13 = masks_all[num_diseased:normal_train_cutoff]

dimgs = images_all_labeled[:diseased_train_cutoff]
dmasks = masks_all[:diseased_train_cutoff]

while len(image_list_train) < (2 * (normal_train_cutoff - num_diseased)):
    image_list_train.extend(dimgs)
    mask_list_train_13.extend(dmasks)

del dimgs
del dmasks

d_image_list_cv = images_all_labeled[diseased_train_cutoff:diseased_valid_cutoff]
d_mask_list_cv_13 = masks_all[diseased_train_cutoff:diseased_valid_cutoff]

d_image_list_test = images_all_labeled[diseased_valid_cutoff:num_diseased]
d_mask_list_test_13 = masks_all[diseased_valid_cutoff:num_diseased]


image_list_train = sorted(image_list_train)
mask_list_train_13 = sorted(mask_list_train_13)

d_image_list_cv = sorted(d_image_list_cv)
d_mask_list_cv_13 = sorted(d_mask_list_cv_13)

d_image_list_test = sorted(d_image_list_test)
d_mask_list_test_13 = sorted(d_mask_list_test_13)

def train_net(
    epochs = 50,
    save_name = "UNet_pytorch",
    save_path = "/data/HeidelbergTraining/saved_model_pytorch",
    train_data=None,
    valid_data=None,
    loss_fn=None,
    u_model=None,
    optimizer=None,
    lr_schedule=None,
):
    
    train_loss = []
    val_loss = []
    o_params = None
    
    min_train_loss = 50000
    min_valid_loss = 50000
    plateau = 0
    best_model = None
    
    loss = None
    
    with open("./HeidelbergTraining/saved_model_pytorch/train_status.txt", "w") as status_file:
        print("cleared previous training data from status file")
    
    for i in range(epochs):
        u_model = u_model.train()
        print("epoch %s:"%i)
        loss_train_epoch = 0
        loss_valid_epoch = 0
        train_data=tqdm(train_data)

        for j, (image, mask) in enumerate(train_data):
            optimizer.zero_grad()
            
            image = Variable(image.float()).to(device)
            mask = Variable(mask.long()).to(device)
            output = u_model(image)

            loss = loss_fn(output, mask)
            loss.backward()
            optimizer.step()

            loss_train_epoch += loss.item()
            status = "epoch %s, "%i + "current loss: %s"%loss.item()
            train_data.set_description(status)
        
        u_model = u_model.eval()
        with torch.no_grad():
            for image, mask in valid_data:
                image,mask = Variable(image.float()).to(device), Variable(mask.long()).to(device)

                output = u_model(image)

                loss = loss_fn(output, mask)

                loss_valid_epoch += loss.item()
         

        loss_train_epoch /= len(train_data)
        loss_valid_epoch /= len(valid_data)


        if loss_train_epoch < min_train_loss:
            min_train_loss = loss_train_epoch

        if loss_valid_epoch < min_valid_loss:
            min_valid_loss = loss_valid_epoch
            plateau = 0

            best_model = u_model.state_dict()

        else:
            plateau += 1

        lr_schedule.step() 
        train_loss.append(loss_train_epoch)
        val_loss.append(loss_valid_epoch)
        with open("./HeidelbergTraining/saved_model_pytorch/train_status.txt", "a") as status_file:
            status_file.write("epoch %3d:\n"%i)
            status_file.write("best train loss so far: %s\n"%min_train_loss)
            status_file.write("avg train loss of this epoch: %s\n"%train_loss[-1])
            status_file.write("best valid loss so far: %s\n"%min_valid_loss)
            status_file.write("avg valid loss of this epoch: %s\n"%val_loss[-1])
            status_file.write("------\n\n")
            torch.save(best_model, os.path.join(save_path, save_name))


            if plateau > 9:
                status_file.write("Valid Loss hasn't decreased even after %s"%plateau + " epochs. Ending training now\n")
                status_file.write("The best model had a minimum loss of %s\n"%train_loss[-1])
                break

    with open("./HeidelbergTraining/saved_model_pytorch/train_status.txt", "a") as status_file:
        torch.save(best_model, os.path.join(save_path, save_name))
        status_file.write("\n\nlosses:\n")
        status_file.write("train:\n%s\n\nvalid:\n"%train_loss.__repr__())
        status_file.write(val_loss.__repr__())

        
standard_unet_model = unet_models.unet_model(
    32, 
    num_downs=4, 
    n_classes=13, 
    input_channels=1, 
    name="standard unet",
).to(device)
state_dict = torch.load("./HeidelbergTraining/saved_model_pytorch/standard unet_iter1")
standard_unet_model.load_state_dict(state_dict)

double_unet_model = unet_models.unet_model(
    32, 
    num_downs=4, 
    n_classes=13, 
    input_channels=2, 
    name="double unet",
).to(device)
state_dict = torch.load("./HeidelbergTraining/saved_model_pytorch/double unet_iter1")
double_unet_model.load_state_dict(state_dict)

aug1 = transforms.RandomChoice([
    transforms.RandomHorizontalFlip(),
    transforms.Compose([
        transforms.RandomCrop(size=(256, 256)),
        transforms.Resize((512, 512), interpolation=Image.NEAREST),
        transforms.RandomHorizontalFlip()
    ]),
    transforms.Compose([
        transforms.RandomRotation(15),
        transforms.CenterCrop((256, 256)),
        transforms.Resize((512, 512), interpolation=Image.NEAREST),
        transforms.RandomHorizontalFlip()
    ]),
])

num_batches = 2
iter_train = TData.DataLoader(
    CustomDataset(
        image_list_train, 
        mask_list_train_13, 
        image_specs=img_size, 
        augs=aug1,
        prev_model=standard_unet_model
    ),
    batch_size=num_batches,
    shuffle=True
)
print("num batches: %d"%num_batches)

iter_d_valid = TData.DataLoader(
    CustomDataset(d_image_list_test, d_mask_list_test_13, image_specs=img_size, prev_model=standard_unet_model)
)

iter_d_test = TData.DataLoader(
    CustomDataset(d_image_list_cv, d_mask_list_cv_13, image_specs=img_size, prev_model=standard_unet_model)
)

optimizer = optim.Adam(double_unet_model.parameters(), lr = 0.0003, weight_decay=1e-5)
lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6, 12, 18, 24], gamma=0.5)
loss_fn = nn.CrossEntropyLoss(weight=class_weights_13)

print(class_weights_13)

train_net(
    epochs = 50,
    save_name = "%s_iter1"%double_unet_model.name,
    save_path = "./HeidelbergTraining/saved_model_pytorch",
    train_data=iter_train,
    valid_data=iter_d_valid,
    loss_fn=loss_fn,
    u_model=double_unet_model,
    optimizer=optimizer,
    lr_schedule=lr_schedule,
)
