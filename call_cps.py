import sys

sys.path.append('./TorchSemiSeg/exp.voc/voc8.res50v3+.CPS+CutMix/')
from network import Network
import dataloader
from config import config
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os


# imgs should be 512 x 512


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
model = Network(
    config.num_classes, 
    criterion=criterion, 
    pretrained_model=config.pretrained_model,
    norm_layer=torch.nn.BatchNorm2d
).to(device)

def output_rearrange(output, num_classes):
    argmax_output = output[0][0].detach()[:, :, None]
    for i in range(1, num_classes):
        argmax_output = torch.cat((argmax_output, output[0][i].detach()[:, :, None]), dim=2)
    
    return torch.argmax(argmax_output, dim=-1).cpu().numpy()

# normalize invididual image to have mean 0 and variance 1
def normalize(img):
    img = img.astype(np.float32) / 255.0
    img = img - np.mean(img.ravel())
    img = img / np.std(img.ravel())

    return img

# extract the Nth (img, mask) pair from list containing img and file paths
def get_test_image_mask(imgs, masks, N):
    img = np_rearrange(normalize(cv2.imread(imgs[N])))
    mask = cv2.imread(masks[N], cv2.IMREAD_GRAYSCALE)

    img = torch.from_numpy(np.ascontiguousarray(img))[None, :, :, :].float().to(device)
    mask = torch.from_numpy(np.ascontiguousarray(mask))[None, :, :].long().to(device)

    return (img, mask)

# adds cysts to the output and mask using information from the image and mask
def add_13_label(output, image, mask):
    avg_pooler = nn.AvgPool2d(4, stride=2).to(device)
    reduced_mask = (avg_pooler(mask[None, :, :, :].float()))[0][0].detach().cpu().numpy()
    reduced_output = (avg_pooler(output[None, None, :, :].float()))[0][0].detach().cpu().numpy()
    reduced_image = (avg_pooler(image[:,0:1,:,:]))[0][0].detach().cpu().numpy()

		output = output.cpu().numpy()
    
    threshold = np.percentile(reduced_image, 25)

    a = [reduced_image < threshold][0].astype(np.int8)
    b = [reduced_output == 0][0].astype(np.int8)
    b = ((a + b) // 2).astype(bool)

    a = [reduced_image < threshold][0].astype(np.int8)
    c = [reduced_mask == 0][0].astype(np.int8)
    c = ((a + c) // 2).astype(bool)

    yvals, xvals = np.where(c)
    for (x,y) in zip(xvals, yvals):
        mask[0, y*2:y*2+5, x*2:x*2+5] = 13

    yvals, xvals = np.where(b)
    for (x,y) in zip(xvals, yvals):
        output[y*2:y*2+5,x*2:x*2+5] = 13
    
    return output, mask

# given a list of filepaths for imgs and their respective masks, and an eval_model, this generates predictions 
def get_prediction(imgs, masks, eval_model, ignore_0=False, n_classes=list(range(14))):
		eval_model.eval()
		outputs = []
		for i in range(num_images):
			image,mask = get_test_image_mask(imgs, masks, i)
			output = eval_model.forward(image)
			output = torch.from_numpy(output_rearrange(output, 13))

			output_14, mask_14 = add_13_label(output, image, mask)

			mask_14 = mask_14[0].cpu().numpy()
			image = image[0][0].cpu().numpy()

			outputs.append((image, mask_14, output_14))		
		return outputs


