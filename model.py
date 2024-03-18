import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import open_clip
from PIL import Image
from torch.autograd import Variable
import numpy as np
import h5py
import scipy.io
from tqdm import tqdm
import torchvision.transforms as T
from torch import linalg as LA
inputDim = 512
outputDim = -1
learningRate = 0.01 
epochs = 20
batchSize = 32
subject = 1

transform = T.ToPILImage()

with h5py.File('./subj01/betas_session01_s_zscored.hdf5', "r") as f:
    data_set = f['betas']
    outputDim = data_set.shape[1]*data_set.shape[2]*data_set.shape[3]

class fMRIDataset(Dataset):
    def __init__(self, image_path = 'nsd_stimuli.hdf5', target_path = './subj01/betas_session01_s_zscored.hdf5', nsd_expdesign_path = 'nsd_expdesign.mat'):
        print('init')
        self.images = h5py.File(image_path, 'r')['imgBrick']
        self.fmri_target = h5py.File(target_path, 'r')['betas']
        data = {}
        scipy.io.loadmat(nsd_expdesign_path,data)
        self.master_ordering = data['masterordering'][0]
        self.subject_ims =[]
        for i in range(len(data['subjectim'])):
            self.subject_ims.append(data['subjectim'][i])
        self.CLIP_model, _, self.CLIP_preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def __len__(self):
        return len(self.fmri_target)

    def __getitem__(self, idx):
        subject_im_idx = self.master_ordering[idx]
        coco_idx = self.subject_ims[subject][subject_im_idx]
        image = self.images[coco_idx]
        target = self.fmri_target[idx].flatten()
        target = torch.flatten(torch.from_numpy(self.fmri_target[idx]))
        target = torch.nan_to_num(target)
        temp = self.CLIP_preprocess(transform(image[0]).convert('RGB')).unsqueeze(0).to(self.device)
        image_features = self.CLIP_model.encode_image(temp)
        return image_features.flatten(), target

class LinearModel(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearModel, self).__init__()
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, x):
        out = self.linear(x / (LA.vector_norm(x) + 1e-16))#normalizing the image_features from CLIP
        return out.float()



model = LinearModel(inputDim, outputDim)
device = "cuda" if torch.cuda.is_available() else "cpu"
criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
dataset = fMRIDataset()
test_size = 0.15
val_size = 0.15
test_amount, val_amount = int(len(dataset) * test_size), int(len(dataset) * val_size)

train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
            (len(dataset) - (test_amount + val_amount)), 
            test_amount, 
            val_amount
])

train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batchSize,
            shuffle=True,
)
val_dataloader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batchSize,
            shuffle=True,
)
test_dataloader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batchSize,
            shuffle=True,
)

for epoch in tqdm(range(epochs), desc='Epochs'):
    for images, targets in tqdm(train_dataloader, desc='Inner'):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.float(), targets.float())
        loss.backward()
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))