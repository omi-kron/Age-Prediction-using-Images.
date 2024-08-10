import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import timm
from PIL import Image
from tempfile import TemporaryDirectory
from torchvision.io import read_image
from torchvision.transforms import v2

import pandas as pd
cudnn.benchmark = True
plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset'

# HYPS
batch_size = 8
LR = 1e-3

class FaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        return image, label
    

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    labels = labels.reshape(-1, 1)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_loss > best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {best_loss:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {outputs[j]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
    
def inference(model):
    model.eval()
    res = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_face_dataloader):
            inputs = inputs.to(device)

            outputs = model(inputs)
            res.append(outputs.item())
    return res

train_data_csv = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train.csv'
train_data_dir = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train'
test_data_dir = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/test'
submission_data_csv = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv'
tsfm = v2.Compose([
    v2.Resize(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
face_data = FaceDataset(train_data_csv, train_data_dir, transform=tsfm)


inp, cls = face_data[0]
out = torchvision.utils.make_grid(inp)

train_face_data, val_face_data = torch.utils.data.random_split(face_data, lengths=[0.9, 0.1])

train_face_dataloader = DataLoader(train_face_data, batch_size=batch_size, shuffle=True, num_workers=1)
val_face_dataloader = DataLoader(val_face_data, batch_size=batch_size, shuffle=False, num_workers=1)

datasets = {'train':train_face_data, 'val':val_face_data}
dataloaders = {'train':train_face_dataloader, 'val':val_face_dataloader}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

for inputs, labels in dataloaders['train']:
    print(labels.reshape(-1, 1).shape)
    break

model_timm = timm.create_model('tf_efficientnet_b4.ns_jft_in1k', pretrained=True)

num_ftrs = model_timm.classifier.in_features
model_timm.classifier = nn.Linear(num_ftrs, 1)
model_timm = model_timm.to(device)
print(model_timm)
criterion = nn.L1Loss()
optimizer_ft = optim.Adam(model_timm.parameters(), lr=LR)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_timm = train_model(model_timm, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=35)

test_face_data = FaceDataset(submission_data_csv, test_data_dir, transform=tsfm)
test_face_dataloader = DataLoader(test_face_data, batch_size=1, shuffle=False)

res = inference(model_timm)

sub_csv = pd.read_csv(submission_data_csv)
sub_csv['age'] = res

sub_csv.to_csv('submission.csv',index=False)

visualize_model(model_timm)