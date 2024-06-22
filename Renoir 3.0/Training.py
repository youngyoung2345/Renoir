import os
from google.colab import drive

drive.mount('/content/drive')

project_folder = '/content/drive/MyDrive/Project3'

image = []
label = []

for subdir, _, files in os.walk(project_folder):
    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(subdir, file)
            image.append(image_path)
            
            label_name = os.path.basename(subdir)
            label.append(label_name)
            

import Preprocessing
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split 

BATCH_SIZE = 64

X_train, X_test, y_train, y_test = train_test_split(image, label, test_size = 0.33, random_state = 425)

train_dataset = Preprocessing.CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataset = Preprocessing.CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)


import time
import torch
import torch.nn as nn

from Backbone import VGG19

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

EPOCH = 50

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL = VGG19(in_dim = 4096, out_dim = 20).to(DEVICE)
LOSS = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.adam(MODEL.parameters(), lr = 0.01)


def compute_accuracy_and_loss(model, data_loader, flag, device):
    loss = 0
    
    for batch_idx, (image, label) in enumerate(data_loader):
        image = image.to(device)
        probability = model(image)
    
        loss += LOSS(probability, label)
        
        print (f'Epoch: {epoch:03d} | '
               f'Batch {batch_idx:03d}/{len(train_loader):03d} |')
    
    writer.add_scalar(flag+'loss')
    writer.add_scalar(flag+'accauracy')



start_time = time.time()

for epoch in range(EPOCH):
    MODEL.train()
    
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        probability = MODEL(image)
        
        loss = LOSS(probability, label)
        
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        
        print (f'Epoch: {epoch:03d} | '
               f'Batch {batch_idx:03d}/{len(train_loader):03d} |')
    MODEL.eval()

    with torch.no_grad():
        compute_accuracy_and_loss(MODEL, train_loader, 'training', device=DEVICE)
        compute_accuracy_and_loss(MODEL, test_loader,'test', device=DEVICE)
    
    if epoch%5 == 0:
        PATH = f"/content/drive/MyDrive/Renoir_3.0_{epoch}.pth"  # 에포크별로 파일 이름 지정
        torch.save(MODEL.state_dict(), PATH)
        
    print(f'Epoch: {epoch:03d}/{EPOCH:0}')
    
    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')
    
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')