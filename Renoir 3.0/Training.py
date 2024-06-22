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
            

from torch.utils.data import DataLoader
from Preprocessing import CustomDataset
from sklearn.model_selection import train_test_split 

BATCH_SIZE = 128

X_train, X_test, y_train, y_test = train_test_split(image, label, test_size = 0.33, random_state = 425)

train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True)



import time
import torch
import torch.nn as nn

from Model import Recognizer

EPOCH = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL = Recognizer().to(DEVICE)
LOSS = nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr = 0.01)


def compute_accuracy_and_loss(device, model, data_loader):
    loss, example_num, correct_num = 0, 0, 0
    
    for batch_idx, (image, label) in enumerate(data_loader):
        image = image.to(device)
        probability = model(image)
        
        #Calculate loss using CrossEntropy
    
        loss += LOSS(probability, label)
        
        #Calculate accuracy
        
        _, true_index = torch.max(label, 1)
        _, predict_index = torch.max(probability, 1)
        
        example_num += true_index.size(0)
        correct_num += (true_index == predict_index).sum
        
        print (f'Epoch: {epoch:03d} | '
               f'Batch {batch_idx:03d}/{len(train_loader):03d} |'
               f'Loss: {loss:03f}')
        
    return loss/example_num, correct_num/example_num*100


def save_weight(model, path):
    torch.save(model.state_dict(), path)

'''

Visualizing model architecture by using tensorboard Library

'''

from torch.utils.tensorboard import SummaryWriter

image_for_visualization, label_for_visualization = train_dataset[0]

writer = SummaryWriter()
writer.add_graph(MODEL, image_for_visualization.unsqueeze(0))

'''

Training

'''

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
               f'Batch: {batch_idx:03d}/{len(train_loader):03d} |'
               f'Loss: {loss:03f}')
        
    MODEL.eval()
    with torch.no_grad():
        train_loss, train_acc = compute_accuracy_and_loss(DEVICE, MODEL, train_loader)
        test_loss, test_acc = compute_accuracy_and_loss(DEVICE, MODEL, test_loader)
        
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        writer.flush()
    
    if epoch%10 == 0:
        save_weight(MODEL.VGG19, f"/content/drive/MyDrive/VGG19_{epoch}.pth")
        save_weight(MODEL.ArcFace, f"/content/drive/MyDrive/ArcFace_{epoch}.pth")
        
    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')

elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')

writer.close()

