import torch, torchvision
from torch import nn,optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy
import dataset
from models.model import *

# T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# emnist_data = torchvision.datasets.EMNIST('emnist_data1', 'byclass', transform=T,download=False)

# emnist_dataloader = torch.utils.data.DataLoader(emnist_data,batch_size=128)
# for i, (images,labels) in enumerate(emnist_dataloader):
#     print(i, images.size(), len(labels))
#     break

# batch = next(iter(emnist_dataloader))
# # print(batch)
# # print(len(batch[1]))
# samples = batch[0][127:]
# for i, sample in enumerate(samples):
#     plt.subplot(1,1,i+1)
#     plt.title(batch[1][127])
#     plt.imshow(sample.numpy().reshape((28, 28)))
#     plt.axis('off')
# plt.show()
# class Mnet(nn.Module):
#     def __init__(self):
#         super(Mnet,self).__init__()
#         self.linear1 = nn.Linear(28*28,100)
#         self.linear2 = nn.Linear(100,80)
#         self.final_linear = nn.Linear(80,62)
        
#         self.relu = nn.ReLU()
        
#     def forward(self,images):
#         x = images.view(-1,28*28)
#         x = self.relu(self.linear1(x))
#         x = self.relu(self.linear2(x))
#         x = self.final_linear(x)
#         return x

# model = Mnet()
# cec_loss = nn.CrossEntropyLoss()
# params = model.parameters()
# optimizer = optim.Adam(params=params,lr=0.001)

# n_epochs=3
# n_iterations=0

# vis=Visdom()
# vis_window=vis.line(np.array([0]),np.array([0]))
# for e in range(n_epochs):
#     for i,(images,labels) in enumerate(emnist_dataloader):
#         images = Variable(images)
#         labels = Variable(labels)
#         output = model(images)
        
#         model.zero_grad()
#         loss = cec_loss(output,labels)
#         loss.backward()
        
#         optimizer.step()
#         running_loss += loss.item()
        
#         n_iterations+=1
#         if n_iterations%500 == 0:
#             print(str(5453./n_iterations) + "%")
#             print(running_loss)
    # print(n_epochs)
    # print(loss)

def classification_accuracy(model, data_loader, device):
    correct = 0
    examples = 0
    for (inputs, labels) in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(labels)
        # print(labels.size())
        outputs = model(inputs)
        # print(outputs)
        # print(outputs.size())
        _, top = torch.max(outputs, dim=1)
        # print(top)
        # print(top.size())
        # exit()
        for i in range(len(outputs)):
            if labels[i] == top[i]:
                correct += 1
        examples += len(outputs)
    return float(correct)/examples 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 100
train_loader, val_loader, test_loader = dataset.get_data_loaders(batch_size)
model = NNmodel()
model = model.to(device)
model.load_state_dict(torch.load('models/model.101'))
model.eval()
print(classification_accuracy(model, test_loader, device))



