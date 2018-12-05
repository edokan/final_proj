import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True

import dataset
from models.model import *
from models.ResNet import *

# Parameters to tweak
num_epochs = 100
output_period = 100
batch_size = 100


def run():

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NNmodel()
    model = model.to(device)

    # get the loaders from the dataset.py file
    train_loader, val_loader, test_loader = dataset.get_data_loaders(batch_size)
    # debugging checks
    print(len(train_loader), batch_size)
    # assert(len(train_loader) == batch_size)
    # assert(len(val_loader) == batch_size)
    # assert(len(test_loader) == batch_size)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(params=model.parameters(), weight_decay=.01, lr=.001)


    epoch = 1
    while epoch <= num_epochs:
        running_loss = 0.0


        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' % (
                    epoch, batch_num*1.0/num_train_batches,
                    running_loss/output_period
                    ))
                running_loss = 0.0
                gc.collect()

        gc.collect()
        # save after every epoch

        # evaluation mode
        model.eval()
        # TODO: training and validation error
        print("training accuracy:")
        print(classification_accuracy(model, train_loader, device))
        print("validation accuracy:")
        print(classification_accuracy(model, val_loader, device))

        gc.collect()
        epoch += 1
    torch.save(model.state_dict(), "models/model.%d" % epoch)
    
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






if __name__ == "__main__":
    print("Starting Training")
    run()
    print("Finished Training!")