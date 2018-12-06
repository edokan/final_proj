import torch, torchvision
from torch import nn,optim
from torch.autograd import Variable
import numpy
import dataset
from models.model import *
from torchvision import transforms
from PIL import Image


chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
lookup_dict = {i: chars[i] for i in range(len(chars))}
root_path = 'output_files/'
root_model = 'models/saved/modelv4.101'

transformation = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])


def write_output_file(segmented_input, name):
    # set up the model
    model = model = NNmodel()
    model.load_state_dict(torch.load(root_model))
    model.eval()

    output_file = open(root_path + name + '.txt', 'w+')
    for line in segmented_input:
        for word in line:
            for letter in word:
                output = model(transformation(letter).unsqueeze(0))
                _, top = torch.max(output, dim=1)
                output_file.write(lookup_dict[top[0].item()])
            output_file.write(' ')
        output_file.write('\n')
write_output_file(i, 'x')