import torch
from torchvision import datasets, transforms

# root folders for our image data
root_train = './data/train'
root_validation = './data/val'
root_test = './data/test'

# Transformation to turn images into tensors and normalize this tensor image
transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# Get data split into training, validation, and testing sets
training_data = datasets.ImageFolder(root=root_train, transform=transformation)
validation_data = datasets.ImageFolder(root=root_validation, transform=transformation)
test_data = datasets.ImageFolder(root=root_test, transform=transformation)


def get_data_loaders(batch_size):
    """
    Returns the data loaders to be used for training, validating, and
    testing.


    inputs:
    batch_size - batch size for the data to be used in the neural net

    returns:
    train_loader - DataLoader of training data
    val_loader - Dataloader of Validation data
    test_loader - DataLoader of testing data
    
    """
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return (train_loader, val_loader, test_loader)

